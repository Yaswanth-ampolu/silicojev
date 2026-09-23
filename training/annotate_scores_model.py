#!/usr/bin/env python3
"""Annotate RTL-BenchLS 3q records with `risk` and `urgency` — model-judged.

This driver deliberately contains **no** rule, regex, keyword map, template or
default that can select a label or a probability. Its only jobs are plumbing and
checking:

* select the next batch of three unannotated records;
* print each record's case evidence for a model to read and judge;
* write the model's judgments to a resumable progress file and rebuild the output;
* validate structure — ids, question/gold shape, canonical criteria, probability
  sums and ranges, and that the model's own `label` equals the argmax it supplied.

The scores are produced by the model reading the evidence, one batch at a time.
Every score is written `codex_pseudo_unverified` with `defensible: false`.

Flow:
    python3 training/annotate_scores_model.py --next
    # model reads evidence, writes a judgments JSON file
    python3 training/annotate_scores_model.py --apply /tmp/batch.json
    ...
    python3 training/annotate_scores_model.py --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "dataset/converted/rtl_benchls"
INPUTS = ("records_3q.jsonl", "records_3q_unverified.jsonl")
MERGED_5Q = REPO_ROOT / "dataset/normalized/merged_5q/all.jsonl"

PROGRESS = OUT_DIR / "distillation_progress.jsonl"
OUTPUT = OUT_DIR / "records_5q_distilled_pseudo_unverified.jsonl"
REPORT = OUT_DIR / "distillation_report.json"

LABEL_SOURCE = "codex_pseudo_unverified"
GENERATED_BY = "training/annotate_scores_model.py"
LEVELS = ("0", "1", "2", "3")
BASE_QUESTIONS = ("next_action", "root_cause_type", "evidence_sufficient")
SCORE_QUESTIONS = ("risk", "urgency")
BATCH_SIZE = 3
EVIDENCE_FIELDS = ("issue_title", "issue_body", "issue_labels", "failure_context")


def load_canonical() -> tuple[dict, dict]:
    """The canonical risk/urgency question, read from the merged five-question set."""
    for line in MERGED_5Q.open():
        if not line.strip():
            continue
        questions = json.loads(json.loads(line)["questions"])
        if "risk" in questions and "urgency" in questions:
            return questions["risk"], questions["urgency"]
    sys.exit("no five-question record found in merged_5q/all.jsonl")


def load_inputs() -> list[dict]:
    """Both 3q files, deduplicated by id, input order preserved."""
    rows, seen = [], set()
    for name in INPUTS:
        path = OUT_DIR / name
        if not path.is_file():
            sys.exit(f"missing input: {path}")
        for line in path.open():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            rows.append(record)
    return rows


def tier_of(record: dict, trusted_ids: set[str]) -> str:
    return "trusted_exploratory_3q" if record["id"] in trusted_ids else "exploratory_unverified_3q"


def load_progress() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if PROGRESS.exists():
        for line in PROGRESS.open():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def evidence_of(record: dict, tier: str) -> dict:
    """Everything the model should read to judge this record — nothing repaired."""
    state = json.loads(record["state"])
    gold = json.loads(record["gold"])
    action = gold.get("next_action") or {}
    return {
        "id": record["id"],
        "tier": tier,
        "trust_flags": record["provenance"].get("trust_flags") or [],
        "repository": state.get("repository"),
        "affected_files": state.get("affected_files"),
        "report": {f: state.get(f) for f in EVIDENCE_FIELDS},
        "recommended_action": {
            "probabilities": action.get("probabilities"),
            "label_source": action.get("label_source"),
        },
        "existing_labels": {
            "next_action": action.get("label"),
            "root_cause_type": (gold.get("root_cause_type") or {}).get("label"),
            "evidence_sufficient": (gold.get("evidence_sufficient") or {}).get("label"),
        },
    }


def argmax_level(probs: dict) -> str:
    """Highest-probability level, ties broken toward the lower level."""
    return max(LEVELS, key=lambda k: (probs[k], -int(k)))


def validate_judgment(judgment: dict, record: dict) -> list[str]:
    """Structural checks only. The model chose the numbers; this only checks them."""
    problems = []
    rid = record["id"]
    for qid in SCORE_QUESTIONS:
        entry = judgment.get(qid)
        if not isinstance(entry, dict):
            problems.append(f"{rid}/{qid}: judgment missing")
            continue
        probs = entry.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(LEVELS):
            problems.append(f"{rid}/{qid}: probabilities keys must be {list(LEVELS)}")
            continue
        vals = []
        for k in LEVELS:
            v = probs[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                problems.append(f"{rid}/{qid}: probability {k}={v!r} is not a non-negative number")
            else:
                vals.append(float(v))
        if len(vals) != 4:
            continue
        if abs(sum(vals) - 1.0) > 1e-6:
            problems.append(f"{rid}/{qid}: probabilities sum to {sum(vals)}")
        label = entry.get("label")
        if label not in LEVELS:
            problems.append(f"{rid}/{qid}: label {label!r} is not a level")
        elif label != argmax_level({k: float(probs[k]) for k in LEVELS}):
            problems.append(f"{rid}/{qid}: label {label!r} is not the argmax of the given vector "
                            f"(argmax {argmax_level({k: float(probs[k]) for k in LEVELS})!r})")
        conf = entry.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0.0 <= float(conf) <= 1.0:
            problems.append(f"{rid}/{qid}: confidence {conf!r} is not in [0, 1]")
        if not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip():
            problems.append(f"{rid}/{qid}: rationale is empty")
        ev = entry.get("evidence")
        if not isinstance(ev, list) or not any(str(x).strip() for x in ev):
            problems.append(f"{rid}/{qid}: evidence list is empty")
    return problems


def build_row(record: dict, judgment: dict, batch: int, tier: str,
              risk_q: dict, urgency_q: dict) -> dict:
    questions = json.loads(record["questions"])
    gold = json.loads(record["gold"])
    questions["risk"], questions["urgency"] = risk_q, urgency_q
    annotation = {}
    for qid in SCORE_QUESTIONS:
        j = judgment[qid]
        probs = {k: round(float(j["probabilities"][k]), 6) for k in LEVELS}
        label = j["label"]
        score = round(sum(int(k) * v for k, v in probs.items()), 4)
        gold[qid] = {
            "probabilities": probs,
            "score": score,
            "label": label,
            "confidence": round(float(j["confidence"]), 4),
            "label_source": LABEL_SOURCE,
            "defensible": False,
        }
        audit = {
            "rationale": j["rationale"],
            "evidence": [str(x) for x in j["evidence"]],
        }
        for key in ("missing", "counter_evidence", "basis", "insufficient_evidence"):
            if key in j:
                audit[key] = j[key]
        annotation[qid] = audit

    row = dict(record)
    row["questions"] = json.dumps(questions, ensure_ascii=False, sort_keys=True)
    row["gold"] = json.dumps(gold, ensure_ascii=False, sort_keys=True)
    row["score_annotation"] = {
        "generated_by": GENERATED_BY,
        "method": ("model case-by-case judgment over the pre-repair state; the three existing "
                   "labels are context, never the source of the scores; no script selected any "
                   "label or probability"),
        "batch": batch,
        "source_tier": tier,
        "source_trust_flags": record["provenance"].get("trust_flags") or [],
        "label_source": LABEL_SOURCE,
        "defensible": False,
        "provenance_note": ("model-generated exploratory estimate; not verified, not gold, and not "
                            "to be promoted to a trusted label"),
        **annotation,
    }
    return row


def rebuild_output(done: dict[str, dict], records: list[dict]) -> None:
    rows = [done[r["id"]] for r in records if r["id"] in done]
    OUTPUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def cmd_next(args, records, done, trusted_ids, risk_q, urgency_q) -> int:
    todo = [r for r in records if r["id"] not in done]
    batch = todo[: args.batch_size]
    print(json.dumps({
        "progress": f"{len(done)}/{len(records)}",
        "batch_index": len(done) // args.batch_size + 1,
        "remaining": len(todo),
        "canonical_criteria": {"risk": risk_q, "urgency": urgency_q},
        "batch": [evidence_of(r, tier_of(r, trusted_ids)) for r in batch],
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_apply(args, records, done, trusted_ids, risk_q, urgency_q) -> int:
    judgments = json.loads(Path(args.apply).read_text())
    if not isinstance(judgments, list):
        sys.exit("judgments file must be a JSON list")
    by_id = {j["id"]: j for j in judgments}
    todo = [r for r in records if r["id"] not in done]
    expected = [r["id"] for r in todo[: args.batch_size]]
    if [j.get("id") for j in judgments] != expected:
        sys.exit(f"judgments ids {[j.get('id') for j in judgments]} != next batch {expected}")

    source_by_id = {r["id"]: r for r in records}
    problems = []
    for rid in expected:
        problems += validate_judgment(by_id[rid], source_by_id[rid])
    if problems:
        print(json.dumps({"problems": problems}, indent=2))
        return 1

    batch = len(done) // args.batch_size + 1
    with PROGRESS.open("a") as sink:
        for rid in expected:
            record = source_by_id[rid]
            row = build_row(record, by_id[rid], batch, tier_of(record, trusted_ids),
                            risk_q, urgency_q)
            done[rid] = row
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()
    rebuild_output(done, records)
    print(json.dumps({"batch": batch, "applied": expected,
                      "progress": f"{len(done)}/{len(records)}"}, indent=2))
    return 0


def cmd_report(args, records, done, trusted_ids, risk_q, urgency_q) -> int:
    raw_before = {n: hashlib.sha256((OUT_DIR / n).read_bytes()).hexdigest() for n in INPUTS}
    rebuild_output(done, records)

    source_by_id = {r["id"]: r for r in records}
    problems, seen = [], Counter()
    risk_levels, urg_levels = Counter(), Counter()
    risk_conf, urg_conf = [], []
    insufficient = []
    for row in (done[r["id"]] for r in records if r["id"] in done):
        rid = row["id"]
        seen[rid] += 1
        src = source_by_id[rid]
        if row["state"] != src["state"]:
            problems.append(f"{rid}: state changed")
        for field in ("source", "source_group", "provenance", "outcome"):
            if json.dumps(row.get(field), sort_keys=True) != json.dumps(src.get(field), sort_keys=True):
                problems.append(f"{rid}: {field} changed")
        q, g = json.loads(row["questions"]), json.loads(row["gold"])
        if set(q) != set(BASE_QUESTIONS + SCORE_QUESTIONS):
            problems.append(f"{rid}: question ids are {sorted(q)}")
        if set(g) != set(q):
            problems.append(f"{rid}: gold ids do not match question ids")
        for qid in BASE_QUESTIONS:
            if q.get(qid) != json.loads(src["questions"]).get(qid):
                problems.append(f"{rid}/{qid}: question definition changed")
            if g.get(qid) != json.loads(src["gold"]).get(qid):
                problems.append(f"{rid}/{qid}: gold entry changed")
        if q.get("risk") != risk_q or q.get("urgency") != urgency_q:
            problems.append(f"{rid}: score question wording differs from canonical")
        for qid in SCORE_QUESTIONS:
            entry = g[qid]
            probs = entry["probabilities"]
            if set(probs) != set(LEVELS):
                problems.append(f"{rid}/{qid}: probability keys {sorted(probs)}")
                continue
            if any(v < 0 for v in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6:
                problems.append(f"{rid}/{qid}: probabilities invalid ({sum(probs.values())})")
            if abs(entry["score"] - sum(int(k) * v for k, v in probs.items())) > 1e-6:
                problems.append(f"{rid}/{qid}: score != expectation")
            if entry["label"] != argmax_level(probs):
                problems.append(f"{rid}/{qid}: label is not the argmax")
            if entry["label_source"] != LABEL_SOURCE or entry["defensible"] is not False:
                problems.append(f"{rid}/{qid}: not marked pseudo/indefensible")
            if not row["score_annotation"].get(qid, {}).get("rationale"):
                problems.append(f"{rid}/{qid}: no rationale")
        risk_levels[g["risk"]["label"]] += 1
        urg_levels[g["urgency"]["label"]] += 1
        risk_conf.append(g["risk"]["confidence"])
        urg_conf.append(g["urgency"]["confidence"])
        if row["score_annotation"]["urgency"].get("insufficient_evidence"):
            insufficient.append(rid)

    duplicated = [i for i, n in seen.items() if n > 1]
    if duplicated:
        problems.append(f"duplicate ids: {duplicated[:5]}")
    missing = [r["id"] for r in records if r["id"] not in seen]
    if missing:
        problems.append(f"missing ids: {missing[:5]}")
    if raw_before != {n: hashlib.sha256((OUT_DIR / n).read_bytes()).hexdigest() for n in INPUTS}:
        problems.append("an input file changed during annotation")

    # comparison only — prior labels were never read while judging
    disagreements = []
    prior_files = ("records_5q_pseudo_unverified.jsonl", "records_5q.jsonl")
    for prior_name in prior_files:
        prior_path = OUT_DIR / prior_name
        if not prior_path.is_file():
            continue
        for line in prior_path.open():
            if not line.strip():
                continue
            prior = json.loads(line)
            mine = done.get(prior["id"])
            if mine is None:
                continue
            pg, mg = json.loads(prior["gold"]), json.loads(mine["gold"])
            for qid in SCORE_QUESTIONS:
                if qid in pg and pg[qid].get("label") != mg[qid]["label"]:
                    disagreements.append({"id": prior["id"], "question": qid,
                                          "prior_file": prior_name,
                                          "prior_label": pg[qid].get("label"),
                                          "prior_label_source": pg[qid].get("label_source"),
                                          "new_label": mg[qid]["label"]})

    batches = Counter(row["score_annotation"]["batch"] for row in done.values())
    report = {
        "generated_by": GENERATED_BY,
        "method": "model case-by-case judgment; no rule or default selected any label or probability",
        "inputs": list(INPUTS),
        "input_records": len(records),
        "completed_records": len(done),
        "missing_records": len(missing),
        "batch_size": args.batch_size,
        "completion_by_batch": {str(k): v for k, v in sorted(batches.items())},
        "tiers": {
            "trusted_exploratory_3q": sum(1 for r in done.values()
                                          if r["score_annotation"]["source_tier"] == "trusted_exploratory_3q"),
            "exploratory_unverified_3q": sum(1 for r in done.values()
                                             if r["score_annotation"]["source_tier"] == "exploratory_unverified_3q"),
            "by_trust_flag": dict(Counter(f for r in done.values()
                                          for f in r["score_annotation"]["source_trust_flags"])),
        },
        "risk": {"distribution": dict(sorted(risk_levels.items())),
                 "confidence_mean": round(sum(risk_conf) / len(risk_conf), 4) if risk_conf else None,
                 "confidence_min": min(risk_conf) if risk_conf else None,
                 "confidence_max": max(risk_conf) if risk_conf else None},
        "urgency": {"distribution": dict(sorted(urg_levels.items())),
                    "confidence_mean": round(sum(urg_conf) / len(urg_conf), 4) if urg_conf else None,
                    "confidence_min": min(urg_conf) if urg_conf else None,
                    "confidence_max": max(urg_conf) if urg_conf else None,
                    "insufficient_evidence": len(insufficient)},
        "insufficient_evidence_cases": insufficient,
        "disagreements_with_prior_annotations": {
            "count": len(disagreements),
            "by_prior_file": dict(Counter(d["prior_file"] for d in disagreements)),
            "all": disagreements,
            "note": ("prior labels were never read while judging; they are compared here only, after "
                     "the fact, and never copied into the output"),
        },
        "validation": {"problem_count": len(problems), "problems": problems[:40]},
        "output_file": str(OUTPUT.relative_to(REPO_ROOT)),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"input": len(records), "completed": len(done), "missing": len(missing),
                      "risk": report["risk"]["distribution"],
                      "urgency": report["urgency"]["distribution"],
                      "insufficient_urgency_evidence": len(insufficient),
                      "disagreements_with_prior": len(disagreements),
                      "validation_problems": len(problems), "detail": problems[:10]}, indent=2))
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--next", action="store_true", help="print the next batch of evidence")
    ap.add_argument("--apply", metavar="FILE", help="write model judgments from FILE")
    ap.add_argument("--report", action="store_true", help="rebuild output + write the report")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--reset", action="store_true", help="discard progress first")
    args = ap.parse_args()

    if args.reset and PROGRESS.exists():
        PROGRESS.unlink()

    records = load_inputs()
    tiers = json.loads((OUT_DIR / "usage_sets.json").read_text())
    trusted_ids = set(tiers.get("trusted_exploratory_3q") or [])
    risk_q, urgency_q = load_canonical()
    done = load_progress()
    unknown = set(done) - {r["id"] for r in records}
    if unknown:
        sys.exit(f"progress holds ids that are not inputs: {sorted(unknown)[:3]}")

    if args.next:
        return cmd_next(args, records, done, trusted_ids, risk_q, urgency_q)
    if args.apply:
        return cmd_apply(args, records, done, trusted_ids, risk_q, urgency_q)
    if args.report:
        return cmd_report(args, records, done, trusted_ids, risk_q, urgency_q)
    print(json.dumps({"inputs": len(records), "completed": len(done),
                      "remaining": len(records) - len(done)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
