#!/usr/bin/env python3
"""Driver for the Fixbench-RTL risk/urgency distillation.

This driver contains NO scoring logic. It only:
  * selects the next ordered batch of three records and prints their evidence,
  * validates the model's hand-written judgments from a JSON file
    (schema, probability validity/normalisation, label == argmax with ties
    toward the lower level, score == expectation), writes them out,
  * rebuilds the output file and the report.

Every label, probability and confidence comes from the judgment JSON that the
model writes after reading each batch. The driver never chooses a label or a
probability; the only arithmetic it performs is the expectation of a
distribution the model supplied, plus validation of the model's stated label
against that distribution's argmax.

Usage:
  python3 training/distill_fixbench_scores.py --next
  python3 training/distill_fixbench_scores.py --apply <judgments.json>
  python3 training/distill_fixbench_scores.py --report
  python3 training/distill_fixbench_scores.py --status
"""
import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXDIR = os.path.join(ROOT, "dataset", "converted", "fixbench_rtl")
RECORDS_3Q = os.path.join(FIXDIR, "records_3q.jsonl")
USAGE = os.path.join(FIXDIR, "usage_sets.json")
CANONICAL = os.path.join(ROOT, "dataset", "normalized", "merged_5q", "all.jsonl")
PRIOR_RUBRIC = os.path.join(FIXDIR, "records_5q_pseudo_unverified.jsonl")
OUT = os.path.join(FIXDIR, "records_5q_distilled_pseudo_unverified.jsonl")
PROGRESS = os.path.join(FIXDIR, "distillation_progress.jsonl")
REPORT = os.path.join(FIXDIR, "distillation_report.json")
NORMALIZED = os.path.join(ROOT, "dataset", "normalized")

BATCH_SIZE = 3
LEVELS = ["0", "1", "2", "3"]
PSEUDO = "codex_pseudo_unverified"


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_3q():
    return load_jsonl(RECORDS_3Q)


def load_progress():
    if not os.path.exists(PROGRESS):
        return []
    return load_jsonl(PROGRESS)


def canonical_criteria():
    """Read the canonical risk/urgency question definitions from merged_5q."""
    with open(CANONICAL) as f:
        rec = json.loads(f.readline())
    q = rec["questions"]
    if isinstance(q, str):
        q = json.loads(q)
    return {"risk": q["risk"], "urgency": q["urgency"]}


def record_key(rec):
    st = rec["state"]
    if isinstance(st, str):
        st = json.loads(st)
    return (st["source_index"], st["case_id"], rec["id"])


def next_batch(records, progress):
    done = {(json.loads(r["state"])["case_id"]) for r in progress}
    todo = [r for r in records if json.loads(r["state"])["case_id"] not in done]
    todo.sort(key=lambda r: json.loads(r["state"])["source_index"])
    return todo[:BATCH_SIZE], done


def cmd_next(args):
    records = load_3q()
    progress = load_progress()
    batch, done = next_batch(records, progress)
    canon = canonical_criteria()
    print(json.dumps({
        "completed": len(done),
        "remaining": len(records) - len(done),
        "batch_index": len(done) // BATCH_SIZE + 1,
        "canonical_risk_criteria": canon["risk"]["criteria"],
        "canonical_urgency_criteria": canon["urgency"]["criteria"],
        "records": [evidence_view(r) for r in batch],
    }, indent=1))


def evidence_view(rec):
    st = json.loads(rec["state"])
    g = json.loads(rec["gold"])
    p = rec["provenance"]
    o = rec["outcome"]
    return {
        "id": rec["id"],
        "source_index": st["source_index"],
        "case_id": st["case_id"],
        "bug_description": st["bug_description"],
        "next_action_distribution": g["next_action"]["probabilities"],
        "next_action_label_source": g["next_action"].get("label_source"),
        "root_cause_type": g["root_cause_type"]["probabilities"],
        "replay_class": o.get("replay_class"),
        "repair_evidence_verified": o.get("repair_evidence_verified"),
        "deciding_backend": o.get("deciding_backend"),
        "base_replay_verdict": o.get("base_replay_verdict"),
        "trust_flags": p.get("trust_flags"),
        "bug_family": p.get("bug_family"),
        "quality": p.get("quality"),
        "label_source": p.get("label_source"),
        "buggy_replay_status": st.get("replay", {}).get("status"),
        "buggy_compiler_output": st.get("replay", {}).get("compiler_output", "")[:600],
        "buggy_simulation_output": st.get("replay", {}).get("simulation_output", "")[:600],
    }


def validate_score(name, gold, judgment):
    """Validate one model-supplied score. No label/probability is chosen here."""
    errs = []
    probs = judgment.get("probabilities")
    label = judgment.get("label")
    conf = judgment.get("confidence")
    if not isinstance(probs, dict):
        return ["%s.probabilities must be an object" % name], None, None
    if list(probs.keys()) != LEVELS and set(probs.keys()) != set(LEVELS):
        errs.append("%s.probabilities keys must be exactly %s" % (name, LEVELS))
    for k, v in probs.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            errs.append("%s.probabilities[%s] not numeric" % (name, k))
        elif v < 0:
            errs.append("%s.probabilities[%s] negative" % (name, k))
    if errs:
        return errs, None, None
    total = sum(float(probs[k]) for k in LEVELS)
    if abs(total - 1.0) > 0.002:
        errs.append("%s.probabilities sum %.6f != 1" % (name, total))
    # expectation
    score = round(sum(i * float(probs[LEVELS[i]]) for i in range(4)), 4)
    # argmax, ties toward the lower level
    best = max(LEVELS, key=lambda k: (round(float(probs[k]), 6), -int(k)))
    if str(label) not in LEVELS:
        errs.append("%s.label must be one of %s" % (name, LEVELS))
    elif str(label) != best:
        errs.append("%s.label '%s' != argmax '%s'" % (name, label, best))
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0.0 <= float(conf) <= 1.0):
        errs.append("%s.confidence must be in [0,1]" % name)
    if errs:
        return errs, None, None
    gold_score = {
        "probabilities": {k: float(probs[k]) for k in LEVELS},
        "score": score,
        "label": str(label),
        "confidence": float(conf),
        "label_source": PSEUDO,
        "defensible": False,
    }
    return [], gold_score, score


def build_record(rec, canon, judgment, batch_index):
    st = rec["state"]
    if isinstance(st, str):
        st = json.loads(st)
    base_q = rec["questions"]
    base_q = json.loads(base_q) if isinstance(base_q, str) else copy.deepcopy(base_q)
    base_g = rec["gold"]
    base_g = json.loads(base_g) if isinstance(base_g, str) else copy.deepcopy(base_g)

    q = copy.deepcopy(base_q)
    q["risk"] = copy.deepcopy(canon["risk"])
    q["urgency"] = copy.deepcopy(canon["urgency"])
    g = copy.deepcopy(base_g)

    out = copy.deepcopy(rec)
    out["questions"] = json.dumps(q)
    score_ann = {"batch": batch_index, "label_source": PSEUDO, "defensible": False}
    for name in ("risk", "urgency"):
        j = judgment[name]
        errs, gold_score, score = validate_score(name, g.get(name), j)
        if errs:
            raise ValueError("; ".join(errs))
        g[name] = gold_score
        score_ann[name] = {
            "rationale": j.get("rationale", ""),
            "evidence_considered": j.get("evidence_considered", []),
            "evidence_insufficient": bool(j.get("evidence_insufficient", False)),
            "label_source": PSEUDO,
            "defensible": False,
            "gold": gold_score,
        }
    out["gold"] = json.dumps(g)
    out["score_annotation"] = score_ann
    return out


def cmd_apply(args):
    records = load_3q()
    by_id = {r["id"]: r for r in records}
    by_case = {json.loads(r["state"])["case_id"]: r for r in records}
    progress = load_progress()
    done = {json.loads(r["state"])["case_id"] for r in progress}
    batch, _ = next_batch(records, progress)
    batch_cases = [r["id"] for r in batch]

    with open(args.apply) as f:
        judgments = json.load(f)
    if not isinstance(judgments, list):
        print("ERROR: judgments file must be a JSON list", file=sys.stderr)
        return 2
    batch_index = len(done) // BATCH_SIZE + 1
    canon = canonical_criteria()
    applied = []
    for j in judgments:
        cid = j.get("id")
        if cid not in batch_cases:
            if cid in done:
                print("ERROR: %s already applied (no duplicates allowed)" % cid, file=sys.stderr)
            else:
                print("ERROR: %s is not in the current batch %s" % (cid, batch_cases), file=sys.stderr)
            return 2
        rec = by_id[cid]
        if "risk" not in j or "urgency" not in j:
            print("ERROR: %s missing risk/urgency" % cid, file=sys.stderr)
            return 2
        out = build_record(rec, canon, j, batch_index)
        progress.append(out)
        applied.append(cid)

    progress.sort(key=lambda r: json.loads(r["state"])["source_index"])
    write_jsonl(PROGRESS, progress)
    write_jsonl(OUT, progress)
    print("applied %d record(s); total %d/100; batch %d" % (len(applied), len(progress), batch_index))
    return 0


def cmd_status(args):
    records = load_3q()
    progress = load_progress()
    done = {json.loads(r["state"])["case_id"] for r in progress}
    print("completed %d / %d, remaining %d" % (len(done), len(records), len(records) - len(done)))
    return 0


# ----- report & validation -------------------------------------------------

def normalized_ids():
    ids = set()
    for dirpath, _dirs, files in os.walk(NORMALIZED):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(r, dict) and str(r.get("id", "")).startswith("fixbench:"):
                            ids.add(r["id"])
            except Exception:
                continue
    return ids


def validate_points(records, out_rows, canon):
    by_case = {json.loads(r["state"])["case_id"]: r for r in records}
    seen = [json.loads(r["state"])["case_id"] for r in out_rows]
    uniq = len(seen) == len(set(seen)) == len(records) == 100 and set(seen) == set(by_case)
    pins = {"ids_once_each": uniq, "originals_unchanged": True, "question_ids_match_gold": True,
            "canonical_criteria_equal": True, "probabilities_valid_normalised": True,
            "score_equals_expectation": True, "label_equals_argmax": True,
            "pseudo_unverified_defensible_false": True, "state_byte_identical": True,
            "normalized_untouched": True}
    for r in out_rows:
        cid = json.loads(r["state"])["case_id"]
        src = by_case[cid]
        if r["state"] != src["state"]:
            pins["state_byte_identical"] = False
            pins["originals_unchanged"] = False
        oq = json.loads(src["questions"]); nq = json.loads(r["questions"])
        og = json.loads(src["gold"]); ng = json.loads(r["gold"])
        for k in oq:
            if nq.get(k) != oq[k] or ng.get(k) != og[k]:
                pins["originals_unchanged"] = False
        if set(nq.keys()) != set(ng.keys()) or set(nq.keys()) != {
                "next_action", "root_cause_type", "evidence_sufficient", "risk", "urgency"}:
            pins["question_ids_match_gold"] = False
        for k in ("risk", "urgency"):
            if nq.get(k) != canon[k]:
                pins["canonical_criteria_equal"] = False
            g = ng[k]; probs = g["probabilities"]
            if list(probs.keys()) != LEVELS or any(probs[x] < 0 for x in LEVELS) \
                    or abs(sum(probs[x] for x in LEVELS) - 1.0) > 0.002:
                pins["probabilities_valid_normalised"] = False
            else:
                exp = round(sum(i * probs[LEVELS[i]] for i in range(4)), 4)
                if abs(exp - g["score"]) > 1e-6:
                    pins["score_equals_expectation"] = False
                best = max(LEVELS, key=lambda x: (round(probs[x], 6), -int(x)))
                if str(g["label"]) != best:
                    pins["label_equals_argmax"] = False
            if g.get("label_source") != PSEUDO or g.get("defensible") is not False:
                pins["pseudo_unverified_defensible_false"] = False
    if normalized_ids() & set(by_case.keys()):
        pins["normalized_untouched"] = False
    return [{"check": k, "ok": v} for k, v in pins.items()]


def validate_all():
    records = load_3q()
    by_case = {json.loads(r["state"])["case_id"]: r for r in records}
    out_rows = load_jsonl(OUT) if os.path.exists(OUT) else []
    canon = canonical_criteria()
    errors = []
    seen = set()
    for r in out_rows:
        st = json.loads(r["state"])
        cid = st["case_id"]
        if cid in seen:
            errors.append("duplicate id %s" % cid)
        seen.add(cid)
        src = by_case.get(cid)
        if src is None:
            errors.append("unknown case %s" % cid)
            continue
        # originals unchanged
        if r["state"] != src["state"]:
            errors.append("%s: state string changed" % cid)
        oq = json.loads(src["questions"]); nq = json.loads(r["questions"])
        og = json.loads(src["gold"]); ng = json.loads(r["gold"])
        for k in oq:
            if nq.get(k) != oq[k]:
                errors.append("%s: original question %s changed" % (cid, k))
            if ng.get(k) != og[k]:
                errors.append("%s: original gold %s changed" % (cid, k))
        # five questions / golds match
        if set(nq.keys()) != set(ng.keys()):
            errors.append("%s: question ids != gold ids" % cid)
        if set(nq.keys()) != {"next_action", "root_cause_type", "evidence_sufficient", "risk", "urgency"}:
            errors.append("%s: unexpected question set %s" % (cid, sorted(nq.keys())))
        # canonical criteria
        for k in ("risk", "urgency"):
            if nq.get(k) != canon[k]:
                errors.append("%s: %s criteria != canonical" % (cid, k))
        for k in ("risk", "urgency"):
            g = ng[k]
            probs = g["probabilities"]
            if list(probs.keys()) != LEVELS:
                errors.append("%s.%s: bad level keys" % (cid, k))
                continue
            if any(probs[x] < 0 for x in LEVELS):
                errors.append("%s.%s: negative prob" % (cid, k))
            if abs(sum(probs[x] for x in LEVELS) - 1.0) > 0.002:
                errors.append("%s.%s: probs not normalised" % (cid, k))
            exp = round(sum(i * probs[LEVELS[i]] for i in range(4)), 4)
            if abs(exp - g["score"]) > 1e-6:
                errors.append("%s.%s: score != expectation" % (cid, k))
            best = max(LEVELS, key=lambda x: (round(probs[x], 6), -int(x)))
            if str(g["label"]) != best:
                errors.append("%s.%s: label != argmax" % (cid, k))
            if g.get("label_source") != PSEUDO or g.get("defensible") is not False:
                errors.append("%s.%s: not pseudo/unverified or defensible not false" % (cid, k))
    # completeness
    if len(out_rows) != len(records):
        errors.append("completeness: %d output rows for %d input records" % (len(out_rows), len(records)))
    if seen != set(by_case.keys()):
        errors.append("id set mismatch: missing=%s extra=%s" % (
            sorted(set(by_case) - seen), sorted(seen - set(by_case))))
    # normalized untouched
    leaked = normalized_ids() & set(by_case.keys())
    if leaked:
        errors.append("fixbench ids found under dataset/normalized/: %s" % sorted(leaked))
    return errors, out_rows, records


def label_counts(rows, key):
    counts = {lv: 0 for lv in LEVELS}
    conf_by_label = {lv: [] for lv in LEVELS}
    scores = []
    for r in rows:
        g = json.loads(r["gold"])[key]
        counts[str(g["label"])] += 1
        conf_by_label[str(g["label"])].append(g["confidence"])
        scores.append(g["score"])
    summary = {
        "label_counts": counts,
        "mean_confidence_by_label": {
            lv: (round(sum(v) / len(v), 4) if v else None) for lv, v in conf_by_label.items()
        },
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
    }
    return summary


def cmd_report(args):
    errors, out_rows, records = validate_all()
    canon = canonical_criteria()
    usage = json.load(open(USAGE))
    by_case = {json.loads(r["state"])["case_id"]: r for r in records}
    flagship = set(usage["replay_verified_unflagged_3q"])
    tier = {cid: ("unflagged" if rec["id"] in flagship else "flagged")
            for cid, rec in by_case.items()}

    replay_counts = {}
    trust_counts = {"unflagged": 0, "flagged": 0}
    for r in out_rows:
        st = json.loads(r["state"])
        o = r["outcome"]
        replay_counts[o.get("replay_class")] = replay_counts.get(o.get("replay_class"), 0) + 1
        trust_counts[tier[st["case_id"]]] += 1

    # insufficient evidence
    insuff_risk, insuff_urg = [], []
    for r in out_rows:
        sa = r.get("score_annotation", {})
        st = json.loads(r["state"])
        if sa.get("risk", {}).get("evidence_insufficient"):
            insuff_risk.append(st["case_id"])
        if sa.get("urgency", {}).get("evidence_insufficient"):
            insuff_urg.append(st["case_id"])

    # batches
    counts_by_batch = {}
    for r in out_rows:
        b = str(r.get("score_annotation", {}).get("batch"))
        counts_by_batch[b] = counts_by_batch.get(b, 0) + 1

    # comparison vs prior labels
    snap = None
    snap_path = os.path.join(ROOT, "training", "fixbench_distill_scratch", "prior_labels_snapshot.json")
    if os.path.exists(snap_path):
        snap = json.load(open(snap_path))
    comparison = {"available": snap is not None}
    if snap:
        for name, prior in [("vs_rejected_distilled_run", snap["rejected_distilled"]),
                            ("vs_original_rubric_pseudo", snap["original_rubric"])]:
            risk_dis, urg_dis = [], []
            for r in out_rows:
                cid = json.loads(r["state"])["case_id"]
                if cid not in prior:
                    continue
                g = json.loads(r["gold"])
                if str(g["risk"]["label"]) != str(prior[cid]["risk_label"]):
                    risk_dis.append({"id": cid, "prior": prior[cid]["risk_label"], "new": g["risk"]["label"]})
                if str(g["urgency"]["label"]) != str(prior[cid]["urgency_label"]):
                    urg_dis.append({"id": cid, "prior": prior[cid]["urgency_label"], "new": g["urgency"]["label"]})
            comparison[name] = {
                "n_compared": sum(1 for r in out_rows if json.loads(r["state"])["case_id"] in prior),
                "risk_label_disagreements": risk_dis,
                "n_risk_disagreements": len(risk_dis),
                "urgency_label_disagreements": urg_dis,
                "n_urgency_disagreements": len(urg_dis),
            }

    report = {
        "dataset": "Fixbench-RTL",
        "output_file": OUT,
        "progress_file": PROGRESS,
        "input_records": len(records),
        "completed_records": len(out_rows),
        "missing_records": len(records) - len(out_rows),
        "missing_ids": sorted(set(by_case) - {json.loads(r["state"])["case_id"] for r in out_rows}),
        "counts_by_batch": counts_by_batch,
        "counts_by_replay_class": replay_counts,
        "counts_by_trust_tier": trust_counts,
        "records_with_insufficient_risk_evidence": insuff_risk,
        "records_with_insufficient_urgency_evidence": insuff_urg,
        "n_insufficient_risk_evidence": len(insuff_risk),
        "n_insufficient_urgency_evidence": len(insuff_urg),
        "risk": label_counts(out_rows, "risk"),
        "urgency": label_counts(out_rows, "urgency"),
        "label_source": PSEUDO,
        "defensible": False,
        "validation": {
            "ok": len(errors) == 0,
            "errors": errors,
            "n_checked": len(out_rows),
            "n_input": len(records),
            "ten_point": validate_points(records, out_rows, canon),
        },
        "comparison_vs_prior_labels": comparison,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in (
        "input_records", "completed_records", "missing_records",
        "counts_by_trust_tier", "risk", "urgency",
        "n_insufficient_risk_evidence", "n_insufficient_urgency_evidence",
        "validation", "comparison_vs_prior_labels")}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--apply")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.next:
        return cmd_next(args)
    if args.apply:
        return cmd_apply(args)
    if args.report:
        return cmd_report(args)
    if args.status:
        return cmd_status(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
