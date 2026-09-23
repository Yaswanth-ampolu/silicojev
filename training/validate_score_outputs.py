#!/usr/bin/env python3
"""Independent validation of the model-judged risk/urgency score outputs.

Reads only the files on disk and re-derives every checkable fact. It does not
import the annotation driver and shares none of its logic, so a bug in the
driver cannot hide here.

    python3 training/validate_score_outputs.py rtl_benchls
    python3 training/validate_score_outputs.py fixbench_rtl

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_5Q = REPO_ROOT / "dataset/normalized/merged_5q/all.jsonl"

SOURCES = {
    "rtl_benchls": {
        "dir": REPO_ROOT / "dataset/converted/rtl_benchls",
        "inputs": ("records_3q.jsonl", "records_3q_unverified.jsonl"),
        "output": "records_5q_distilled_pseudo_unverified.jsonl",
        "prefix": "rtlbenchls:",
    },
    "fixbench_rtl": {
        "dir": REPO_ROOT / "dataset/converted/fixbench_rtl",
        "inputs": ("records_3q.jsonl",),
        "output": "records_5q_distilled_pseudo_unverified.jsonl",
        "prefix": "fixbench:",
    },
}
BASE_QUESTIONS = ("next_action", "root_cause_type", "evidence_sufficient")
SCORE_QUESTIONS = ("risk", "urgency")
LEVELS = ("0", "1", "2", "3")
CHECKS = (
    "ids_once_each", "originals_unchanged", "five_question_ids_match_gold",
    "canonical_criteria_equal", "probabilities_valid", "score_equals_expectation",
    "label_equals_argmax_low_tie", "pseudo_unverified_defensible_false",
    "state_byte_identical", "provenance_and_audit_present", "normalized_dir_untouched",
)


def canonical() -> tuple[dict, dict]:
    for line in MERGED_5Q.open():
        if line.strip():
            q = json.loads(json.loads(line)["questions"])
            if "risk" in q and "urgency" in q:
                return q["risk"], q["urgency"]
    sys.exit("canonical score questions not found in merged_5q")


def argmax_level(probs: dict) -> str:
    return max(LEVELS, key=lambda k: (probs[k], -int(k)))


def load_source(cfg) -> dict[str, dict]:
    rows, seen = {}, set()
    for name in cfg["inputs"]:
        for line in (cfg["dir"] / name).open():
            if line.strip():
                r = json.loads(line)
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows[r["id"]] = r
    return rows


class Report:
    def __init__(self) -> None:
        self.ok = {name: True for name in CHECKS}
        self.problems: list[str] = []

    def fail(self, check: str, msg: str) -> None:
        self.ok[check] = False
        self.problems.append(f"[{check}] {msg}")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in SOURCES:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(SOURCES)}}}")
    cfg = SOURCES[sys.argv[1]]
    risk_q, urgency_q = canonical()
    source = load_source(cfg)
    out_path = cfg["dir"] / cfg["output"]
    rep = Report()

    if not out_path.is_file():
        sys.exit(f"missing output: {out_path}")
    rows = [json.loads(l) for l in out_path.open() if l.strip()]

    seen = Counter()
    risk_levels, urg_levels = Counter(), Counter()
    insuff = 0
    for row in rows:
        rid = row["id"]
        seen[rid] += 1
        src = source.get(rid)
        if src is None:
            rep.fail("ids_once_each", f"{rid}: not a source id")
            continue
        if not rid.startswith(cfg["prefix"]):
            rep.fail("ids_once_each", f"{rid}: missing prefix {cfg['prefix']}")

        if row.get("state") != src.get("state"):
            rep.fail("state_byte_identical", f"{rid}: state string changed")
        for field in ("source", "source_group", "provenance", "outcome"):
            if field in src or field in row:
                if json.dumps(row.get(field), sort_keys=True) != json.dumps(src.get(field), sort_keys=True):
                    rep.fail("provenance_and_audit_present", f"{rid}: {field} changed")

        sq, sg = json.loads(src["questions"]), json.loads(src["gold"])
        q, g = json.loads(row["questions"]), json.loads(row["gold"])
        for qid in BASE_QUESTIONS:
            if qid in sq and q.get(qid) != sq.get(qid):
                rep.fail("originals_unchanged", f"{rid}/{qid}: question definition changed")
            if qid in sg and g.get(qid) != sg.get(qid):
                rep.fail("originals_unchanged", f"{rid}/{qid}: gold entry changed")
        if set(q) != set(BASE_QUESTIONS + SCORE_QUESTIONS):
            rep.fail("five_question_ids_match_gold", f"{rid}: question ids {sorted(q)}")
        if set(g) != set(q):
            rep.fail("five_question_ids_match_gold", f"{rid}: gold ids != question ids")
        if q.get("risk") != risk_q or q.get("urgency") != urgency_q:
            rep.fail("canonical_criteria_equal", f"{rid}: score wording != canonical")

        for qid in SCORE_QUESTIONS:
            entry = g.get(qid) or {}
            probs = entry.get("probabilities")
            if not isinstance(probs, dict) or set(probs) != set(LEVELS):
                rep.fail("probabilities_valid", f"{rid}/{qid}: keys {probs and sorted(probs)}")
                continue
            if any(v < 0 for v in probs.values()) or abs(sum(probs.values()) - 1) > 0.002:
                rep.fail("probabilities_valid", f"{rid}/{qid}: sum {sum(probs.values())}")
            exp = sum(int(k) * v for k, v in probs.items())
            if abs(entry.get("score", -1) - exp) > 0.002:
                rep.fail("score_equals_expectation", f"{rid}/{qid}: {entry.get('score')} != {exp:.4f}")
            if entry.get("label") != argmax_level(probs):
                rep.fail("label_equals_argmax_low_tie",
                         f"{rid}/{qid}: {entry.get('label')!r} != {argmax_level(probs)!r}")
            conf = entry.get("confidence")
            if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
                rep.fail("probabilities_valid", f"{rid}/{qid}: confidence {conf!r}")
            if entry.get("label_source") != "codex_pseudo_unverified" or entry.get("defensible") is not False:
                rep.fail("pseudo_unverified_defensible_false", f"{rid}/{qid}: not pseudo/indefensible")

        ann = row.get("score_annotation") or {}
        for qid in SCORE_QUESTIONS:
            if not (ann.get(qid) or {}).get("rationale"):
                rep.fail("provenance_and_audit_present", f"{rid}/{qid}: no rationale")
            if not (ann.get(qid) or {}).get("evidence"):
                rep.fail("provenance_and_audit_present", f"{rid}/{qid}: no evidence")
        if ann.get("urgency", {}).get("insufficient_evidence"):
            insuff += 1
        risk_levels[g["risk"]["label"]] += 1
        urg_levels[g["urgency"]["label"]] += 1

    dupes = [i for i, n in seen.items() if n > 1]
    missing = [i for i in source if i not in seen]
    extra = [i for i in seen if i not in source]
    for msg in (f"duplicate ids: {dupes[:5]}" if dupes else "",
                f"missing source ids: {missing[:5]}" if missing else "",
                f"unexpected ids: {extra[:5]}" if extra else ""):
        if msg:
            rep.fail("ids_once_each", msg)

    dirty = subprocess.run(["git", "status", "--porcelain", "--", "dataset/normalized"],
                           cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        rep.fail("normalized_dir_untouched", dirty)

    print(json.dumps({
        "source": sys.argv[1],
        "output": str(out_path.relative_to(REPO_ROOT)),
        "records": len(rows),
        "source_ids": len(source),
        "risk_distribution": dict(sorted(risk_levels.items())),
        "urgency_distribution": dict(sorted(urg_levels.items())),
        "urgency_insufficient_evidence": insuff,
        "checks": rep.ok,
        "problems": len(rep.problems),
        "problem_detail": rep.problems[:40],
    }, indent=2))
    return 1 if rep.problems else 0


if __name__ == "__main__":
    sys.exit(main())
