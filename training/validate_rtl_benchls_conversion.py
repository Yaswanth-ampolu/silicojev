#!/usr/bin/env python3
"""Independently validate the converted RTL-BenchLS Task 3 records.

Re-reads the written JSONL files from disk (it does not trust the converter's
in-memory state), checks them against dataset/SILICOJEV_SCHEMA.json, and
re-verifies the state-leakage rules against the raw source and repo_cache.

Usage:
    python3 training/validate_rtl_benchls_conversion.py
    python3 training/validate_rtl_benchls_conversion.py --no-network
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_rtl_benchls import (  # noqa: E402
    BASE_QUESTION_IDS,
    RepoCache,
    parse_patch,
    render_windows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "dataset/raw/github/RTL-BenchLS"
DEFAULT_CONVERTED = REPO_ROOT / "dataset/converted/rtl_benchls"
SCHEMA = REPO_ROOT / "dataset/SILICOJEV_SCHEMA.json"

FILES = {
    "records_3q.jsonl": 3,
    "records_3q_unverified.jsonl": 3,
    "records_5q.jsonl": 5,
    "records_5q_pseudo_unverified.jsonl": 5,
}
TYPE_BY_QUESTION = {
    "next_action": "choice",
    "root_cause_type": "choice",
    "evidence_sufficient": "noul",
    "risk": "score",
    "urgency": "score",
}


def check_schema_shape(row: dict) -> list[str]:
    """Required SilicoJev fields and JSON-string encoding."""
    errors = []
    for field in ("id", "state", "questions", "gold", "provenance"):
        if field not in row:
            errors.append(f"missing required field {field}")
    for field in ("state", "questions", "gold"):
        if not isinstance(row.get(field), str):
            errors.append(f"{field} must be a JSON string")
            continue
        try:
            json.loads(row[field])
        except json.JSONDecodeError as exc:
            errors.append(f"{field} is not valid JSON: {exc}")
    prov = row.get("provenance") or {}
    for field in ("license", "synthetic"):
        if field not in prov:
            errors.append(f"provenance.{field} missing")
    if not isinstance(prov.get("synthetic"), bool):
        errors.append("provenance.synthetic must be boolean")
    return errors


def check_questions(row: dict, n_expected: int) -> list[str]:
    errors = []
    qs = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    if len(qs) != n_expected:
        errors.append(f"expected {n_expected} questions, found {len(qs)}")
    if set(qs) != set(gold):
        errors.append("question ids and gold ids differ")
    for qid, q in qs.items():
        if q.get("type") != TYPE_BY_QUESTION.get(qid):
            errors.append(f"{qid}: type {q.get('type')} unexpected")
        crit = q.get("criteria")
        if q["type"] == "choice":
            if not isinstance(crit, dict) or not crit:
                errors.append(f"{qid}: choice criteria must be a non-empty object")
            probs = gold[qid]["probabilities"]
            if set(probs) != set(crit):
                errors.append(f"{qid}: probability keys do not match criteria")
        if q["type"] == "noul":
            if set(crit or {}) != {"false", "true"}:
                errors.append(f"{qid}: noul criteria must be false/true")
        if q["type"] == "score":
            if not isinstance(crit, list) or len(crit) != 4:
                errors.append(f"{qid}: score criteria must have 4 levels")
            probs = gold[qid]["probabilities"]
            if set(probs) != {str(i) for i in range(4)}:
                errors.append(f"{qid}: score keys must be 0..3")
        probs = gold[qid]["probabilities"]
        if abs(sum(probs.values()) - 1.0) > 0.02:
            errors.append(f"{qid}: probabilities sum to {sum(probs.values()):.3f}")
        if any(v < 0 for v in probs.values()):
            errors.append(f"{qid}: negative probability")
        if not gold[qid].get("label_source"):
            errors.append(f"{qid}: missing label_source")
    return errors


def check_leakage(row: dict, case: dict, cache: RepoCache) -> list[str]:
    """Every byte of state must be base-revision and pre-repair."""
    issues = []
    state_text = row["state"]
    state = json.loads(state_text)

    if case["head_commit"] in state_text:
        issues.append("head_commit appears in state")
    pr = case.get("pr_info") or {}
    for field in ("title", "body"):
        text = (pr.get(field) or "").strip()
        if len(text) >= 40 and text[:40] in state_text:
            issues.append(f"pr_info.{field} appears in state")
    for label in (case.get("issue_info") or {}).get("labels") or []:
        if label.startswith("Status:") and label in state_text:
            issues.append(f"post-repair label {label} appears in state")
    for banned in ("patches", "head_commit", "additions", "deletions", "lec_status"):
        if banned in state:
            issues.append(f"state exposes key {banned}")

    for entry in state["rtl_context"]:
        if not entry.get("content"):
            continue
        blob = cache.read(case["repository"], case["base_commit"], entry["path"])
        if blob is None:
            issues.append(f"{entry['path']}: cannot read base blob")
            continue
        windows = [tuple(w) for w in entry.get("windows") or []]
        if not windows:
            issues.append(f"{entry['path']}: no window ranges recorded")
            continue
        n_lines = blob.count("\n") + 1
        previous_end = 0
        for start, end in windows:
            if not (1 <= start <= end <= n_lines + 1):
                issues.append(f"{entry['path']}: window {start}-{end} outside base blob")
            if start <= previous_end:
                issues.append(f"{entry['path']}: windows overlap at line {start}")
            previous_end = end
        # The state text must be exactly the base-revision rendering of those
        # windows, so no head-revision line can be present.
        if render_windows(blob, windows, entry["path"]) != entry["content"]:
            issues.append(f"{entry['path']}: state text is not the rendered base window")

    # Repaired text may only reach the state if it also exists at base.
    trivial = re.compile(r"^[\s\)\};,]*(end|begin|else|endif|\}|\{)?[\s\)\};,]*$")
    for entry in case.get("patches") or []:
        path = entry.get("filename")
        blob = cache.read(case["repository"], case["base_commit"], path) or ""
        _, added, _ = parse_patch(entry.get("patch") or "")
        for line in added:
            stripped = line.strip()
            if len(stripped) < 12 or trivial.match(stripped):
                continue
            if stripped in blob:
                continue
            section = next((f["content"] for f in state["rtl_context"] if f["path"] == path), "")
            if stripped in section:
                issues.append(f"{path}: repaired line present in state: {stripped[:60]}")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--converted-dir", type=Path, default=DEFAULT_CONVERTED)
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()

    schema = json.loads(SCHEMA.read_text())
    cases = {c["task_id"]: c
             for c in json.loads((args.dataset_root / "data/repo_issue_108_cases.json").read_text())["cases"]}
    cache = RepoCache(args.dataset_root / "repo_cache", allow_network=not args.no_network)

    results: dict[str, dict] = {}
    all_ids: list[str] = []
    per_question_source: Counter = Counter()
    projections: dict[str, dict] = {}

    for name, n_questions in FILES.items():
        path = args.converted_dir / name
        problems, count = [], 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            task_id = row["id"].split(":", 1)[1]
            case = cases.get(task_id)
            if case is None:
                problems.append(f"{row['id']}: not a source case")
                continue
            if row["id"] != f"rtlbenchls:{task_id}":
                problems.append(f"{row['id']}: id does not match task_id {task_id}")
            if row.get("source") != "RTL-BenchLS":
                problems.append(f"{row['id']}: unexpected source")
            if row.get("source_group") != f"rtlbenchls:{case['repository']}":
                problems.append(f"{row['id']}: source_group does not match repository")
            if "outcome" not in row or not isinstance(row["outcome"], dict):
                problems.append(f"{row['id']}: outcome must be an object")
            for err in check_schema_shape(row) + check_questions(row, n_questions):
                problems.append(f"{row['id']}: {err}")
            for issue in check_leakage(row, case, cache):
                problems.append(f"{row['id']}: LEAKAGE {issue}")
            all_ids.append(row["id"])
            for qid, g in json.loads(row["gold"]).items():
                per_question_source[f"{qid}:{g['label_source']}"] += 1
            if n_questions == 5:
                gold = json.loads(row["gold"])
                questions = json.loads(row["questions"])
                projections[row["id"]] = {
                    "questions": json.dumps({q: questions[q] for q in BASE_QUESTION_IDS},
                                            sort_keys=True),
                    "gold": json.dumps({q: gold[q] for q in BASE_QUESTION_IDS}, sort_keys=True),
                }
        results[name] = {"records": count, "problems": problems[:20],
                         "problem_count": len(problems)}

    # The three-question projection of every 5q record must equal the 3q record.
    projection_errors = []
    three_q: dict[str, dict] = {}
    for name in ("records_3q.jsonl", "records_3q_unverified.jsonl"):
        for line in (args.converted_dir / name).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                three_q[row["id"]] = {"questions": row["questions"], "gold": row["gold"]}
    for rid, proj in projections.items():
        base = three_q.get(rid)
        if base is None:
            projection_errors.append(f"{rid}: 5q record has no 3q counterpart")
            continue
        if any(base[k] != proj[k] for k in ("questions", "gold")):
            projection_errors.append(f"{rid}: 3q projection differs from the 3q record")

    counts = Counter(all_ids)
    duplicates = [i for i, n in counts.items() if n > 1]
    total_3q = results["records_3q.jsonl"]["records"] + results["records_3q_unverified.jsonl"]["records"]

    # usage_sets.json must be internally consistent with the files it indexes, and
    # no evaluation-excluded case may appear in a trusted set.
    usage_problems = []
    usage = json.loads((args.converted_dir / "usage_sets.json").read_text())
    file_ids = {}
    for name in FILES:
        file_ids[name] = {json.loads(l)["id"]
                          for l in (args.converted_dir / name).read_text().splitlines() if l.strip()}
    excluded = set(usage.get("eval_excluded") or [])
    if len(excluded) != len(usage.get("eval_excluded") or []):
        usage_problems.append("eval_excluded lists an id twice")
    for rid in sorted(excluded):
        if rid not in usage.get("eval_excluded_reasons", {}):
            usage_problems.append(f"{rid}: excluded without a reason")
        if not any(rid in ids for ids in file_ids.values()):
            usage_problems.append(f"{rid}: excluded id is in no output file")
        if rid in file_ids["records_3q.jsonl"] and rid not in set(
                usage.get("eval_excluded_from_trusted_3q") or []):
            usage_problems.append(f"{rid}: excluded case stays in the trusted 3q file unlisted")
    for key in ("trusted_exploratory_3q", "trusted_exploratory_5q"):
        bad = sorted(set(usage.get(key) or []) & excluded)
        if bad:
            usage_problems.append(f"{key}: contains evaluation-excluded cases {bad}")
    stray = sorted(set(usage.get("trusted_exploratory_3q") or []) - file_ids["records_3q.jsonl"])
    if stray:
        usage_problems.append(f"trusted_exploratory_3q: ids not in records_3q.jsonl {stray}")
    stray = sorted(set(usage.get("trusted_exploratory_5q") or []) - file_ids["records_5q.jsonl"])
    if stray:
        usage_problems.append(f"trusted_exploratory_5q: ids not in records_5q.jsonl {stray}")
    off = sorted(set(usage.get("trusted_exploratory_5q") or [])
                 - set(usage.get("trusted_exploratory_3q") or []))
    if off:
        usage_problems.append(f"trusted_exploratory_5q is not a subset of trusted 3q {off}")

    summary = {
        "schema_used": str(SCHEMA.relative_to(REPO_ROOT)),
        "schema_title": schema.get("title"),
        "files": results,
        "three_q_total": total_3q,
        "three_q_ids_unique": len({json.loads(l)['id']
                                   for n in ("records_3q.jsonl", "records_3q_unverified.jsonl")
                                   for l in (args.converted_dir / n).read_text().splitlines() if l.strip()}),
        "five_q_total": results["records_5q.jsonl"]["records"]
        + results["records_5q_pseudo_unverified.jsonl"]["records"],
        "duplicate_ids_across_3q_and_5q": sorted(set(duplicates)),
        "duplicate_note": ("ids repeating across a 3q file and a 5q file are the same case "
                           "with the two score questions added; they must not repeat inside one file"),
        "projection_errors": projection_errors,
        "usage_set_problems": usage_problems,
        "usage_set_sizes": {k: len(v) for k, v in usage.items() if isinstance(v, list)},
        "label_source_by_question": dict(sorted(per_question_source.items())),
        "total_problems": (sum(v["problem_count"] for v in results.values())
                           + len(projection_errors) + len(usage_problems)),
    }
    print(json.dumps(summary, indent=2))
    if summary["total_problems"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
