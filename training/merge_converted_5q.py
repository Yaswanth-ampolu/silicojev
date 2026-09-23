#!/usr/bin/env python3
"""Merge the distilled risk/urgency scores onto the description-bearing records.

This mirrors ``training/merge_silicojev_questions.py`` for the two converted
sources (RTL-BenchLS Task 3 and Fixbench-RTL).  The description-carrying base
records are the three-question files; the distilled file adds exactly ``risk``
and ``urgency``.  The merge re-checks that invariant and then writes one
canonical five-question dataset record shape, one file per source plus a
combined ``all.jsonl``.

Nothing here is label promotion.  Every added score stays
``codex_pseudo_unverified`` with ``defensible: false``; the base three questions,
the gold labels and ``state`` are copied through untouched, and no source file
is rewritten.

Usage:
    python3 training/merge_converted_5q.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "dataset/converted/merged_5q"

BASE_QUESTION_IDS = {"next_action", "root_cause_type", "evidence_sufficient"}
ADDED_QUESTION_IDS = {"risk", "urgency"}

SOURCES: dict[str, dict[str, Any]] = {
    "rtl_benchls": {
        "dir": REPO_ROOT / "dataset/converted/rtl_benchls",
        "inputs": ("records_3q.jsonl", "records_3q_unverified.jsonl"),
        "distilled": "records_5q_distilled_pseudo_unverified.jsonl",
    },
    "fixbench_rtl": {
        "dir": REPO_ROOT / "dataset/converted/fixbench_rtl",
        "inputs": ("records_3q.jsonl",),
        "distilled": "records_5q_distilled_pseudo_unverified.jsonl",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def parse_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def merge_row(base: dict[str, Any], distilled: dict[str, Any], source_path: Path) -> dict[str, Any]:
    if base.get("id") != distilled.get("id"):
        raise ValueError(f"ID mismatch: {base.get('id')} != {distilled.get('id')}")

    if base["state"] != distilled["state"]:
        raise ValueError(f"{base['id']}: distilled file changed the state/description")

    base_questions = parse_json(base["questions"])
    base_gold = parse_json(base["gold"])
    dist_questions = parse_json(distilled["questions"])
    dist_gold = parse_json(distilled["gold"])

    if set(base_questions) != BASE_QUESTION_IDS:
        raise ValueError(f"{base['id']}: base question IDs are {sorted(base_questions)}")
    if set(dist_questions) != BASE_QUESTION_IDS | ADDED_QUESTION_IDS:
        raise ValueError(f"{base['id']}: distilled question IDs are {sorted(dist_questions)}")
    if set(base_gold) != BASE_QUESTION_IDS:
        raise ValueError(f"{base['id']}: base gold IDs are {sorted(base_gold)}")
    if set(dist_gold) != BASE_QUESTION_IDS | ADDED_QUESTION_IDS:
        raise ValueError(f"{base['id']}: distilled gold IDs are {sorted(dist_gold)}")

    for qid in BASE_QUESTION_IDS:
        if base_questions[qid] != dist_questions[qid]:
            raise ValueError(f"{base['id']}: distilled file changed original question {qid}")
        if base_gold[qid] != dist_gold[qid]:
            raise ValueError(f"{base['id']}: distilled file changed original gold label {qid}")

    for qid in ADDED_QUESTION_IDS:
        gold = dist_gold[qid]
        if gold.get("label_source") != "codex_pseudo_unverified" or gold.get("defensible") is not False:
            raise ValueError(f"{base['id']}/{qid}: added score is not marked pseudo/unverified")

    # Start from the description-bearing base row, then add exactly the two
    # score questions, their golds, and the audit trail. The base questions and
    # golds are left as the base file wrote them.
    merged = dict(base)
    merged_questions = dict(base_questions)
    merged_questions.update({qid: dist_questions[qid] for qid in sorted(ADDED_QUESTION_IDS)})
    merged_gold = dict(base_gold)
    merged_gold.update({qid: dist_gold[qid] for qid in sorted(ADDED_QUESTION_IDS)})
    merged["questions"] = json.dumps(merged_questions, ensure_ascii=False, sort_keys=True)
    merged["gold"] = json.dumps(merged_gold, ensure_ascii=False, sort_keys=True)

    if "score_annotation" in distilled:
        merged["score_annotation"] = distilled["score_annotation"]

    provenance = dict(merged.get("provenance") or {})
    provenance.update({
        "pseudo_questions_merged": sorted(ADDED_QUESTION_IDS),
        "pseudo_label_status": "codex_pseudo_unverified",
        "pseudo_source_file": str(source_path.relative_to(REPO_ROOT)),
    })
    merged["provenance"] = provenance
    return merged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[dict[str, Any]]] = {}
    for name, spec in SOURCES.items():
        base: dict[str, dict[str, Any]] = {}
        for filename in spec["inputs"]:
            for row in read_jsonl(spec["dir"] / filename):
                if row["id"] in base:
                    raise ValueError(f"{name}: duplicate id across base inputs: {row['id']}")
                base[row["id"]] = row
        distilled_path = spec["dir"] / spec["distilled"]
        distilled = {row["id"]: row for row in read_jsonl(distilled_path)}
        if set(base) != set(distilled):
            missing = sorted(set(base) - set(distilled))[:5]
            extra = sorted(set(distilled) - set(base))[:5]
            raise ValueError(f"{name}: id sets differ; missing={missing} extra={extra}")
        if len(distilled) != len(read_jsonl(distilled_path)):
            raise ValueError(f"{name}: duplicate ids in distilled file")
        rows = [merge_row(base[rid], distilled[rid], distilled_path) for rid in sorted(base)]
        merged[name] = rows
        write_jsonl(OUT_DIR / f"{name}.jsonl", rows)

    all_rows = [row for name in sorted(merged) for row in merged[name]]
    write_jsonl(OUT_DIR / "all.jsonl", all_rows)

    risk_labels = Counter()
    urgency_labels = Counter()
    insufficient = Counter()
    for row in all_rows:
        gold = parse_json(row["gold"])
        risk_labels[gold["risk"]["label"]] += 1
        urgency_labels[gold["urgency"]["label"]] += 1
        ann = row.get("score_annotation") or {}
        for qid in ADDED_QUESTION_IDS:
            if (ann.get(qid) or {}).get("insufficient_evidence") or (ann.get(qid) or {}).get("evidence_insufficient"):
                insufficient[qid] += 1

    summary = {
        "format": "SilicoJev/Laya typed-decisions compatible JSONL",
        "base_question_ids": sorted(BASE_QUESTION_IDS),
        "added_question_ids": sorted(ADDED_QUESTION_IDS),
        "total_cases": len(all_rows),
        "total_decisions": len(all_rows) * 5,
        "source_counts": {name: len(rows) for name, rows in sorted(merged.items())},
        "source_group_counts": dict(sorted(Counter(row.get("source") for row in all_rows).items())),
        "risk_label_counts": dict(sorted(risk_labels.items())),
        "urgency_label_counts": dict(sorted(urgency_labels.items())),
        "insufficient_evidence_counts": dict(sorted(insufficient.items())),
        "original_question_integrity": "verified_exact_match",
        "state_integrity": "verified_byte_identical_to_base",
        "pseudo_label_status": "codex_pseudo_unverified",
        "standalone": "not merged into dataset/normalized/; exploratory use only",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "README.md").write_text(
        "# Merged five-question dataset (converted sources)\n\n"
        "The two converted sources — RTL-BenchLS Task 3 and Fixbench-RTL — merged "
        "onto their description-bearing base records, so every row carries all "
        "five SilicoJev questions: the three source-derived base questions "
        "(`next_action`, `root_cause_type`, `evidence_sufficient`) plus the "
        "distilled `risk` and `urgency` scores.\n\n"
        "`all.jsonl` is the combined set; `rtl_benchls.jsonl` and "
        "`fixbench_rtl.jsonl` are the per-source subsets. Each row preserves the "
        "base record's `state` (the case description) byte-identically, and the "
        "three base questions and golds are verified unchanged.\n\n"
        "The two added score labels are model judgments marked "
        "`codex_pseudo_unverified` with `defensible: false`. They are for "
        "exploratory training only and must not be used as final evaluation "
        "truth. This directory is a standalone artefact: it is not merged into "
        "`dataset/normalized/` and has not been fed to Laya.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
