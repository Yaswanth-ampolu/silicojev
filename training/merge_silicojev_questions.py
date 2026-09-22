#!/usr/bin/env python3
"""Merge SilicoJev's original questions with its two pseudo score questions.

The base records contain the three source-derived questions.  The pseudo file
must contain the same records unchanged plus exactly ``risk`` and ``urgency``.
This script checks that invariant before writing a canonical five-question
dataset in the same JSONL record shape used by the SilicoJev/Laya trainer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASE_QUESTION_IDS = {"evidence_sufficient", "next_action", "root_cause_type"}
ADDED_QUESTION_IDS = {"risk", "urgency"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_row(base: dict[str, Any], pseudo: dict[str, Any], source: Path) -> dict[str, Any]:
    if base.get("id") != pseudo.get("id"):
        raise ValueError(f"ID mismatch: {base.get('id')} != {pseudo.get('id')}")

    base_questions = parse_json(base["questions"])
    base_gold = parse_json(base["gold"])
    pseudo_questions = parse_json(pseudo["questions"])
    pseudo_gold = parse_json(pseudo["gold"])

    if set(base_questions) != BASE_QUESTION_IDS:
        raise ValueError(f"{base['id']}: base question IDs are {sorted(base_questions)}")
    if set(pseudo_questions) != BASE_QUESTION_IDS | ADDED_QUESTION_IDS:
        raise ValueError(f"{base['id']}: pseudo question IDs are {sorted(pseudo_questions)}")
    if set(base_gold) != BASE_QUESTION_IDS or set(pseudo_gold) != BASE_QUESTION_IDS | ADDED_QUESTION_IDS:
        raise ValueError(f"{base['id']}: question/gold IDs do not match expected sets")

    for qid in BASE_QUESTION_IDS:
        if base_questions[qid] != pseudo_questions[qid]:
            raise ValueError(f"{base['id']}: pseudo file changed original question {qid}")
        if base_gold[qid] != pseudo_gold[qid]:
            raise ValueError(f"{base['id']}: pseudo file changed original gold label {qid}")

    merged = dict(base)
    merged_questions = dict(base_questions)
    merged_questions.update({qid: pseudo_questions[qid] for qid in sorted(ADDED_QUESTION_IDS)})
    merged_gold = dict(base_gold)
    merged_gold.update({qid: pseudo_gold[qid] for qid in sorted(ADDED_QUESTION_IDS)})
    merged["questions"] = dump_json(merged_questions)
    merged["gold"] = dump_json(merged_gold)

    # Preserve the annotation audit trail without replacing the source-derived
    # provenance or pretending that pseudo labels are independently validated.
    for key in ("pseudo_annotation_notes", "pseudo_annotation_review"):
        if key in pseudo:
            merged[key] = pseudo[key]
    provenance = dict(merged.get("provenance") or {})
    provenance.update({
        "pseudo_questions_merged": sorted(ADDED_QUESTION_IDS),
        "pseudo_label_status": "codex_pseudo_unverified",
        "pseudo_source_file": str(source),
    })
    merged["provenance"] = provenance
    return merged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset/normalized",
    )
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset/normalized/pseudo_scores",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset/normalized/merged_5q",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        base_path = args.normalized_dir / f"{split}.jsonl"
        pseudo_path = args.pseudo_dir / f"{split}_with_pseudo_scores.jsonl"
        base_rows = read_jsonl(base_path)
        pseudo_rows = read_jsonl(pseudo_path)
        base_by_id = {row["id"]: row for row in base_rows}
        pseudo_by_id = {row["id"]: row for row in pseudo_rows}
        if len(base_by_id) != len(base_rows) or len(pseudo_by_id) != len(pseudo_rows):
            raise ValueError(f"{split}: duplicate record IDs")
        if set(base_by_id) != set(pseudo_by_id):
            raise ValueError(f"{split}: base and pseudo record IDs differ")
        merged_by_split[split] = [
            merge_row(base_by_id[row_id], pseudo_by_id[row_id], pseudo_path)
            for row_id in sorted(base_by_id)
        ]
        write_jsonl(args.output_dir / f"{split}.jsonl", merged_by_split[split])

    all_rows = [row for split in ("train", "validation", "test") for row in merged_by_split[split]]
    write_jsonl(args.output_dir / "all.jsonl", all_rows)
    summary = {
        "format": "SilicoJev/Laya typed-decisions compatible JSONL",
        "base_question_ids": sorted(BASE_QUESTION_IDS),
        "added_question_ids": sorted(ADDED_QUESTION_IDS),
        "total_cases": len(all_rows),
        "total_decisions": len(all_rows) * 5,
        "split_counts": {split: len(rows) for split, rows in merged_by_split.items()},
        "decision_counts": {split: len(rows) * 5 for split, rows in merged_by_split.items()},
        "source_counts": dict(sorted(Counter(row.get("source", "unknown") for row in all_rows).items())),
        "quality_counts": dict(sorted(Counter((row.get("provenance") or {}).get("quality", "unknown") for row in all_rows).items())),
        "original_question_integrity": "verified_exact_match",
        "pseudo_label_status": "codex_pseudo_unverified",
        "training_note": "Use risk/urgency for exploratory training only until independently validated; do not use them as final evaluation truth.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# SilicoJev merged five-question dataset\n\n"
        "This dataset preserves the three original SilicoJev questions and adds "
        "the distilled `risk` and `urgency` score questions by matching stable "
        "record IDs. The original questions and labels were verified unchanged.\n\n"
        "The two added score labels are marked `codex_pseudo_unverified`; they "
        "are suitable for exploratory training but not final evaluation truth.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
