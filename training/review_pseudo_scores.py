#!/usr/bin/env python3
"""Audit and merge teacher-generated SilicoJev risk/urgency scores.

This tool performs objective checks and corrections only. It does not claim to
know whether a teacher's semantic judgment is true. Every generated label
remains explicitly unverified, and weak or ambiguous cases are flagged for
later review instead of being silently promoted to ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RISK_CRITERIA = [
    "Benign: read-only inspection, parsing, linting, or clearly reversible analysis.",
    "Low: routine simulation, formal checking, or isolated writes within the project scope.",
    "Moderate: modifying RTL, testbench, constraints, or configuration in a way that could affect results.",
    "High: destructive, irreversible, security-sensitive, out-of-scope, or potentially damaging action.",
]
URGENCY_CRITERIA = [
    "No time pressure: the issue can wait indefinitely.",
    "Routine: handle during the normal development queue.",
    "Elevated: should be investigated within the same week.",
    "Critical: blocks an important flow, threatens tapeout/release/security, or requires action within the same day.",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def normalize(probs: dict[str, Any], keys: list[str]) -> tuple[dict[str, float], bool]:
    values = {key: max(0.0, float(probs.get(key, 0.0))) for key in keys}
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0:
        values = {key: 1.0 / len(keys) for key in keys}
        return values, True
    normalized = {key: value / total for key, value in values.items()}
    changed = any(abs(normalized[key] - float(probs.get(key, 0.0))) > 1e-6 for key in keys)
    return normalized, changed


def review_score(gold: dict[str, Any], question: dict[str, Any], question_id: str) -> tuple[dict[str, Any], list[str], list[str]]:
    flags: list[str] = []
    corrections: list[str] = []
    criteria = question.get("criteria")
    expected_criteria = RISK_CRITERIA if question_id == "risk" else URGENCY_CRITERIA
    if question.get("type") != "score" or criteria != expected_criteria:
        flags.append("question_definition_mismatch")

    keys = [str(index) for index in range(len(criteria or expected_criteria))]
    probs, normalized = normalize(gold.get("probabilities", {}), keys)
    if normalized:
        corrections.append(f"{question_id}.probabilities")
    label = max(keys, key=lambda key: (probs[key], -int(key)))
    if str(gold.get("label")) != label:
        corrections.append(f"{question_id}.label")
    score = sum(int(key) * probs[key] for key in keys)
    if abs(float(gold.get("score", score)) - score) > 1e-5:
        corrections.append(f"{question_id}.score")

    confidence = gold.get("confidence")
    if confidence is None or not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        flags.append("invalid_confidence")
    if confidence is not None and float(confidence) < 0.5:
        flags.append("low_confidence")
    if question_id == "urgency" and label == "0":
        # Absence of a deadline does not prove that an active bug can wait
        # indefinitely. Keep the teacher label, but make this ambiguity visible.
        flags.append("urgency_no_time_pressure_not_proven")

    reviewed = dict(gold)
    reviewed["probabilities"] = {key: round(value, 6) for key, value in probs.items()}
    reviewed["score"] = round(score, 6)
    reviewed["label"] = label
    reviewed["label_source"] = "codex_pseudo_unverified"
    return reviewed, flags, corrections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    annotated_rows = load_jsonl(args.annotated)
    base_by_id = {row["id"]: row for row in base_rows}
    annotated_by_id = {row["id"]: row for row in annotated_rows}
    if len(base_by_id) != len(base_rows) or len(annotated_by_id) != len(annotated_rows):
        raise ValueError("Duplicate IDs found in base or annotated data")
    if set(base_by_id) != set(annotated_by_id):
        raise ValueError("Annotated IDs do not exactly match base IDs")

    shard_ids: list[str] = []
    shard_counts: dict[str, int] = {}
    for shard in sorted(args.shard_dir.glob("shard_*.jsonl")):
        rows = load_jsonl(shard)
        ids = [row["id"] for row in rows]
        shard_counts[shard.stem] = len(ids)
        shard_ids.extend(ids)
    shard_counter = Counter(shard_ids)
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("Duplicate IDs across pseudo-annotation shards")
    if set(shard_ids) != set(base_by_id):
        raise ValueError("Pseudo-annotation shards do not cover the base dataset exactly")

    reviewed_rows: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    correction_counts: Counter[str] = Counter()
    source_flags: dict[str, Counter[str]] = defaultdict(Counter)
    for base in base_rows:
        row = annotated_by_id[base["id"]]
        # The annotation pass may add fields, but must not rewrite existing
        # fields other than the intentional questions/gold extension.
        for key, value in base.items():
            if key in {"questions", "gold"}:
                continue
            if row.get(key) != value:
                raise ValueError(f"Original field changed for {base['id']}: {key}")

        questions = parse_json(row["questions"])
        gold = parse_json(row["gold"])
        if set(questions) != set(gold):
            raise ValueError(f"Question/gold IDs differ for {base['id']}")
        if not {"risk", "urgency"}.issubset(questions):
            raise ValueError(f"Missing risk/urgency questions for {base['id']}")

        reviewed_gold = dict(gold)
        row_flags: list[str] = []
        row_corrections: list[str] = []
        for question_id in ("risk", "urgency"):
            revised, flags, corrections = review_score(gold[question_id], questions[question_id], question_id)
            reviewed_gold[question_id] = revised
            row_flags.extend(flags)
            row_corrections.extend(corrections)

        source = str(row.get("source", "unknown"))
        if source == "OriGen":
            row_flags.append("bronze_unreplayed_teacher_label")
        for flag in row_flags:
            flag_counts[flag] += 1
            source_flags[source][flag] += 1
        for correction in row_corrections:
            correction_counts[correction] += 1

        reviewed = dict(row)
        reviewed["questions"] = json.dumps(questions, ensure_ascii=False, sort_keys=True)
        reviewed["gold"] = json.dumps(reviewed_gold, ensure_ascii=False, sort_keys=True)
        reviewed["pseudo_annotation_review"] = {
            "status": "structurally_reviewed_semantically_unverified",
            "flags": sorted(set(row_flags)),
            "corrected_fields": sorted(set(row_corrections)),
            "semantic_review_note": "Teacher judgments were not independently verified; flagged ambiguity remains unresolved.",
        }
        reviewed_rows.append(reviewed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_all = args.output_dir / "all_with_pseudo_scores_reviewed.jsonl"
    with output_all.open("w") as handle:
        for row in reviewed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    id_to_split = {}
    for split in ("train", "validation", "test"):
        for row in load_jsonl(args.output_dir.parent / f"{split}.jsonl"):
            id_to_split[row["id"]] = split
    split_counts = Counter()
    split_paths = {}
    for split in ("train", "validation", "test"):
        split_path = args.output_dir / f"{split}_with_pseudo_scores.jsonl"
        split_paths[split] = str(split_path)
        with split_path.open("w") as handle:
            for row in reviewed_rows:
                if id_to_split.get(row["id"]) == split:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    split_counts[split] += 1

    report = {
        "base": str(args.base),
        "annotated": str(args.annotated),
        "merged_all": str(output_all),
        "merged_splits": split_paths,
        "records": len(reviewed_rows),
        "shard_counts": shard_counts,
        "shard_unique_ids": len(shard_counter),
        "original_fields_preserved": True,
        "objective_corrections": dict(sorted(correction_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "source_flag_counts": {source: dict(sorted(counts.items())) for source, counts in sorted(source_flags.items())},
        "split_counts": dict(sorted(split_counts.items())),
        "semantic_status": "unverified_teacher_pseudo_labels",
        "score_training_warning": "Do not use these labels as final validation/test truth until independently audited.",
    }
    report_path = args.shard_dir / "review_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
