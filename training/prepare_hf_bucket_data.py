#!/usr/bin/env python3
"""Validate and split the four public Hugging Face SilicoJev JSONL files.

The base SilicoJev split is preserved exactly from the repository. Converted
corpora are split by their recorded repository/project/family groups, never by
individual row. Source records are copied without editing state/questions/gold;
training_weights is additive metadata consumed by the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_QUESTIONS = {
    "evidence_sufficient": "noul",
    "next_action": "choice",
    "root_cause_type": "choice",
    "risk": "score",
    "urgency": "score",
}
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {
    "silicojev_5q.jsonl": 6248,
    "fixbench_rtl_5q.jsonl": 100,
    "rtl_benchls_5q.jsonl": 108,
    "veribugbench_5q.jsonl": 2443,
}
DEFAULT_LABEL_WEIGHTS = {
    # Evidence directly supported by validated repairs, benchmark structure,
    # or explicit human annotations.
    "verified_repair": 1.0,
    "benchmark_bug_family": 1.0,
    "validated_mutation": 1.0,
    "manual_bug_label": 1.0,
    "manual_issue_label": 1.0,
    "historical_bug": 1.0,
    "inferred_from_verified_repair": 0.75,
    # Teacher/rubric/incomplete-repair targets are useful for exploration, but
    # deliberately contribute less than the stronger labels above.
    "llm_judge": 0.25,
    "distilled_unverified": 0.25,
    "codex_pseudo_unverified": 0.20,
    "repair_pair_unvalidated": 0.20,
    "synthetic_bug_pattern": 0.20,
}


def parse_json(value: Any, where: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: invalid embedded JSON: {exc}") from exc
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(group: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def load_base_split_ids(split_dir: Path) -> dict[str, str]:
    id_to_split: dict[str, str] = {}
    for split in SPLITS:
        path = split_dir / f"{split}.jsonl"
        for row in read_jsonl(path):
            rid = str(row.get("id", ""))
            if not rid or rid in id_to_split:
                raise ValueError(f"Missing or duplicate base id in {path}: {rid!r}")
            id_to_split[rid] = split
    return id_to_split


def validate_record(row: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    where = f"{path}:{row.get('id', '<missing-id>')}"
    if not row.get("id") or "state" not in row:
        raise ValueError(f"{where}: id/state is missing")
    questions = parse_json(row.get("questions"), f"{where}.questions")
    gold = parse_json(row.get("gold"), f"{where}.gold")
    if not isinstance(questions, dict) or set(questions) != set(EXPECTED_QUESTIONS):
        raise ValueError(f"{where}: expected exactly five SilicoJev questions")
    if not isinstance(gold, dict) or set(gold) != set(EXPECTED_QUESTIONS):
        raise ValueError(f"{where}: gold keys do not match the five questions")

    for qid, expected_type in EXPECTED_QUESTIONS.items():
        question = questions[qid]
        if question.get("type") != expected_type or not question.get("instructions"):
            raise ValueError(f"{where}/{qid}: invalid type or missing instructions")
        criteria = question.get("criteria")
        if expected_type == "choice" and (not isinstance(criteria, dict) or not criteria):
            raise ValueError(f"{where}/{qid}: choice criteria must be a nonempty map")
        if expected_type == "score" and (not isinstance(criteria, list) or not 2 <= len(criteria) <= 10):
            raise ValueError(f"{where}/{qid}: score criteria must have 2-10 ordered levels")
        probabilities = gold[qid].get("probabilities")
        expected_keys = (
            list(criteria) if expected_type == "choice"
            else ["false", "true"] if expected_type == "noul"
            else [str(i) for i in range(len(criteria))]
        )
        if not isinstance(probabilities, dict) or set(probabilities) != set(expected_keys):
            raise ValueError(f"{where}/{qid}: probability keys do not match criteria")
        values = [float(probabilities[k]) for k in expected_keys]
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-3:
            raise ValueError(f"{where}/{qid}: probabilities must be nonnegative and sum to 1")
    return questions, gold


def make_splits(
    source_rows: dict[str, list[dict[str, Any]]],
    base_split_dir: Path,
    fixbench_groups_path: Path,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_ids = load_base_split_ids(base_split_dir)
    bucket_base_ids = {str(row["id"]) for row in source_rows["silicojev_5q.jsonl"]}
    if base_ids.keys() != bucket_base_ids:
        missing = sorted(base_ids.keys() - bucket_base_ids)[:5]
        extra = sorted(bucket_base_ids - base_ids.keys())[:5]
        raise ValueError(f"Bucket SilicoJev IDs differ from repository splits; missing={missing}, extra={extra}")

    fix_groups = json.loads(fixbench_groups_path.read_text(encoding="utf-8"))["case_group"]
    outputs = {split: [] for split in SPLITS}
    excluded_rows: list[dict[str, str]] = []
    assignments: dict[str, dict[str, str]] = {name: {} for name in source_rows}
    group_splits: dict[str, dict[str, str]] = {name: {} for name in source_rows}

    for filename, rows in source_rows.items():
        seen: set[str] = set()
        if len(rows) != EXPECTED_COUNTS[filename]:
            raise ValueError(f"{filename}: expected {EXPECTED_COUNTS[filename]} rows, got {len(rows)}")
        for row in rows:
            rid = str(row["id"])
            if rid in seen:
                raise ValueError(f"Duplicate id {rid} in {filename}")
            seen.add(rid)
            questions, gold = validate_record(row, Path(filename))
            if (row.get("provenance") or {}).get("eval_excluded") is True:
                excluded_rows.append({
                    "id": rid,
                    "source": str(row.get("source", "unknown")),
                    "reason": str((row.get("provenance") or {}).get("eval_exclusion_reason", "source marks eval_excluded")),
                })
                continue

            if filename == "silicojev_5q.jsonl":
                split = base_ids[rid]
                group = f"base:{row.get('source_group', rid)}"
            elif filename == "rtl_benchls_5q.jsonl":
                split = (row.get("provenance") or {}).get("split")
                if split not in SPLITS:
                    raise ValueError(f"{rid}: RTL-BenchLS record lacks a valid repository-disjoint split")
                group = f"rtlbenchls:{row.get('source_group') or rid}"
            elif filename == "fixbench_rtl_5q.jsonl":
                group_name = fix_groups.get(rid, rid)
                group = f"fixbench:{group_name}"
                split = stable_split(group, seed)
            else:
                provenance = row.get("provenance") or {}
                group_name = provenance.get("project_id") or row.get("source_group") or rid
                group = f"veribugbench:{group_name}"
                split = stable_split(group, seed)

            previous = group_splits[filename].setdefault(group, split)
            if previous != split:
                raise ValueError(f"Group leakage in {filename}: {group} crosses splits")

            copied = dict(row)
            copied["training_weights"] = {
                qid: float(DEFAULT_LABEL_WEIGHTS.get(str(gold[qid].get("label_source", "")), 0.20))
                for qid in questions
            }
            outputs[split].append(copied)
            assignments[filename][rid] = split

    # A repeated ID across sources would silently duplicate one learning case.
    all_ids = [rid for rows in source_rows.values() for rid in (str(row["id"]) for row in rows)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Record IDs overlap across source files")

    split_groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for filename, groups in group_splits.items():
        for group, split in groups.items():
            if group in split_groups[split]:
                continue
            split_groups[split].add(group)
    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            overlap = split_groups[split_a] & split_groups[split_b]
            if overlap:
                raise ValueError(f"Group leakage across {split_a}/{split_b}: {sorted(overlap)[:5]}")

    metadata = {
        "seed": seed,
        "split_policy": {
            "silicojev_5q.jsonl": "preserve tracked train/validation/test ID assignments",
            "rtl_benchls_5q.jsonl": "preserve converter repository-disjoint split",
            "fixbench_rtl_5q.jsonl": "stable hash split by audited lineage family",
            "veribugbench_5q.jsonl": "stable hash split by source project_id",
        },
        "label_source_weights": DEFAULT_LABEL_WEIGHTS,
        "unknown_label_source_weight": 0.20,
        "split_group_counts": {split: len(groups) for split, groups in split_groups.items()},
        "source_record_counts": {name: len(rows) for name, rows in source_rows.items()},
        "excluded_record_count": len(excluded_rows),
        "excluded_records": excluded_rows,
    }
    return outputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--base-split-dir", type=Path, default=REPO_ROOT / "dataset/normalized/merged_5q")
    parser.add_argument("--fixbench-groups", type=Path, default=REPO_ROOT / "dataset/converted/fixbench_rtl/split_groups.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    expected_files = (
        "silicojev_5q.jsonl",
        "fixbench_rtl_5q.jsonl",
        "rtl_benchls_5q.jsonl",
        "veribugbench_5q.jsonl",
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {args.output_dir}")
    source_rows = {name: read_jsonl(args.input_dir / name) for name in expected_files}
    splits, metadata = make_splits(source_rows, args.base_split_dir, args.fixbench_groups, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split, rows in splits.items():
        rows.sort(key=lambda row: str(row["id"]))
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                source_counts[split][str(row.get("source", "unknown"))] += 1
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metadata["split_records"] = {split: len(rows) for split, rows in splits.items()}
    metadata["source_counts_by_split"] = {
        split: dict(sorted(source_counts[split].items())) for split in SPLITS
    }
    metadata["input_files"] = {
        name: {"sha256": sha256(args.input_dir / name), "bytes": (args.input_dir / name).stat().st_size}
        for name in expected_files
    }
    metadata["total_records"] = sum(len(rows) for rows in splits.values())
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
