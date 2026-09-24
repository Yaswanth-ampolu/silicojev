from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_hf_bucket_data as prepare


def record(rid: str, source: str, **extra):
    questions = {
        "evidence_sufficient": {"type": "noul", "instructions": "Does evidence suffice?", "criteria": {}},
        "next_action": {"type": "choice", "instructions": "What next?", "criteria": {"inspect": "Inspect", "test": "Test"}},
        "root_cause_type": {"type": "choice", "instructions": "Cause?", "criteria": {"logic": "Logic", "reset": "Reset"}},
        "risk": {"type": "score", "instructions": "Risk?", "criteria": ["low", "high"]},
        "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "high"]},
    }
    probabilities = {
        "evidence_sufficient": {"false": 0.2, "true": 0.8},
        "next_action": {"inspect": 0.7, "test": 0.3},
        "root_cause_type": {"logic": 0.6, "reset": 0.4},
        "risk": {"0": 0.9, "1": 0.1},
        "urgency": {"0": 0.8, "1": 0.2},
    }
    gold = {qid: {"probabilities": values, "label_source": "manual_issue_label"} for qid, values in probabilities.items()}
    return {"id": rid, "source": source, "state": {"buggy": "only"},
            "questions": questions, "gold": gold, **extra}


def write_rows(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PrepareSplitTest(unittest.TestCase):
    def test_four_splits_keep_groups_and_original_test_ids_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "base_splits"; split_dir.mkdir()
            base_rows = [record("base_train", "silico", source_group="a"),
                         record("base_val1", "silico", source_group="b"),
                         record("base_val2", "silico", source_group="c"),
                         record("base_test", "silico", source_group="d")]
            write_rows(split_dir / "train.jsonl", [base_rows[0]])
            write_rows(split_dir / "validation.jsonl", base_rows[1:3])
            write_rows(split_dir / "test.jsonl", [base_rows[3]])
            fix_groups = root / "fix_groups.json"
            fix_groups.write_text(json.dumps({"case_group": {"fix_id": "family"}}))
            rows = {
                "silicojev_5q.jsonl": base_rows,
                "fixbench_rtl_5q.jsonl": [record("fix_id", "fixbench")],
                "rtl_benchls_5q.jsonl": [record("rtl_id", "rtlbench", source_group="repo", provenance={"split": "validation"})],
                "veribugbench_5q.jsonl": [record("veri_id", "veribug", provenance={"project_id": "project"})],
            }
            counts = {name: len(data) for name, data in rows.items()}
            with patch.dict(prepare.EXPECTED_COUNTS, counts, clear=True):
                splits, meta = prepare.make_splits(rows, split_dir, fix_groups, seed=7)
            id_split = {row["id"]: split for split, data in splits.items() for row in data}
            self.assertEqual(id_split["base_train"], "train")
            self.assertEqual(id_split["base_test"], "test")
            self.assertIn(id_split["base_val1"], {"validation", "calibration"})
            self.assertIn(id_split["base_val2"], {"validation", "calibration"})
            self.assertNotEqual(id_split["base_val1"], "test")
            self.assertNotEqual(id_split["base_val2"], "test")
            self.assertEqual(sum(len(data) for data in splits.values()), 7)
            self.assertEqual(set(splits), {"train", "validation", "calibration", "test"})
            self.assertEqual(meta["split_group_counts"]["test"], len({row["id"] for row in splits["test"]}))
            self.assertGreater(meta["split_group_counts"]["calibration"] + meta["split_group_counts"]["validation"], 0)

    def test_manifest_helpers_report_distribution_and_weight_mass(self):
        rows = [record("one", "source-a"), record("two", "source-b")]
        for row in rows:
            row["training_weights"] = {qid: 0.25 for qid in row["gold"]}
        stats = prepare.split_statistics(rows)
        mass = prepare.training_mass(rows)
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["decisions"], 10)
        self.assertEqual(stats["source_distribution"], {"source-a": 1, "source-b": 1})
        self.assertAlmostEqual(mass["by_source"]["source-a"]["weighted_mass"], 1.25)
        self.assertAlmostEqual(mass["by_source"]["source-a"]["weighted_mass_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
