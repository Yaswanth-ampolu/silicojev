from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline_utils import (
    LengthBucketBatchSampler,
    clean_calibration_config,
    content_fingerprint,
    optimizer_steps_per_epoch,
    permute_choice_item,
    prune_step_checkpoints,
    remap_choice_target,
    resume_compatibility,
    total_optimizer_steps,
    validate_calibration_config,
    warmup_cosine_multiplier,
)
from decision_metrics import distribution_observation, summarize


class PipelineHelpersTest(unittest.TestCase):
    def test_calibration_removes_option_override_and_validates_range(self):
        cfg = clean_calibration_config({"temperature": [1, 1, 1], "temperature_by_options": {"choice:2": 3}}, [0.8, 1.2, 2.0])
        self.assertNotIn("temperature_by_options", cfg)
        validate_calibration_config(cfg)
        with self.assertRaises(ValueError):
            clean_calibration_config({}, [0.1, 1.0, 1.0])

    def test_calibration_config_rejects_conflict(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            validate_calibration_config({"temperature": [1, 1, 1], "temperature_by_options": {}})

    def test_scheduler_steps_cover_partial_batches_and_accumulation(self):
        self.assertEqual(optimizer_steps_per_epoch(17, 4, 2), 3)
        self.assertEqual(total_optimizer_steps(17, 4, 2, 4), 12)
        self.assertEqual(total_optimizer_steps(17, 4, 2, 4, 5), 5)
        self.assertEqual(warmup_cosine_multiplier(0, 100, 0.05), 0.2)
        self.assertAlmostEqual(warmup_cosine_multiplier(100, 100, 0.05), 0.0)

    def test_data_fingerprint_does_not_depend_on_parent_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "a" / "train.jsonl", root / "b" / "train.jsonl"
            left.parent.mkdir(); right.parent.mkdir()
            left.write_text('{"id":"x"}\n'); right.write_text('{"id":"x"}\n')
            self.assertEqual(content_fingerprint([left], {"split": "train"}), content_fingerprint([right], {"split": "train"}))

    def test_resume_allows_epoch_extension_but_rejects_semantic_change(self):
        ok, errors = resume_compatibility({"epochs": 4, "dtype": "bf16", "num_workers": 2},
                                          {"epochs": 5, "dtype": "bf16", "num_workers": 8})
        self.assertTrue(ok, errors)
        ok, errors = resume_compatibility({"epochs": 4, "dtype": "bf16"}, {"epochs": 5, "dtype": "fp16"})
        self.assertFalse(ok)
        self.assertTrue(any("dtype" in error for error in errors))

    def test_length_sampler_keeps_each_example_once_and_shuffles(self):
        sampler = LengthBucketBatchSampler(list(range(1, 101)), batch_size=4, seed=9, bucket_multiplier=3)
        sampler.set_epoch(0); first = list(sampler)
        sampler.set_epoch(1); second = list(sampler)
        flattened = [index for batch in first for index in batch]
        self.assertEqual(sorted(flattened), list(range(100)))
        self.assertNotEqual(first, second)

    def test_choice_permutation_reorders_tokens_markers_and_target_together(self):
        sample = {"ids": [101, 11, 12, 21, 22, 31, 102, 81, 82],
                  "option_spans": [(1, 3), (3, 5), (5, 6)], "separator": 6,
                  "markers": [1, 3, 5], "target": [0.1, 0.7, 0.2], "label": 1, "qtype_name": "choice"}
        permuted = permute_choice_item(sample, [2, 0, 1])
        self.assertEqual(permuted["ids"], [101, 31, 11, 12, 21, 22, 102, 81, 82])
        self.assertEqual(permuted["target"], [0.2, 0.1, 0.7])
        self.assertEqual(permuted["label"], 2)
        self.assertEqual(permuted["markers"], [1, 2, 4])
        self.assertEqual(sample["target"], [0.1, 0.7, 0.2])
        self.assertEqual(remap_choice_target([0.1, 0.7, 0.2], [2, 0, 1]), [0.2, 0.1, 0.7])
        ordinal = {**sample, "qtype_name": "score"}
        self.assertIs(permute_choice_item(ordinal, [2, 0, 1]), ordinal)

    def test_checkpoint_pruning_protects_latest_and_keeps_requested_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in range(1, 6):
                (root / f"step_{step:08d}").mkdir()
            (root / "latest").symlink_to("step_00000005")
            removed = prune_step_checkpoints(root, keep=1)
            self.assertEqual({p.name for p in removed}, {"step_00000001", "step_00000002", "step_00000003"})
            self.assertTrue((root / "step_00000005").exists())
            self.assertTrue((root / "step_00000004").exists())

    def test_shared_decision_metrics_and_question_summary(self):
        obs = distribution_observation("choice", [0.8, 0.2], [0.7, 0.3])
        self.assertEqual(obs["correct"], 1.0)
        self.assertAlmostEqual(obs["soft_accuracy"], 0.62)
        summary = summarize([obs])
        self.assertEqual(summary["n"], 1)
        self.assertAlmostEqual(summary["brier_score"], 0.02)


if __name__ == "__main__":
    unittest.main()
