from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

try:
    from train_silicojev import load_resume, loss_from_logits, optimizer_learning_rates, save_checkpoint
except (ImportError, ModuleNotFoundError) as exc:  # Laya is an external runtime dependency.
    load_resume = loss_from_logits = optimizer_learning_rates = save_checkpoint = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


class Config:
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "config.json").write_text("{}")


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.config = Config()


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = torch.nn.Linear(4, 2)


class Tokenizer:
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer.json").write_text("{}")


@unittest.skipIf(save_checkpoint is None, f"trainer dependencies unavailable: {IMPORT_ERROR}")
class CheckpointSmokeTest(unittest.TestCase):
    def test_encoder_and_head_learning_rates_are_reported_by_name(self):
        encoder = torch.nn.Parameter(torch.ones(1))
        head = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([
            {"params": [encoder], "lr": 2.5e-5, "name": "encoder"},
            {"params": [head], "lr": 1e-4, "name": "head"},
        ])
        self.assertEqual(optimizer_learning_rates(optimizer), {"encoder": 2.5e-5, "head": 1e-4})

    def test_cpu_rlcd_loss_backward_accumulation_and_optimizer_steps(self):
        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            for _micro in range(2):
                features = torch.randn(2, 4)
                logits = model.head(model.encoder.linear(features))
                batch = {"marker_mask": torch.ones(2, 2, dtype=torch.bool),
                         "target": torch.tensor([[0.8, 0.2], [0.25, 0.75]]),
                         "qtype": torch.zeros(2, dtype=torch.long),
                         "sample_weight": torch.ones(2)}
                loss = loss_from_logits(logits, batch, sigma=0.4, model=model) / 2
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
            optimizer.step()
        self.assertGreaterEqual(optimizer.state[next(model.parameters())]["step"].item(), 3)

    def test_optimizer_checkpoint_roundtrip_and_metrics_append(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = ToyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
            loss = model.head(model.encoder.linear(torch.ones(1, 4))).sum()
            loss.backward(); optimizer.step(); scheduler.step()
            state = {"epoch": 0, "batch_in_epoch": 1, "global_step": 1,
                     "training_config": {"epochs": 2, "dtype": "bf16"}}
            metrics = root / "metrics.jsonl"
            saved = save_checkpoint(root, "step_00000001", model, Tokenizer(), {"temperature": [1, 1, 1]},
                                    optimizer, scheduler, None, state, metrics)
            self.assertTrue((saved / "model.safetensors").is_file())
            self.assertTrue((root / "latest").is_symlink())
            self.assertEqual(json.loads(metrics.read_text().splitlines()[0])["type"], "checkpoint")

            restored = ToyModel()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0)
            restored_state = load_resume(saved, restored, restored_optimizer, restored_scheduler, None, torch.device("cpu"))
            self.assertEqual(restored_state["global_step"], 1)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
            restored_optimizer.zero_grad(set_to_none=True)
            resumed_loss = restored.head(restored.encoder.linear(torch.ones(1, 4))).sum()
            resumed_loss.backward(); restored_optimizer.step(); restored_scheduler.step()
            restored_state["global_step"] += 1
            save_checkpoint(root, "step_00000002", restored, Tokenizer(), {"temperature": [1, 1, 1]},
                            restored_optimizer, restored_scheduler, None, restored_state, metrics)
            self.assertEqual(restored_state["global_step"], 2)
            self.assertEqual(len(metrics.read_text().splitlines()), 2)
            self.assertEqual((root / "latest").resolve().name, "step_00000002")


if __name__ == "__main__":
    unittest.main()
