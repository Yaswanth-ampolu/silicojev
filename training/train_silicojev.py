#!/usr/bin/env python3
"""Resumable A100/H100 fine-tuning for SilicoJev.

The script keeps the Laya architecture and typed output contract, but trains on
normalized SilicoJev JSONL records. Checkpoints are self-contained and expose a
stable ``latest`` link for notebook restarts.
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter, deque
import fcntl
from functools import partial
import hashlib
from itertools import islice
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYA_SOURCE = Path(os.environ.get("SILICOJEV_LAYA_SOURCE", REPO_ROOT / "external" / "laya")).resolve()
if str(LAYA_SOURCE) not in sys.path:
    sys.path.insert(0, str(LAYA_SOURCE))

from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, DecisionModel, build_sequence, proper_reward, render_options  # noqa: E402
from decision_metrics import distribution_observation, summarize  # noqa: E402
from pipeline_utils import (  # noqa: E402
    LengthBucketBatchSampler,
    clean_calibration_config,
    content_fingerprint,
    jsonl_fingerprint,
    optimizer_steps_per_epoch,
    permute_choice_item,
    prune_step_checkpoints,
    resume_compatibility,
    total_optimizer_steps,
    validate_calibration_config,
    warmup_cosine_multiplier,
)


def parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def target_for(q: dict[str, Any], gold_q: dict[str, Any]) -> list[float]:
    qtype = q["type"]
    criteria = q.get("criteria", {})
    probs = gold_q.get("probabilities", {})
    if qtype == "choice":
        keys = list(criteria.keys())
        target = [float(probs.get(key, 0.0)) for key in keys]
    elif qtype == "noul":
        target = [float(probs.get("false", 0.5)), float(probs.get("true", 0.5))]
    elif qtype == "score":
        target = [float(probs.get(str(i), 0.0)) for i in range(len(criteria))]
    else:
        raise ValueError(f"Unsupported question type: {qtype}")
    total = sum(target)
    if not target or any(not np.isfinite(value) or value < 0 for value in target) or total <= 0 or abs(total - 1.0) > 1e-3:
        raise ValueError("Target probabilities must be finite, nonnegative, and sum to 1")
    return [value / total for value in target]


def build_items(rows: list[dict[str, Any]], tokenizer, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = []
    skipped: Counter[str] = Counter()
    skipped_by_question: Counter[str] = Counter()
    skipped_by_source: Counter[str] = Counter()
    lengths: dict[str, list[int]] = {"state": [], "head": [], "total": []}
    truncation = Counter()
    for row in rows:
        try:
            state = parse_json(row["state"])
            questions = parse_json(row["questions"])
            gold = parse_json(row["gold"])
        except (KeyError, TypeError, json.JSONDecodeError):
            skipped["malformed_data"] += 1
            skipped_by_source[str(row.get("source") or "unknown")] += 1
            continue
        if not isinstance(questions, dict) or not isinstance(gold, dict):
            skipped["malformed_data"] += 1
            skipped_by_source[str(row.get("source") or "unknown")] += 1
            continue
        for qid, q in questions.items():
            reason = None
            if qid not in gold:
                reason = "missing_target"
            elif not isinstance(q, dict):
                reason = "malformed_question"
            elif q.get("type") not in QTYPES:
                reason = "unsupported_question_type"
            if reason:
                skipped[reason] += 1
                skipped_by_question[qid] += 1
                skipped_by_source[str(row.get("source") or "unknown")] += 1
                continue
            try:
                target = target_for(q, gold[qid])
                if len(target) < 2 or any(value < 0 for value in target) or not np.isfinite(target).all() or abs(sum(target) - 1.0) > 1e-4:
                    raise ValueError("invalid probability target")
            except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError):
                reason = "invalid_target"
                skipped[reason] += 1
                skipped_by_question[qid] += 1
                skipped_by_source[str(row.get("source") or "unknown")] += 1
                continue
            try:
                internal_q = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
                instruction_text = f"{internal_q['t']} question: {str(internal_q['ins']).replace(tokenizer.mask_token, ' ')}"
                instruction_tokens = len(tokenizer(instruction_text, add_special_tokens=False)["input_ids"])
                option_tokens = sum(
                    1 + min(48, len(tokenizer(" " + option.replace(tokenizer.mask_token, " "), add_special_tokens=False)["input_ids"]))
                    for option in render_options(internal_q)
                )
                untruncated_head_tokens = instruction_tokens + option_tokens
                seq, markers = build_sequence(
                    tokenizer, state, internal_q,
                    int(cfg.get("max_len", 1024)),
                    int(cfg.get("head_max_len", 256)),
                )
            except (KeyError, TypeError, ValueError, IndexError, RuntimeError, AttributeError):
                reason = "tokenization_or_sequence_construction"
                skipped[reason] += 1
                skipped_by_question[qid] += 1
                skipped_by_source[str(row.get("source") or "unknown")] += 1
                continue
            if len(markers) != len(target):
                reason = "marker_option_count_mismatch"
                skipped[reason] += 1
                skipped_by_question[qid] += 1
                skipped_by_source[str(row.get("source") or "unknown")] += 1
                continue
            sep_positions = [i for i in range(markers[-1] + 1, len(seq)) if seq[i] == tokenizer.sep_token_id]
            last_end = sep_positions[0] if sep_positions else len(seq)
            spans = [(markers[i], markers[i + 1] if i + 1 < len(markers) else last_end) for i in range(len(markers))]
            state_ids = tokenizer(json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state, add_special_tokens=False)["input_ids"]
            head_tokens = untruncated_head_tokens
            lengths["state"].append(len(state_ids))
            lengths["head"].append(head_tokens)
            lengths["total"].append(len(seq))
            truncation["state_truncated"] += int(len(state_ids) > max(0, int(cfg.get("max_len", 1024)) - last_end - 2))
            truncation["head_truncated"] += int(head_tokens > int(cfg.get("head_max_len", 256)))
            items.append({
                "ids": seq,
                "markers": markers,
                "option_spans": spans,
                "separator": last_end,
                "option_count": len(markers),
                "qtype": QTYPES[q["type"]],
                "qtype_name": q["type"],
                "target": target,
                "label": int(np.argmax(target)),
                "case_id": row.get("id"),
                "question_id": qid,
                "weight": float((row.get("training_weights") or {}).get(qid, 1.0)),
                "source": str(row.get("source") or "unknown"),
                "label_source": str(gold[qid].get("label_source") or "unspecified"),
            })
    if not items:
        raise ValueError("No tokenizable training items were produced")
    def percentile(values: list[int]) -> dict[str, float | int]:
        return ({f"p{p}": float(np.percentile(values, p)) for p in (50, 75, 90, 95, 99)} | {"max": max(values)}) if values else {}

    return items, {
        "count": dict(skipped),
        "by_question_id": dict(skipped_by_question),
        "by_source": dict(skipped_by_source),
        "lengths": {key: percentile(value) for key, value in lengths.items()},
        "truncation_percent": {
            key: 100.0 * value / max(len(items), 1) for key, value in truncation.items()
        },
    }


def collate(items: list[dict[str, Any]], pad_id: int, choice_permutation_prob: float = 0.0) -> dict[str, torch.Tensor]:
    if choice_permutation_prob:
        augmented = []
        for item in items:
            item_rng = random.Random(int(item.get("augmentation_seed", 0)))
            if item.get("qtype_name") == "choice" and item_rng.random() < choice_permutation_prob:
                order = list(range(item["option_count"]))
                item_rng.shuffle(order)
                item = permute_choice_item(item, order)
            augmented.append(item)
        items = augmented
    n = len(items)
    length = max(len(item["ids"]) for item in items)
    kmax = max(len(item["markers"]) for item in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, kmax), dtype=torch.long)
    marker_mask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    qtype = torch.zeros(n, dtype=torch.long)
    label = torch.zeros(n, dtype=torch.long)
    sample_weight = torch.ones(n, dtype=torch.float32)
    for i, item in enumerate(items):
        ids[i, :len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        attention[i, :len(item["ids"])] = 1
        k = len(item["markers"])
        marker_pos[i, :k] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[i, :k] = True
        target[i, :len(item["target"])] = torch.tensor(item["target"], dtype=torch.float32)
        qtype[i] = item["qtype"]
        label[i] = item["label"]
        sample_weight[i] = float(item.get("weight", 1.0))
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": qtype,
        "label": label,
        "sample_weight": sample_weight,
    }


class DecisionDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_from_logits(logits: torch.Tensor, batch: dict[str, torch.Tensor], sigma: float, model) -> torch.Tensor:
    mask = batch["marker_mask"]
    target = batch["target"]
    qtype = batch["qtype"]
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float().clamp_min(1.0)
    eps = torch.randn((4,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(advantage * logp).mean(0)
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    weights = batch.get("sample_weight", torch.ones_like(loss_ce)).float().clamp_min(0.0)
    denominator = weights.sum().clamp_min(1e-8)
    return ((loss_rl + loss_ce) * weights).sum() / denominator


@torch.no_grad()
def evaluate(
    model,
    items: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
    batch_size: int,
    temperatures: list[float] | None = None,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    total = 0
    observations: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(items), batch_size):
        chosen = items[start:start + batch_size]
        batch = move_batch(collate(chosen, pad_id), device)
        logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
        mask = batch["marker_mask"]
        target = batch["target"]
        if temperatures is not None:
            scales = torch.as_tensor(temperatures, dtype=torch.float32, device=device)[batch["qtype"]]
            logits = logits.float() / scales[:, None]
        loss = -(target * torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        probabilities = torch.softmax(logits.float().masked_fill(~mask, -1e4), -1).cpu().numpy()
        targets = target.cpu().numpy()
        total += len(chosen)
        loss_sum += float(loss.item()) * len(chosen)
        for i, item in enumerate(chosen):
            k = len(item["target"])
            obs = distribution_observation(item["qtype_name"], probabilities[i, :k], targets[i, :k])
            if item["qtype_name"] == "score":
                pred_score = float(np.dot(np.arange(k), probabilities[i, :k]))
                gold_score = float(np.dot(np.arange(k), targets[i, :k]))
                obs["score_mae"] = abs(pred_score - gold_score)
                obs["within_1_level"] = float(abs(pred_score - gold_score) <= 1.0)
            observations.append(obs)
            by_type.setdefault(item["qtype_name"], []).append(obs)
    model.train()
    result = summarize(observations)
    result["loss"] = loss_sum / max(total, 1)
    result["items"] = total
    result["by_question_type"] = {name: summarize(rows) for name, rows in sorted(by_type.items())}
    return result


@torch.no_grad()
def collect_calibration_logits(
    model,
    items: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
    batch_size: int,
) -> list[tuple[int, np.ndarray, list[float]]]:
    """Collect uncalibrated validation logits and soft targets.

    Calibration must see logits before ``Agent`` applies the checkpoint's
    temperature.  Keeping this pass here also makes the calibration split and
    the training tokenizer/model configuration exactly match.
    """
    model.eval()
    collected: list[tuple[int, np.ndarray, list[float]]] = []
    for start in range(0, len(items), batch_size):
        chosen = items[start:start + batch_size]
        batch = move_batch(collate(chosen, pad_id), device)
        logits, _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
        logits = logits.float().cpu()
        for row_idx, item in enumerate(chosen):
            k = len(item["target"])
            collected.append((
                int(item["qtype"]),
                logits[row_idx, :k].numpy(),
                list(item["target"]),
            ))
    model.train()
    return collected


def fit_one_temperature(
    samples: list[tuple[np.ndarray, list[float]]],
    minimum: float = 0.5,
    maximum: float = 5.0,
) -> float:
    """Fit one scalar temperature by validation cross-entropy."""
    if len(samples) < 10:
        return 1.0
    kmax = max(len(logits) for logits, _ in samples)
    logits = torch.full((len(samples), kmax), -1e4, dtype=torch.float32)
    targets = torch.zeros((len(samples), kmax), dtype=torch.float32)
    for row_idx, (row_logits, row_target) in enumerate(samples):
        k = len(row_logits)
        logits[row_idx, :k] = torch.as_tensor(row_logits, dtype=torch.float32)
        targets[row_idx, :len(row_target)] = torch.as_tensor(row_target, dtype=torch.float32)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = -(targets * torch.log_softmax(logits / temperature, dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    value = float(log_temperature.exp().detach().clamp(minimum, maximum).item())
    return value if np.isfinite(value) else 1.0


def fit_calibration_temperatures(
    model,
    items: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], dict[str, Any]]:
    """Fit Laya-compatible temperatures for choice, score, and noul heads."""
    samples = collect_calibration_logits(model, items, pad_id, device, batch_size)
    by_type: dict[int, list[tuple[np.ndarray, list[float]]]] = {0: [], 1: [], 2: []}
    for qtype, logits, target in samples:
        by_type.setdefault(qtype, []).append((logits, target))
    temperatures = []
    counts = {}
    for qtype, name in ((0, "choice"), (1, "score"), (2, "noul")):
        selected = by_type.get(qtype, [])
        counts[name] = len(selected)
        try:
            temperatures.append(fit_one_temperature(selected))
        except (FloatingPointError, RuntimeError, ValueError):
            temperatures.append(1.0)
    status = {
        "enabled": True,
        "method": "calibration_nll_lbfgs",
        "split": "calibration",
        "items_by_type": counts,
        "score_enabled": counts["score"] > 0,
        "score_note": (
            "Score temperature fitted from held-out calibration score items."
            if counts["score"] > 0
            else "Score is disabled: the normalized SilicoJev data contains no defensible ordered score labels."
        ),
    }
    return temperatures, status


def gpu_memory_metrics(device: torch.device | None = None) -> dict[str, float]:
    """Return allocator memory in GiB without synchronizing the CUDA device."""
    if not torch.cuda.is_available():
        return {
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_peak_gb": 0.0,
        }
    return {
        "gpu_allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "gpu_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "gpu_peak_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one complete JSON record; metrics survive restarts and are never reset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def optimizer_learning_rates(optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def report_cuda_oom(exc: RuntimeError, epoch: int, micro_step: int, global_step: int, batch_size: int) -> None:
    if "out of memory" not in str(exc).lower():
        return
    memory = gpu_memory_metrics(torch.device("cuda") if torch.cuda.is_available() else None)
    print(
        f"[oom] epoch={epoch} micro_step={micro_step} global_step={global_step} "
        f"micro_batch_size={batch_size} allocated={memory['gpu_allocated_gb']:.2f}GiB "
        f"reserved={memory['gpu_reserved_gb']:.2f}GiB peak={memory['gpu_peak_gb']:.2f}GiB",
        flush=True,
    )


def save_model_artifacts(model, tokenizer, cfg: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    model.encoder.config.save_pretrained(directory / "encoder")
    tokenizer.save_pretrained(directory / "tokenizer")
    (directory / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))


def save_checkpoint(
    root: Path,
    name: str,
    model,
    tokenizer,
    cfg: dict[str, Any],
    optimizer,
    scheduler,
    scaler,
    state: dict[str, Any],
    metrics_path: Path | None = None,
    update_latest: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final = root / name
    temp = root / f".tmp_{name}_{os.getpid()}"
    print(f"[checkpoint] saving step={state.get('global_step')} path={final}", flush=True)
    backup = root / f".tmp_previous_{name}_{os.getpid()}"
    try:
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir(parents=True)
        save_model_artifacts(model, tokenizer, cfg, temp)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "state": state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, temp / "training_state.pt")
        (temp / "checkpoint_meta.json").write_text(json.dumps(state, indent=2, default=str))
        required = ("model.safetensors", "training_state.pt", "checkpoint_meta.json", "rl_agent_config.json", "encoder/config.json")
        if any(not (temp / item).is_file() or (temp / item).stat().st_size == 0 for item in required):
            raise RuntimeError(f"Checkpoint verification failed in {temp}")
        if final.exists():
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(final, backup)
        os.replace(temp, final)
        if backup.exists():
            shutil.rmtree(backup)
        if update_latest:
            latest_tmp = root / f".latest_{os.getpid()}"
            if latest_tmp.exists() or latest_tmp.is_symlink():
                latest_tmp.unlink()
            latest_tmp.symlink_to(final.name)
            os.replace(latest_tmp, root / "latest")
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    print(f"[checkpoint] saved path={final}", flush=True)
    if metrics_path is not None:
        checkpoint_event = {
            "type": "checkpoint",
            "step": state.get("global_step"),
            "epoch": state.get("epoch"),
            "path": str(final),
        }
        if state.get("elapsed_seconds") is not None:
            checkpoint_event["elapsed_seconds"] = state["elapsed_seconds"]
        append_jsonl(metrics_path, checkpoint_event)
    return final


def load_resume(path: Path, model, optimizer, scheduler, scaler, device: torch.device) -> dict[str, Any]:
    weights = load_file(str(path / "model.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True)
    ckpt = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    state = ckpt["state"]
    random.setstate(ckpt["python_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    torch.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return state


def _main_pipeline() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a Laya typed-decision checkpoint for SilicoJev")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-model-id", default="convaiinnovations/laya-typed-decisions")
    ap.add_argument("--base-model-revision", default="1a793eb568e6718f15941d08f85432581df534e3")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder-lr", type=float, default=2.5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--gradient-checkpointing", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--fused-adamw", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--attention-backend", choices=("auto", "sdpa", "flash_attention_2"), default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--choice-permutation-prob", type=float, default=0.0)
    ap.add_argument("--max-skip-ratio", type=float, default=0.02)
    ap.add_argument("--best-metric", choices=("brier_score", "loss", "accuracy", "soft_accuracy", "ece"), default="brier_score")
    ap.add_argument("--best-mode", choices=("min", "max"), default="min")
    args = ap.parse_args()
    if args.epochs < 1 or min(args.micro_batch_size, args.grad_accum_steps) < 1:
        ap.error("epochs, micro-batch-size, and grad-accum-steps must be positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.choice_permutation_prob <= 1:
        ap.error("warmup-ratio must be in [0,1); choice-permutation-prob must be in [0,1]")
    if args.num_workers < 0 or args.prefetch_factor < 1 or args.keep_checkpoints < 0:
        ap.error("num-workers/keep-checkpoints must be nonnegative and prefetch-factor positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SilicoJev training")
    device = torch.device("cuda")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but this GPU/PyTorch build does not support it")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    amp_enabled = args.dtype != "fp32"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / ".training.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"Another trainer holds {root / '.training.lock'}") from exc
    atexit.register(lock_handle.close)
    for abandoned in root.glob(".tmp_*"):
        if abandoned.is_dir():
            shutil.rmtree(abandoned)
        elif abandoned.is_file() or abandoned.is_symlink():
            abandoned.unlink()
    model_dir = args.model_dir.resolve()
    train_path, valid_path = args.train.resolve(), args.validation.resolve()
    calibration_path = (args.calibration or valid_path.with_name("calibration.jsonl")).resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Separate calibration split is required at {calibration_path}; validation must not fit temperatures.")
    _fix_tokenizer_config(str(model_dir))
    cfg_path = model_dir / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.update({"max_len": args.max_len, "head_max_len": args.head_max_len, "fine_tuned": True,
                "model_name": "silicojev", "amp_dtype": args.dtype,
                "base_model_id": args.base_model_id, "base_model_revision": args.base_model_revision})
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    train_rows, valid_rows, calibration_rows = load_rows(train_path), load_rows(valid_path), load_rows(calibration_path)

    data_parts = {
        "train": jsonl_fingerprint(train_path, "train"),
        "validation": jsonl_fingerprint(valid_path, "validation"),
        "calibration": jsonl_fingerprint(calibration_path, "calibration"),
    }
    test_path = args.test.resolve() if args.test else valid_path.with_name("test.jsonl")
    if not test_path.is_file():
        raise FileNotFoundError(f"Untouched final test split is required at {test_path}")
    data_parts["test"] = jsonl_fingerprint(test_path, "test")
    data_fp = hashlib.sha256(json.dumps(data_parts, sort_keys=True).encode()).hexdigest()
    model_files = [model_dir / "model.safetensors", cfg_path, model_dir / "encoder" / "config.json"]
    tokenizer_dir = model_dir / "tokenizer"
    if tokenizer_dir.is_dir():
        model_files.extend(path for path in tokenizer_dir.iterdir() if path.is_file())
    missing_model_files = [path for path in model_files[:3] if not path.is_file()]
    if missing_model_files:
        raise FileNotFoundError("Base checkpoint is incomplete: " + ", ".join(str(path) for path in missing_model_files))
    model_fp = content_fingerprint(model_files,
                                   {"base_model_id": args.base_model_id, "base_model_revision": args.base_model_revision})
    train_config = {
        "max_len": args.max_len, "head_max_len": args.head_max_len,
        "micro_batch_size": args.micro_batch_size, "grad_accum_steps": args.grad_accum_steps,
        "dtype": args.dtype, "seed": args.seed, "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr, "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "sigma_start": args.sigma_start, "sigma_end": args.sigma_end,
        "choice_permutation_prob": args.choice_permutation_prob,
        "best_metric": args.best_metric,
        "best_mode": args.best_mode,
    }
    train_identity = {**train_config, "epochs": args.epochs, "max_steps": args.max_steps}
    train_items, train_analysis = build_items(train_rows, tokenizer, cfg)
    valid_items, valid_analysis = build_items(valid_rows, tokenizer, cfg)
    calibration_items, calibration_analysis = build_items(calibration_rows, tokenizer, cfg)
    for item_index, item in enumerate(train_items):
        item["item_index"] = item_index
    for name, items, report in (("train", train_items, train_analysis), ("validation", valid_items, valid_analysis),
                                ("calibration", calibration_items, calibration_analysis)):
        skipped = sum(report["count"].values()); ratio = skipped / max(skipped + len(items), 1)
        print(f"[{name}] usable={len(items):,} skipped={skipped:,} ({ratio:.2%}) reasons={report['count']}", flush=True)
        if report["count"]:
            print(f"[{name}] skipped_by_question={report['by_question_id']} by_source={report['by_source']}", flush=True)
        print(f"[{name}] token_lengths={report['lengths']} truncation={report['truncation_percent']}", flush=True)
        if ratio > args.max_skip_ratio:
            raise ValueError(f"{name} skip ratio {ratio:.2%} exceeds {args.max_skip_ratio:.2%}")

    if args.attention_backend == "auto":
        try:
            import flash_attn  # noqa: F401
            attention = "flash_attention_2"
        except ImportError:
            attention = "sdpa"
            print("[attention] FlashAttention 2 unavailable; using SDPA", flush=True)
    else:
        attention = args.attention_backend
    try:
        encoder_cfg = AutoConfig.from_pretrained(model_dir / "encoder")
        encoder = AutoModel.from_config(encoder_cfg, attn_implementation=attention)
    except (ImportError, RuntimeError, ValueError) as exc:
        if args.attention_backend != "auto" or attention == "sdpa":
            raise
        print(f"[attention] FlashAttention 2 failed ({exc}); falling back to SDPA", flush=True)
        attention = "sdpa"
        encoder_cfg = AutoConfig.from_pretrained(model_dir / "encoder")
        encoder = AutoModel.from_config(encoder_cfg, attn_implementation="sdpa")
    cfg["attention_backend"] = attention
    model = DecisionModel(encoder, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device="cpu"), strict=True)
    model.encoder.config.reference_compile = False
    if args.gradient_checkpointing == "auto":
        gradient_checkpointing = torch.cuda.mem_get_info(device)[0] / (1024**3) < 60.0
    else:
        gradient_checkpointing = args.gradient_checkpointing == "on"
    cfg["gradient_checkpointing"] = gradient_checkpointing
    train_identity["resolved_attention_backend"] = attention
    train_identity["gradient_checkpointing"] = gradient_checkpointing
    if gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = False
    model.to(device); model.train()
    enc_params = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    groups = [{"params": enc_params, "lr": args.encoder_lr, "name": "encoder"},
              {"params": head_params, "lr": args.head_lr, "name": "head"}]
    fused = args.fused_adamw != "off"
    try:
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, fused=fused)
    except Exception as exc:
        if args.fused_adamw == "on":
            raise
        print(f"[optimizer] fused AdamW unavailable ({exc}); using standard AdamW", flush=True)
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay); fused = False
    train_identity["fused_adamw"] = fused
    steps_epoch = optimizer_steps_per_epoch(len(train_items), args.micro_batch_size, args.grad_accum_steps)
    total_steps = total_optimizer_steps(len(train_items), args.micro_batch_size, args.grad_accum_steps, args.epochs, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_cosine_multiplier(s, total_steps, args.warmup_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")
    metrics_path = root / "metrics.jsonl"
    tb = None
    if SummaryWriter:
        try:
            tb = SummaryWriter(str(root / "tensorboard")); atexit.register(tb.close)
        except (ImportError, OSError) as exc:
            print(f"[tensorboard] unavailable: {exc}", flush=True)

    resume_state = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0}
    resume_status = "fresh run" if args.resume != "none" else "disabled"
    if args.resume != "none":
        resume_path = root / "latest" if args.resume == "auto" else Path(args.resume)
        if resume_path.exists():
            resolved = resume_path.resolve(); meta = json.loads((resolved / "checkpoint_meta.json").read_text())
            expected = {"data_fingerprint": data_fp, "model_fingerprint": model_fp,
                        "base_model_id": args.base_model_id, "base_model_revision": args.base_model_revision}
            errors = [f"{k}: saved={meta.get(k)!r}, current={v!r}" for k, v in expected.items() if meta.get(k) != v]
            compatible, config_errors = resume_compatibility(meta.get("training_config", {}), train_identity)
            errors.extend(config_errors)
            if errors:
                raise RuntimeError("[resume] incompatibilities; refusing resume:\n  - " + "\n  - ".join(errors))
            resume_state = load_resume(resolved, model, optimizer, scheduler, scaler, device)
            resume_status = str(resolved)
            print(f"[resume] checkpoint={resolved} starting_step={resume_state['global_step']} starting_epoch={resume_state['epoch'] + 1}", flush=True)
    for group, name in zip(optimizer.param_groups, ("encoder", "head")):
        group["name"] = name

    properties = torch.cuda.get_device_properties(device)
    pad_id = tokenizer.pad_token_id or 0
    print("=" * 72, flush=True); print("SilicoJev Training", flush=True); print("=" * 72, flush=True)
    print(f"GPU: {properties.name} | CUDA: {torch.version.cuda} | VRAM: {properties.total_memory / 1024**3:.1f} GiB", flush=True)
    print(f"Base: {args.base_model_id}@{args.base_model_revision} | attention={attention} | precision={args.dtype}", flush=True)
    print(f"Examples(decisions): train={len(train_items):,} validation={len(valid_items):,} calibration={len(calibration_items):,}", flush=True)
    print(f"Test split: held out for final evaluation only ({data_parts['test'][:12]} fingerprint)", flush=True)
    print(f"Epochs={args.epochs} batches/epoch={math.ceil(len(train_items)/args.micro_batch_size)} optimizer_steps/epoch={steps_epoch} total_steps={total_steps}", flush=True)
    data_parallel_world_size = 1
    print(f"Visible GPUs={torch.cuda.device_count()} | data-parallel world size={data_parallel_world_size}", flush=True)
    print(f"Micro batch={args.micro_batch_size} accumulation={args.grad_accum_steps} effective batch={args.micro_batch_size*args.grad_accum_steps*data_parallel_world_size}", flush=True)
    print(f"LR encoder={args.encoder_lr:g} head={args.head_lr:g} weight_decay={args.weight_decay:g} warmup={args.warmup_ratio:.1%}", flush=True)
    print(f"max_len={args.max_len} head_max_len={args.head_max_len} grad_checkpointing={gradient_checkpointing} fused_adamw={fused}", flush=True)
    print(f"workers={args.num_workers} prefetch={args.prefetch_factor} pin_memory={args.pin_memory} output={root} resume={resume_status}", flush=True)
    print(f"DATA_FINGERPRINT={data_fp} MODEL_FINGERPRINT={model_fp}", flush=True)
    print("=" * 72, flush=True)
    append_jsonl(metrics_path, {"type":"run", "data_fingerprint":data_fp, "data_fingerprints":data_parts,
                                "model_fingerprint":model_fp, "training_config":train_identity,
                                "base_model_id":args.base_model_id, "base_model_revision":args.base_model_revision,
                                "attention_backend":attention, "gradient_checkpointing":gradient_checkpointing,
                                "fused_adamw":fused})
    torch.cuda.reset_peak_memory_stats(device)
    sampler = LengthBucketBatchSampler([len(item["ids"]) for item in train_items], args.micro_batch_size, args.seed)
    loader_options: dict[str, Any] = {"dataset": DecisionDataset(train_items), "batch_sampler": sampler,
                                      "collate_fn": partial(collate, pad_id=pad_id, choice_permutation_prob=args.choice_permutation_prob), "num_workers": args.num_workers,
                                      "pin_memory": bool(args.pin_memory)}
    if args.num_workers:
        loader_options.update(prefetch_factor=args.prefetch_factor, persistent_workers=True)
    loader = DataLoader(**loader_options)
    best_value = None
    if resume_state.get("global_step", 0) > 0 and (root / "best" / "checkpoint_meta.json").exists():
        best_value = json.loads((root / "best" / "checkpoint_meta.json").read_text()).get("best_metric_value")
    run_start = time.perf_counter(); total_samples = total_tokens = 0
    completed_epoch = int(resume_state["epoch"])

    for epoch in range(int(resume_state["epoch"]), args.epochs):
        if args.max_steps and resume_state["global_step"] >= args.max_steps:
            break
        for item in train_items:
            item["augmentation_seed"] = args.seed + (epoch + 1) * 1_000_003 + item["item_index"] * 9_176
        sampler.set_epoch(epoch); first_batch = int(resume_state["batch_in_epoch"]) if epoch == int(resume_state["epoch"]) else 0
        optimizer.zero_grad(set_to_none=True); accum = 0; epoch_loss = 0.0; epoch_n = 0
        epoch_actual_tokens = epoch_padded_tokens = 0
        rolling = deque(maxlen=100)
        remaining_loader = enumerate(islice(loader, first_batch, None), start=first_batch)
        progress = tqdm(remaining_loader, total=len(loader) - first_batch,
                        desc=f"Epoch {epoch+1}/{args.epochs}", unit="micro", dynamic_ncols=True)
        for batch_index, cpu_batch in progress:
            cpu_batch_tokens = int(cpu_batch["attention_mask"].sum().item())
            cpu_padded_tokens = int(cpu_batch["input_ids"].numel())
            epoch_actual_tokens += cpu_batch_tokens
            epoch_padded_tokens += cpu_padded_tokens
            batch = move_batch(cpu_batch, device)
            window_start = (batch_index // args.grad_accum_steps) * args.grad_accum_steps
            window_size = min(args.grad_accum_steps, len(loader) - window_start)
            progress_sigma = min(1.0, resume_state["global_step"] / max(total_steps - 1, 1))
            sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress_sigma
            try:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
                    raw_loss = loss_from_logits(logits, batch, sigma, model)
                    loss = raw_loss / window_size
                scaler.scale(loss).backward(); accum += 1
                loss_value = float(raw_loss.detach().item()); n = int(batch["label"].numel())
                epoch_loss += loss_value * n; epoch_n += n; total_samples += n; total_tokens += cpu_batch_tokens
                rolling.append((loss_value * n, n))
                last_micro = batch_index + 1 == len(loader)
                updated = accum >= window_size
                if updated:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    did_optimizer_update = not scaler.is_enabled() or scaler.get_scale() >= old_scale
                    optimizer.zero_grad(set_to_none=True)
                    if did_optimizer_update:
                        scheduler.step()
                        resume_state["global_step"] += 1
                    accum = 0
                    if did_optimizer_update and args.checkpoint_every > 0 and resume_state["global_step"] % args.checkpoint_every == 0:
                        state = {"epoch": epoch, "batch_in_epoch": batch_index + 1, "global_step": resume_state["global_step"],
                                 "data_fingerprint": data_fp, "data_fingerprints": data_parts, "model_fingerprint": model_fp,
                                 "base_model_id": args.base_model_id, "base_model_revision": args.base_model_revision,
                                 "training_config": train_identity, "train_items": len(train_items),
                                 "validation_items": len(valid_items), "calibration_items": len(calibration_items)}
                        save_checkpoint(root, f"step_{resume_state['global_step']:08d}", model, tokenizer, cfg,
                                        optimizer, scheduler, scaler, state, metrics_path)
                        prune_step_checkpoints(root, args.keep_checkpoints)
                    should_log = did_optimizer_update and (resume_state["global_step"] % 10 == 0 or last_micro or (args.max_steps and resume_state["global_step"] >= args.max_steps))
                    if should_log:
                        elapsed = max(time.perf_counter() - run_start, 1e-9); rates = optimizer_learning_rates(optimizer); memory = gpu_memory_metrics(device)
                        record = {"type": "train", "epoch": epoch+1, "step": resume_state["global_step"],
                                  "micro_step": epoch * len(loader) + batch_index + 1, "loss": loss_value,
                                  "average_loss": epoch_loss / max(epoch_n, 1), "encoder_lr": rates["encoder"], "head_lr": rates["head"],
                                  "mean_actual_tokens_per_batch": epoch_actual_tokens / max(batch_index - first_batch + 1, 1),
                                  "mean_padded_tokens_per_batch": epoch_padded_tokens / max(batch_index - first_batch + 1, 1),
                                  "padding_percent": 100.0 * (1.0 - epoch_actual_tokens / max(epoch_padded_tokens, 1)),
                                  "samples_per_second": total_samples / elapsed, "tokens_per_second": total_tokens / elapsed,
                                  "sigma": sigma, "base_model_id": args.base_model_id, "base_model_revision": args.base_model_revision,
                                  "data_fingerprint": data_fp, "model_fingerprint": model_fp, **memory, "elapsed_seconds": elapsed}
                        append_jsonl(metrics_path, record)
                        if tb:
                            for key in ("loss", "average_loss", "encoder_lr", "head_lr", "samples_per_second"):
                                tag = "throughput/samples_per_second" if key == "samples_per_second" else f"train/{key}"
                                tb.add_scalar(tag, record[key], record["step"])
                            tb.add_scalar("gpu/allocated_gb", memory["gpu_allocated_gb"], record["step"])
                            tb.add_scalar("gpu/reserved_gb", memory["gpu_reserved_gb"], record["step"]); tb.flush()
                elapsed = max(time.perf_counter() - run_start, 1e-9); memory = gpu_memory_metrics(device); rates = optimizer_learning_rates(optimizer)
                rolling_n = sum(count for _, count in rolling); rolling_loss = sum(value for value, _ in rolling) / max(rolling_n, 1)
                progress.set_postfix(step=resume_state["global_step"], loss=f"{loss_value:.4f}", avg=f"{epoch_loss/max(epoch_n,1):.4f}",
                                     roll=f"{rolling_loss:.4f}", enc_lr=f"{rates['encoder']:.2g}", head_lr=f"{rates['head']:.2g}",
                                     vram=f"{memory['gpu_allocated_gb']:.1f}G", reserved=f"{memory['gpu_reserved_gb']:.1f}G",
                                     peak=f"{memory['gpu_peak_gb']:.1f}G", **{"samples/s":f"{total_samples/elapsed:.1f}", "tokens/s":f"{total_tokens/elapsed:.0f}"})
                if updated and args.max_steps and resume_state["global_step"] >= args.max_steps:
                    break
            except RuntimeError as exc:
                report_cuda_oom(exc, epoch+1, batch_index+1, resume_state["global_step"], args.micro_batch_size)
                raise

        validation_temperatures, validation_calibration = fit_calibration_temperatures(
            model, calibration_items, pad_id, device, args.micro_batch_size
        )
        metrics = evaluate(model, valid_items, pad_id, device, args.micro_batch_size, validation_temperatures)
        validation_record = {"type": "validation", "epoch": epoch+1, "step": resume_state["global_step"], **metrics,
                             "calibration_temperatures": validation_temperatures,
                             "base_model_id": args.base_model_id, "base_model_revision": args.base_model_revision,
                             "data_fingerprint": data_fp, "model_fingerprint": model_fp}
        append_jsonl(metrics_path, validation_record)
        print(f"[validation] epoch={epoch+1} step={resume_state['global_step']} loss={metrics['loss']:.6f} "
              f"acc={metrics['accuracy']:.4f} soft_acc={metrics['soft_accuracy']:.4f} brier={metrics['brier_score']:.6f} "
              f"ece={metrics['ece']:.6f}", flush=True)
        if tb:
            for metric in ("loss", "accuracy", "soft_accuracy", "brier_score", "ece"):
                tb.add_scalar(f"validation/{metric}", metrics[metric], resume_state["global_step"])
            tb.flush()
        value = metrics[args.best_metric]
        improved = value is not None and (best_value is None or (value < best_value if args.best_mode == "min" else value > best_value))
        common_state = {"epoch": epoch+1, "batch_in_epoch": 0, "global_step": resume_state["global_step"],
                        "data_fingerprint": data_fp, "data_fingerprints": data_parts, "model_fingerprint": model_fp,
                        "base_model_id": args.base_model_id, "base_model_revision": args.base_model_revision,
                        "training_config": train_identity, "train_items": len(train_items), "validation_items": len(valid_items),
                        "calibration_items": len(calibration_items), "train_loss": epoch_loss/max(epoch_n,1), "validation": metrics}
        if improved:
            old_value = best_value; best_value = float(value)
            best_cfg = clean_calibration_config(cfg, validation_temperatures)
            best_cfg["calibration"] = validation_calibration
            validate_calibration_config(best_cfg)
            best_state = {**common_state, "best_metric": args.best_metric, "best_metric_mode": args.best_mode,
                          "best_metric_value": best_value, "calibration": validation_calibration,
                          "calibration_temperatures": validation_temperatures}
            best_path = save_checkpoint(
                root,
                "best",
                model,
                tokenizer,
                best_cfg,
                optimizer,
                scheduler,
                scaler,
                best_state,
                metrics_path,
                update_latest=False,
            )
            print(
                f"[best] metric={args.best_metric} "
                f"old={old_value} "
                f"new={best_value:.6f} "
                f"path={best_path}",
                flush=True,
            )
            append_jsonl(metrics_path, {"type":"best", "step":resume_state["global_step"], "epoch":epoch+1,
                                        "metric":args.best_metric, "value":best_value, "path":str(best_path),
                                        "data_fingerprint":data_fp, "model_fingerprint":model_fp,
                                        "base_model_id":args.base_model_id, "base_model_revision":args.base_model_revision})
        epoch_state = {**common_state, "elapsed_seconds": time.perf_counter()-run_start}
        save_checkpoint(root, "latest_checkpoint", model, tokenizer, cfg, optimizer, scheduler, scaler, epoch_state, metrics_path)
        completed_epoch = epoch+1; resume_state["batch_in_epoch"] = 0
        if args.max_steps and resume_state["global_step"] >= args.max_steps:
            break

    fitted, calibration_info = fit_calibration_temperatures(model, calibration_items, pad_id, device, args.micro_batch_size)
    cfg = clean_calibration_config(cfg, fitted)
    cfg["calibration"] = calibration_info
    validate_calibration_config(cfg)
    calibration_event = {"type":"calibration", "step":resume_state["global_step"], "temperatures":fitted,
                         "details":calibration_info, "calibration_fingerprint":data_parts["calibration"],
                         "base_model_id":args.base_model_id, "base_model_revision":args.base_model_revision,
                         "data_fingerprint":data_fp, "model_fingerprint":model_fp}
    append_jsonl(metrics_path, calibration_event)
    final_metrics = evaluate(model, valid_items, pad_id, device, args.micro_batch_size, fitted)
    final_state = {"epoch":completed_epoch, "batch_in_epoch":0, "global_step":resume_state["global_step"],
                   "data_fingerprint":data_fp, "data_fingerprints":data_parts, "model_fingerprint":model_fp,
                   "base_model_id":args.base_model_id, "base_model_revision":args.base_model_revision,
                   "training_config":train_identity, "train_items":len(train_items), "validation_items":len(valid_items),
                   "calibration_items":len(calibration_items), "final_validation":final_metrics,
                   "calibration":calibration_info, "calibration_temperatures":fitted}
    save_checkpoint(root, "final", model, tokenizer, cfg, optimizer, scheduler, scaler, final_state, metrics_path)
    if "test" in data_parts:
        test_items, test_analysis = build_items(load_rows(test_path), tokenizer, cfg)
        skipped = sum(test_analysis["count"].values())
        if skipped / max(skipped + len(test_items), 1) > args.max_skip_ratio:
            raise ValueError("test split skip ratio exceeds --max-skip-ratio")
        test_metrics = evaluate(model, test_items, pad_id, device, args.micro_batch_size, fitted)
        append_jsonl(metrics_path, {"type":"test", "step":resume_state["global_step"], "metrics":test_metrics,
                                    "test_fingerprint":data_parts["test"], "base_model_id":args.base_model_id,
                                    "base_model_revision":args.base_model_revision, "data_fingerprint":data_fp,
                                    "model_fingerprint":model_fp})
        print(f"[test] held-out final-only loss={test_metrics['loss']:.6f} brier={test_metrics['brier_score']:.6f} ece={test_metrics['ece']:.6f}", flush=True)
    append_jsonl(metrics_path, {"type":"final", "step":resume_state["global_step"], "epoch":completed_epoch,
                                "validation":final_metrics, "calibration":calibration_info,
                                "base_model_id":args.base_model_id, "base_model_revision":args.base_model_revision,
                                "data_fingerprint":data_fp, "model_fingerprint":model_fp})
    print(f"[calibration] split=calibration temperatures={fitted} inherited_option_temperatures_removed=True", flush=True)


main = _main_pipeline


if __name__ == "__main__":
    _main_pipeline()
