#!/usr/bin/env python3
"""Resumable single-A100 fine-tuning for SilicoJev.

The script keeps the Laya architecture and typed output contract, but trains on
normalized SilicoJev JSONL records. Checkpoints are self-contained and expose a
stable ``latest`` link for notebook restarts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYA_SOURCE = Path(os.environ.get("SILICOJEV_LAYA_SOURCE", REPO_ROOT / "external" / "laya")).resolve()
if str(LAYA_SOURCE) not in sys.path:
    sys.path.insert(0, str(LAYA_SOURCE))

from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_model, build_sequence, proper_reward  # noqa: E402


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


def fingerprint(paths: list[Path], config: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode())
        h.update(path.read_bytes())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()


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
    if total <= 0:
        return [1.0 / len(target)] * len(target)
    return [value / total for value in target]


def build_items(rows: list[dict[str, Any]], tokenizer, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    items = []
    skipped = 0
    for row in rows:
        state = parse_json(row["state"])
        questions = parse_json(row["questions"])
        gold = parse_json(row["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                skipped += 1
                continue
            target = target_for(q, gold[qid])
            internal_q = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
            seq, markers = build_sequence(
                tokenizer,
                state,
                internal_q,
                int(cfg.get("max_len", 1024)),
                int(cfg.get("head_max_len", 256)),
            )
            if len(markers) != len(target):
                skipped += 1
                continue
            items.append({
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q["type"]],
                "target": target,
                "label": int(np.argmax(target)),
                "case_id": row.get("id"),
                "question_id": qid,
                "weight": float((row.get("training_weights") or {}).get(qid, 1.0)),
            })
    if not items:
        raise ValueError("No tokenizable training items were produced")
    return items, skipped


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
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
def evaluate(model, items: list[dict[str, Any]], pad_id: int, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for start in range(0, len(items), batch_size):
        batch = move_batch(collate(items[start:start + batch_size], pad_id), device)
        logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
        mask = batch["marker_mask"]
        target = batch["target"]
        loss = -(target * torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        pred = logits.float().masked_fill(~mask, -1e4).argmax(-1)
        correct += int((pred == batch["label"]).sum().item())
        total += batch["label"].numel()
        loss_sum += float(loss.item()) * batch["label"].numel()
    model.train()
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "items": total}


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


def fit_validation_temperatures(
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
        "method": "validation_nll_lbfgs",
        "split": "validation",
        "items_by_type": counts,
        "score_enabled": counts["score"] > 0,
        "score_note": (
            "Score temperature fitted from validation score items."
            if counts["score"] > 0
            else "Score is disabled: the normalized SilicoJev data contains no defensible ordered score labels."
        ),
    }
    return temperatures, status


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
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final = root / name
    temp = root / f".tmp_{name}_{os.getpid()}"
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
    if final.exists():
        shutil.rmtree(final)
    os.replace(temp, final)
    latest_tmp = root / f".latest_{os.getpid()}"
    if latest_tmp.exists() or latest_tmp.is_symlink():
        latest_tmp.unlink()
    latest_tmp.symlink_to(final.name)
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    os.replace(latest_tmp, latest)
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
    print(f"Resumed from {path} at epoch={state['epoch']} batch={state['batch_in_epoch']} step={state['global_step']}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=16)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--keep-checkpoints", type=int, default=3)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--encoder-lr", type=float, default=2.5e-5)
    ap.add_argument("--head-lr", type=float, default=1.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SilicoJev training")
    device = torch.device("cuda")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but this GPU/PyTorch build does not support it")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    amp_enabled = args.dtype != "fp32"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    model_dir = args.model_dir.resolve()
    _fix_tokenizer_config(str(model_dir))
    cfg_path = model_dir / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len
    cfg["gradient_checkpointing"] = True
    cfg["fine_tuned"] = True
    cfg["model_name"] = "silicojev"
    cfg["amp_dtype"] = args.dtype

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    train_rows = load_rows(args.train)
    valid_rows = load_rows(args.validation)
    data_fp = fingerprint([args.train.resolve(), args.validation.resolve()], {
        "model": str(model_dir),
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "dtype": args.dtype,
        "seed": args.seed,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
    })
    train_items, train_skipped = build_items(train_rows, tokenizer, cfg)
    valid_items, valid_skipped = build_items(valid_rows, tokenizer, cfg)
    print(f"Prepared train={len(train_items)} items (skipped={train_skipped}), validation={len(valid_items)} (skipped={valid_skipped})")
    print(f"Dataset fingerprint: {data_fp}")

    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device="cpu"), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    enc_params = [p for name, p in model.named_parameters() if "encoder." in name]
    head_params = [p for name, p in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    updates_per_epoch = max(1, (len(train_items) + args.micro_batch_size - 1) // args.micro_batch_size // args.grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates_per_epoch * args.epochs), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "fp16")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    resume_state = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0}
    if args.resume != "none":
        resume_path = root / "latest" if args.resume == "auto" else Path(args.resume)
        if resume_path.exists():
            resolved = resume_path.resolve()
            meta = json.loads((resolved / "checkpoint_meta.json").read_text())
            if meta.get("dataset_fingerprint") != data_fp:
                raise RuntimeError("Checkpoint dataset fingerprint differs from the current dataset")
            resume_state = load_resume(resolved, model, optimizer, scheduler, scaler, device)

    pad_id = tokenizer.pad_token_id or 0
    started = time.time()
    completed_epoch = int(resume_state["epoch"])
    for epoch in range(int(resume_state["epoch"]), args.epochs):
        order = list(range(len(train_items)))
        random.Random(args.seed + epoch).shuffle(order)
        first_batch = int(resume_state["batch_in_epoch"]) if epoch == int(resume_state["epoch"]) else 0
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        epoch_loss = 0.0
        epoch_items = 0
        for batch_start in range(0, len(order), args.micro_batch_size):
            if batch_start < first_batch:
                continue
            chosen = [train_items[i] for i in order[batch_start:batch_start + args.micro_batch_size]]
            if not chosen:
                continue
            batch = move_batch(collate(chosen, pad_id), device)
            progress = epoch / max(1, args.epochs - 1)
            sigma = 0.4 + (0.1 - 0.4) * progress
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
                loss = loss_from_logits(logits, batch, sigma, model) / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum += 1
            epoch_loss += float(loss.item()) * args.grad_accum_steps * len(chosen)
            epoch_items += len(chosen)
            is_last = batch_start + args.micro_batch_size >= len(order)
            if accum % args.grad_accum_steps == 0 or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum = 0
                resume_state["global_step"] += 1
                if resume_state["global_step"] % args.checkpoint_every == 0:
                    state = {
                        "epoch": epoch,
                        "batch_in_epoch": batch_start + args.micro_batch_size,
                        "global_step": resume_state["global_step"],
                        "dataset_fingerprint": data_fp,
                        "training_config": vars(args),
                        "train_items": len(train_items),
                        "validation_items": len(valid_items),
                    }
                    save_checkpoint(root, f"step_{resume_state['global_step']:08d}", model, tokenizer, cfg, optimizer, scheduler, scaler, state)
                    print(f"checkpoint step={resume_state['global_step']} epoch={epoch + 1} batch={batch_start + args.micro_batch_size}", flush=True)
                if args.max_steps and resume_state["global_step"] >= args.max_steps:
                    break
        metrics = evaluate(model, valid_items, pad_id, device, args.micro_batch_size)
        state = {
            "epoch": epoch + 1,
            "batch_in_epoch": 0,
            "global_step": resume_state["global_step"],
            "dataset_fingerprint": data_fp,
            "training_config": vars(args),
            "train_items": len(train_items),
            "validation_items": len(valid_items),
            "train_loss": epoch_loss / max(epoch_items, 1),
            "validation": metrics,
            "elapsed_seconds": time.time() - started,
        }
        save_checkpoint(root, f"epoch_{epoch + 1:04d}", model, tokenizer, cfg, optimizer, scheduler, scaler, state)
        print(json.dumps({"epoch": epoch + 1, "train_loss": state["train_loss"], "validation": metrics}, indent=2), flush=True)
        resume_state["batch_in_epoch"] = 0
        completed_epoch = epoch + 1
        if args.max_steps and resume_state["global_step"] >= args.max_steps:
            break

    fitted_temperatures, calibration = fit_validation_temperatures(
        model,
        valid_items,
        pad_id,
        device,
        args.micro_batch_size,
    )
    cfg["temperature"] = fitted_temperatures
    cfg["calibration"] = calibration
    print(json.dumps({"temperatures": fitted_temperatures, "calibration": calibration}, indent=2), flush=True)

    final_state = {
        "epoch": min(args.epochs, completed_epoch),
        "batch_in_epoch": 0,
        "global_step": resume_state["global_step"],
        "dataset_fingerprint": data_fp,
        "training_config": vars(args),
        "train_items": len(train_items),
        "validation_items": len(valid_items),
        "final_validation": evaluate(model, valid_items, pad_id, device, args.micro_batch_size),
        "calibration": calibration,
        "elapsed_seconds": time.time() - started,
    }
    save_checkpoint(root, "final", model, tokenizer, cfg, optimizer, scheduler, scaler, final_state)
    print(json.dumps(final_state, indent=2, default=str))


if __name__ == "__main__":
    main()
