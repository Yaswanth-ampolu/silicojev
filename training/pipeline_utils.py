"""Small deterministic helpers for reproducible SilicoJev training."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


def content_fingerprint(paths: Iterable[Path], metadata: dict[str, Any] | None = None) -> str:
    """Hash named file contents without incorporating machine-specific paths."""
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    digest.update(json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), default=str).encode())
    return digest.hexdigest()


def jsonl_fingerprint(path: Path, split_name: str) -> str:
    return content_fingerprint([path], {"schema": "silicojev-5q-v2", "split": split_name})


def optimizer_steps_per_epoch(example_count: int, micro_batch_size: int, grad_accum_steps: int) -> int:
    if min(example_count, micro_batch_size, grad_accum_steps) <= 0:
        raise ValueError("example_count, micro_batch_size, and grad_accum_steps must be positive")
    micro_batches = math.ceil(example_count / micro_batch_size)
    return max(1, math.ceil(micro_batches / grad_accum_steps))


def total_optimizer_steps(
    example_count: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    max_steps: int = 0,
) -> int:
    steps = optimizer_steps_per_epoch(example_count, micro_batch_size, grad_accum_steps) * epochs
    return min(steps, max_steps) if max_steps > 0 else steps


def warmup_cosine_multiplier(step: int, total_steps: int, warmup_ratio: float) -> float:
    if total_steps <= 0:
        return 1.0
    warmup = min(total_steps, max(0, int(math.ceil(total_steps * warmup_ratio))))
    if warmup and step < warmup:
        return max(1e-8, float(step + 1) / warmup)
    decay_steps = max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def resume_compatibility(
    saved: dict[str, Any],
    current: dict[str, Any],
    extendable: tuple[str, ...] = ("epochs", "max_steps"),
    runtime_only: tuple[str, ...] = ("num_workers", "prefetch_factor", "pin_memory"),
) -> tuple[bool, list[str]]:
    """Reject semantic changes; permit monotonic run extension and loader tuning."""
    problems: list[str] = []
    for key in sorted(set(saved) | set(current)):
        if key in runtime_only or key not in saved or key not in current or saved[key] == current[key]:
            continue
        if key in extendable and current[key] >= saved[key]:
            continue
        problems.append(f"{key}: checkpoint={saved.get(key)!r}, requested={current.get(key)!r}")
    return not problems, problems


class LengthBucketBatchSampler:
    """Shuffle globally, sort within random pools, then emit similar-length batches."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 42, bucket_multiplier: int = 50):
        if batch_size < 1 or bucket_multiplier < 1:
            raise ValueError("batch_size and bucket_multiplier must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_multiplier = bucket_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        pool_size = self.batch_size * self.bucket_multiplier
        batches: list[list[int]] = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start:start + pool_size], key=lambda i: self.lengths[i])
            batches.extend(pool[b:b + self.batch_size] for b in range(0, len(pool), self.batch_size))
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def clean_calibration_config(cfg: dict[str, Any], temperatures: list[float]) -> dict[str, Any]:
    """Replace Laya calibration, whose per-option values otherwise take precedence."""
    if len(temperatures) != 3 or any(not math.isfinite(float(t)) or not 0.5 <= float(t) <= 5.0 for t in temperatures):
        raise ValueError("Laya temperatures must contain three finite values within [0.5, 5.0]")
    result = dict(cfg)
    result["temperature"] = [float(t) for t in temperatures]
    result.pop("temperature_by_options", None)
    return result


def validate_calibration_config(cfg: dict[str, Any]) -> None:
    temps = cfg.get("temperature")
    if not isinstance(temps, list) or len(temps) != 3:
        raise ValueError("Checkpoint config must have exactly three SilicoJev temperatures")
    if "temperature_by_options" in cfg:
        raise ValueError("Conflicting temperature_by_options would override SilicoJev temperatures")
    if any(not math.isfinite(float(t)) or not 0.5 <= float(t) <= 5.0 for t in temps):
        raise ValueError("Checkpoint temperatures must be finite and within Laya's [0.5, 5.0] range")


def prune_step_checkpoints(root: Path, keep: int, protected: Iterable[Path] = ()) -> list[Path]:
    """Remove old step_* directories only; never touch best/final/latest targets."""
    protected_resolved = {Path(p).resolve() for p in protected}
    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        protected_resolved.add(latest.resolve())
    checkpoints = sorted(
        (p for p in root.glob("step_[0-9]*") if p.is_dir() and not p.name.startswith(".tmp_")),
        key=lambda p: p.name,
        reverse=True,
    )
    removed = []
    retained = 0
    for checkpoint in checkpoints:
        if checkpoint.resolve() in protected_resolved:
            continue
        if retained < max(0, keep):
            retained += 1
            continue
        import shutil
        shutil.rmtree(checkpoint)
        removed.append(checkpoint)
    return removed


def remap_choice_target(target: list[float], order: list[int]) -> list[float]:
    """Return target probabilities in a permuted option order."""
    if sorted(order) != list(range(len(target))):
        raise ValueError("option order must be a permutation of target indices")
    return [float(target[index]) for index in order]


def permute_choice_item(item: dict[str, Any], order: list[int]) -> dict[str, Any]:
    """Reorder complete marker+option token spans and permute targets identically."""
    if item.get("qtype_name") != "choice":
        return item
    spans = item["option_spans"]
    if sorted(order) != list(range(len(spans))):
        raise ValueError("invalid choice permutation")
    result = dict(item)
    prefix = item["ids"][:spans[0][0]]
    segments = [item["ids"][start:end] for start, end in spans]
    suffix = item["ids"][item["separator"]:]
    ids = list(prefix)
    new_spans = []
    markers = []
    for index in order:
        start = len(ids)
        ids.extend(segments[index])
        new_spans.append((start, len(ids)))
        markers.append(start)
    ids.extend(suffix)
    result.update(ids=ids, option_spans=new_spans, markers=markers,
                  target=remap_choice_target(item["target"], order))
    result["label"] = int(max(range(len(result["target"])), key=result["target"].__getitem__))
    return result
