"""Shared probability metrics for SilicoJev training-time validation and evaluation."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def finite(value: float, default: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else default


def normalize(values: Iterable[float]) -> np.ndarray:
    probs = np.asarray(list(values), dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    return probs / total if total > 0 else np.full(len(probs), 1.0 / max(len(probs), 1))


def ece(confidence: list[float], correct: list[float], bins: int = 15) -> float:
    if not confidence:
        return 0.0
    conf = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(conf[mask].mean()) - float(correctness[mask].mean()))
    return finite(result)


def distribution_observation(
    qtype: str,
    predicted: Iterable[float],
    target: Iterable[float],
) -> dict[str, float | str]:
    """Compute evaluator-compatible exact/soft and proper-probability metrics."""
    pred = normalize(predicted)
    gold = normalize(target)
    if pred.shape != gold.shape:
        raise ValueError(f"Prediction/target distribution sizes differ: {pred.shape} != {gold.shape}")
    pred_label = int(pred.argmax())
    gold_label = int(gold.argmax())
    if qtype == "noul":
        correct = float((pred[1] >= 0.5) == (gold[1] >= 0.5))
    else:
        correct = float(pred_label == gold_label)
    return {
        "qtype": qtype,
        "correct": correct,
        "confidence": float(pred.max()) if len(pred) else 0.0,
        "soft_accuracy": float(np.dot(pred, gold)),
        "brier_score": float(np.square(pred - gold).sum()),
        "kl_divergence": float(np.sum(gold * np.log(np.clip(gold, 1e-12, 1.0) / np.clip(pred, 1e-12, 1.0)))),
        "total_variation": float(0.5 * np.abs(pred - gold).sum()),
    }


def summarize(records: list[dict[str, Any]], latencies_ms: list[float] | None = None) -> dict[str, Any]:
    if not records:
        result: dict[str, Any] = {
            "n": 0,
            "accuracy": None,
            "soft_accuracy": None,
            "brier_score": None,
            "kl_divergence": None,
            "total_variation": None,
            "ece": None,
        }
    else:
        correct = [float(row["correct"]) for row in records]
        result = {
            "n": len(records),
            "accuracy": finite(np.mean(correct)),
            "soft_accuracy": finite(np.mean([row["soft_accuracy"] for row in records])),
            "brier_score": finite(np.mean([row["brier_score"] for row in records])),
            "kl_divergence": finite(np.mean([row["kl_divergence"] for row in records])),
            "total_variation": finite(np.mean([row["total_variation"] for row in records])),
            "ece": ece([float(row["confidence"]) for row in records], correct),
        }
        score_records = [row for row in records if "score_mae" in row]
        result["score_mae"] = finite(np.mean([row["score_mae"] for row in score_records])) if score_records else None
        result["within_1_level"] = finite(np.mean([row["within_1_level"] for row in score_records])) if score_records else None
    if latencies_ms:
        result["latency_p50_ms"] = finite(np.percentile(latencies_ms, 50))
        result["latency_p95_ms"] = finite(np.percentile(latencies_ms, 95))
    else:
        result["latency_p50_ms"] = None
        result["latency_p95_ms"] = None
    return result
