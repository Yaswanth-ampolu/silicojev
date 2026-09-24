#!/usr/bin/env python3
"""Evaluate a SilicoJev checkpoint with Laya-compatible benchmark metrics.

The report intentionally separates exact classification from distributional
quality. Gold records may be soft distributions, so accuracy alone is not a
complete evaluation. Score metrics are emitted only when ordered score labels
are actually present; the current normalized SilicoJev release documents score
as disabled instead of inventing a numeric target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from decision_metrics import distribution_observation, normalize, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYA_SOURCE = Path(os.environ.get("SILICOJEV_LAYA_SOURCE", REPO_ROOT / "external" / "laya")).resolve()
sys.path.insert(0, str(LAYA_SOURCE))
from laya import Agent  # noqa: E402


SCORE_DISABLED_NOTE = (
    "Score is disabled for the current normalized SilicoJev data: no source "
    "provides a defensible ordered score label. Choice and noul metrics are "
    "reported; score/RPS is not claimed or fabricated."
)


def parse_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def distribution_for_answer(q: dict[str, Any], answer: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    qtype = q["type"]
    if qtype == "choice":
        keys = list(q.get("criteria", {}).keys())
        return keys, normalize(float(answer.get("probabilities", {}).get(k, 0.0)) for k in keys)
    if qtype == "noul":
        p_true = float(answer.get("noul", 0.5))
        return ["false", "true"], normalize([1.0 - p_true, p_true])
    keys = [str(i) for i in range(len(q.get("criteria", [])))]
    return keys, normalize(float(answer.get("probabilities", {}).get(k, 0.0)) for k in keys)


def distribution_for_gold(q: dict[str, Any], gold: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    qtype = q["type"]
    probs = gold.get("probabilities", {})
    if qtype == "choice":
        keys = list(q.get("criteria", {}).keys())
    elif qtype == "noul":
        keys = ["false", "true"]
    else:
        keys = [str(i) for i in range(len(q.get("criteria", [])))]
    return keys, normalize(float(probs.get(k, 0.0)) for k in keys)


def observation(q: dict[str, Any], answer: dict[str, Any], gold: dict[str, Any], source: str) -> dict[str, Any]:
    pred_keys, pred = distribution_for_answer(q, answer)
    gold_keys, target = distribution_for_gold(q, gold)
    if pred_keys != gold_keys:
        raise ValueError(f"Question criteria changed between prediction and gold: {pred_keys} != {gold_keys}")
    record: dict[str, Any] = {
        **distribution_observation(q["type"], pred, target),
        "source": source,
        "label_source": str(gold.get("label_source") or "unspecified"),
    }
    if q["type"] == "score":
        predicted_score = float(answer.get("score", np.dot(np.arange(len(pred)), pred)))
        gold_score = float(gold["score"]) if "score" in gold else float(np.dot(np.arange(len(target)), target))
        record["score_mae"] = abs(predicted_score - gold_score)
        record["within_1_level"] = float(abs(predicted_score - gold_score) <= 1.0)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    agent = Agent(str(args.model_dir), device=args.device)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    records: list[dict[str, Any]] = []
    source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qtype_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_latencies: dict[str, list[float]] = defaultdict(list)
    all_latencies: list[float] = []
    predictions = []
    score_count = 0

    for row in rows:
        state = parse_json(row["state"])
        questions = parse_json(row["questions"])
        gold = parse_json(row["gold"])
        source = str(row.get("source") or "unknown")
        started = time.perf_counter()
        result = agent.predict(state, questions)
        latency_ms = (time.perf_counter() - started) * 1000.0
        all_latencies.append(latency_ms)
        source_latencies[source].append(latency_ms)
        predictions.append({
            "id": row.get("id"),
            "source": source,
            "latency_ms": round(latency_ms, 3),
            "usage": result.get("usage", {}),
            "answers": result["answers"],
        })
        for qid, q in questions.items():
            if qid not in gold:
                continue
            if q["type"] == "score":
                score_count += 1
            record = observation(q, result["answers"][qid], gold[qid], source)
            records.append(record)
            source_records[source].append(record)
            qtype_records[q["type"]].append(record)
            label_source_records[record["label_source"]].append(record)

    score_enabled = score_count > 0
    report = {
        "model_dir": str(args.model_dir),
        "data": str(args.data),
        "cases": len(rows),
        "decisions": len(records),
        "model_config": {
            "model_name": agent.cfg.get("model_name", "unknown"),
            "temperature": agent.cfg.get("temperature", [1.0, 1.0, 1.0]),
            "calibration": agent.cfg.get("calibration", {"enabled": False}),
        },
        "metrics": summarize(records, all_latencies),
        "by_question_type": {
            qtype: summarize(qrecords)
            for qtype, qrecords in sorted(qtype_records.items())
        },
        "by_label_source": {
            label_source: summarize(label_records)
            for label_source, label_records in sorted(label_source_records.items())
        },
        "per_source": {
            source: summarize(source_records[source], source_latencies[source])
            for source in sorted(source_records)
        },
        "score": {
            "enabled": score_enabled,
            "metrics": summarize(qtype_records["score"]) if score_enabled else None,
            "note": None if score_enabled else SCORE_DISABLED_NOTE,
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    source_output = args.output.with_name(args.output.stem + "_by_source.json")
    source_output.write_text(json.dumps({
        "model_dir": str(args.model_dir),
        "data": str(args.data),
        "score": report["score"],
        "per_source": report["per_source"],
    }, indent=2))
    print(json.dumps({
        "cases": len(rows),
        "decisions": len(records),
        "metrics": report["metrics"],
        "by_question_type": report["by_question_type"],
        "per_source": report["per_source"],
        "score": report["score"],
        "source_report": str(source_output),
    }, indent=2))


if __name__ == "__main__":
    main()
