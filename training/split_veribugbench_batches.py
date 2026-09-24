#!/usr/bin/env python3
"""Split oversized judgment batches into readable chunks.

Why: the pilot batch that worked was 33 KB. Several project batches are
1.7-8.3 MB, which cannot be read in a single tool call, so a judge handed one
either truncates silently or gives up. This rewrites any batch above
MAX_BYTES into `batch_NNa.json`, `batch_NNb.json`, ... each under the cap.

Existing in-flight batches are NOT touched: only batches exceeding the cap are
rewritten, and they are rewritten to NEW filenames (`NN a/b/c`), leaving the
original `batch_NN_<project>.json` in place.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "dataset/converted/veribugbench/batches"
MAX_BYTES = 400_000
MAX_RECORDS = 40


def chunk(payload: dict) -> list[dict]:
    """Greedy chunk by both record count and serialised size."""
    chunks, cur, cur_bytes = [], [], 0
    head = {k: v for k, v in payload.items() if k != "records"}
    head_bytes = len(json.dumps(head))
    for rec in payload["records"]:
        rb = len(json.dumps(rec))
        if cur and (cur_bytes + rb + head_bytes > MAX_BYTES or len(cur) >= MAX_RECORDS):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(rec)
        cur_bytes += rb
    if cur:
        chunks.append(cur)
    return chunks


def main() -> None:
    made, skipped = [], []
    for f in sorted(BATCHES.glob("batch_*.json")):
        if f.name == "index.json":
            continue
        size = f.stat().st_size
        if size <= MAX_BYTES:
            skipped.append((f.name, size))
            continue
        payload = json.loads(f.read_text())
        parts = chunk(payload)
        for i, part in enumerate(parts):
            sub = {**{k: v for k, v in payload.items() if k != "records"},
                   "records": part,
                   "split_of": f.name,
                   "split_part": f"{i+1}/{len(parts)}"}
            out = f.with_name(f.stem + chr(ord("a") + i) + ".json")
            out.write_text(json.dumps(sub, indent=1))
            made.append((out.name, out.stat().st_size, len(part)))
    print(f"split batches created: {len(made)}")
    for n, s, c in made:
        print(f"   {n:52s} {s/1024:7.1f} KB  {c} records")
    print(f"left alone (under cap): {len(skipped)}")


if __name__ == "__main__":
    main()
