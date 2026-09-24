#!/usr/bin/env python3
"""Build one judgment batch per project from the VeriBugBench 3q skeleton.

Design notes
------------
* Batches are grouped BY PROJECT (`source_group`). The testbench is a
  project-level artifact, so stating it once per batch instead of once per
  record cuts the payload from 168 MB to ~24 MB.
* The judge is shown ONLY pre-decision evidence: the buggy RTL and the
  testbench. Deliberately withheld: `fault_operator`, `fault_site`,
  `reference_rtl`, the `oracle*.txt` files, and the raw instance_id (which
  encodes the operator). Withholding the operator is what makes the judge's
  output a genuine judgement rather than a read of the answer -- and it is
  what lets us later score the judge against that ground truth.
* This script chooses NO labels. It only reshapes input.

Output: dataset/converted/veribugbench/batches/batch_<NN>_<group>.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKEL = ROOT / "dataset/converted/veribugbench/skeleton_3q.jsonl"
OUT = ROOT / "dataset/converted/veribugbench/batches"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = defaultdict(list)
    testbenches: dict[str, str] = {}

    with SKEL.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            st = json.loads(rec["state"])
            g = rec["source_group"]
            groups[g].append({
                "id": rec["id"],
                "project": st["project"],
                "buggy_rtl": st["buggy_rtl"],
                "upstream_repo": st["source_group"],
            })
            testbenches[g] = st["testbench"] or ""

    questions = None
    with SKEL.open() as fh:
        questions = json.loads(json.loads(fh.readline())["questions"])

    index = []
    for i, (g, items) in enumerate(sorted(groups.items())):
        safe = g.split("/", 1)[-1]
        payload = {
            "batch_index": i,
            "source_group": g,
            "project": items[0]["project"],
            "upstream": items[0]["upstream_repo"],
            "testbench": testbenches[g],
            "questions": questions,
            "records": items,
        }
        p = OUT / f"batch_{i:02d}_{safe}.json"
        p.write_text(json.dumps(payload, indent=1))
        index.append({
            "batch_index": i,
            "file": str(p.relative_to(ROOT)),
            "source_group": g,
            "records": len(items),
            "bytes": p.stat().st_size,
        })

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    tot = sum(x["records"] for x in index)
    mb = sum(x["bytes"] for x in index) / 1e6
    print(f"batches={len(index)}  records={tot}  total={mb:.1f} MB")
    print(f"smallest={min(x['records'] for x in index)}  largest={max(x['records'] for x in index)}")
    print(f"index: {OUT/'index.json'}")


if __name__ == "__main__":
    main()
