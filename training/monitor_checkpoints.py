#!/usr/bin/env python3
"""Print the latest SilicoJev checkpoint and GPU status."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    args = ap.parse_args()
    latest = args.checkpoint_root / "latest"
    if latest.exists():
        resolved = latest.resolve()
        meta = resolved / "checkpoint_meta.json"
        print(json.dumps({
            "latest": str(resolved),
            "metadata": json.loads(meta.read_text()) if meta.exists() else None,
        }, indent=2, default=str))
    else:
        print(json.dumps({"latest": None}, indent=2))
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ], text=True)
        print(output.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"nvidia-smi unavailable: {exc}")


if __name__ == "__main__":
    main()
