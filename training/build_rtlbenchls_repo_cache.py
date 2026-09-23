#!/usr/bin/env python3
"""Materialize the Task 3 upstream repository cache.

The Task 3 records reference buggy `base_commit` and fixed `head_commit`
revisions in nine upstream GitHub repositories. This script builds a
`repo_cache/<owner>_<repo>/` git store that contains exactly those commit
objects, using shallow + blobless fetches so the cache stays small. Blobs are
fetched lazily on `git show`, which is how the converter reads base-commit RTL.

Usage:
    python3 training/build_rtlbenchls_repo_cache.py
    python3 training/build_rtlbenchls_repo_cache.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "dataset/raw/github/RTL-BenchLS"
BATCH = 20


def load_commits(dataset_root: Path) -> dict[str, set[str]]:
    path = dataset_root / "data/repo_issue_108_cases.json"
    data = json.loads(path.read_text())
    cases = data.get("cases", data)
    per: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        for key in ("base_commit", "head_commit"):
            sha = (case.get(key) or "").strip()
            if sha:
                per[case["repository"]].add(sha)
    return dict(per)


def slug(repo: str) -> str:
    return repo.replace("/", "_")


def run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=False
    )


def has_commit(cache: Path, sha: str) -> bool:
    return run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=cache).returncode == 0


def build_repo(repo: str, shas: set[str], cache_root: Path, verbose: bool) -> dict:
    dest = cache_root / slug(repo)
    if not (dest / ".git").is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q", "."], cwd=dest)
        run(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=dest)
        run(["git", "config", "remote.origin.promisor", "true"], cwd=dest)
        run(["git", "config", "remote.origin.partialclonefilter", "blob:none"], cwd=dest)

    missing = sorted(sha for sha in shas if not has_commit(dest, sha))
    for i in range(0, len(missing), BATCH):
        batch = missing[i : i + BATCH]
        proc = run(
            [
                "git", "fetch", "-q", "--depth", "1", "--filter=blob:none",
                "--no-tags", "origin", *batch,
            ],
            cwd=dest,
        )
        if proc.returncode != 0 and verbose:
            print(f"    fetch batch failed: {proc.stderr.strip()[:300]}", file=sys.stderr)

    still_missing = [sha for sha in shas if not has_commit(dest, sha)]
    size = run(["bash", "-c", f"du -sh {dest} | cut -f1"]).stdout.strip()
    return {
        "repository": repo,
        "cache_path": str(dest.relative_to(REPO_ROOT)),
        "required_commits": len(shas),
        "missing_commits": still_missing,
        "disk": size,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cache_root = args.cache_root or (args.dataset_root / "repo_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    per_repo = load_commits(args.dataset_root)

    reports = []
    for repo in sorted(per_repo, key=lambda r: -len(per_repo[r])):
        shas = per_repo[repo]
        if args.verify_only:
            dest = cache_root / slug(repo)
            missing = sorted(s for s in shas if not has_commit(dest, s)) if (dest / ".git").is_dir() else sorted(shas)
            reports.append(
                {
                    "repository": repo,
                    "cache_path": str(dest.relative_to(REPO_ROOT)),
                    "required_commits": len(shas),
                    "missing_commits": missing,
                    "disk": None,
                }
            )
            status = "OK" if not missing else f"MISSING {len(missing)}"
            print(f"  {repo:34s} {len(shas):3d} commits  {status}")
            continue
        print(f"  fetching {repo} ({len(shas)} commits) ...", flush=True)
        report = build_repo(repo, shas, cache_root, args.verbose)
        status = "OK" if not report["missing_commits"] else f"MISSING {len(report['missing_commits'])}"
        print(f"    {status}  {report['disk']}")
        reports.append(report)

    out = cache_root / "cache_manifest.json"
    out.write_text(json.dumps({"repositories": reports}, indent=2))
    total_required = sum(r["required_commits"] for r in reports)
    total_missing = sum(len(r["missing_commits"]) for r in reports)
    print(f"\n{len(reports)} repositories, {total_required} required commits, {total_missing} missing")
    print(f"manifest: {out}")
    if total_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
