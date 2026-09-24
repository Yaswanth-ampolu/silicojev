#!/usr/bin/env python3
"""VeriBugBench v1.0 -> SilicoJev 3q skeleton.

Produces the *structure* of a 3q record: id, source, source_group, state,
questions, outcome, provenance. Gold labels are NOT produced here.

Rationale for the split (project guardrail): Python may extract facts from
artifacts and validate, but no script, regex, keyword map or default is
allowed to choose a label or a probability. Everything this file writes is an
extracted fact or a byte-for-byte copy of an existing corpus constant.

LEAKAGE RULES enforced here
---------------------------
`state` is pre-decision evidence ONLY. Three things in VeriBugBench would give
the answer away and are therefore kept out of `state`:

  1. `operator_id` (e.g. "ExprDelete") -- names the injected fault, i.e. the
     answer to `root_cause_type`. Lives in `outcome`.
  2. the reference (correct) RTL -- it *is* the repair. Lives in `outcome`.
  3. `oracle*.txt` / `output.txt` -- golden expected output. Never read.
  4. the instance_id itself ("decoder_3_to_8_ExprDelete_108_22") encodes the
     operator. `id` is therefore opaque (sha1) and the readable id is kept in
     `provenance` for traceability, not in `state`.

Output: dataset/converted/veribugbench/skeleton_3q.jsonl
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "dataset/raw/github/VeriBugBench"
OUT = Path(__file__).resolve().parent.parent / "dataset/converted/veribugbench"
V1 = BENCH / "dataset/v1.0"

# Byte-identical `questions` blob shared by all 6,284 existing 3q records
# (sha256 prefix e21825c4cbf6). Copied verbatim so converted records are
# schema-compatible with the rest of the corpus.
CANONICAL_QUESTIONS_SHA = "e21825c4cbf6"


def load_canonical_questions(repo_root: Path) -> str:
    """Reuse the corpus's own questions string, verbatim, or refuse to run."""
    src = repo_root / "dataset/normalized/all.jsonl"
    with src.open() as fh:
        row = json.loads(fh.readline())
    blob = row["questions"]
    got = hashlib.sha256(blob.encode()).hexdigest()[:12]
    if got != CANONICAL_QUESTIONS_SHA:
        sys.exit(
            f"FATAL: canonical questions blob changed (expected "
            f"{CANONICAL_QUESTIONS_SHA}, got {got}). Refusing to emit records "
            f"that would be inconsistent with the corpus."
        )
    return blob


# ---------------------------------------------------------------- fault site
_TOK = re.compile(r"[A-Za-z_]\w*|\d+'[bhd][0-9a-fA-FxXzZ_]+|\d+|[^\s\w]")
# Port-declaration reformatting: PyVerilog rewrites `output a, b;` into
# `output a; output b;`, which shows up as `,` -> `; output` hunks. These are
# formatting, not the injected fault, and must be filtered out.
_PORT_REFORMAT = re.compile(r"^[;,]?\s*(?:output|input|inout)$")


def _tokens(text: str) -> list[str]:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return _TOK.findall(text)


# ------------------------------------------------- operator-name redaction
# VeriBugBench's "enhanced" testbenches were authored to target specific
# mutation operators and NAME them in comments. Measured scope: 1 project
# (decoder_3_to_8, 20 records) contains `// Targets: If-Else Move (NSubMove)`.
# That comment hands the judge `root_cause_type` for free, so operator names
# are redacted inside comments. Code is never touched -- only comments -- so
# the design under test is unchanged.
OPERATOR_IDS = [
    "ExprDelete", "ExprInsert", "ExprUpdate", "AssignN2B", "EdgeFlip",
    "EdgeInsert", "NSubDelete", "NSubInsert_IF", "NSubInsert", "NSubMove",
    "NSubUpdate", "PointerInsert", "AssignB2N", "EdgeDelete",
    "NSubInsert_Case", "PointerDelete", "PointerUpdate", "PartselectDelete",
    "PartselectUpdate", "PartselectInsert", "IOFlip1", "IOFlip2",
]
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def redact_operator_names(text: str | None) -> tuple[str | None, list[str]]:
    """Redact mutation-operator names appearing inside comments only."""
    if not text:
        return text, []
    found: list[str] = []

    def _repl(m: re.Match) -> str:
        out = m.group(0)
        for op in OPERATOR_IDS:
            if op in out:
                found.append(op)
                out = out.replace(op, "[operator-redacted]")
        return out

    return _COMMENT.sub(_repl, text), sorted(set(found))


def fault_site(ref_tokens: list[str], mutant_src: str) -> dict:
    """Locate the injected fault by token diff, filtering reformatting hunks.

    HEURISTIC. Recorded for audit only -- never used to pick a label.
    `ref_tokens` is pre-tokenised and cached per project (45 projects, 2443
    instances) because re-tokenising the reference per instance dominated
    runtime.
    """
    a, b = ref_tokens, _tokens(mutant_src)
    if max(len(a), len(b)) > 4000:
        # SequenceMatcher is quadratic; bail out rather than stall the run.
        return {
            "method": "skipped_token_stream_too_large",
            "authoritative": False,
            "usable_as_localisation": False,
            "reference_tokens": len(a),
            "mutant_tokens": len(b),
            "hunks": [],
        }
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    substantive, formatting = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ref_txt = " ".join(a[i1:i2]).strip()
        mut_txt = " ".join(b[j1:j2]).strip()
        if _PORT_REFORMAT.match(mut_txt) or _PORT_REFORMAT.match(ref_txt):
            formatting += 1
            continue
        substantive.append({"op": tag, "reference": ref_txt, "mutant": mut_txt})
    return {
        "method": "token_diff_vs_reference_filtering_port_reformatting",
        "authoritative": False,
        # A single-fault mutant should differ semantically in exactly one place.
        # Anything else means PyVerilog's reformatting could NOT be separated
        # from the fault, so the diff is noise and must not be read as a
        # localisation. Measured: clean in only ~1 case in 9.
        "usable_as_localisation": len(substantive) == 1,
        "substantive_hunks": len(substantive),
        "formatting_hunks_filtered": formatting,
        "hunks": substantive[:8],
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    questions_blob = load_canonical_questions(repo_root)

    projects = {}
    with (BENCH / "metadata/projects.csv").open() as fh:
        for row in csv.DictReader(fh):
            projects[row["project_id"]] = row

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "skeleton_3q.jsonl"

    stats = {"emitted": 0, "mutant_missing": 0, "tb_missing": 0, "ref_missing": 0}
    missing_projects: dict[str, int] = {}
    ref_cache: dict[str, tuple] = {}
    fault_skipped = 0

    with (BENCH / "metadata/instances.csv").open() as fh, out_path.open("w") as out:
        for row in csv.DictReader(fh):
            iid = row["instance_id"]
            proj = row["project_id"]
            mutant_rel = row["mutant_artifact"]
            mutant_path = BENCH / mutant_rel
            if not mutant_path.exists():
                stats["mutant_missing"] += 1
                missing_projects[proj] = missing_projects.get(proj, 0) + 1
                continue

            proj_row = projects.get(proj, {})
            root = BENCH / proj_row.get("benchmark_root", "")
            manifest_path = root / "veribugbench_v_1_0.yaml"
            tb_name, origin_files = None, []
            if manifest_path.exists():
                try:
                    import yaml

                    man = yaml.safe_load(manifest_path.read_text()) or {}
                    tb_name = man.get("testbench")
                    origin_files = man.get("origin_source_files") or []
                except Exception:
                    pass

            tb_src = None
            if tb_name:
                tb_path = root / tb_name
                if tb_path.exists():
                    tb_src = tb_path.read_text(errors="replace")
            if tb_src is None:
                stats["tb_missing"] += 1

            ref_src, ref_rel = None, None
            if proj in ref_cache:
                ref_rel, ref_tokens = ref_cache[proj]
            else:
                for o in origin_files:
                    p = root / o
                    if p.exists():
                        ref_rel = str(p.relative_to(BENCH))
                        ref_src = p.read_text(errors="replace")
                        break
                ref_tokens = _tokens(ref_src) if ref_src is not None else None
                ref_cache[proj] = (ref_rel, ref_tokens)
            if ref_tokens is None:
                stats["ref_missing"] += 1

            mutant_src = mutant_path.read_text(errors="replace")

            # Redact operator names that upstream left in comments.
            tb_src, tb_redacted = redact_operator_names(tb_src)
            mutant_src, rtl_redacted = redact_operator_names(mutant_src)
            redacted = sorted(set(tb_redacted) | set(rtl_redacted))

            sim_cmd = None
            sim_path = root / "vcs_sim_command"
            if sim_path.exists():
                sim_cmd = sim_path.read_text(errors="replace").strip()

            # ---- state: pre-decision evidence only -------------------------
            state = {
                "project": proj,
                "source_group": row["source_group"],
                "buggy_rtl": mutant_src,
                "testbench": tb_src,
                "tool": "vcs" if sim_cmd else None,
                "sim_command": sim_cmd,
                "failure_context": {
                    "artefact": "injected single-fault mutant on the reference revision",
                    "oracle_output_available": False,  # deliberately never read
                    "no_tool_log_available": True,
                    "reported_symptom": (
                        "the project testbench does not match its golden reference "
                        "on this revision"
                    ),
                },
                "previous_actions": [],
            }

            # ---- outcome: post-decision truth ------------------------------
            outcome = {
                "resolved": None,
                "verification": "previously_validated",
                "fault_operator": row["operator_id"],
                "mutant_artifact": mutant_rel,
                "reference_rtl": ref_rel,
            }
            if ref_tokens is not None:
                outcome["fault_site"] = fault_site(ref_tokens, mutant_src)

            # ---- provenance ------------------------------------------------
            provenance = {
                "license": (
                    "MIT (VeriBugBench software); bundled RTL carries upstream "
                    "project licenses - see THIRD_PARTY_NOTICES.md"
                ),
                "synthetic": True,
                "quality": "silver_benchmark",
                "source_instance_id": iid,
                "source_group": row["source_group"],
                "project_id": proj,
                "implementation_schema": row["implementation_schema"],
                "upstream_manuscript": "Mantra (DAC 2023), doi:10.1109/DAC56929.2023.10247962",
                "benchmark": "VeriBugBench v1.0",
            }
            if redacted:
                provenance["state_sanitization"] = {
                    "redacted_operator_names_in_comments": redacted,
                    "reason": "upstream testbench comments named the target mutation operator",
                }

            record = {
                "id": "veribugbench:" + hashlib.sha1(iid.encode()).hexdigest()[:16],
                "source": "VeriBugBench",
                "source_group": f"veribugbench:{row['source_group']}/{proj}",
                "state": json.dumps(state, sort_keys=True),
                "questions": questions_blob,
                "gold": None,  # filled by the judgment stage
                "outcome": outcome,
                "provenance": provenance,
            }
            out.write(json.dumps(record) + "\n")
            stats["emitted"] += 1

    print(json.dumps({
        **stats,
        "output": str(out_path),
        "missing_by_project": missing_projects,
    }, indent=2))


if __name__ == "__main__":
    main()
