#!/usr/bin/env python3
"""Build Laya-compatible SilicoJev decision records from the selected sources.

This is deliberately conservative: repaired code and patches are used only to
derive labels, never placed in the model state. Every row keeps provenance and
label quality so later training can filter it explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTION_CRITERIA = {
    "rtl": "Inspect the RTL implementation or generated hardware logic",
    "testbench": "Inspect stimulus, testbench, UVM, or verification code",
    "waveform": "Inspect waveform and cycle-level signal behavior",
    "formal": "Run or inspect formal properties and counterexamples",
    "specification": "Check the protocol, design specification, or requirements",
    "simulation": "Run a targeted simulation or a different simulator configuration",
    "constraints": "Inspect timing, clock, pin, or synthesis constraints",
    "ask_human": "Request review from a hardware expert",
    "abstain": "There is insufficient evidence to choose a direction",
}

ROOT_CAUSE_CRITERIA = {
    "syntax_compile": "Syntax, compilation, elaboration, or l-value failure",
    "type_width": "Type, width, signedness, cast, or operator mismatch",
    "combinational_logic": "Incorrect combinational or arithmetic logic",
    "sequential_assignment": "Clocked assignment, blocking/non-blocking, or latch issue",
    "reset_initialization": "Reset, initialization, or unknown-state issue",
    "timing_protocol": "Timing, handshake, clock-domain, or protocol sequencing issue",
    "state_machine": "FSM transition, state encoding, or control-flow issue",
    "formal_property": "Assertion, property, invariant, or formal-model issue",
    "security": "Hardware security, vulnerability, or information-flow issue",
    "unknown": "The root-cause category is not known from the available evidence",
}

SOURCE_LICENSES = {
    "hwe-bench": "Apache-2.0 metadata; upstream repositories retained",
    "HierSVA": "Solderpad Hardware License 2.1 plus BaseJump provenance",
    "RootCause-Bench": "Apache-2.0 repository; fixture audit required",
    "Inspect-Eval-ChipBench": "MIT harness; benchmark artifact audit required",
    "OriGen": "GPL-3.0 dataset metadata",
    "RTL-augmented": "MIT",
}


def clip(value: Any, limit: int = 8000) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def read_text(path: Path, limit: int = 8000) -> str:
    try:
        return clip(path.read_text(errors="replace"), limit)
    except OSError:
        return ""


def root_cause_key(text: Any) -> str:
    s = str(text or "").lower()
    patterns = [
        ("security", "security"),
        ("formal_property", "assertion|property|sva|formal|counterexample|invariant"),
        ("reset_initialization", "reset|initialization|uninitialized|unknown state|x state"),
        ("timing_protocol", "timing|clock domain|cdc|handshake|protocol|race|setup|hold"),
        ("state_machine", "state machine|fsm|transition|state encoding"),
        ("syntax_compile", "syntax|compile|compilation|elaborat|l-value|lvalue"),
        ("type_width", "type error|width|bit width|signed|unsigned|cast|operator"),
        ("sequential_assignment", "blocking|non-blocking|latch|always_ff|always_comb|sequential"),
        ("combinational_logic", "combinational|arithmetic|carry|and|or|xor|logic"),
    ]
    for key, pattern in patterns:
        if re.search(pattern, s):
            return key
    return "unknown"


def action_for_file(path: str) -> str:
    s = path.lower().replace("\\", "/")
    if any(x in s for x in ("/dv/", "/tb/", "testbench", "/test/", "/uvm/", "verif")):
        return "testbench"
    if any(x in s for x in (".sva", "/assert", "formal")):
        return "formal"
    if s.endswith((".xdc", ".sdc")) or "/constraint" in s:
        return "constraints"
    if "/doc" in s or s.endswith((".rst", ".md", ".txt")):
        return "specification"
    if s.endswith((".v", ".sv", ".vh", ".svh", ".vhd", ".vhdl", ".scala")) or "/rtl/" in s:
        return "rtl"
    if any(x in s for x in ("sim", "makefile", ".tcl")):
        return "simulation"
    return "rtl"


def distribution(keys: Iterable[str], active: Iterable[str]) -> dict[str, float]:
    keys = list(keys)
    active = [x for x in dict.fromkeys(active) if x in keys]
    if not active:
        active = ["abstain"] if "abstain" in keys else [keys[-1]]
    p = 1.0 / len(active)
    return {key: (p if key in active else 0.0) for key in keys}


def questions() -> dict[str, dict[str, Any]]:
    return {
        "next_action": {
            "type": "choice",
            "instructions": "What diagnostic direction should be investigated next?",
            "criteria": ACTION_CRITERIA,
        },
        "root_cause_type": {
            "type": "choice",
            "instructions": "Which root-cause category best describes the available evidence?",
            "criteria": ROOT_CAUSE_CRITERIA,
        },
        "evidence_sufficient": {
            "type": "noul",
            "instructions": "Is there enough evidence to choose a likely root cause?",
            "criteria": {
                "false": "Evidence is insufficient or unvalidated",
                "true": "Evidence is sufficient and supported by a tool or trusted label",
            },
        },
    }


def make_case(
    *,
    case_id: str,
    source: str,
    group: str,
    state: dict[str, Any],
    active_actions: Iterable[str],
    root_key: str,
    evidence: bool,
    label_source: str,
    quality: str,
    synthetic: bool,
    outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    qs = questions()
    gold = {
        "next_action": {
            "probabilities": distribution(ACTION_CRITERIA, active_actions),
            "label_source": label_source,
        },
        "root_cause_type": {
            "probabilities": distribution(ROOT_CAUSE_CRITERIA, [root_key]),
            "label_source": label_source,
        },
        "evidence_sufficient": {
            "probabilities": {"false": 0.0 if evidence else 1.0, "true": 1.0 if evidence else 0.0},
            "label_source": label_source,
        },
    }
    return {
        "id": case_id,
        "source": source,
        "source_group": group,
        "state": json.dumps(state, ensure_ascii=False, sort_keys=True),
        "questions": json.dumps(qs, ensure_ascii=False, sort_keys=True),
        "gold": json.dumps(gold, ensure_ascii=False, sort_keys=True),
        "outcome": outcome or {},
        "provenance": {
            "license": SOURCE_LICENSES.get(source, "audit required"),
            "synthetic": synthetic,
            "quality": quality,
            "label_source": label_source,
        },
    }


def load_hwe(root: Path) -> list[dict[str, Any]]:
    path = root / "hwe_bench_full.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        base = x.get("base") or {}
        repo = (base.get("repo") or {}).get("full_name") or f"{x.get('org', '')}/{x.get('repo', '')}"
        files = x.get("modified_files") or []
        actions = [action_for_file(p) for p in files]
        state = {
            "repository": repo,
            "base_commit": (base.get("sha") or ""),
            "issue_title": clip(x.get("title"), 1200),
            "issue_body": clip(x.get("body"), 5000),
            "tool_context": {
                "reproducer_path": x.get("reproducer_path"),
                "reproducer_signal": x.get("reproducer_signal"),
                "simulation_cost": x.get("simulation_cost"),
                "benchmark_level": x.get("level1"),
                "bug_scope": x.get("level2"),
            },
            "previous_actions": [],
        }
        rows.append(make_case(
            case_id=f"hwe:{x.get('id') or x.get('url')}",
            source="hwe-bench",
            group=f"hwe:{repo}",
            state=state,
            active_actions=actions,
            root_key=root_cause_key(" ".join([str(x.get("title", "")), str(x.get("body", "")), str(x.get("level2", ""))])),
            evidence=bool(x.get("reproducer_signal") or x.get("test_patch")),
            label_source="verified_repair",
            quality="gold",
            synthetic=False,
            outcome={
                "resolved": True,
                "verification": "hwe_fail_to_pass",
                "modified_files": files,
                "lines_added": x.get("lines_added"),
                "lines_removed": x.get("lines_removed"),
            },
        ))
    return rows


def load_rootcause(root: Path) -> list[dict[str, Any]]:
    rows = []
    for csv_name, kind in (("error_list.csv", "runtime"), ("error_list_synth.csv", "synthesis")):
        path = root / csv_name
        if not path.exists():
            continue
        with path.open(newline="", errors="replace") as f:
            for row in csv.DictReader(f):
                bug_id = str(row.get("Bug ID", "")).strip()
                prefix = "bug_" if kind == "runtime" else "synth_bug_"
                case_dir = root / ("runtime_errors" if kind == "runtime" else "synthesis_errors") / f"{prefix}{bug_id}"
                rtl = "\n\n".join(read_text(p, 5000) for p in sorted((case_dir / "rtl").glob("*")) if p.is_file())
                tb = "\n\n".join(read_text(p, 3000) for p in sorted((case_dir / "testbench").glob("*")) if p.is_file())
                state = {
                    "source_kind": kind,
                    "language": row.get("Language"),
                    "bug_type": row.get("Type of Bug"),
                    "description": clip(row.get("Description") or row.get("Human Error message"), 2500),
                    "error_message": clip(row.get("Error Message"), 3000),
                    "question_context": clip(row.get("Question"), 3000),
                    "rtl_context": rtl,
                    "testbench_context": tb,
                    "previous_actions": [],
                }
                text = " ".join(str(v) for v in row.values())
                rows.append(make_case(
                    case_id=f"rootcause:{kind}:{bug_id}",
                    source="RootCause-Bench",
                    group=f"rootcause:{kind}:{bug_id}",
                    state=state,
                    active_actions=["rtl"],
                    root_key=root_cause_key(text),
                    evidence=True,
                    label_source="manual_bug_label",
                    quality="silver",
                    synthetic=False,
                    outcome={"resolved": None, "verification": "benchmark_case"},
                ))
    return rows


def load_origen(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(errors="replace").splitlines()):
        if not line.strip():
            continue
        x = json.loads(line)
        instruction = x.get("Instruction", "")
        state = {
            "task": clip(instruction, 11000),
            "previous_actions": [],
            "response_available": False,
        }
        rows.append(make_case(
            case_id=f"origen:{idx}",
            source="OriGen",
            group=f"origen:{hashlib.sha1(instruction.encode()).hexdigest()[:16]}",
            state=state,
            active_actions=["rtl"],
            root_key=root_cause_key(instruction),
            evidence=bool(re.search(r"error message|error|syntax|compile", instruction, re.I)),
            label_source="repair_pair_unvalidated",
            quality="bronze",
            synthetic=True,
            outcome={"resolved": None, "verification": "not_replayed"},
        ))
    return rows


def load_rtl_augmented(root: Path) -> list[dict[str, Any]]:
    rows = []
    for info_path in root.rglob("augment_info.json"):
        try:
            info = json.loads(info_path.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("simulation_status") != "sim_ok":
            continue
        case_dir = info_path.parent
        rtl_files = sorted(case_dir.glob("original_*.v")) + sorted(case_dir.glob("original_*.sv"))
        rtl = "\n\n".join(read_text(p, 9000) for p in rtl_files)
        state = {
            "repository": info.get("repo"),
            "module": info.get("module"),
            "rtl_context": rtl,
            "simulation_log": read_text(case_dir / "sim_log.txt", 3000),
            "tool": "simulator",
            "previous_actions": [],
        }
        repo = str(info.get("repo") or "unknown")
        module = str(info.get("module") or case_dir.parent.name)
        bug_type = str(info.get("bug_type") or "unknown")
        rows.append(make_case(
            case_id=f"rtl-augmented:{repo}:{module}:{bug_type}",
            source="RTL-augmented",
            group=f"rtl-augmented:{repo}:{module}",
            state=state,
            active_actions=["rtl"],
            root_key=root_cause_key(bug_type),
            evidence=True,
            label_source="validated_mutation",
            quality="silver_validated",
            synthetic=True,
            outcome={
                "resolved": True,
                "verification": "sim_ok",
                "bug_type": bug_type,
                "files_modified": info.get("files_modified", []),
            },
        ))
    return rows


def load_hiersva(root: Path) -> list[dict[str, Any]]:
    rows = []
    for variant, synthetic in (("buggy_rtl", True), ("buggy_rtl2", False)):
        for rtl_path in sorted((root / variant).rglob("*.sv")):
            bug_description = read_text(rtl_path.with_suffix(".md"), 2500)
            code = read_text(rtl_path, 9000)
            state = {
                "module": rtl_path.stem,
                "hierarchy_group": rtl_path.parent.name,
                "rtl_context": code,
                "previous_actions": [],
            }
            rows.append(make_case(
                case_id=f"hiersva:{variant}:{rtl_path.relative_to(root)}",
                source="HierSVA",
                group=f"hiersva:{rtl_path.parent.name}:{rtl_path.stem}",
                state=state,
                active_actions=["rtl"],
                root_key=root_cause_key(bug_description),
                evidence=True,
                label_source="synthetic_bug_pattern" if synthetic else "historical_bug",
                quality="silver_formal_context",
                synthetic=synthetic,
                outcome={"resolved": None, "verification": "formal_benchmark"},
            ))
    return rows


def load_chipbench(root: Path) -> list[dict[str, Any]]:
    rows = []
    for prompt_path in sorted((root / "src" / "chipbench" / "data" / "debug").glob("*/*_prompt.txt")):
        prompt = read_text(prompt_path, 11000)
        stem = prompt_path.name.removesuffix("_prompt.txt")
        bug_type = prompt_path.parent.name + " " + stem
        state = {
            "benchmark_task": "chipbench_debug",
            "bug_family": prompt_path.parent.name,
            "problem": clip(prompt, 11000),
            "tool": "Icarus/functional simulation",
            "previous_actions": [],
        }
        rows.append(make_case(
            case_id=f"chipbench:{prompt_path.parent.name}:{stem}",
            source="Inspect-Eval-ChipBench",
            group=f"chipbench:{prompt_path.parent.name}:{stem}",
            state=state,
            active_actions=["rtl"],
            root_key=root_cause_key(bug_type),
            evidence=True,
            label_source="benchmark_bug_family",
            quality="silver_benchmark",
            synthetic=True,
            outcome={"resolved": None, "verification": "reference_and_test_artifacts_present"},
        ))
    return rows


def split_name(group: str) -> str:
    bucket = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % 100
    if bucket < 75:
        return "train"
    if bucket < 88:
        return "validation"
    return "test"


def validate(row: dict[str, Any]) -> list[str]:
    errors = []
    try:
        state = json_value(row["state"])
        qs = json_value(row["questions"])
        gold = json_value(row["gold"])
    except (KeyError, TypeError):
        return ["missing state/questions/gold"]
    if not state:
        errors.append("empty state")
    if not isinstance(qs, dict) or not qs:
        errors.append("questions must be a non-empty object")
    if set(qs or {}) != set(gold or {}):
        errors.append("question/gold IDs differ")
    for qid, q in (qs or {}).items():
        if q.get("type") not in {"choice", "noul", "score"}:
            errors.append(f"{qid}: invalid type")
            continue
        crit = q.get("criteria")
        if q["type"] == "choice" and (not isinstance(crit, dict) or not crit or len(crit) > 64):
            errors.append(f"{qid}: invalid choice criteria")
        if q["type"] == "noul" and not isinstance(crit, dict):
            errors.append(f"{qid}: invalid noul criteria")
        if q["type"] == "score" and (not isinstance(crit, list) or not 2 <= len(crit) <= 10):
            errors.append(f"{qid}: invalid score criteria")
        probs = (gold.get(qid) or {}).get("probabilities", {})
        if not probs or any(float(v) < 0 for v in probs.values()) or sum(float(v) for v in probs.values()) <= 0:
            errors.append(f"{qid}: invalid probabilities")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--include-origen", action="store_true")
    args = ap.parse_args()
    root = args.dataset_root
    out = args.output_dir or (root / "normalized")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows += load_hwe(root / "raw/hf/hwe-bench")
    rows += load_hiersva(root / "raw/hf/HierSVA")
    rows += load_rootcause(root / "raw/github/root-cause-bench")
    rows += load_chipbench(root / "raw/github/Inspect-Eval-ChipBench")
    rows += load_rtl_augmented(root / "raw/hf/rtl-augmented")
    if args.include_origen:
        rows += load_origen(root / "raw/hf/origen-dataset-debug/origen_debug.jsonl")

    dedup: dict[str, dict[str, Any]] = {}
    invalid = []
    for row in rows:
        errors = validate(row)
        if errors:
            invalid.append({"id": row.get("id"), "errors": errors})
            continue
        dedup[row["id"]] = row
    rows = list(dedup.values())

    for split in ("train", "validation", "test"):
        path = out / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                if split_name(row["source_group"]) == split:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "all.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "validation_errors.json").write_text(json.dumps(invalid, indent=2))

    counts = Counter(row["source"] for row in rows)
    split_counts = Counter(split_name(row["source_group"]) for row in rows)
    quality_counts = Counter(row["provenance"]["quality"] for row in rows)
    summary = {
        "total_cases": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "invalid_count": len(invalid),
        "include_origen": args.include_origen,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
