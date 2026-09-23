#!/usr/bin/env python3
"""Replay each Fixbench-RTL testbench against the buggy and the corrected RTL.

Evidence discipline:

- the replay of `buggycode` is pre-repair evidence and may enter the model state;
- the replay of `correctcode` is verification only. Its log is written to the
  outcome/provenance side of a record and never to the state. Both replays are
  kept in the result file; the converter decides what is allowed to surface.

The raw `Fixbench-RTL.json` is opened read-only. Replay workspaces are written to
a cache directory that can be deleted and rebuilt.

Backends, in order of preference: Icarus Verilog (open, fast, no licence), then
Questa/ModelSim via `vlog` + `vsim` for the SystemVerilog the open tool rejects.
A licence file is located automatically and passed through `LM_LICENSE_FILE`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "dataset/raw/hf/Fixbench-RTL/Fixbench-RTL.json"
DEFAULT_CACHE = Path(__file__).resolve().parent / "fixbench_replay_cache"

COMPILE_TIMEOUT = 180
RUN_TIMEOUT = 90
TOOL_INPUT_LIMIT = 4000  # characters of tool output kept per stream

PASS_MARK = re.compile(r"=+\s*Passed\b[^=\n]*?=+", re.I)
FAIL_MARK = re.compile(r"=+\s*Failed\b[^=\n]*?=+", re.I)
FAIL_LINE = re.compile(r"^\s*(?:\*\*)?\s*(?:FAIL|Failed)\b", re.I)
# `=====Test completed with 69 / 100 failures=====` and the denominator-less
# `=====Test completed with  4 failures=====` are verdicts too, not chatter: the
# testbench states the failure count itself. An `x` placeholder (it occurs once
# in this corpus) states nothing and is left unparsed on purpose.
TB_SUMMARY = re.compile(
    r"=+\s*Test completed with\s+(x|\d+)\s*(?:/\s*\d+)?\s*failures?\s*=+", re.I)
QUESTA_SUMMARY = re.compile(r"^#\s*Errors:\s*(\d+)", re.M)
SIM_ERROR = re.compile(
    r"^\s*(?:\*\*\s*)?(?:Error|Fatal|\$fatal)\b|error-\[|%Error|RUN_FAILED|"
    r"Errors:\s*[1-9]", re.I | re.M)
SOURCE_FILE = re.compile(r"[A-Za-z_0-9./]+\.(?:sv|svh|v|vh)\b")
# A tooling failure is not a verdict. These happen when the design cannot be
# loaded/elaborated or the compiled model is missing: they say nothing about
# whether the testbench passed, so they must never be recorded as a failure.
TOOL_ERROR = re.compile(
    r"Error loading design|Unable to open input file|Failed to load|"
    r"vsim-\d+.*(?:fatal|error)|Design unit .* not found|"
    r"cannot find (?:the )?(?:design|module)|Unknown module type|"
    r"bad option|cannot open|no such file", re.I)
COMPILE_ERROR = re.compile(r"error|syntax error|not (?:declared|supported)|"
                           r"invalid module|cannot be driven|malformed", re.I)


def testbench_tops(testbench: str) -> list[str]:
    """Candidate top modules for the supplied testbench.

    The testbench file often declares helper modules in addition to the root, and
    the root is not always called `tb`. A module that is instantiated inside the
    file cannot be the root, so those are dropped; conventional names come first.
    """
    declared = re.findall(r"^\s*module\s+([A-Za-z_]\w*)", testbench, re.M)
    if not declared:
        return []
    instantiated = set()
    for name in declared:
        pattern = rf"(?<![.\w]){re.escape(name)}\s*(?:#\s*\([^;()]*\)\s*)?([A-Za-z_]\w*)\s*\("
        if re.search(pattern, testbench):
            instantiated.add(name)
    roots = [name for name in declared if name not in instantiated] or declared[-1:]
    roots.sort(key=lambda n: (0 if re.search(r"test ?bench|^tb$|^top$", n, re.I) else 1,
                              declared.index(n)))
    return roots


def clip(text: str, limit: int = TOOL_INPUT_LIMIT) -> str:
    text = (text or "").replace("\x00", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def run(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None) -> dict:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        return {"cmd": cmd, "returncode": proc.returncode, "timed_out": False,
                "stdout": clip(proc.stdout), "stderr": clip(proc.stderr)}
    except subprocess.TimeoutExpired as exc:
        return {"cmd": cmd, "returncode": None, "timed_out": True,
                "stdout": clip(exc.stdout.decode() if isinstance(exc.stdout, bytes)
                               else (exc.stdout or "")),
                "stderr": clip(exc.stderr.decode() if isinstance(exc.stderr, bytes)
                               else (exc.stderr or ""))}
    except FileNotFoundError:
        return {"cmd": cmd, "returncode": None, "timed_out": False,
                "stdout": "", "stderr": "tool not found"}


def last_marker_end(pattern: re.Pattern, text: str) -> int:
    """End offset of the last match, or -1. A transcript's final verdict line is
    the testbench's conclusion, so the last marker wins."""
    end = -1
    for match in pattern.finditer(text):
        end = match.end()
    return end


def verdict_signals(text: str) -> dict:
    """Pass/fail signals in one simulator transcript.

    Kept separate from the verdict so the audit can see *why* a verdict was
    reached, and so a transcript that says both things is visible rather than
    silently resolved.
    """
    signals = {"pass_at": last_marker_end(PASS_MARK, text), "fail_at": -1,
               "summary": None, "questa_errors": None, "error_lines": 0}
    fail_at = last_marker_end(FAIL_MARK, text)
    for match in FAIL_LINE.finditer(text):
        fail_at = max(fail_at, match.end())
    summary = TB_SUMMARY.search(text)
    if summary:
        signals["summary"] = summary.group(1)
        if summary.group(1) == "0":
            signals["pass_at"] = max(signals["pass_at"], summary.end())
        elif summary.group(1).isdigit():
            fail_at = max(fail_at, summary.end())
    questo = QUESTA_SUMMARY.search(text)
    if questo:
        signals["questa_errors"] = int(questo.group(1))
    signals["error_lines"] = len(SIM_ERROR.findall(text))
    if signals["error_lines"]:
        fail_at = max(fail_at, 0)
    signals["fail_at"] = fail_at
    signals["verdict"] = ("failed" if fail_at >= 0 else
                          "passed" if signals["pass_at"] >= 0 else "unknown")
    return signals


def verdict_of(compile_step: dict | None, run_step: dict | None) -> tuple[str, dict]:
    """One of: compile_failed, passed, failed, timeout, tool_error, unknown.

    `failed` means the testbench or the simulator reported a failure of the
    design. `tool_error` means the tool could not load or run the model at all,
    which is not evidence about the design.
    """
    if compile_step is None or "tool not found" in (compile_step.get("stderr") or ""):
        return "tool_missing", {}
    if compile_step["timed_out"]:
        return "timeout", {}
    if compile_step["returncode"] not in (0, None):
        return "compile_failed", {}
    if run_step is None:
        return "unknown", {}
    combined = f"{run_step['stdout']}\n{run_step['stderr']}"
    signals = verdict_signals(combined)
    if signals["verdict"] != "unknown":
        return signals["verdict"], signals
    if TOOL_ERROR.search(combined):
        return "tool_error", signals
    if run_step["timed_out"]:
        return "timeout", signals
    if run_step["returncode"] not in (0, None):
        return "failed", signals
    return "unknown", signals


def first_error(attempt: dict) -> str | None:
    """First diagnostic line of an attempt, for the audit trail."""
    compile_step = attempt.get("compile") or {}
    run_step = attempt.get("run") or {}
    for stream in (compile_step.get("stderr"), compile_step.get("stdout"),
                   run_step.get("stderr"), run_step.get("stdout")):
        for line in (stream or "").splitlines():
            if re.search(r"error|fatal|not found|cannot|could not|missing", line, re.I):
                return line.strip()[:200]
    return None


def failure_signature(attempt: dict) -> tuple[str, ...]:
    """How a side failed, with source filenames and numbers normalised away.

    Two sides sharing a signature failed in the same *way* (same messages, not
    merely the same verdict), which is what distinguishes a failure caused by
    the buggy revision from one that survives the repair unchanged.
    """
    compile_step = attempt.get("compile") or {}
    run_step = attempt.get("run") or {}
    text = "\n".join(filter(None, [compile_step.get("stderr"), compile_step.get("stdout"),
                                   run_step.get("stdout"), run_step.get("stderr")]))
    lines = []
    for line in text.splitlines():
        if not re.search(r"error|unknown module|missing|fail|fatal|mismatch", line, re.I):
            continue
        norm = SOURCE_FILE.sub("<src>", line)
        norm = re.sub(r"\d+", "N", norm)
        lines.append(re.sub(r"\s+", " ", norm).strip().lower())
    return tuple(sorted(set(lines)))


def iverilog_replay(case_dir: Path, source: str, name: str) -> dict:
    vvp = f"{name}.vvp"
    # The design file is compiled before the testbench: 13 of the supplied
    # testbenches `import` a package that is declared in the design file, and
    # both simulators resolve package imports while parsing, so the declaring
    # file has to come first. Module visibility is order-independent.
    compile_step = run(["iverilog", "-g2012", "-o", vvp, source, "tb.sv"], case_dir, COMPILE_TIMEOUT)
    compile_step["argv"] = compile_step["cmd"]
    if compile_step["returncode"] != 0 or compile_step["timed_out"]:
        verdict, signals = verdict_of(compile_step, None)
        return {"backend": "iverilog", "compile": compile_step, "run": None,
                "verdict": verdict, "signals": signals}
    run_step = run(["vvp", vvp], case_dir, RUN_TIMEOUT)
    verdict, signals = verdict_of(compile_step, run_step)
    return {"backend": "iverilog", "compile": compile_step, "run": run_step,
            "verdict": verdict, "signals": signals}


def questa_env() -> dict | None:
    env = dict(os.environ)
    if not env.get("LM_LICENSE_FILE") and not env.get("MGLS_LICENSE_FILE"):
        candidates = [Path("/usr/local/questasim/license.dat"),
                      Path("/usr/local/questa/license.dat"),
                      Path("/opt/questa/license.dat")]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            return None
        env["LM_LICENSE_FILE"] = str(found)
    return env


def questa_replay(case_dir: Path, source: str, name: str, vlog: str, vsim: str,
                  tops: list[str]) -> dict:
    env = questa_env()
    if env is None:
        return {"backend": "questa", "compile": None, "run": None, "verdict": "tool_missing"}
    lib = "work"  # one library per run directory: each replay owns its own work lib
    (case_dir / "modelsim.ini").touch()
    vlib = run(["vlib", lib], case_dir, COMPILE_TIMEOUT, env)
    if vlib["returncode"] != 0:
        return {"backend": "questa", "compile": vlib, "run": None, "verdict": "tool_missing"}
    compile_step = run([vlog, "-sv", "-quiet", "-work", lib, source, "tb.sv"],
                       case_dir, COMPILE_TIMEOUT, env)
    compile_step["argv"] = compile_step["cmd"]
    if compile_step["returncode"] != 0 or compile_step["timed_out"]:
        verdict, signals = verdict_of(compile_step, None)
        return {"backend": "questa", "compile": compile_step, "run": None,
                "verdict": verdict, "signals": signals}
    attempts = []
    chosen = None
    for top in tops or ["tb"]:
        run_step = run([vsim, "-c", "-quiet", "-do", "run -all; quit -f", top],
                       case_dir, RUN_TIMEOUT, env)
        verdict, signals = verdict_of(compile_step, run_step)
        attempts.append({"top": top, "verdict": verdict})
        chosen = {"backend": "questa", "top": top, "compile": compile_step,
                  "run": run_step, "verdict": verdict, "signals": signals,
                  "tops_tried": list(attempts)}
        if verdict != "tool_error":
            break
    return chosen


def tool_version(tool: str, args: list[str], cwd: Path, env: dict | None = None) -> str:
    step = run([tool] + args, cwd, 60, env)
    text = f"{step['stdout']}\n{step['stderr']}".strip()
    return text.splitlines()[0].strip() if text else "unknown"


VERDICT_PRIORITY = {"passed": 0, "failed": 0, "compile_failed": 1, "timeout": 2,
                    "tool_error": 3, "unknown": 4, "tool_missing": 5}
FAILING_VERDICTS = ("compile_failed", "failed", "timeout")


def classify_replay(result: dict) -> dict:
    """Whether the observed failure is attributable to the buggy revision.

    A failure is only evidence about the bug if the repair target removes it, so
    the class is decided by *pairs* of runs — buggy and corrected, under the
    **same simulator**. Pairing per simulator matters: a revision can fail to
    build under one tool and run fine under the other, and only a within-tool
    pair isolates the defect.

    - `verified_failure_at_base` — some simulator fails the buggy revision and
      passes the corrected one: the failure tracks the defect. `deciding_backend`
      names the simulator the claim is made under.
    - `testbench_passes_on_buggy_code` — no pair verifies, and some simulator ran
      the supplied testbench against the buggy revision and it passed: the
      testbench does not detect the defect, so it corroborates nothing.
    - `cannot_build_either_revision` — no pair verifies and *every* verdict
      obtained for either revision is a compile failure (typically an external
      package or an unsupported construct that neither supplied file provides).
    - `failure_survives_repair` — no pair verifies, and some simulator reported a
      testbench failure for both revisions.
    - `no_usable_verdict` — none of the above: the supplied pair yields no
      verdict line under any available simulator.
    - `tool_missing` — no simulator available.
    """
    buggy_by, correct_by = result["buggy_by_backend"], result["correct_by_backend"]
    shared = [b for b in buggy_by if b in correct_by]
    verified = [b for b in shared if buggy_by[b]["verdict"] in FAILING_VERDICTS
                and correct_by[b]["verdict"] == "passed"]
    ran_and_passed = [b for b, a in buggy_by.items() if a["verdict"] == "passed"]
    both_failed = [b for b in shared if buggy_by[b]["verdict"] in FAILING_VERDICTS
                   and correct_by[b]["verdict"] in FAILING_VERDICTS]
    verdicts = [a["verdict"] for a in list(buggy_by.values()) + list(correct_by.values())]
    result["deciding_backend"] = verified[0] if verified else None
    result["verified_backends"] = verified
    result["both_failed_backends"] = both_failed
    # A testbench that passes the buggy revision on any simulator does not detect
    # the defect, even when a stricter simulator's compile failure does track it.
    result["buggy_passes_on_backends"] = ran_and_passed
    result["all_compile_failed"] = bool(verdicts) and all(
        v == "compile_failed" for v in verdicts)
    buggy, correct = result["buggy"]["verdict"], result["correct"]["verdict"]
    result["same_failure_signature"] = (
        failure_signature(result["buggy"]) == failure_signature(result["correct"])
        if buggy in FAILING_VERDICTS and correct in FAILING_VERDICTS else None)
    if not verdicts or "tool_missing" in verdicts:
        klass = "tool_missing"
    elif verified:
        klass = "verified_failure_at_base"
    elif ran_and_passed:
        klass = "testbench_passes_on_buggy_code"
    elif result["all_compile_failed"]:
        klass = "cannot_build_either_revision"
    elif both_failed:
        klass = "failure_survives_repair"
    else:
        klass = "no_usable_verdict"
    result["attribution"] = klass
    result["failure_attributable_to_bug"] = klass == "verified_failure_at_base"
    # Evidence may only be read from the buggy replay when the failure tracks the
    # defect: see the log-channel gate in the converter.
    result["log_channel_usable"] = bool(result["failure_attributable_to_bug"])
    return result


def replay_case(index: int, case: dict, root: Path, backends: tuple[str, ...],
                vlog: str, vsim: str) -> dict:
    base_dir = root / f"case_{index:03d}"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    tops = testbench_tops(case["testbench"])
    result = {"index": index, "checkout_dir": base_dir.name, "testbench_tops": tops,
              "buggy_modules": re.findall(r"^\s*module\s+([A-Za-z_]\w*)", case["buggycode"], re.M),
              "correct_modules": re.findall(r"^\s*module\s+([A-Za-z_]\w*)", case["correctcode"], re.M),
              "attempts": {}, "buggy": None, "correct": None}

    for label, source in (("buggy", "dut.sv"), ("correct", "fix.sv")):
        # Each replay owns its own directory and `work` library so a stale model
        # from one run can never be loaded by another.
        run_dir = base_dir / label
        run_dir.mkdir(parents=True)
        (run_dir / "dut.sv").write_text(case["buggycode"])
        (run_dir / "fix.sv").write_text(case["correctcode"])
        (run_dir / "tb.sv").write_text(case["testbench"])
        attempts: list[dict] = []
        for backend in backends:
            if backend == "iverilog" and shutil.which("iverilog"):
                attempt = iverilog_replay(run_dir, source, label)
            elif backend == "questa" and vsim and Path(vsim).exists():
                attempt = questa_replay(run_dir, source, label, vlog, vsim, tops)
            else:
                attempt = {"backend": backend, "compile": None, "run": None,
                           "verdict": "tool_missing"}
            attempts.append(attempt)
            if attempt["verdict"] in ("passed", "failed"):
                break
        best = min(attempts, key=lambda a: VERDICT_PRIORITY[a["verdict"]])
        result[label] = best
        # Per-backend detail: classification pairs buggy against corrected within
        # one simulator, and the state records the commands of the deciding one.
        result[f"{label}_by_backend"] = {
            a["backend"]: {"verdict": a["verdict"], "top": a.get("top"),
                           "first_error": first_error(a),
                           "compile": a.get("compile"), "run": a.get("run"),
                           "tops_tried": a.get("tops_tried")}
            for a in attempts}
        result["attempts"][label] = [{"backend": a["backend"], "verdict": a["verdict"],
                                      "top": a.get("top"),
                                      "first_error": first_error(a)} for a in attempts]
    return classify_replay(result)
    return classify_replay(result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backends", default="iverilog,questa")
    args = ap.parse_args()

    if not args.raw.is_file():
        sys.exit(f"missing raw dataset: {args.raw} (download it from the public "
                 f"Hugging Face dataset KSU-HW-SEC/Fixbench-RTL)")
    cases = json.loads(args.raw.read_text())
    if not isinstance(cases, list):
        sys.exit(f"unexpected raw shape in {args.raw}: {type(cases).__name__}")

    vlog = shutil.which("vlog") or "/usr/local/questasim/linux_x86_64/vlog"
    vsim = shutil.which("vsim") or "/usr/local/questasim/linux_x86_64/vsim"
    args.cache.mkdir(parents=True, exist_ok=True)
    env = questa_env()
    versions = {
        "iverilog": tool_version("iverilog", ["-V"], args.cache) if shutil.which("iverilog") else None,
        "questa": tool_version(vsim, ["-version"], args.cache, env) if Path(vsim).exists() else None,
        "questa_license": (env or {}).get("LM_LICENSE_FILE"),
    }

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    todo = cases if args.limit is None else cases[: args.limit]
    results = []
    for index, case in enumerate(todo):
        results.append(replay_case(index, case, args.cache, backends, vlog, vsim))
        if (index + 1) % 10 == 0:
            print(f"  replayed {index + 1}/{len(todo)}", file=sys.stderr)

    payload = {
        "raw": str(args.raw),
        "raw_sha256": __import__("hashlib").sha256(args.raw.read_bytes()).hexdigest(),
        "cases": len(results),
        "backends_requested": list(backends),
        "tool_versions": versions,
        "verdicts": {
            "buggy": dict(sorted(Counter(r["buggy"]["verdict"] for r in results).items())),
            "correct": dict(sorted(Counter(r["correct"]["verdict"] for r in results).items())),
        },
        "attribution": dict(sorted(Counter(r["attribution"] for r in results).items())),
        "failure_attributable_to_bug": sum(1 for r in results if r["failure_attributable_to_bug"]),
        "log_channel_usable": sum(1 for r in results if r["log_channel_usable"]),
        "testbench_passes_on_buggy_code": sum(
            1 for r in results if r["buggy_passes_on_backends"]),
        "same_failure_signature_pairs": sum(1 for r in results if r["same_failure_signature"]),
        "both_sides_failed_on_a_backend": sum(1 for r in results if r["both_failed_backends"]),
        "results": results,
    }
    out = args.cache / "replay_results.json"
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in
                      ("cases", "verdicts", "attribution", "failure_attributable_to_bug",
                       "testbench_passes_on_buggy_code", "both_sides_failed_on_a_backend",
                       "tool_versions")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
