#!/usr/bin/env python3
"""Convert RTL-BenchLS Task 3 repository-issue cases into SilicoJev records.

Each Task 3 case is a real reported issue fixed by a real merged PR. The model
state is built only from pre-repair evidence: repository, task id, base commit,
issue title/body/triage labels, the buggy RTL at the base commit, the affected
file names, and the benchmark's source provenance. The patch, the head commit,
the corrected RTL, the PR description and post-repair status are used only to
derive and verify labels and are stored in `outcome`/`provenance`.

Outputs (under dataset/converted/rtl_benchls/):

  records_3q.jsonl                    three-question records with no pseudo label
  records_3q_unverified.jsonl         three-question records with >=1 pseudo label
  records_5q.jsonl                    five-question records whose risk and urgency
                                      are anchored in explicit issue evidence
  records_5q_pseudo_unverified.jsonl  five-question records whose risk/urgency are
                                      transparent rubric estimates only
  conversion_report.json              counts, splits, rule traces, validation
  README.md                           method, rubric and audit notes

Usage:
    python3 training/convert_rtl_benchls.py
    python3 training/convert_rtl_benchls.py --no-network
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import ACTION_CRITERIA, ROOT_CAUSE_CRITERIA  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "dataset/raw/github/RTL-BenchLS"
DEFAULT_OUT = REPO_ROOT / "dataset/converted/rtl_benchls"

SOURCE = "RTL-BenchLS"
LICENSE = "CC-BY-4.0 (RTL-BenchLS); upstream repository licenses apply"

BASE_QUESTION_IDS = ("next_action", "root_cause_type", "evidence_sufficient")

RISK_CRITERIA = [
    "Benign: read-only inspection, parsing, linting, or clearly reversible analysis.",
    "Low: routine simulation, formal checking, or isolated writes within the project scope.",
    "Moderate: modifying RTL, testbench, constraints, or configuration in a way that could affect results.",
    "High: destructive, irreversible, security-sensitive, out-of-scope, or potentially damaging action.",
]

URGENCY_CRITERIA = [
    "No time pressure; can wait indefinitely.",
    "Routine; handle within the normal queue.",
    "Elevated; should be handled within the same week.",
    "Critical; requires action within the same day.",
]

# Context budget for the base-revision RTL placed in the state. Files that fit
# are included whole. Larger files are represented by bounded, non-overlapping
# context windows whose bounds are recorded in the state, so an omitted region
# never silently hides the affected code.
RTL_FILE_LIMIT = 40000
RTL_MIN_FILE_CHARS = 4000    # floor on a file's share of the per-case budget
RTL_TOTAL_LIMIT = 100000
RTL_HEADER_LINES = 60        # always include the file header (module ports/params)
RTL_WINDOW_LINES = 260       # preferred size of each additional context window
RTL_MIN_WINDOW_LINES = 80    # never shrink a context window below this
RTL_MAX_WINDOWS = 8          # per-file cap
RTL_MERGE_GAP = 40           # merge windows separated by fewer than this many lines

# Repository-disjoint splits. Groups keep one repository - and one bug lineage
# (openhwgroup cores plus the pulp-platform cells they instantiate) - inside a
# single split, and hold lowRISC/opentitan out entirely.
SPLIT_GROUPS = {
    "train": [
        "lowRISC/ibex",
        "analogdevicesinc/hdl",
        "openrisc/mor1kx",
        "YosysHQ/picorv32",
        "black-parrot/black-parrot",
    ],
    "validation": [
        "openhwgroup/cv32e40p",
        "openhwgroup/cvfpu",
        "pulp-platform/common_cells",
    ],
    "test": ["lowRISC/opentitan"],
}

POST_REPAIR_LABEL = re.compile(r"^Status:", re.I)

# ------------------------------------------------------------------ risk rules
# A rule fires only on explicit issue content - never on the mere existence,
# age, importance or eventual success of a bug report.
SECURITY_VULNERABILITY = re.compile(
    r"security (hole|vulnerabilit|flaw|issue)|privilege escalation|"
    r"allows? (an? )?(attacker|unprivileged|malicious)|unauthori[sz]ed (access|write)|"
    r"exploit|information (leak|flow)|side.?channel|insecure|"
    r"bypass(es)? (the )?(pmp|lock|security|access|protection|permission)",
    re.I,
)
# Access-control correctness defects (a CSR/field that should be read-only or
# permission-checked but is not). Security-relevant, but not a demonstrated
# vulnerability, so they score below SECURITY_VULNERABILITY.
ACCESS_CONTROL_CORRECTNESS = re.compile(
    r"illegal write to read.?only|access rights? (violat|check|error)|"
    r"read.?only (field|csr|register)|permission (check|violat)|lock.?bit|smepmp",
    re.I,
)
DESTRUCTIVE_BEHAVIOR = re.compile(
    r"silently corrupts?|corrupts? (the )?(memory|data|state|register)|"
    r"data loss|loses? data|bricks?|permanently|destroys?|"
    r"hangs? (the )?(system|core|bus|machine)|hang(ing)? (condition|the|up)|"
    r"\bhangs?\b|gets? stuck|stuck (with|in) |never (ending|terminates|completes|"
    r"decrements|de-?asserts?|updates?)|"
    r"deadlock(s)? (the )?|lock(s|ed)? up (the )?(system|core|bus)|unrecoverable",
    re.I,
)
SAFETY_PROTOCOL_CRITICAL = re.compile(
    r"protocol (violation|error|critical)|violates? the (protocol|specification|spec|handshake)|"
    r"breaks? the (protocol|handshake|bus protocol|axi|ahb|tlul|wishbone)|"
    r"safety.?critical|functional safety|spec.?compliance failure|"
    r"does not (comply|conform) with (the )?(spec|specification|isa)",
    re.I,
)
SYSTEM_IMPACT = re.compile(
    r"cannot (be )?(boot|program|synthesi[sz]e|be used)|"
    r"fails? to (boot|elaborate|compile|program)|"
    r"breaks? the (build|flow|test|simulation|design)|"
    r"blocks? (the )?(release|tapeout|sign.?off|milestone|build|flow|integration)|"
    r"regression (in|of|breaks?)|fails? (all|every) (test|simulation)|"
    r"incorrect (results?|output|behavior) (for|in) (all|most|every)|"
    r"whole (system|core|chip)|denial of service",
    re.I,
)
PRODUCTION_BLOCKING_LABEL = {
    "Earlgrey-PROD Candidate": "production candidate milestone",
    "prodc-integration": "production integration milestone",
    "NoECO_IntegratePostM5": "post-M5 integration milestone",
    "Type:Spec-Compliance": "specification non-compliance",
}
DEFERRED_LABEL = {
    "Type:FutureRelease": "explicitly deferred to a future release",
    "WAIVED:CV32E40P": "explicitly waived for this core",
}
PRIORITY_LABEL = re.compile(r"^Priority:P(\d)", re.I)
MILESTONE_LABEL = re.compile(r"^Milestone:(.+)$", re.I)
EXPLICIT_LOCATION = re.compile(r"[\w/]+\.(sv|svh|v|vh)\b|\bline \d+", re.I)

URGENCY_CRITICAL = {"0": 0.03, "1": 0.12, "2": 0.35, "3": 0.50}
URGENCY_ELEVATED = {"0": 0.07, "1": 0.28, "2": 0.45, "3": 0.20}
URGENCY_BACKLOG = {"0": 0.70, "1": 0.25, "2": 0.05, "3": 0.00}
URGENCY_P1 = {"0": 0.05, "1": 0.12, "2": 0.48, "3": 0.35}
URGENCY_P2 = {"0": 0.12, "1": 0.52, "2": 0.30, "3": 0.06}

RISK_HIGH_SECURITY = {"0": 0.02, "1": 0.06, "2": 0.37, "3": 0.55}
RISK_ACCESS_CONTROL = {"0": 0.03, "1": 0.10, "2": 0.62, "3": 0.25}
RISK_MODERATE = {"0": 0.05, "1": 0.15, "2": 0.65, "3": 0.15}
RISK_LOW_INSPECTION = {"0": 0.80, "1": 0.16, "2": 0.04, "3": 0.00}
RISK_BASELINE = {"0": 0.55, "1": 0.35, "2": 0.09, "3": 0.01}

URGENCY_ANCHORED = {"U_PRIORITY_P1", "U_PRIORITY_P2", "U_PRODUCTION_LABEL",
                    "U_MILESTONE_LABEL", "U_DEFERRED_LABEL", "U_SECURITY_VULNERABILITY",
                    "U_ACCESS_CONTROL", "U_DESTRUCTIVE_IMPACT", "U_PROTOCOL_CRITICAL",
                    "U_SYSTEM_IMPACT"}
RISK_ANCHORED = {"R_SECURITY_VULNERABILITY", "R_ACCESS_CONTROL", "R_PROTOCOL_CRITICAL",
                 "R_DESTRUCTIVE_IMPACT", "R_SYSTEM_IMPACT"}
RISK_RUBRIC = {"R_READONLY_ACTION"}


def urgency_rule(case: dict) -> tuple[str, dict, str, str]:
    issue = case.get("issue_info") or {}
    labels = issue.get("labels") or []
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"

    for label in labels:
        m = PRIORITY_LABEL.match(label)
        if m:
            dist = URGENCY_P1 if m.group(1) == "1" else URGENCY_P2
            return (f"U_PRIORITY_P{m.group(1)}", dist, "manual_issue_label",
                    f"issue carries tracker priority label {label}")
    for label, why in PRODUCTION_BLOCKING_LABEL.items():
        if label in labels:
            return ("U_PRODUCTION_LABEL", URGENCY_ELEVATED, "manual_issue_label",
                    f"issue label {label} ties the fix to a {why}")
    for label in labels:
        m = MILESTONE_LABEL.match(label)
        if m:
            return ("U_MILESTONE_LABEL", URGENCY_ELEVATED, "manual_issue_label",
                    f"issue label {label} schedules the work against a milestone")
    for label, why in DEFERRED_LABEL.items():
        if label in labels:
            return ("U_DEFERRED_LABEL", URGENCY_BACKLOG, "manual_issue_label",
                    f"issue label {label}: {why}")
    if SECURITY_VULNERABILITY.search(text):
        return ("U_SECURITY_VULNERABILITY", URGENCY_CRITICAL, "manual_issue_label",
                "issue states a concrete security vulnerability")
    if ACCESS_CONTROL_CORRECTNESS.search(text):
        return ("U_ACCESS_CONTROL", URGENCY_ELEVATED, "manual_issue_label",
                "issue states an access-control/read-only-field correctness defect")
    if DESTRUCTIVE_BEHAVIOR.search(text):
        return ("U_DESTRUCTIVE_IMPACT", URGENCY_CRITICAL, "manual_issue_label",
                "issue states destructive or unrecoverable behaviour")
    if SAFETY_PROTOCOL_CRITICAL.search(text):
        return ("U_PROTOCOL_CRITICAL", URGENCY_ELEVATED, "manual_issue_label",
                "issue states a protocol/specification-critical failure")
    if SYSTEM_IMPACT.search(text):
        return ("U_SYSTEM_IMPACT", URGENCY_ELEVATED, "manual_issue_label",
                "issue states a clear system/build/production impact")
    return ("U_NONE", URGENCY_BACKLOG, "codex_pseudo_unverified",
            "no priority, schedule or impact evidence in the issue; rubric baseline only")


def risk_rule(case: dict, action_probs: dict[str, float]) -> tuple[str, dict, str, str]:
    issue = case.get("issue_info") or {}
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"

    if SECURITY_VULNERABILITY.search(text):
        return ("R_SECURITY_VULNERABILITY", RISK_HIGH_SECURITY, "manual_issue_label",
                "issue reports a security vulnerability, so acting on this logic is security-sensitive")
    if ACCESS_CONTROL_CORRECTNESS.search(text):
        return ("R_ACCESS_CONTROL", RISK_ACCESS_CONTROL, "manual_issue_label",
                "issue reports an access-control/read-only-field defect; repair-grade changes "
                "to security-relevant logic can affect results")
    if SAFETY_PROTOCOL_CRITICAL.search(text):
        return ("R_PROTOCOL_CRITICAL", RISK_MODERATE, "manual_issue_label",
                "issue reports a protocol/specification-critical failure; repair-grade changes affect results")
    if DESTRUCTIVE_BEHAVIOR.search(text):
        return ("R_DESTRUCTIVE_IMPACT", RISK_MODERATE, "manual_issue_label",
                "issue reports destructive/unrecoverable behaviour; handling can affect results")
    if SYSTEM_IMPACT.search(text):
        return ("R_SYSTEM_IMPACT", RISK_MODERATE, "manual_issue_label",
                "issue reports a system/build-level impact; changes are result-affecting")
    dominant, prob = max(action_probs.items(), key=lambda kv: kv[1])
    if prob >= 0.75 and dominant in {"rtl", "waveform", "specification"}:
        return ("R_READONLY_ACTION", RISK_LOW_INSPECTION, "criteria_applied_verified_action",
                f"repair-pinned next_action '{dominant}' (p={prob}) is read-only inspection, "
                "which the published risk criteria place at benign/low")
    return ("R_NONE", RISK_BASELINE, "codex_pseudo_unverified",
            "no security, destructive or system-impact evidence in the issue and no single "
            "dominant next action; rubric baseline only")


# ------------------------------------------------------- next-action rule table
ACTION_RULES: list[tuple[str, str, str, float, str]] = [
    ("rtl", "patch", r".", 3.0, "affected files are RTL and the merged fix edits them"),
    ("specification", "issue", r"spec|specification|\bisa\b|risc-v|compliance|section \d|chapter",
     1.2, "issue references a specification or standard"),
    ("specification", "label", r"^Type:Spec-Compliance$", 1.5, "issue is labelled spec-compliance"),
    ("testbench", "issue", r"testbench|\btb\b|\buvm\b|\bdv\b|regression|test suite|test case|stimulus",
     1.0, "issue references test or DV collateral"),
    ("testbench", "label", r"^Component:DV$", 1.2, "issue is labelled as a DV issue"),
    ("simulation", "issue", r"simulat|verilator|\bvcs\b|xcelium|questa|iverilog|waveform dump",
     0.9, "issue references a simulator or simulation run"),
    ("waveform", "issue", r"waveform|cycle.?(level|accurate)|timing diagram|signal trace|scope shot",
     0.9, "issue references cycle-level or waveform behaviour"),
    ("formal", "issue", r"assertion|\bsva\b|formal|property|counterexample|jasper|conformal|\blec\b",
     1.1, "issue references formal/assertion evidence"),
    ("constraints", "issue", r"\bxdc\b|\bsdc\b|constraint|false path|multicycle|timing closure|synthesi[sz]",
     0.8, "issue references constraints or synthesis"),
    ("ask_human", "label", r"^Type:Question$", 0.6, "issue is an open question rather than a defect report"),
    ("rtl", "label", r"^Component:RTL$", 1.5, "issue is triaged to the RTL component"),
]

# ------------------------------------------------------- root-cause rule table
ROOT_CAUSE_RULES: list[tuple[str, str, float, str]] = [
    ("security", r"security|vulnerab|privilege|access right|read.?only|lock.?bit|"
                 r"escalat|integrity|information flow|smepmp|\bpmp\b",
     "both", 2.0, "change or report concerns access-control / security logic"),
    ("formal_property", r"assertion|property|\bsva\b|formal|counterexample|coverpoint|invariant",
     "both", 2.0, "change or report concerns assertions/formal properties"),
    ("reset_initialization", r"\breset|\brst\b|rst_|initiali[sz]|uninitiali[sz]|unknown state|"
                             r"x-prop|\bx state|dont.?care state",
     "both", 1.8, "change or report concerns reset/initialisation"),
    ("type_width", r"width|sign.?extend|zero.?extend|truncat|signed|unsigned|overflow|"
                   r"underflow|bit.?width|\bmsb\b|\blsb\b",
     "both", 1.8, "change or report concerns width/signedness"),
    ("timing_protocol", r"handshake|ready.?valid|valid.?ready|\bcdc\b|clock domain|synchroni[sz]|"
                        r"metastab|\brace\b|setup time|hold time|timeout|protocol|backpressure",
     "both", 1.8, "change or report concerns timing/handshake/protocol"),
    ("state_machine", r"state machine|\bfsm\b|next.?state|state encoding|state transition|"
                      r"controller",
     "both", 1.8, "change or report concerns FSM/control flow"),
    ("syntax_compile", r"\bsyntax|compile|compilation|elaborat|l-value|undeclared|"
                       r"missing semicolon|parse error|does not compile|naming error|"
                       r"\bports?\b|instantiat|module interface",
     "both", 2.2, "change or report concerns syntax/elaboration/interface"),
    ("sequential_assignment", r"non.?blocking|blocking assignment|latch|always_ff|always_comb|"
                              r"flip.?flop|pipeline register|write.?back|registered output|clocked",
     "both", 1.6, "change or report concerns clocked/sequential assignment"),
    ("combinational_logic", r"combinational|arithmetic|adder|multiplier|selector|\bmux\b|"
                            r"decode|logic|compar|datapath|encod|assign|shift|mask|"
                            r"illegal|counter|priorit",
     "both", 1.2, "change or report concerns combinational/decoding logic"),
]

PATCH_STRONG_WEIGHT = 1.8
PATCH_WEIGHT = 1.0
ISSUE_WEIGHT = 0.35
TITLE_WEIGHT = 0.8
DECISIVE = 0.60
CLEAR = 0.40


def clip(text: str, limit: int) -> tuple[str, bool]:
    text = (text or "").replace("\x00", " ")
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[truncated]", True


def anchor_lines(file_lines: list[str], issue_title: str, issue_body: str) -> list[int]:
    """Line numbers the reporter's own wording points at, best anchor first.

    Uses only pre-repair evidence: identifiers named in the issue that also occur
    in this file. Identifiers named in the title, and identifiers that are rare
    in the file, rank higher because they localise better.
    """
    text = "\n".join(file_lines)
    stop = {"this", "that", "with", "when", "have", "should", "from", "will", "been",
            "they", "there", "which", "would", "could", "what", "then", "than", "some",
            "only", "does", "also", "very", "must", "same", "make", "case", "here",
            "more", "less", "code", "line", "issue", "error", "value", "result",
            "because", "however", "expect", "expected", "correct", "output", "input"}
    title_ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", issue_title))
    body_ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", issue_body))
    ranked: list[tuple[int, int, int]] = []  # (not_in_title, occurrences, line)
    for ident in title_ids | body_ids:
        if ident.lower() in stop:
            continue
        occurrences = len(re.findall(rf"\b{re.escape(ident)}\b", text))
        if not occurrences:
            continue
        match = re.search(rf"^\s*.*\b{re.escape(ident)}\b", text, re.M)
        if not match:
            continue
        line = text[: match.start()].count("\n") + 1
        ranked.append((0 if ident in title_ids else 1, occurrences, line))
    ranked.sort()
    lines: list[int] = []
    seen: set[int] = set()
    for _, _, line in ranked:
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def _chars_per_line(file_lines: list[str]) -> float:
    return max(sum(len(line) + 1 for line in file_lines) / max(len(file_lines), 1), 1.0)


def select_windows(file_lines: list[str], anchors: list[int],
                   char_budget: int | None = None) -> list[tuple[int, int]]:
    """Bounded windows over one file: header, even spread, then evidence sites.

    Priority order is the top of the file (module declaration, ports, parameters,
    localparams — one window deep, never less than RTL_HEADER_LINES), then an even
    spread across the file, then the sites of identifiers the reporter named. The
    spread outranks the reporter's wording because a bug report is typically wrong
    about where the defect lives; an ablation over these 108 cases found issue-named
    anchors landing on 526 of 768 repair hunks against 534 for the spread alone, so
    anchors only get the capacity a spread leaves over. Only pre-repair evidence
    decides placement; the repaired region never does.
    """
    n = len(file_lines)
    if n == 0:
        return []
    allowance = n
    size = RTL_WINDOW_LINES
    if char_budget is not None:
        allowance = max(1, int(char_budget / _chars_per_line(file_lines)))
        size = max(RTL_MIN_WINDOW_LINES, min(RTL_WINDOW_LINES, allowance // 4))
    half = size // 2

    def around(line: int) -> tuple[int, int]:
        start = max(1, line - half)
        end = min(n, start + size - 1)
        return max(1, end - size + 1), end

    candidates = [(1, min(max(RTL_HEADER_LINES, size), n))]
    candidates += [around(int(n * k / 4)) for k in (1, 2, 3)]
    candidates += [around(line) for line in anchors]

    chosen: list[tuple[int, int]] = []
    for window in candidates:
        if any(window[0] >= c[0] and window[1] <= c[1] for c in chosen):
            continue
        chosen.append(window)
        if len(chosen) >= RTL_MAX_WINDOWS:
            break
    while len(chosen) > 1 and sum(b - a + 1 for a, b in chosen) > allowance:
        chosen.pop()

    chosen.sort()
    merged: list[tuple[int, int]] = []
    for start, end in chosen:
        if merged and start - merged[-1][1] <= RTL_MERGE_GAP:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged[:RTL_MAX_WINDOWS]


def render_windows(blob_text: str, windows: list[tuple[int, int]], path: str) -> str:
    """Deterministic rendering of window ranges over a base blob.

    The validator re-derives the expected text with this same function, so a
    state can only ever contain exact base-revision lines.
    """
    lines = blob_text.splitlines()
    n = len(lines)
    out: list[str] = []
    prev_end = 0
    for start, end in windows:
        if prev_end and start > prev_end + 1:
            out.append(f"// ... lines {prev_end + 1}-{start - 1} of {n} not shown ...")
        out.append(f"// context window: lines {start}-{end} of {n} in {path}")
        out.extend(lines[start - 1 : end])
        prev_end = end
    if prev_end and prev_end < n:
        out.append(f"// ... lines {prev_end + 1}-{n} of {n} not shown ...")
    return "\n".join(out)


def patch_region_lines(patch: str) -> list[tuple[int, int]]:
    """Base-file line ranges touched by a patch, from its hunk headers.

    Audit only: used to measure whether a context window happens to cover the
    repaired region. It never influences window placement.
    """
    ranges = []
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))?", patch, re.M):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        ranges.append((start, start + max(count, 1) - 1))
    return ranges


def covered_lines(windows: list[tuple[int, int]]) -> set[int]:
    covered: set[int] = set()
    for start, end in windows:
        covered.update(range(start, end + 1))
    return covered


def old_top_slice_coverage(blobs: dict, paths: list[str],
                           file_limit: int, total_limit: int) -> dict[str, set[int]]:
    """Line numbers the previous top-anchored scheme would have delivered.

    Reconstructs the replaced behaviour — a prefix slice of at most file_limit
    characters per file, drawn from one shared per-case budget, files in path
    order — so the report can compare the delivered context against it. Audit
    only; nothing depends on it.
    """
    budget = total_limit
    covered: dict[str, set[int]] = {}
    for path in paths:
        text = blobs.get(path)
        if text is None:
            continue
        limit = min(file_limit, budget)
        shown = len(text[:limit].splitlines()) if limit > 0 else 0
        covered[path] = set(range(1, shown + 1))
        budget -= min(len(text), limit)
    return covered


class RepoCache:
    """Read blob content at a commit from the blobless repo_cache."""

    def __init__(self, root: Path, allow_network: bool = True):
        self.root = root
        self.allow_network = allow_network
        self._text: dict[tuple[str, str, str], str | None] = {}
        self._diffs: dict[tuple[str, str, str, str], tuple[list[str], list[str]]] = {}
        self.checkout_failures: list[dict] = []

    def _git(self, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)

    def repo_dir(self, repo: str) -> Path:
        return self.root / repo.replace("/", "_")

    def has_repo(self, repo: str) -> bool:
        return (self.repo_dir(repo) / ".git").is_dir()

    def has_commit(self, repo: str, commit: str) -> bool:
        if not self.has_repo(repo):
            return False
        return self._git(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                         self.repo_dir(repo)).returncode == 0

    def read(self, repo: str, commit: str, path: str) -> str | None:
        key = (repo, commit, path)
        if key in self._text:
            return self._text[key]
        dest = self.repo_dir(repo)
        if not self.has_repo(repo):
            self.checkout_failures.append({"repository": repo, "reason": "repository not cached"})
            self._text[key] = None
            return None
        proc = self._git(["git", "show", f"{commit}:{path}"], dest)
        if proc.returncode != 0 and self.allow_network:
            self._git(["git", "fetch", "-q", "--filter=blob:none", "origin", commit], dest)
            proc = self._git(["git", "show", f"{commit}:{path}"], dest)
        text = proc.stdout if proc.returncode == 0 else None
        if text is None:
            self.checkout_failures.append({"repository": repo, "commit": commit,
                                           "path": path, "reason": "git show failed"})
        self._text[key] = text
        return text

    def changed_lines(self, repo: str, base: str, head: str, path: str) -> tuple[list[str], list[str]]:
        key = (repo, base, head, path)
        if key in self._diffs:
            return self._diffs[key]
        removed, added = [], []
        if self.has_repo(repo):
            proc = self._git(["git", "diff", "--no-color", "-U0", base, head, "--", path],
                             self.repo_dir(repo))
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    if line.startswith(("---", "+++", "@@")):
                        continue
                    if line.startswith("-"):
                        removed.append(line[1:])
                    elif line.startswith("+"):
                        added.append(line[1:])
        self._diffs[key] = (removed, added)
        return removed, added


def parse_patch(patch: str) -> tuple[list[str], list[str], list[str]]:
    removed, added, context = [], [], []
    for line in patch.splitlines():
        if line.startswith(("@@", "---", "+++")):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
        elif line.startswith(" "):
            context.append(line[1:])
    return removed, added, context


def norm(lines: list[str]) -> Counter:
    return Counter(x.strip() for x in lines if x.strip())


def norm_ws(lines: list[str]) -> Counter:
    """Line comparison that ignores internal whitespace runs.

    Some patches in the source were rendered against a slightly different
    revision than `base_commit`, so a cosmetic spacing delta would otherwise
    invalidate an entire case.
    """
    return Counter(re.sub(r"\s+", " ", x.strip()) for x in lines if x.strip())


def build_state(case: dict, rtl_files: list[dict], redactions: list[str]) -> dict:
    issue = case.get("issue_info") or {}
    labels = [l for l in (issue.get("labels") or []) if not POST_REPAIR_LABEL.match(l)]
    dropped = sorted(set(issue.get("labels") or []) - set(labels))
    if dropped:
        redactions.append(f"dropped post-repair issue labels: {dropped}")
    redactions.extend([
        "excluded pr_info.title/body (describe the completed fix)",
        "excluded patches, head_commit and corrected RTL",
        "excluded lec_status, source_info.verified and additions/deletions (post-repair)",
    ])
    source_info = case.get("source_info") or {}
    return {
        "repository": case["repository"],
        "task_id": case["task_id"],
        "base_commit": case["base_commit"],
        "base_ref": case.get("base_ref"),
        "issue_title": clip(issue.get("title") or "", 1200)[0],
        "issue_body": clip(issue.get("body") or "", 6000)[0],
        "issue_labels": labels,
        "affected_files": list(case.get("verilog_files") or []),
        "source_provenance": {
            "source_file": source_info.get("source_file"),
            "line_number": source_info.get("line_number"),
        },
        "rtl_context": rtl_files,
        "failure_context": {
            "artefact": "repository bug-fix task on the base revision",
            "reported_symptom": clip(issue.get("title") or "", 400)[0],
            "no_tool_log_available": True,
        },
        "previous_actions": [],
    }


def action_distribution(case: dict) -> tuple[dict[str, float], dict[str, list[str]], int]:
    issue = case.get("issue_info") or {}
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}"
    labels = "\n".join(issue.get("labels") or [])
    weights: dict[str, float] = defaultdict(float)
    traces: dict[str, list[str]] = {"patch": [], "issue": [], "label": []}
    label_hits = 0
    for action, channel, pattern, weight, why in ACTION_RULES:
        hay = labels if channel == "label" else text
        if pattern == "." or re.search(pattern, hay, re.I | re.M):
            weights[action] += weight
            traces[channel].append(f"{action}: +{weight} ({why})")
            if channel == "label":
                label_hits += 1
    total = sum(weights.values()) or 1.0
    return {k: round(weights.get(k, 0.0) / total, 4) for k in ACTION_CRITERIA}, traces, label_hits


def root_cause_distribution(case: dict, patch_removed: list[str], patch_added: list[str]
                            ) -> tuple[dict[str, float], dict[str, list[str]], bool, bool]:
    """Return (probabilities, traces, patch_anchored, decisive).

    Evidence channels, strongest first: the verified repair (the semantics of the
    accepted change), the reporter's issue title (their own concise diagnosis),
    and the issue body/labels.
    """
    issue = case.get("issue_info") or {}
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    labels = "\n".join(issue.get("labels") or [])
    patch_text = "\n".join(patch_removed + patch_added)

    scores: dict[str, float] = defaultdict(float)
    traces: dict[str, list[str]] = {"patch": [], "title": [], "issue": []}
    patch_hits: dict[str, float] = {}
    for category, pattern, where, weight, why in ROOT_CAUSE_RULES:
        if where in ("patch", "both") and re.search(pattern, patch_text, re.I):
            scores[category] += weight * PATCH_WEIGHT
            patch_hits[category] = weight
            traces["patch"].append(f"{category}: +{weight} ({why})")
        if re.search(pattern, title, re.I):
            scores[category] += round(weight * TITLE_WEIGHT, 3)
            traces["title"].append(f"{category}: +{round(weight * TITLE_WEIGHT, 3)} ({why})")
        hay = body + "\n" + labels
        if where in ("issue", "both") and re.search(pattern, hay, re.I):
            scores[category] += round(weight * ISSUE_WEIGHT, 3)
            traces["issue"].append(f"{category}: +{round(weight * ISSUE_WEIGHT, 3)} ({why})")

    if not scores:
        probs = {k: 0.0 for k in ROOT_CAUSE_CRITERIA}
        probs["unknown"] = 1.0
        traces["patch"].append("no rule fired on the repair, title or report; "
                               "unknown is the only defensible answer")
        return probs, traces, False, False

    total = sum(scores.values())
    probs = {k: scores.get(k, 0.0) / total for k in ROOT_CAUSE_CRITERIA}
    patch_anchored = bool(patch_hits)
    top = max(probs.items(), key=lambda kv: kv[1])

    if top[1] < CLEAR:
        # Neither the repair nor the report clearly supports one category.
        probs = {k: v * 0.6 for k, v in probs.items()}
        probs["unknown"] = probs.get("unknown", 0.0) + 0.4
        traces["patch"].append("no category reaches 0.40; the source does not clearly "
                               "support one, so unknown leads")
    else:
        reserve = (0.05 if patch_hits.get(top[0], 0.0) >= PATCH_STRONG_WEIGHT
                   else (0.15 if patch_anchored else 0.30))
        probs = {k: v * (1 - reserve) for k, v in probs.items()}
        probs["unknown"] = probs.get("unknown", 0.0) + reserve
        traces["patch"].append(f"reserved {reserve} mass for unknown")

    s = sum(probs.values())
    probs = {k: round(v / s, 4) for k, v in probs.items()}
    dominant = max(probs.items(), key=lambda kv: kv[1])[0]
    decisive = patch_anchored and dominant != "unknown" and max(probs.values()) >= DECISIVE
    return probs, traces, patch_anchored, decisive


def evidence_label(case: dict, state: dict) -> tuple[dict[str, float], str, str]:
    """Is the pre-repair evidence enough to select a likely root cause?"""
    issue = case.get("issue_info") or {}
    body = issue.get("body") or ""
    text = f"{issue.get('title') or ''} {body}"

    visible_identifiers: set[str] = set()
    included_paths = set()
    for f in state["rtl_context"]:
        if f.get("content"):
            included_paths.add(f["path"])
            visible_identifiers.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", f["content"]))
    stop = {"this", "that", "with", "when", "have", "should", "from", "will", "been", "they",
            "there", "which", "would", "could", "what", "then", "than", "some", "only", "does",
            "also", "very", "must", "same", "make", "case", "here", "more", "less"}
    named = sorted(i for i in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text))
                   if i in visible_identifiers and i.lower() not in stop)
    cites_spec = bool(re.search(r"spec|section \d|compliance|risc-v|\bisa\b", text, re.I))

    file_mentions = [m for m in re.findall(r"[\w/]+\.(?:sv|svh|v|vh)\b", text, re.I)
                     if m in included_paths or any(p.endswith(m) for p in included_paths)]
    if file_mentions:
        return ({"false": 0.10, "true": 0.90}, "manual_issue_label",
                f"issue explicitly names affected file {file_mentions[0]}, which is included "
                "at the base revision")
    if EXPLICIT_LOCATION.search(text) and (named or cites_spec):
        return ({"false": 0.18, "true": 0.82}, "manual_issue_label",
                "issue states an explicit RTL location and its wording matches the included RTL")
    if named:
        return ({"false": 0.25, "true": 0.75}, "inferred_from_verified_repair",
                f"issue names {len(named)} symbol(s) that exist in the included base RTL")
    if len(body.strip()) < 200:
        return ({"false": 0.78, "true": 0.22}, "codex_pseudo_unverified",
                "issue body does not localise the defect")
    if cites_spec:
        return ({"false": 0.45, "true": 0.55}, "codex_pseudo_unverified",
                "issue cites a specification but names no symbol present in the included RTL")
    return ({"false": 0.55, "true": 0.45}, "codex_pseudo_unverified",
            "issue reports a symptom without naming a symbol present in the included RTL")


def score_gold(probs: dict[str, float], label_source: str) -> dict:
    label = max(probs.items(), key=lambda kv: kv[1])[0]
    score = sum(int(k) * v for k, v in probs.items())
    return {
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "label": label,
        "score": round(score, 4),
        "confidence": round(max(probs.values()), 4),
        "label_source": label_source,
    }


def questions_3q() -> dict:
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


def questions_5q() -> dict:
    q = questions_3q()
    q["risk"] = {
        "type": "score",
        "instructions": "How risky would the recommended diagnostic or agent action be?",
        "criteria": RISK_CRITERIA,
    }
    q["urgency"] = {
        "type": "score",
        "instructions": "How quickly does this hardware issue require attention?",
        "criteria": URGENCY_CRITERIA,
    }
    return q


def verify_patch(case: dict, cache: RepoCache) -> dict:
    """Objectively check the supplied patch against real git objects.

    Some PRs bundle extra edits, so the head commit may contain more than the
    issue patch. Verification therefore checks that the patch is consistent with
    base/head blobs and is a subset of the real base->head diff, and records
    whether it is an exact match.
    """
    repo, base, head = case["repository"], case["base_commit"], case["head_commit"]
    commits_available = cache.has_commit(repo, base) and cache.has_commit(repo, head)
    per_file, exact, subset, applies = [], True, True, True
    for entry in case.get("patches") or []:
        path = entry.get("filename")
        removed, added, _ = parse_patch(entry.get("patch") or "")
        base_blob = cache.read(repo, base, path)
        head_blob = cache.read(repo, head, path)
        real_removed, real_added = cache.changed_lines(repo, base, head, path)
        in_base = base_blob is not None and all(
            x.strip() in base_blob for x in removed if x.strip())
        in_head = head_blob is not None and all(
            x.strip() in head_blob for x in added if x.strip())
        file_exact = norm(removed) == norm(real_removed) and norm(added) == norm(real_added)
        file_subset = (
            not (norm_ws(removed) - norm_ws(real_removed))
            and not (norm_ws(added) - norm_ws(real_added))
        )
        exact = exact and file_exact
        subset = subset and file_subset
        applies = applies and in_base and in_head
        per_file.append({
            "filename": path,
            "patch_removed": len(removed), "patch_added": len(added),
            "git_removed": len(real_removed), "git_added": len(real_added),
            "exact_match": file_exact, "subset_of_diff": file_subset,
            "removed_lines_in_base": in_base, "added_lines_in_head": in_head,
            "base_blob_available": base_blob is not None,
            "head_blob_available": head_blob is not None,
        })
    return {
        "commits_available": commits_available,
        "patch_applies_to_base_and_head": applies,
        "patch_is_subset_of_git_diff": subset,
        "patch_exactly_equals_git_diff": exact,
        "head_contains_extra_changes": not exact,
        "per_file": per_file,
    }


def split_for(repo: str) -> str:
    for split, repos in SPLIT_GROUPS.items():
        if repo in repos:
            return split
    return "unassigned"


def convert_case(case: dict, cache: RepoCache,
                 rtl_file_limit: int = RTL_FILE_LIMIT,
                 rtl_total_limit: int = RTL_TOTAL_LIMIT) -> tuple[dict, dict]:
    repo, base = case["repository"], case["base_commit"]
    verification = verify_patch(case, cache)
    verified = (verification["commits_available"]
                and verification["patch_applies_to_base_and_head"]
                and verification["patch_is_subset_of_git_diff"])

    redactions: list[str] = []
    rtl_files, missing, truncated = [], [], []
    issue = case.get("issue_info") or {}
    issue_title = issue.get("title") or ""
    issue_body = issue.get("body") or ""
    paths = sorted(case.get("verilog_files") or [])
    blobs = {p: cache.read(repo, base, p) for p in paths}
    for path, text in blobs.items():
        if text is None:
            missing.append(path)
    usable = [p for p in paths if blobs[p] is not None]
    # The per-case character budget is shared, so decide who gets it from
    # pre-repair evidence: files the issue names first, then small files, which
    # can be included whole for very little budget.
    def budget_priority(path: str) -> tuple:
        base_name = path.rsplit("/", 1)[-1]
        stem = base_name.rsplit(".", 1)[0]
        haystacks = (issue_title, issue_body)
        named = any(base_name in h or (len(stem) > 3 and stem in h) for h in haystacks)
        return (0 if named else 1, len(blobs[path] or ""), path)

    budget = rtl_total_limit
    remaining = len(usable)
    entries: dict[str, dict] = {}
    for path in sorted(usable, key=budget_priority):
        text = blobs[path]
        lines = text.splitlines()
        # Hold back a floor for the files still to come, so no file can be left
        # with no context at all, but never cut this file below its equal share.
        reserve = RTL_MIN_FILE_CHARS * max(remaining - 1, 0)
        cap = min(rtl_file_limit, budget)
        if remaining > 1:
            cap = min(cap, max(budget - reserve, budget // remaining))
        remaining -= 1
        if cap <= 0:
            windows: list[tuple[int, int]] = []
        elif len(text) <= cap:
            windows = [(1, len(lines))] if lines else []
        else:
            windows = select_windows(lines, anchor_lines(lines, issue_title, issue_body),
                                     char_budget=cap)
        content = render_windows(text, windows, path)
        # Merging or banner overhead can still overshoot the cap; shrink from the
        # tail (highest line numbers) until the file fits.
        while windows and len(content) > cap:
            start, end = windows[-1]
            if end > start:
                windows[-1] = (start, start + (end - start) // 2)
            else:
                windows.pop()
            content = render_windows(text, windows, path)
        budget -= len(content)
        whole = windows == [(1, len(lines))] if windows else False
        if windows and not whole:
            truncated.append(path)
        entries[path] = {
            "path": path,
            "content": content,
            "mode": "whole" if whole else ("windowed" if windows else "omitted"),
            "truncated": not whole,
            "lines": len(lines),
            "total_chars": len(text),
            "windows": [[a, b] for a, b in windows],
        }
    rtl_files = [entries[p] for p in paths if p in entries]
    if missing:
        redactions.append(f"unavailable at base commit: {missing}")

    state = build_state(case, rtl_files, redactions)
    action_probs, action_traces, label_hits = action_distribution(case)

    patch_removed, patch_added = [], []
    for entry in case.get("patches") or []:
        r, a, _ = parse_patch(entry.get("patch") or "")
        patch_removed += r
        patch_added += a
    root_probs, root_traces, patch_anchored, decisive = root_cause_distribution(
        case, patch_removed, patch_added)
    ev_probs, ev_source, ev_why = evidence_label(case, state)

    dominant_action, action_p = max(action_probs.items(), key=lambda kv: kv[1])
    # A verified repair is evidence about the repair, not automatically a verified
    # decision label. Judgement questions are marked as inferred even when the
    # repair behind them is objectively verified.
    if label_hits and action_p >= 0.4:
        action_source = "manual_issue_label"
    elif verified and dominant_action == "rtl" and action_p >= 0.4:
        action_source = "inferred_from_verified_repair"
    else:
        action_source = "codex_pseudo_unverified"
    if verified and decisive:
        root_source = "inferred_from_verified_repair"
    else:
        root_source = "codex_pseudo_unverified"

    gold3 = {
        "next_action": {"probabilities": action_probs, "label_source": action_source},
        "root_cause_type": {"probabilities": root_probs, "label_source": root_source},
        "evidence_sufficient": {"probabilities": ev_probs, "label_source": ev_source},
    }

    urgency_id, urgency, urgency_source, urgency_why = urgency_rule(case)
    risk_id, risk, risk_source, risk_why = risk_rule(case, action_probs)
    anchored_urgency = urgency_id in URGENCY_ANCHORED
    anchored_risk = risk_id in RISK_ANCHORED

    if anchored_urgency and anchored_risk:
        tier = "5q_verified"
    elif anchored_urgency or anchored_risk or risk_id in RISK_RUBRIC:
        tier = "5q_pseudo"
    else:
        tier = "3q_only"

    all_sources = [action_source, root_source, ev_source]
    body_quotes = issue_body_leak_diagnostic(case, cache)
    trust_flags: list[str] = []
    if body_quotes:
        trust_flags.append("self_answering_issue_body")
    if not verified:
        trust_flags.append("patch_inconsistent_with_git_diff")
    eval_excluded = bool(trust_flags)

    # Audit only: does the delivered context happen to cover the repaired region,
    # and how does that compare with the previous top-anchored slice? The patch
    # decides nothing here except which lines to look for afterwards.
    old_coverage = old_top_slice_coverage(blobs, paths, rtl_file_limit, rtl_total_limit)
    visibility = {}
    for entry in rtl_files:
        ranges = sorted(r for e in case.get("patches") or []
                        if e.get("filename") == entry["path"]
                        for r in patch_region_lines(e.get("patch") or ""))
        if not ranges:
            continue
        blob = cache.read(repo, base, entry["path"]) or ""
        n_lines = blob.count("\n") + 1
        target = {ln for a, b in ranges for ln in range(a, b + 1)}
        windows = [tuple(w) for w in entry.get("windows") or []]
        covered = covered_lines(windows)
        old_covered = old_coverage.get(entry["path"], set())
        first = set(range(ranges[0][0], ranges[0][1] + 1))
        visibility[entry["path"]] = {
            "patched_lines": len(target),
            "in_windowed_context": len(target & covered),
            "in_old_top_slice": len(target & old_covered),
            "hunks": len(ranges),
            "hunks_in_windowed_context": sum(1 for a, b in ranges
                                             if set(range(a, b + 1)) & covered),
            "hunks_in_old_top_slice": sum(1 for a, b in ranges
                                          if set(range(a, b + 1)) & old_covered),
            "hunks_beyond_old_top_slice": sum(1 for a, b in ranges
                                              if not set(range(a, b + 1)) & old_covered),
            "hunks_recovered_by_windowed_context": sum(
                1 for a, b in ranges
                if set(range(a, b + 1)) & covered and not set(range(a, b + 1)) & old_covered),
            "first_hunk_in_windowed_context": bool(first & covered),
            "first_hunk_in_old_top_slice": bool(first & old_covered),
            "file_lines": n_lines,
        }
    patch_audit = {
        "hunks": sum(v["hunks"] for v in visibility.values()),
        "lines": sum(v["patched_lines"] for v in visibility.values()),
        "lines_windowed": sum(v["in_windowed_context"] for v in visibility.values()),
        "lines_old": sum(v["in_old_top_slice"] for v in visibility.values()),
        "hunks_windowed": sum(v["hunks_in_windowed_context"] for v in visibility.values()),
        "hunks_old": sum(v["hunks_in_old_top_slice"] for v in visibility.values()),
        "hunks_beyond_old": sum(v["hunks_beyond_old_top_slice"] for v in visibility.values()),
        "hunks_recovered": sum(v["hunks_recovered_by_windowed_context"] for v in visibility.values()),
        "patched_files": len(visibility),
        "patched_files_windowed": sum(1 for v in visibility.values()
                                      if v["in_windowed_context"] == v["patched_lines"]),
        "patched_files_old": sum(1 for v in visibility.values()
                                 if v["in_old_top_slice"] == v["patched_lines"]),
        "patched_files_first_hunk_visible": sum(1 for v in visibility.values()
                                                     if v["first_hunk_in_windowed_context"]),
        "patched_files_first_hunk_visible_old": sum(1 for v in visibility.values()
                                                         if v["first_hunk_in_old_top_slice"]),
    }

    record = {
        "id": f"rtlbenchls:{case['task_id']}",
        "source": SOURCE,
        "source_group": f"rtlbenchls:{repo}",
        "state": json.dumps(state, ensure_ascii=False, sort_keys=True),
        "questions": json.dumps(questions_3q(), ensure_ascii=False, sort_keys=True),
        "gold": json.dumps(gold3, ensure_ascii=False, sort_keys=True),
        "outcome": {
            "resolved": True,
            "verification": ("merged_pr_patch_verified_against_base_and_head" if verified
                             else "merged_pr_patch_not_reproduced"),
            "patch_verification": verification,
            "base_commit": base,
            "head_commit": case["head_commit"],
            "patched_files": sorted(f for f in {e.get('filename') for e in case['patches']} if f),
            "additions": case.get("additions"),
            "deletions": case.get("deletions"),
            "lec_status": case.get("lec_status"),
            "pr_number": case.get("pr_number"),
            "pr_url": case.get("pr_url"),
            "issue_number": case.get("issue_number"),
            "issue_url": case.get("issue_url"),
            "merged_at": (case.get("pr_info") or {}).get("merged_at"),
        },
        "provenance": {
            "license": LICENSE,
            "synthetic": False,
            "quality": ("gold" if (verified and decisive and not any(
                s == "codex_pseudo_unverified" for s in all_sources)) else "silver_verified_repair"),
            "label_source": (all_sources[0] if len(set(all_sources)) == 1 else "mixed"),
            "repair_evidence_verified": verified,
            "repair_evidence": "verified_repair_evidence" if verified else "unverified_patch",
            "trust_flags": trust_flags,
            "eval_excluded": eval_excluded,
            "eval_exclusion_reason": ("; ".join(trust_flags) if trust_flags else None),
            "bug_family": max(root_probs.items(), key=lambda kv: kv[1])[0],
            "benchmark": "RTL-BenchLS Task 3",
            "task_id": case["task_id"],
            "split": split_for(repo),
            "label_rationale": {
                "next_action": (
                    f"dominant '{dominant_action}' (p={action_p}); the merged fix edits RTL at the "
                    "base commit, which corroborates the RTL direction rather than being its only basis"
                ),
                "root_cause_type": (root_traces["patch"] + root_traces["title"]
                                    + root_traces["issue"])[-1],
                "evidence_sufficient": ev_why,
                "risk": risk_why,
                "urgency": urgency_why,
            },
            "rule_traces": {
                "next_action": action_traces,
                "root_cause_type": root_traces,
                "risk_rule": risk_id,
                "urgency_rule": urgency_id,
            },
            "state_redactions": redactions,
            "patch_region_visibility": visibility,
            "score_tier": tier,
        },
    }

    if tier != "3q_only":
        gold5 = dict(gold3)
        gold5["risk"] = score_gold(risk, risk_source if tier == "5q_verified" else "codex_pseudo_unverified")
        gold5["urgency"] = score_gold(urgency, urgency_source if tier == "5q_verified" else "codex_pseudo_unverified")
        record["questions"] = json.dumps(questions_5q(), ensure_ascii=False, sort_keys=True)
        record["gold"] = json.dumps(gold5, ensure_ascii=False, sort_keys=True)
        if tier == "5q_pseudo":
            record["provenance"]["pseudo_label_status"] = "codex_pseudo_unverified"
            record["provenance"]["pseudo_questions"] = ["risk", "urgency"]
            record["provenance"]["unverified_reason"] = (
                "risk and urgency are transparent rubric estimates; not source-anchored "
                "or tool-verified"
            )

    audit = {
        "task_id": case["task_id"],
        "repository": repo,
        "split": split_for(repo),
        "missing_base_files": missing,
        "truncated_files": truncated,
        "windowed_files": sum(1 for f in rtl_files if f.get("mode") == "windowed"),
        "omitted_files": [f["path"] for f in rtl_files if f.get("mode") == "omitted"],
        "whole_files": sum(1 for f in rtl_files if f.get("mode") == "whole"),
        "verified": verified,
        "patch_exact": verification["patch_exactly_equals_git_diff"],
        "patch_subset": verification["patch_is_subset_of_git_diff"],
        "head_extra_changes": verification["head_contains_extra_changes"],
        "patch_anchored": patch_anchored,
        "root_decisive": decisive,
        "urgency_rule": urgency_id,
        "risk_rule": risk_id,
        "score_tier": tier,
        "action_source": action_source,
        "root_source": root_source,
        "evidence_source": ev_source,
        "trust_flags": trust_flags,
        "eval_excluded": eval_excluded,
        "patch_visibility": visibility,
        "patch_audit": patch_audit,
    }
    return record, audit


def leakage_check(record: dict, case: dict, cache: RepoCache) -> list[str]:
    """Hard check: the state may carry only base-revision, pre-repair content."""
    issues = []
    state = record["state"]
    if case["head_commit"] in state:
        issues.append("head_commit appears in state")
    pr = case.get("pr_info") or {}
    for field in ("title", "body"):
        text = (pr.get(field) or "").strip()
        if len(text) >= 40 and text[:40] in state:
            issues.append(f"pr_info.{field} text appears in state")
    for label in (case.get("issue_info") or {}).get("labels") or []:
        if label.startswith("Status:") and label in state:
            issues.append(f"post-repair label {label} appears in state")
    if case.get("lec_status") and str(case["lec_status"]) in state:
        issues.append("lec_status appears in state")
    if (case.get("source_info") or {}).get("verified") and '"verified"' in state:
        issues.append("source_info.verified appears in state")
    for entry in json.loads(state)["rtl_context"]:
        if not entry.get("content"):
            continue
        blob = cache.read(case["repository"], case["base_commit"], entry["path"])
        if blob is None:
            issues.append(f"cannot re-read {entry['path']} at base commit")
            continue
        windows = [tuple(w) for w in entry.get("windows") or []]
        if not windows:
            issues.append(f"{entry['path']}: no window ranges recorded")
            continue
        n_lines = blob.count("\n") + 1
        for start, end in windows:
            if not (1 <= start <= end <= n_lines + 1):
                issues.append(f"{entry['path']}: window {start}-{end} outside the base blob")
        expected = render_windows(blob, windows, entry["path"])
        if expected != entry["content"]:
            issues.append(f"{entry['path']}: state text is not the rendered base-revision window")
    return issues


def issue_body_leak_diagnostic(case: dict, cache: RepoCache) -> list[str]:
    """Diagnostic only: does the issue body already quote the repair?

    Such records are still valid - the issue text is genuine pre-repair evidence -
    but downstream training/eval may want to filter them.
    """
    body = ((case.get("issue_info") or {}).get("body") or "").strip()
    if not body:
        return []
    trivial = re.compile(r"^[\s\)\};,]*(end|begin|else|endif|\}|\{)?[\s\)\};,]*$")
    hits = []
    for entry in case.get("patches") or []:
        _, added, _ = parse_patch(entry.get("patch") or "")
        for line in added:
            stripped = line.strip()
            if len(stripped) < 12 or trivial.match(stripped):
                continue
            if stripped in body:
                hits.append(stripped[:88])
    return hits


def validate(record: dict, expect_five: bool) -> list[str]:
    errors = []
    for field in ("id", "source", "source_group", "state", "questions", "gold",
                  "outcome", "provenance"):
        if field not in record:
            errors.append(f"missing {field}")
    try:
        state = json.loads(record["state"])
        qs = json.loads(record["questions"])
        gold = json.loads(record["gold"])
    except (TypeError, json.JSONDecodeError) as exc:
        return errors + [f"unparsable json string: {exc}"]
    if not state:
        errors.append("empty state")
    for banned in ("patches", "head_commit", "additions", "deletions", "lec_status"):
        if banned in state:
            errors.append(f"state contains banned key {banned}")
    if set(qs) != set(gold):
        errors.append("question/gold ids differ")
    if expect_five and set(qs) != set(BASE_QUESTION_IDS) | {"risk", "urgency"}:
        errors.append("5q record does not have exactly the five questions")
    if not expect_five and set(qs) != set(BASE_QUESTION_IDS):
        errors.append("3q record does not have exactly the three questions")
    for qid, q in qs.items():
        if q.get("type") not in {"choice", "noul", "score"}:
            errors.append(f"{qid}: bad type")
        probs = (gold.get(qid) or {}).get("probabilities") or {}
        if not probs:
            errors.append(f"{qid}: no probabilities")
            continue
        if any(v < 0 for v in probs.values()) or abs(sum(probs.values()) - 1.0) > 0.02:
            errors.append(f"{qid}: probabilities do not sum to 1")
        if q["type"] in {"choice", "noul"} and set(probs) != set(q.get("criteria") or {}):
            errors.append(f"{qid}: probability keys do not match criteria")
        if q["type"] == "score" and set(probs) != {str(i) for i in range(4)}:
            errors.append(f"{qid}: score keys must be 0..3")
        if not (gold.get(qid) or {}).get("label_source"):
            errors.append(f"{qid}: missing label_source")
    if record["provenance"].get("synthetic") is None:
        errors.append("provenance.synthetic missing")
    if not record["provenance"].get("split"):
        errors.append("provenance.split missing")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--rtl-file-limit", type=int, default=RTL_FILE_LIMIT)
    ap.add_argument("--rtl-total-limit", type=int, default=RTL_TOTAL_LIMIT)
    args = ap.parse_args()

    cases = json.loads((args.dataset_root / "data/repo_issue_108_cases.json").read_text())["cases"]
    cache = RepoCache(args.dataset_root / "repo_cache", allow_network=not args.no_network)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    records, audits, invalid, leaky, skipped, body_leaks = [], [], [], [], [], []
    for case in cases:
        try:
            record, audit = convert_case(case, cache, args.rtl_file_limit, args.rtl_total_limit)
        except Exception as exc:  # noqa: BLE001 - record and continue
            skipped.append({"task_id": case.get("task_id"), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        qs = json.loads(record["questions"])
        errs = validate(record, expect_five=("risk" in qs))
        if errs:
            invalid.append({"id": record["id"], "errors": errs})
            continue
        leaks = leakage_check(record, case, cache)
        if leaks:
            leaky.append({"id": record["id"], "issues": leaks})
            continue
        hits = issue_body_leak_diagnostic(case, cache)
        if hits:
            body_leaks.append({"id": record["id"], "proposed_fix_lines_in_issue_body": hits})
        records.append(record)
        audits.append(audit)

    def tier_of(r: dict) -> str:
        return r["provenance"]["score_tier"]

    weak3 = [r for r in records
             if any(json.loads(r["gold"])[q]["label_source"] == "codex_pseudo_unverified"
                    for q in BASE_QUESTION_IDS)]
    strong3 = [r for r in records if r not in weak3]
    q5 = [r for r in records if tier_of(r) == "5q_verified"]
    q5_pseudo = [r for r in records if tier_of(r) == "5q_pseudo"]
    q3_only = [r for r in records if tier_of(r) == "3q_only"]

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def project_3q(record: dict) -> dict:
        """The three-question projection of a record (drops the score questions)."""
        trimmed = dict(record)
        qs = json.loads(record["questions"])
        gold = json.loads(record["gold"])
        trimmed["questions"] = json.dumps(
            {q: qs[q] for q in BASE_QUESTION_IDS}, ensure_ascii=False, sort_keys=True)
        trimmed["gold"] = json.dumps(
            {q: gold[q] for q in BASE_QUESTION_IDS}, ensure_ascii=False, sort_keys=True)
        return trimmed

    write(out / "records_3q.jsonl", [project_3q(r) for r in strong3])
    write(out / "records_3q_unverified.jsonl", [project_3q(r) for r in weak3])
    write(out / "records_5q.jsonl", q5)
    write(out / "records_5q_pseudo_unverified.jsonl", q5_pseudo)

    # Independent post-write audit of what actually landed on disk.
    expected_questions = {
        "records_3q.jsonl": 3,
        "records_3q_unverified.jsonl": 3,
        "records_5q.jsonl": 5,
        "records_5q_pseudo_unverified.jsonl": 5,
    }
    post_audit = {}
    for name, n_questions in expected_questions.items():
        path = out / name
        ids, problems = [], []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(row["id"])
            qs = json.loads(row["questions"])
            gold = json.loads(row["gold"])
            if len(qs) != n_questions:
                problems.append(f"{row['id']}: {len(qs)} questions")
            if set(qs) != set(gold):
                problems.append(f"{row['id']}: question/gold mismatch")
            if row["source"] != SOURCE or not row["source_group"].startswith("rtlbenchls:"):
                problems.append(f"{row['id']}: bad source/source_group")
            if not row["provenance"].get("label_source"):
                problems.append(f"{row['id']}: no label_source")
        post_audit[name] = {
            "records": len(ids),
            "unique_ids": len(set(ids)),
            "questions_per_record": n_questions,
            "problems": problems[:20],
        }

    def tally(rows: list[dict], qids=BASE_QUESTION_IDS) -> dict:
        def probs(r, q):
            return json.loads(r["gold"])[q]["probabilities"]
        return {
            "repositories": dict(sorted(Counter(r["source_group"] for r in rows).items())),
            "splits": dict(sorted(Counter(r["provenance"]["split"] for r in rows).items())),
            "root_cause": dict(sorted(Counter(
                max(probs(r, "root_cause_type").items(), key=lambda kv: kv[1])[0]
                for r in rows).items())),
            "next_action": dict(sorted(Counter(
                max(probs(r, "next_action").items(), key=lambda kv: kv[1])[0]
                for r in rows).items())),
            "evidence_sufficient": dict(sorted(Counter(
                max(probs(r, "evidence_sufficient").items(), key=lambda kv: kv[1])[0]
                for r in rows).items())),
            "label_source": dict(sorted(Counter(
                json.loads(r["gold"])[q]["label_source"] for r in rows for q in qids).items())),
        }

    def label_counts(rows: list[dict], qids=BASE_QUESTION_IDS) -> dict:
        return dict(sorted(Counter(
            json.loads(r["gold"])[q]["label_source"] for r in rows for q in qids).items()))

    # Usage sets. "Trusted" requires objectively verified repair evidence, no
    # evaluation exclusion, and no pseudo decision label.
    trusted_3q = [r for r in strong3 if not r["provenance"]["eval_excluded"]]
    excluded_trusted = [r for r in strong3 if r["provenance"]["eval_excluded"]]
    trusted_3q_ids = {r["id"] for r in trusted_3q}
    trusted_5q = [r for r in q5 if r["id"] in trusted_3q_ids]
    flagged = [r for r in records if r["provenance"]["eval_excluded"]]
    membership = {}
    for name, rows in (("records_3q.jsonl", strong3), ("records_3q_unverified.jsonl", weak3),
                       ("records_5q.jsonl", q5), ("records_5q_pseudo_unverified.jsonl", q5_pseudo)):
        for r in rows:
            membership.setdefault(r["id"], []).append(name)
    usage_sets = {
        "trusted_exploratory_3q": sorted(trusted_3q_ids),
        "trusted_exploratory_5q": sorted(r["id"] for r in trusted_5q),
        "exploratory_unverified_3q": sorted(r["id"] for r in weak3),
        "exploratory_pseudo_5q": sorted(r["id"] for r in q5_pseudo),
        "eval_excluded": sorted(r["id"] for r in flagged),
        "eval_excluded_reasons": {
            r["id"]: r["provenance"]["eval_exclusion_reason"] for r in flagged
        },
        "eval_excluded_files": {r["id"]: sorted(membership.get(r["id"], [])) for r in flagged},
        "eval_excluded_from_trusted_3q": sorted(r["id"] for r in excluded_trusted),
        "policy": (
            "trusted = records_3q.jsonl minus every eval_excluded case; trusted 5q is "
            "records_5q.jsonl intersected with the trusted 3q ids. Cases flagged "
            "self_answering_issue_body or patch_inconsistent_with_git_diff stay in the "
            "output files marked eval_excluded=true but must not be used for training or "
            "evaluation; they are listed in eval_excluded regardless of which file holds "
            "them. 'Trusted' means the repair evidence is objectively verified and no "
            "label is a full pseudo guess, not that the judgement labels are correct."
        ),
    }
    (out / "usage_sets.json").write_text(json.dumps(usage_sets, indent=2))

    # Windowed context: does the delivered context cover the repaired region?
    vis = [v for a in audits for v in a["patch_visibility"].values()]
    pa = [a["patch_audit"] for a in audits if a["patch_audit"]["hunks"]]
    covered_new = sum(v["in_windowed_context"] for v in vis)
    covered_old = sum(v["in_old_top_slice"] for v in vis)
    total_patched = sum(v["patched_lines"] for v in vis)
    hunks_new = sum(v["hunks_in_windowed_context"] for v in vis)
    hunks_old = sum(v["hunks_in_old_top_slice"] for v in vis)
    hunks_beyond_old = sum(v["hunks_beyond_old_top_slice"] for v in vis)
    hunks_recovered = sum(v["hunks_recovered_by_windowed_context"] for v in vis)
    total_hunks = sum(v["hunks"] for v in vis)
    report = {
        "source": SOURCE,
        "source_file": str((args.dataset_root / "data/repo_issue_108_cases.json").relative_to(REPO_ROOT)),
        "repo_cache": str((args.dataset_root / "repo_cache").relative_to(REPO_ROOT)),
        "generated_by": "training/convert_rtl_benchls.py",
        "rtl_context_budget": {"per_file_chars": args.rtl_file_limit,
                               "per_case_chars": args.rtl_total_limit},
        "total_source_cases": len(cases),
        "converted_total": len(records),
        "converted_3q_verified": len(strong3),
        "converted_3q_unverified": len(weak3),
        "converted_5q_verified": len(q5),
        "converted_5q_pseudo_unverified": len(q5_pseudo),
        "three_question_only": len(q3_only),
        "skipped": skipped,
        "invalid": invalid,
        "possible_state_leakage": leaky,
        "missing_base_files": {a["task_id"]: a["missing_base_files"]
                               for a in audits if a["missing_base_files"]},
        "checkout_failures": cache.checkout_failures[:40],
        "truncated_state_cases": sum(1 for a in audits if a["truncated_files"]),
        "windowed_file_count": sum(a["windowed_files"] for a in audits),
        "whole_file_count": sum(a["whole_files"] for a in audits),
        "omitted_file_count": sum(len(a["omitted_files"]) for a in audits),
        "omitted_files": {a["task_id"]: a["omitted_files"] for a in audits if a["omitted_files"]},
        "context_window_coverage": {
            "patched_lines_total": total_patched,
            "covered_by_windowed_context": covered_new,
            "covered_by_old_top_anchored_slice": covered_old,
            "windowed_coverage_percent": round(100.0 * covered_new / max(total_patched, 1), 2),
            "old_coverage_percent": round(100.0 * covered_old / max(total_patched, 1), 2),
            "repair_hunks_total": total_hunks,
            "hunks_in_windowed_context": hunks_new,
            "hunks_in_old_top_anchored_slice": hunks_old,
            "hunks_beyond_old_top_anchored_slice": hunks_beyond_old,
            "hunks_old_slice_cannot_show": hunks_beyond_old - hunks_recovered,
            "hunks_recovered_by_windowed_context": hunks_recovered,
            "hunk_coverage_percent": round(100.0 * hunks_new / max(total_hunks, 1), 2),
            "old_hunk_coverage_percent": round(100.0 * hunks_old / max(total_hunks, 1), 2),
            "patched_files_total": sum(a["patched_files"] for a in pa),
            "patched_files_with_first_hunk_visible": sum(a["patched_files_first_hunk_visible"] for a in pa),
            "patched_files_with_first_hunk_visible_old": sum(a["patched_files_first_hunk_visible_old"] for a in pa),
            "patched_files_fully_visible": sum(a["patched_files_windowed"] for a in pa),
            "patched_files_fully_visible_old": sum(a["patched_files_old"] for a in pa),
            "cases_audited": len(pa),
            "note": (
                "Audit only. Window placement uses pre-repair evidence only (the top of the "
                "file, an even spread, then identifiers named in the issue); the repaired "
                "region is never used to choose a window, and patch line numbers never enter "
                "the state. A file that fits the budget is included whole. The old-slice "
                "columns reconstruct the replaced top-anchored prefix slice."
            ),
        },
        "usage_sets": {
            "trusted_exploratory_3q": len(trusted_3q),
            "trusted_exploratory_5q": len(trusted_5q),
            "exploratory_unverified_3q": len(weak3),
            "exploratory_pseudo_5q": len(q5_pseudo),
            "eval_excluded": len(flagged),
            "eval_excluded_from_trusted_3q": len(excluded_trusted),
            "file": "usage_sets.json",
        },
        "trust_flags": dict(sorted(Counter(
            flag for a in audits for flag in a["trust_flags"]).items())),
        "eval_excluded_cases": [
            {"task_id": a["task_id"], "flags": a["trust_flags"]}
            for a in audits if a["eval_excluded"]
        ],
        "patch_verified": sum(1 for a in audits if a["verified"]),
        "patch_exact_match": sum(1 for a in audits if a["patch_exact"]),
        "head_contains_extra_changes": [a["task_id"] for a in audits if a["head_extra_changes"]],
        "patch_not_reproduced": [a["task_id"] for a in audits if not a["verified"]],
        "issue_body_quotes_patch_lines": body_leaks,
        "issue_body_quotes_patch_lines_count": len(body_leaks),
        "label_source_counts_3q_questions": label_counts(records),
        "label_source_counts_5q_verified": label_counts(
            q5, BASE_QUESTION_IDS + ("risk", "urgency")),
        "cases_with_pseudo_label_per_question": {
            q: sum(1 for r in records
                   if json.loads(r["gold"])[q]["label_source"] == "codex_pseudo_unverified")
            for q in BASE_QUESTION_IDS
        },
        "pseudo_label_count": sum(
            1 for r in records for v in json.loads(r["gold"]).values()
            if v["label_source"] == "codex_pseudo_unverified"),
        "inferred_label_count": sum(
            1 for r in records for v in json.loads(r["gold"]).values()
            if v["label_source"] == "inferred_from_verified_repair"),
        "manual_issue_label_count": sum(
            1 for r in records for v in json.loads(r["gold"]).values()
            if v["label_source"] == "manual_issue_label"),
        "criteria_applied_count": sum(
            1 for r in records for v in json.loads(r["gold"]).values()
            if v["label_source"] == "criteria_applied_verified_action"),
        "cases_with_any_pseudo_label": len(weak3),
        "cases_fully_verified": len(strong3),
        "five_q_records_with_all_base_labels_verified": sum(
            1 for r in q5
            if all(json.loads(r["gold"])[q]["label_source"] != "codex_pseudo_unverified"
                   for q in BASE_QUESTION_IDS)),
        "five_q_records_with_a_pseudo_base_label": sum(
            1 for r in q5
            if any(json.loads(r["gold"])[q]["label_source"] == "codex_pseudo_unverified"
                   for q in BASE_QUESTION_IDS)),
        "rule_usage": {
            "urgency": dict(sorted(Counter(a["urgency_rule"] for a in audits).items())),
            "risk": dict(sorted(Counter(a["risk_rule"] for a in audits).items())),
        },
        "repository_split": SPLIT_GROUPS,
        "repository_split_counts": {
            split: sum(1 for r in records if r["provenance"]["split"] == split)
            for split in SPLIT_GROUPS
        },
        "split_policy": (
            "repository-disjoint: no repository appears in more than one split; openhwgroup "
            "cores and the pulp-platform cells they instantiate form one lineage; "
            "lowRISC/opentitan is held out entirely as test"
        ),
        "file_inventory": {
            "records_3q.jsonl": len(strong3),
            "records_3q_unverified.jsonl": len(weak3),
            "records_5q.jsonl": len(q5),
            "records_5q_pseudo_unverified.jsonl": len(q5_pseudo),
        },
        "file_layout": (
            "The 3q files carry exactly the three base questions and together cover every "
            "converted case exactly once. The 5q files carry all five questions; the "
            "three-question projection of a 5q record is byte-identical to its 3q counterpart."
        ),
        "post_write_audit": post_audit,
        "tallies_3q_verified": tally(strong3),
        "tallies_3q_unverified": tally(weak3),
        "tallies_5q_verified": tally(q5, BASE_QUESTION_IDS + ("risk", "urgency")),
        "tallies_5q_pseudo": tally(q5_pseudo, BASE_QUESTION_IDS + ("risk", "urgency")),
        "score_question_policy": {
            "verified_tier": (
                "urgency and risk each require explicit issue evidence: a Priority/Milestone/"
                "security/destructive/system-impact statement or schedule label"
            ),
            "pseudo_tier": (
                "transparent rubric estimate (issue impact wording, or published risk criteria "
                "applied to a repair-pinned read-only action); both score labels are marked "
                "codex_pseudo_unverified"
            ),
            "excluded": "records with neither issue-anchored nor rubric risk/urgency evidence keep three questions only",
        },
        "audit_index": [
            {"task_id": a["task_id"], "repository": a["repository"], "split": a["split"],
             "score_tier": a["score_tier"], "urgency_rule": a["urgency_rule"],
             "risk_rule": a["risk_rule"], "verified": a["verified"],
             "patch_exact": a["patch_exact"], "truncated": bool(a["truncated_files"])}
            for a in audits
        ],
    }
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "total_source_cases": report["total_source_cases"],
        "converted_total": report["converted_total"],
        "converted_3q_verified": report["converted_3q_verified"],
        "converted_3q_unverified": report["converted_3q_unverified"],
        "converted_5q_verified": report["converted_5q_verified"],
        "converted_5q_pseudo_unverified": report["converted_5q_pseudo_unverified"],
        "three_question_only": report["three_question_only"],
        "skipped": len(skipped),
        "invalid": len(invalid),
        "possible_state_leakage": len(leaky),
        "missing_base_files": len(report["missing_base_files"]),
        "checkout_failures": len(cache.checkout_failures),
        "patch_verified": report["patch_verified"],
        "issue_body_quotes_patch_lines": len(body_leaks),
        "trusted_exploratory_3q": len(trusted_3q),
        "trusted_exploratory_5q": len(trusted_5q),
        "windowed_coverage_percent": report["context_window_coverage"]["windowed_coverage_percent"],
        "old_coverage_percent": report["context_window_coverage"]["old_coverage_percent"],
        "hunk_coverage_percent": report["context_window_coverage"]["hunk_coverage_percent"],
        "old_hunk_coverage_percent": report["context_window_coverage"]["old_hunk_coverage_percent"],
        "hunks_beyond_old_top_anchored_slice": report["context_window_coverage"]["hunks_beyond_old_top_anchored_slice"],
        "hunks_recovered_by_windowed_context": report["context_window_coverage"]["hunks_recovered_by_windowed_context"],
        "patched_files_with_first_hunk_visible": report["context_window_coverage"]["patched_files_with_first_hunk_visible"],
        "patched_files_with_first_hunk_visible_old": report["context_window_coverage"]["patched_files_with_first_hunk_visible_old"],
        "windowed_file_count": report["windowed_file_count"],
        "whole_file_count": report["whole_file_count"],
        "omitted_file_count": report["omitted_file_count"],
        "label_source_counts_3q_questions": report["label_source_counts_3q_questions"],
        "label_source_counts_5q_verified": report["label_source_counts_5q_verified"],
        "eval_excluded_cases": len(report["eval_excluded_cases"]),
    }, indent=2))


if __name__ == "__main__":
    main()
