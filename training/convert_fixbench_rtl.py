#!/usr/bin/env python3
"""Convert Fixbench-RTL into SilicoJev/Laya-compatible decision records.

Input : `dataset/raw/hf/Fixbench-RTL/Fixbench-RTL.json` (read-only, 100 records:
        `bug`, `buggycode`, `correctcode`, `testbench`)
Replay: `training/fixbench_replay_cache/replay_results.json`, built by
        `training/build_fixbench_replay.py`. Without it every case is marked
        unvalidated and nothing is labelled `verified_repair`.

Evidence discipline (the whole point of this conversion):

- the state carries only pre-repair material: the bug description, the buggy
  RTL, the supplied testbench, and the replay log *of the buggy run*;
- `correctcode` never enters the state, in whole or as a diff. It is used as a
  repair target, as corroboration when deriving a root-cause category, and as
  outcome metadata;
- `source_info`-style shortcuts are not trusted: the corrected replay is
  verification only, and a case is only `verified_repair` when the base revision
  demonstrably fails *and* the corrected revision passes the same replay.

Outputs (dataset/converted/fixbench_rtl/): records_3q.jsonl, records_5q.jsonl,
records_5q_pseudo_unverified.jsonl, usage_sets.json, split_groups.json,
trusted_label_provenance.md, conversion_report.json, README.md.
Records are self-contained; nothing is merged into dataset/normalized/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "dataset/raw/hf/Fixbench-RTL/Fixbench-RTL.json"
DEFAULT_REPLAY = Path(__file__).resolve().parent / "fixbench_replay_cache/replay_results.json"
DEFAULT_OUT = REPO_ROOT / "dataset/converted/fixbench_rtl"

SOURCE = "Fixbench-RTL"
GROUP = "fixbench_rtl"
LICENSE = "CC-BY-4.0 (Fixbench-RTL); upstream source licences apply"
BASE_QUESTION_IDS = ("next_action", "root_cause_type", "evidence_sufficient")

# Reused verbatim from training/prepare_dataset.py so the corpus shares one
# taxonomy.
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
EVIDENCE_CRITERIA = {
    "false": "Evidence is insufficient or unvalidated",
    "true": "Evidence is sufficient and supported by a tool or trusted label",
}
# The score rubric is copied verbatim from the existing SilicoJev training set
# (training/review_pseudo_scores.py), so a Fixbench record's risk/urgency
# question is the same question the merged 6,248-case set asks.
RISK_CRITERIA = [
    "Benign: read-only inspection, parsing, linting, or clearly reversible analysis.",
    "Low: routine simulation, formal checking, or isolated writes within the project scope.",
    "Moderate: modifying RTL, testbench, constraints, or configuration in a way that could affect results.",
    "High: destructive, irreversible, security-sensitive, out-of-scope, or potentially damaging action.",
]
URGENCY_CRITERIA = [
    "No time pressure: the issue can wait indefinitely.",
    "Routine: handle during the normal development queue.",
    "Elevated: should be investigated within the same week.",
    "Critical: blocks an important flow, threatens tapeout/release/security, or requires action within the same day.",
]

ACTION_INSTRUCTIONS = "What diagnostic direction should be investigated next?"
ROOT_INSTRUCTIONS = "Which root-cause category best describes the available evidence?"
EVIDENCE_INSTRUCTIONS = "Is there enough evidence to choose a likely root cause?"
RISK_INSTRUCTIONS = ("How risky is the recommended next action in this repository? "
                     "Choose one of the four ordered levels.")
URGENCY_INSTRUCTIONS = ("How urgent is this case? Choose one of the four ordered levels.")

# Thresholds. One-hot is reserved for evidence that names exactly one category.
ONE_HOT_MIN = 0.80
ONE_HOT_MARGIN = 0.35
DECISIVE_MIN = 0.55          # below this the record keeps mass on `unknown`
UNKNOWN_RESERVE = {"high": 0.03, "medium": 0.12, "low": 0.30}

STATE_LOG_LIMIT = 4000
BUG_LIMIT = 4000
RTL_LIMIT = 60000

# --------------------------------------------------------------------------
# Root-cause signatures, strongest evidence first.
#
# `log` rules match the replay log of the *buggy* revision (pre-repair, tool
# produced). `bug` rules match the human-written description. `fix` rules match
# the corrected-code diff and are corroboration only: a diff shows what changed,
# not what class of defect the change belongs to, so a `fix`-only match never
# reaches the verified label source.
ROOT_RULES = [
    # (category, channel, weight, regex, why)
    #
    # Two channels only: the buggy replay log (tool-produced, pre-repair) and the
    # human description. The corrected-code channel is deliberately absent. A diff
    # shows what changed, not what class of defect the change belongs to, and the
    # patterns that could be written against it (an operator changed, a reset is
    # mentioned) fire on nearly every case, adding a constant offset that only
    # flattens the distribution. `correctcode` is still used elsewhere: to check
    # that no repaired content reached the state, and to record the outcome.
    ("syntax_compile", "log", 3.0, r"error-\d|syntax error|malformed|invalid module|"
     r"not declared within module|unexpected|near .*: syntax|Invalid module item|"
     r"Superfluous comma|must have a value|must have a default value",
     "replay log reports a syntax/elaboration failure"),
    ("syntax_compile", "bug", 2.5, r"syntax|compile|compilation|elaborat|missing semicolon|"
     r"typo|parse|not supported|multiple drivers|invalid for a single",
     "description states a syntax/compile problem"),
    ("type_width", "log", 2.5, r"not a valid l-value|width|signed|unsigned|"
     r"requires an explicit cast|part-select|bit-select|"
     r"cannot be driven by primitives|is not a port of|not a port of",
     "replay log reports a type/width/l-value/port failure"),
    ("type_width", "bug", 2.5, r"\bwidth\b|bit width|signedness|unsigned|\bcast\b|"
     r"l-value|port list|\bdirection\b|declared as wire|part-select|bit-select",
     "description states a type/width/port mismatch"),
    ("reset_initialization", "bug", 2.5, r"\breset\b|initiali[sz]|unknown state|"
     r"\bx\b state|power-?on|\bpor\b|flush|not initialized|uninitialised",
     "description states a reset/initialisation problem"),
    ("timing_protocol", "bug", 2.5, r"handshake|protocol|deadlock|credit|acknowledg|"
     r"clock domain|\bcdc\b|race|glitch|latency|stall|timeout|wait-state|"
     r"backpressure|ready/valid|flow control|out-of-order|ordering",
     "description states a protocol/timing/synchronisation problem"),
    ("timing_protocol", "log", 1.5, r"deadlock|timeout|never (?:reached|finish)|hang",
     "replay log reports a hang/deadlock"),
    ("state_machine", "bug", 2.5, r"state machine|\bfsm\b|state encoding|transition|"
     r"trap|unreachable|dead code|control-?flow|\bdecode\b|one-?hot|gray code",
     "description states a state-machine/control-flow problem"),
    ("sequential_assignment", "bug", 2.5, r"blocking|non-?blocking|register|pipeline|"
     r"\bff\b|flip-?flop|sequential|latch|stage|clocked|posedge|negedge|"
     r"same cycle|different cycle|always block|always_ff|always_comb",
     "description states a clocked-assignment/register problem"),
    ("sequential_assignment", "log", 2.0, r"not a valid l-value|cannot be driven|"
     r"is declared here as wire|driven in an always_ff",
     "replay log reports an illegal procedural assignment"),
    ("combinational_logic", "bug", 1.8, r"combinational|arithmetic|carry|adder|subtract|"
     r"Boolean|\blogic\b|mux|multiplex|compar|equality|\bsum\b|product|shift|"
     r"operator|priority|encode|parity|expression|truth table",
     "description states a logic/arithmetic defect"),
    ("formal_property", "bug", 2.0, r"assertion|property|\bsva\b|invariant|formal|"
     r"counterexample|cover property|specification requires",
     "description states an assertion/property issue"),
    ("formal_property", "log", 2.5, r"assertion|property_spec|\bsva\b|invariant",
     "replay log reports an assertion failure"),
    ("security", "bug", 2.5, r"security|vulnerab|fault[- ]injection|\bfi\b|adversary|"
     r"tamper|hamming distance|ledger|attack|side[- ]channel|redundanc|shadow|"
     r"integrity|unprotected state|protective mask|glitch",
     "description states a security/integrity concern"),
]

# Action rules. The buggy replay log is pre-repair evidence about where to look.
ACTION_RULES = [
    ("rtl", "log", 3.0, r"error-\d|syntax error|malformed|invalid module|"
     r"not declared within module|Invalid module item|not a valid l-value|"
     r"cannot be driven|not supported",
     "the failure is reported against the design source itself"),
    # The description is the agent's brief and usually names the faulty construct;
    # the log establishes only that behaviour was wrong, not where. When both speak
    # the resulting distribution stays soft rather than naming one direction, which
    # is the honest shape for "look at the RTL or look at the waveform".
    #
    # A description counts as naming a construct when it carries an identifier
    # (`end_cnt`, `mul_en_out_reg[1]`) or a design noun. Fixbench's bug text is an
    # explicit defect description, so this fires on most cases by design; the
    # label never comes from the patch, which is what the "do not just label it
    # rtl" caution is about.
    ("rtl", "bug", 1.6, r"`[A-Za-z_]\w*`|\b[a-z]\w*_[a-z]\w*\b|"
     r"\bmodule\b|\bsignal\b|\bregister\b|\brtl\b|\blogic\b|"
     r"\bassignment\b|\bimplementation\b|\bstatement\b|\bbranch\b|"
     r"\bport\b|\bexpression\b|\binstantiat|\bwire\b|\bassign|\balways\b|"
     r"\boutput\b|\binput\b|\bcounter\b|\baccumulat|\bcompar|\bencode\b|"
     r"\bdecoder\b|\bmux\b|\bshift\b|\badder\b|\bpointer\b|\bflop\b|"
     r"\bvalue\b|\bdriven\b|\bassert",
     "the description names design internals"),
    # A bare `Failed` marker says the testbench objected, not that a waveform was
    # examined: that mis-fire produced confident `waveform` labels on cases whose
    # description named the defective construct outright. Require actual values or
    # times in the transcript before claiming cycle-level evidence.
    ("waveform", "log", 1.5, r"expected\s*[=:]|got\s*[=:]|\bmismatch\b|"
     r"\bHD=|\bFIF=|\bvs\b|at cycle|cycle \d|@\s*\d+\s*(?:ns|ps|us)|"
     r"Time\s*\d+\s*:", "the transcript carries cycle-level values or times"),
    ("waveform", "bug", 1.4, r"\bcycle\b|\btiming\b|\blatency\b|\bposedge\b|"
     r"\bnegedge\b|\bclock\b|\bwaveform\b|same cycle|different cycle|"
     r"\bdeadlock\b|\brace\b|\blivelock\b|half-period|one cycle",
     "the description is about cycle-level behaviour"),
    ("simulation", "log", 1.0, r"deadlock|never (?:reached|finish)|timeout|"
     r"ERROR|Fatal|\$fatal|\$finish called|===+\s*Failed",
     "the transcript reports a failure without localising it, so a targeted "
     "re-run or a different simulator configuration is warranted"),
    ("simulation", "bug", 0.8, r"simulat|stimulus|test ?bench|reproduc|watchdog",
     "the description points at the simulation setup"),
    ("testbench", "bug", 1.4, r"test ?bench|\btb\b|checker|scoreboard|monitor|"
     r"coverage|sequence|stimulus|expected value|self-?check",
     "the description involves the verification code"),
    ("testbench", "log", 1.2, r"Unknown module type|not a task name|"
     r"Could not find the package|Class or package .* not found|Unresolved|"
     r"is not a port of|not found in `|port.*not found",
     "the testbench does not fit the design it is meant to drive"),
    ("formal", "bug", 1.0, r"assertion|\bproperty\b|\bsva\b|formal|invariant|"
     r"counterexample", "assertion/property reasoning is indicated"),
    ("formal", "log", 0.6, r"assertion|property|sva",
     "the failure surfaced as an assertion"),
    ("specification", "bug", 1.2, r"specification|\bspec\b|requirement|"
     r"standard|per the|according to the|should be|expected to|documented|"
     r"reference model", "the description cites a specification"),
    ("constraints", "bug", 1.0, r"synthesis|constraint|\bxdc\b|\bsdc\b|timing constraint|"
     r"vivado|quartus|place|route|fpga", "physical/synthesis constraints are implicated"),
    ("constraints", "log", 0.5, r"synthesis|constraint|unconstrained",
     "synthesis/tool configuration is implicated"),
]

SECURITY_CONTENT = re.compile(
    r"security|vulnerab|fault[- ]injection|\bfi\b|adversary|tamper|"
    r"hamming distance|attack|side[- ]channel|crypto|cipher", re.I)
# Boilerplate that must never drive a label. The license banner is genuinely part
# of the Questa transcript ("...information that is the property of..."), and
# "Signed-off-by:" makes the description look like it mentions signedness. Both
# were observed firing rules before these filters existed.
LEGAL_BOILERPLATE = re.compile(
    r"secrets and commercial or financial information|licensed under|"
    r"spdx-license|all rights reserved|copyright|solderpad hardware licence|"
    r"this file is part of|\babnf\b", re.I)
SIGNOFF_BOILERPLATE = re.compile(
    r"^\s*(?:signed-off-by|co-authored-by|acked-by|reviewed-by|tested-by|"
    r"reported-by|suggested-by|cc)\s*:|^\s*-{3,}\s*$|^\s*diff --git", re.I)


def rule_text(text: str, channel: str) -> str:
    """Text as the label rules should see it, with boilerplate removed.

    Only the rule input is cleaned: the model state keeps the description and the
    verbatim tool output, since that is what an agent would actually be given.
    """
    if not text:
        return ""
    boilerplate = SIGNOFF_BOILERPLATE if channel == "bug" else LEGAL_BOILERPLATE
    kept = [line for line in text.splitlines() if not boilerplate.search(line)]
    return "\n".join(kept)
PR_TEXT = re.compile(r"this pr|this patch|signed-off-by|fixes #|revert\b|\bpr #\d+", re.I)


def clip(text: str, limit: int) -> str:
    text = (text or "").replace("\x00", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def stable_case_id(index: int, case: dict) -> str:
    """A stable id for a record that has no id of its own.

    Fixbench ships no per-case identifier, so the id is the source index plus a
    content hash. The hash makes it verifiable that the id refers to those exact
    four fields, and the index keeps it readable.
    """
    blob = json.dumps({k: case.get(k) for k in ("bug", "buggycode", "correctcode", "testbench")},
                      sort_keys=True, ensure_ascii=False).encode()
    return f"case_{index:03d}_{hashlib.sha256(blob).hexdigest()[:8]}"


def channel_text(case: dict, channel: str, replay: dict | None) -> str:
    """Text available to the label rules for one evidence channel.

    `log` is everything the **buggy** side printed, under every simulator that
    ran it. Both simulators' output about the buggy revision is pre-repair
    evidence: it is what an agent would have seen, whether or not the corrected
    revision later confirms the failure. Confirmation is a separate question and
    is handled by the label-source gate, not by hiding evidence.

    `fix` is the corrected code, and exists only so the rules can *corroborate*
    a reading; a `fix`-only match never reaches a verified label source.
    """
    if channel == "bug":
        return rule_text(case.get("bug") or "", "bug")
    if channel == "fix":
        return rule_text(case.get("correctcode") or "", "fix")
    if channel == "log":
        parts = []
        for backend, attempt in sorted((replay or {}).get("buggy_by_backend", {}).items()):
            for key in ("compile", "run"):
                step = attempt.get(key) or {}
                parts += [step.get("stdout") or "", step.get("stderr") or ""]
        return rule_text("\n".join(part for part in parts if part), "log")
    return ""


def replay_state(replay: dict | None) -> dict:
    """The replay facts a label is allowed to lean on, from the buggy side only.

    `verified` is the only field that mentions the corrected revision, and it is
    a boolean about *evidence*, not a repair hint: it says the failure was also
    shown to be removed. Nothing here contains corrected code, a diff, or the
    corrected transcript.
    """
    replay = replay or {}
    buggy_by = replay.get("buggy_by_backend") or {}
    return {
        "class": replay.get("attribution", "not_replayed"),
        "verified": bool(replay.get("failure_attributable_to_bug")),
        "deciding_backend": replay.get("deciding_backend"),
        "verified_backends": list(replay.get("verified_backends") or []),
        "buggy_verdicts": {b: a.get("verdict") for b, a in sorted(buggy_by.items())},
        "testbench_passed_buggy_on": list(replay.get("buggy_passes_on_backends") or []),
        "buggy_verdict": (replay.get("buggy") or {}).get("verdict", "not_replayed"),
    }


def weighted_distribution(weights: dict[str, float], unknown_reserve: float) -> dict[str, float]:
    """Normalise, holding `unknown_reserve` for the unknown category if needed."""
    total = sum(weights.values())
    if total <= 0:
        return {k: (1.0 if k == "unknown" else 0.0) for k in ROOT_CAUSE_CRITERIA}
    probs = {k: weights.get(k, 0.0) / total for k in ROOT_CAUSE_CRITERIA}
    if unknown_reserve > 0:
        remaining = 1.0 - unknown_reserve
        probs = {k: v * remaining for k, v in probs.items()}
        probs["unknown"] = probs.get("unknown", 0.0) + unknown_reserve
    return {k: round(v, 4) for k, v in probs.items()}


def root_cause_distribution(case: dict, replay: dict | None
                            ) -> tuple[dict[str, float], list[str], str]:
    """Return (probabilities, trace, evidence_channel).

    `evidence_channel` names the strongest channel that fired: `log` (the buggy
    replay), `bug` (the human description), `fix` (the corrected diff), or none.
    The `unknown` reserve is raised whenever the buggy-side failure was not shown
    to track the defect: a log that reports a build error caused by a missing
    external package, or by a testbench that does not fit the RTL, does not
    localise the defect.
    """
    weights: dict[str, float] = defaultdict(float)
    trace: list[str] = []
    channels_hit: set[str] = set()
    best_channel = "none"
    state = replay_state(replay)
    for category, channel, weight, pattern, why in ROOT_RULES:
        hay = channel_text(case, channel, replay)
        match = re.search(pattern, hay, re.I | re.M) if hay else None
        if match:
            weights[category] += weight
            channels_hit.add(channel)
            trace.append(f"{category}: +{weight} ({channel}: {why}) "
                         f"[matched {match.group(0)[:40]!r}]")
    if not weights:
        return (weighted_distribution({}, UNKNOWN_RESERVE["low"]),
                ["no rule fired"], best_channel)
    if "log" in channels_hit:
        best_channel = "log"
    elif "bug" in channels_hit:
        best_channel = "bug"
    elif "fix" in channels_hit:
        best_channel = "fix"
    # What the replay phase establishes, independent of the compiler's wording. A
    # build failure whose removal was demonstrated is syntax/elaboration evidence
    # by construction; a run-time failure means the design elaborates, so the log
    # is not evidence of a syntax problem at all.
    verified_phases = [state["buggy_verdicts"].get(b) for b in state.get("verified_backends", [])]
    if "compile_failed" in verified_phases:
        weights["syntax_compile"] += 3.0
        channels_hit.add("log")
        trace.append("syntax_compile: +3 (replay: the buggy revision does not build "
                     "and the corrected revision builds, so the failure is a "
                     "syntax/elaboration failure whatever the compiler's wording)")
    elif "failed" in verified_phases:
        trace.append("syntax_compile: not credited (replay: the buggy revision "
                     "builds and fails at run time, so this is not a build failure)")
    if not state["verified"]:
        trace.append(f"verification unavailable (replay class {state['class']}): the "
                     "failure was not shown to track the defect, so the unknown "
                     "reserve is raised")
    top = max(weights.items(), key=lambda kv: kv[1])[1]
    distinct = len([c for c, w in weights.items() if w >= top * 0.5])
    reserve = (UNKNOWN_RESERVE["high"] if distinct == 1 and top >= 4.0
               else UNKNOWN_RESERVE["medium"] if top >= 3.0 else UNKNOWN_RESERVE["low"])
    if not state["verified"]:
        reserve = max(reserve, UNKNOWN_RESERVE["medium"])
    if best_channel == "fix":
        reserve = UNKNOWN_RESERVE["high"]
    return weighted_distribution(weights, reserve), trace, best_channel


def action_distribution(case: dict, replay: dict | None
                        ) -> tuple[dict[str, float], list[str]]:
    """Weights for `next_action` over the *pre-repair* state.

    `next_action` is the direction to investigate from the buggy input, so only
    the buggy-side replay and the description feed it. The corrected replay never
    does: knowing which direction eventually paid off is exactly the answer that
    must not leak into the label.
    """
    weights: dict[str, float] = defaultdict(float)
    trace: list[str] = []
    for action, channel, weight, pattern, why in ACTION_RULES:
        hay = channel_text(case, channel, replay)
        match = re.search(pattern, hay, re.I | re.M) if hay else None
        if match:
            weights[action] += weight
            trace.append(f"{action}: +{weight} ({channel}: {why}) "
                         f"[matched {match.group(0)[:40]!r}]")
    if not weights:
        weights["abstain"] = 1.0
        trace.append("abstain: +1 (no channel gave a direction)")
    # A testbench that passes the buggy revision is itself the thing to look at:
    # the stimulus does not exercise the defect, so more testing is the honest
    # next direction rather than an RTL change.
    state = replay_state(replay)
    if state["testbench_passed_buggy_on"]:
        weights["testbench"] += 2.0
        trace.append("testbench: +2 (the supplied testbench passes the buggy "
                     "revision under " + ", ".join(state["testbench_passed_buggy_on"])
                     + ", so the stimulus does not exercise the defect)")
    if state["class"] == "cannot_build_either_revision":
        weights["simulation"] += 1.5
        trace.append("simulation: +1.5 (the buggy revision and the supplied "
                     "testbench do not elaborate together, so the build/simulation "
                     "setup has to be settled before any diagnosis)")
    if state["class"] == "no_usable_verdict":
        weights["simulation"] += 1.0
        trace.append("simulation: +1 (the replay produced no verdict line, so a "
                     "working observing run comes first)")
    total = sum(weights.values())
    probs = {k: round(weights.get(k, 0.0) / total, 4) for k in ACTION_CRITERIA}
    return probs, trace


def one_hot_or_soft(probs: dict[str, float]) -> tuple[dict[str, float], str]:
    """Collapse to one-hot only when a single category clearly dominates."""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ranked[0], ranked[1]
    if top[1] >= ONE_HOT_MIN and top[1] - second[1] >= ONE_HOT_MARGIN:
        return {k: (1.0 if k == top[0] else 0.0) for k in probs}, "one_hot"
    return probs, "soft"


def risk_estimates(case: dict) -> tuple[int, str]:
    """Rubric estimate for the recommended action's risk.

    Every Fixbench case is local diagnosis: read the buggy RTL, run the supplied
    testbench in a scratch directory. That is at most a routine simulation. The
    level is a rubric application, never an observation from the source, so it is
    only ever written to the pseudo file.
    """
    return 1, ("rubric: the pinned action is RTL inspection plus a routine local "
               "simulation, which is level 1 (Low) and never level 2+ because no "
               "case asks for an edit to shared RTL, constraints or configuration")


def urgency_estimates(case: dict) -> tuple[int, str]:
    """Rubric estimate for urgency.

    No Fixbench record states a deadline, a release, or production impact, so the
    honest rubric answer is level 0. Technical difficulty or severity is not
    urgency.
    """
    return 0, ("rubric: no deadline, release or production statement exists in the "
               "record, so urgency is level 0 (No time pressure); difficulty or "
               "technical severity is deliberately not treated as urgency")


def build_questions(n: int) -> dict:
    qs = {
        "next_action": {"type": "choice", "instructions": ACTION_INSTRUCTIONS,
                        "criteria": ACTION_CRITERIA},
        "root_cause_type": {"type": "choice", "instructions": ROOT_INSTRUCTIONS,
                            "criteria": ROOT_CAUSE_CRITERIA},
        "evidence_sufficient": {"type": "noul", "instructions": EVIDENCE_INSTRUCTIONS,
                                "criteria": EVIDENCE_CRITERIA},
    }
    if n == 5:
        qs["risk"] = {"type": "score", "instructions": RISK_INSTRUCTIONS,
                      "criteria": list(RISK_CRITERIA)}
        qs["urgency"] = {"type": "score", "instructions": URGENCY_INSTRUCTIONS,
                         "criteria": list(URGENCY_CRITERIA)}
    return qs


def score_gold(level: int, label_source: str) -> dict:
    """A score answer in the shape the existing SilicoJev files use.

    A hard level would be a claim this rubric cannot support, so the estimate is
    a one-hot value with explicit confidence, plus the expectation `score` and
    the argmax `label` that `review_pseudo_scores.py` re-derives.
    """
    probabilities = {str(i): (1.0 if i == level else 0.0) for i in range(4)}
    expectation = sum(int(key) * value for key, value in probabilities.items())
    return {
        "probabilities": probabilities,
        "label": str(level),
        "score": round(expectation, 6),
        "confidence": probabilities[str(level)],
        "label_source": label_source,
    }


BUGGY_REPLAY_NOTE = {
    "compile_failed": "the buggy revision does not build together with the supplied "
                      "testbench, and no run was possible",
    "failed": "the supplied testbench reports a failure against the buggy revision",
    "passed": "the supplied testbench passes against the buggy revision, so it does "
              "not detect the defect",
    "timeout": "the replay against the buggy revision did not finish in time",
    "tool_error": "the simulator could not load the design, so no run happened",
    "unknown": "the buggy replay produced no verdict line",
    "not_replayed": "this case was not replayed",
}


def replay_block(replay: dict | None) -> dict:
    """The buggy-revision replay, shaped for the model state.

    Only the `buggy` side is ever exposed, and the note is written from the buggy
    side alone. The corrected replay, its verdict and the fact that it passes
    stay in `outcome`/`provenance`: the state may not contain a statement that a
    corrected version exists or passes.
    """
    if replay is None:
        return {"status": "not_replayed",
                "note": BUGGY_REPLAY_NOTE["not_replayed"],
                "per_backend": {}}
    buggy = replay.get("buggy") or {}
    compile_step = buggy.get("compile") or {}
    run_step = buggy.get("run") or {}
    status = buggy.get("verdict", "unknown")
    per_backend = {b: (a or {}).get("verdict")
                   for b, a in sorted((replay.get("buggy_by_backend") or {}).items())}
    note = BUGGY_REPLAY_NOTE.get(status, BUGGY_REPLAY_NOTE["unknown"])
    if len(set(per_backend.values())) > 1:
        note += ("; the simulators disagree: " + ", ".join(
            f"{b} {v}" for b, v in per_backend.items()))
    return {
        "status": status,
        "note": note,
        "per_backend": per_backend,
        "backend": buggy.get("backend"),
        "top_module": buggy.get("top"),
        "compile_command": " ".join(compile_step.get("cmd") or []) or None,
        "compile_returncode": compile_step.get("returncode"),
        "run_command": " ".join(run_step.get("cmd") or []) or None,
        "run_returncode": run_step.get("returncode"),
        "timed_out": bool(compile_step.get("timed_out") or run_step.get("timed_out")),
        "compiler_output": clip(streams(compile_step), STATE_LOG_LIMIT),
        "simulation_output": clip(streams(run_step), STATE_LOG_LIMIT),
    }


def streams(step: dict) -> str:
    """Both output streams of a tool step, labelled, stdout first.

    Preferring one stream silently drops the other, and which one carries the
    message differs per tool: iverilog reports diagnostics on stderr, Questa's
    transcript lands on stdout. Labelling them keeps the state complete and lets
    the validator re-derive the field.
    """
    parts = []
    for name in ("stdout", "stderr"):
        text = (step.get(name) or "").strip()
        if text:
            parts.append(f"[{name}]\n{text}")
    return "\n".join(parts)


def build_state(index: int, case: dict, case_id: str, replay: dict | None,
                tool_versions: dict, raw_path: str, raw_sha: str) -> dict:
    bug = case.get("bug") or ""
    return {
        "case_id": case_id,
        "source_index": index,
        "bug_description": clip(bug, BUG_LIMIT),
        "buggy_rtl": clip(case.get("buggycode") or "", RTL_LIMIT),
        "testbench": clip(case.get("testbench") or "", RTL_LIMIT),
        "replay": replay_block(replay),
        "source_metadata": {
            "dataset": SOURCE,
            "dataset_file": raw_path,
            "dataset_sha256": raw_sha,
            "license": LICENSE,
            "top_module": (replay or {}).get("buggy_modules", [None])[:1] or None,
            "toolchain": {k: v for k, v in tool_versions.items() if v},
            "input_fields_used": ["bug", "buggycode", "testbench"],
        },
        "previous_actions": [],
    }


def convert_case(index: int, case: dict, replay: dict | None, tool_versions: dict,
                 raw_path: str, raw_sha: str) -> tuple[dict, dict, dict]:
    case_id = stable_case_id(index, case)
    rid = f"fixbench:{case_id}"
    bug = case.get("bug") or ""
    state_replay = replay_state(replay)
    replay_verified = state_replay["verified"]
    attribution = state_replay["class"]
    tb_passed_buggy = bool(state_replay["testbench_passed_buggy_on"])
    buggy_verdict = state_replay["buggy_verdict"]
    correct_verdict = ((replay or {}).get("correct") or {}).get("verdict", "not_replayed")

    root_probs, root_trace, root_channel = root_cause_distribution(case, replay)
    root_probs, root_shape = one_hot_or_soft(root_probs)
    dominant_root = max(root_probs.items(), key=lambda kv: kv[1])[0]
    dominant_root_p = root_probs[dominant_root]

    action_probs, action_trace = action_distribution(case, replay)
    action_probs, action_shape = one_hot_or_soft(action_probs)
    dominant_action = max(action_probs.items(), key=lambda kv: kv[1])[0]

    # Label sources, decided by which channel actually drove the winning category
    # rather than by a second keyword list. `verified_repair` requires the replay
    # to have shown that the failure tracks the defect; `manual_bug_label` means
    # the winning category is stated by the human description; anything the rules
    # inferred from wording alone is a reading and says so.
    def drove(trace: list[str], winner: str, channel: str) -> bool:
        return any(entry.startswith(f"{winner}:") and f"({channel}:" in entry
                   for entry in trace)

    root_from_bug = drove(root_trace, dominant_root, "bug")
    root_from_log = drove(root_trace, dominant_root, "log")
    action_from_bug = drove(action_trace, dominant_action, "bug")
    action_from_log = drove(action_trace, dominant_action, "log")

    if (replay_verified and dominant_root != "unknown" and dominant_root_p >= DECISIVE_MIN
            and (root_from_bug or root_from_log)):
        root_source = "verified_repair"
    elif root_from_bug:
        root_source = "manual_bug_label"
    else:
        root_source = "codex_pseudo_unverified"

    if replay_verified and action_shape == "one_hot" and action_from_log:
        action_source = "verified_repair"
    elif action_from_bug:
        action_source = "manual_bug_label"
    else:
        action_source = "codex_pseudo_unverified"

    # Evidence sufficiency: `true` only where a tool or a trusted label supports
    # the evidence, per the canonical criteria wording. A testbench that passes
    # the buggy revision is positive evidence that the tool output does *not*
    # localise the defect, so it is the one case that leans further towards
    # `false` than a bare unverified replay does.
    trusted_label = root_from_bug or root_source == "verified_repair"
    evidence_true = bool(replay_verified or trusted_label)
    evidence_source = ("verified_repair" if replay_verified
                       else "manual_bug_label" if trusted_label
                       else "codex_pseudo_unverified")
    if replay is None:
        evidence_probs = {"false": 1.0, "true": 0.0}
    elif replay_verified and trusted_label:
        evidence_probs = {"false": 0.05, "true": 0.95}
    elif replay_verified:
        evidence_probs = {"false": 0.2, "true": 0.8}
    elif tb_passed_buggy:
        evidence_probs = {"false": 0.9, "true": 0.1}
    elif attribution == "cannot_build_either_revision":
        evidence_probs = {"false": 0.85, "true": 0.15}
    elif trusted_label:
        evidence_probs = {"false": 0.25, "true": 0.75}
    else:
        evidence_probs = {"false": 0.9, "true": 0.1}

    gold3 = {
        "next_action": {"probabilities": action_probs, "label_source": action_source},
        "root_cause_type": {"probabilities": root_probs, "label_source": root_source},
        "evidence_sufficient": {"probabilities": evidence_probs,
                                "label_source": evidence_source},
    }

    state = build_state(index, case, case_id, replay, tool_versions, raw_path, raw_sha)
    leaks = leakage_check(state, case)
    flags: list[str] = []
    if PR_TEXT.search(bug):
        flags.append("bug_text_is_upstream_pr_text")
    if SECURITY_CONTENT.search(bug):
        flags.append("security_relevant_content")
    if tb_passed_buggy:
        flags.append("testbench_does_not_detect_the_bug")
    if replay is None:
        flags.append("not_replayed")
    elif attribution == "cannot_build_either_revision":
        flags.append("cannot_build_either_revision")
    elif attribution == "failure_survives_repair":
        flags.append("failure_survives_repair")
    elif attribution == "no_usable_verdict":
        flags.append("no_usable_verdict")

    risk_level, risk_why = risk_estimates(case)
    urgency_level, urgency_why = urgency_estimates(case)

    record = {
        "id": rid,
        "source": SOURCE,
        "source_group": GROUP,
        "state": json.dumps(state, ensure_ascii=False, sort_keys=True),
        "questions": json.dumps(build_questions(3), ensure_ascii=False, sort_keys=True),
        "gold": json.dumps(gold3, ensure_ascii=False, sort_keys=True),
        "outcome": {
            "case_id": case_id,
            "resolved": True,
            "repair_available": True,
            "repair_evidence_verified": replay_verified,
            "replay_class": attribution,
            "deciding_backend": state_replay["deciding_backend"],
            "base_replay_verdict": buggy_verdict,
            "corrected_replay_verdict": correct_verdict,
            "base_replay_verdicts_by_backend": state_replay["buggy_verdicts"],
            "testbench_passed_buggy_on": state_replay["testbench_passed_buggy_on"],
            "corrected_replay_note": ("verification only: the corrected revision was "
                                      "replayed to check the repair, and that log and "
                                      "verdict are deliberately not part of the state"),
            "correctcode_sha256": hashlib.sha256(
                (case.get("correctcode") or "").encode()).hexdigest(),
        },
        "provenance": {
            "license": LICENSE,
            "synthetic": False,
            "benchmark": SOURCE,
            "case_id": case_id,
            "source_index": index,
            "bug_family": dominant_root,
            "quality": ("repair_evidence_verified" if replay_verified
                        else "replayed_not_verified" if replay is not None
                        else "unvalidated"),
            "label_source": root_source,
            "repair_evidence_verified": replay_verified,
            "replay_class": attribution,
            "deciding_backend": state_replay["deciding_backend"],
            "evidence_channel": root_channel,
            "state_redactions": [
                "excluded correctcode and any diff against it",
                "excluded the corrected-revision replay log, commands and verdict",
                "excluded any statement that the corrected revision builds or passes",
                "excluded the attribution class from the state: it is derived by "
                "comparing the two revisions, so it lives in provenance only",
                "excluded post-repair PR/commit metadata; the bug text is kept "
                "verbatim because it is the input a real agent would be given",
            ],
            "rule_traces": {"root_cause": root_trace, "next_action": action_trace},
            "probability_shape": {"root_cause": root_shape, "next_action": action_shape},
            "trust_flags": flags,
            "risk": {"level": risk_level, "rationale": risk_why,
                     "defensible": False, "label_source": "codex_pseudo_unverified"},
            "urgency": {"level": urgency_level, "rationale": urgency_why,
                        "defensible": False, "label_source": "codex_pseudo_unverified",
                        "flags": (["urgency_no_time_pressure_not_proven"]
                                  if urgency_level == 0 else [])},
            "leakage_findings": leaks,
        },
    }
    audit = {
        "case_id": case_id, "index": index,
        "replay_verified": replay_verified, "replay_class": attribution,
        "deciding_backend": state_replay["deciding_backend"],
        "buggy_verdict": buggy_verdict, "correct_verdict": correct_verdict,
        "buggy_verdicts_by_backend": state_replay["buggy_verdicts"],
        "testbench_passed_buggy_on": state_replay["testbench_passed_buggy_on"],
        "evidence_channel": root_channel,
        "root_cause": dominant_root, "root_source": root_source,
        "action": dominant_action, "action_source": action_source,
        "evidence_source": evidence_source, "evidence_true": evidence_true,
        "flags": flags, "leakage": leaks,
        "root_shape": root_shape, "action_shape": action_shape,
    }
    return record, audit, gold3


def _evaluation_exclusions(records: list[dict], replay_by_index: dict) -> dict:
    """Cases whose *labels* should not be trusted even if the replay verified.

    Three mechanisms can make a case unsafe for evaluation:

    - the bug text is upstream PR/commit prose, which describes the fix rather
      than the defect as it was found;
    - the testbench passes the buggy revision, so the tool evidence in the state
      does not point at the defect at all;
    - the pair cannot be built, or the failure survives the repair, so nothing in
      the replay confirms the defect.

    A record may legitimately remain in the 3q file — the state is still valid
    pre-repair evidence — but it must not be scored as a verified case.
    """
    excluded: dict[str, list[str]] = {}
    for record in records:
        prov = record["provenance"]
        reasons = [flag for flag in prov["trust_flags"]
                   if flag in ("bug_text_is_upstream_pr_text",
                               "testbench_does_not_detect_the_bug",
                               "cannot_build_either_revision",
                               "failure_survives_repair",
                               "no_usable_verdict")]
        if not reasons:
            continue
        excluded[record["id"]] = {
            "index": prov["source_index"],
            "reasons": reasons,
            "replay_class": prov["replay_class"],
            "repair_evidence_verified": prov["repair_evidence_verified"],
            "still_in_records_3q": True,
        }
    return excluded


def state_lines(text: str) -> set[str]:
    """Stripped lines of a state field, dropping a clipped final line."""
    if not text:
        return set()
    rows = text.splitlines()
    if rows and rows[-1].endswith("...[truncated]"):
        rows = rows[:-1]
    return {row.strip() for row in rows if row.strip()}


def leakage_check(state: dict, case: dict) -> list[str]:
    """Mechanical check that no repaired content reached the state.

    Line-exact rather than substring: a repaired line can be a *prefix* of a
    buggy line (a trailing comma or indentation differs) and a substring test
    reports that as a leak. The decisive check is structural — every line of the
    state's RTL field must be a line of the buggy revision, so nothing from any
    other revision can appear there.
    """
    findings: list[str] = []
    correct = (case.get("correctcode") or "").strip()
    if not correct:
        return ["source case has no correctcode to compare against"]
    buggy_lines = state_lines(case.get("buggycode"))
    tb_lines = state_lines(case.get("testbench"))

    for field, source, label in (("buggy_rtl", buggy_lines, "buggy revision"),
                                 ("testbench", tb_lines, "supplied testbench")):
        extra = state_lines(state.get(field) or "") - source
        for line in sorted(extra)[:3]:
            findings.append(f"{field}: line absent from the {label}: {line[:70]}")

    # Comment-only and trivial lines collide with the buggy revision by chance.
    repaired_only = {l.strip() for l in correct.splitlines()
                     if len(l.strip()) >= 12 and not l.strip().startswith("//")} - buggy_lines
    replay = state.get("replay") or {}
    for name, text in (("buggy_rtl", state.get("buggy_rtl")),
                       ("testbench", state.get("testbench")),
                       ("replay.compiler_output", replay.get("compiler_output")),
                       ("replay.simulation_output", replay.get("simulation_output"))):
        for line in sorted(state_lines(text) & repaired_only):
            if name == "testbench" and line in tb_lines:
                continue  # supplied stimulus that happens to match a repaired line
            findings.append(f"{name}: repaired-only line present: {line[:70]}")

    if correct in json.dumps(state, ensure_ascii=False):
        findings.append("full correctcode text present in state")
    commands = " ".join(filter(None, [replay.get("compile_command"),
                                      replay.get("run_command")]))
    if "fix.sv" in commands:
        findings.append("state replay command references the corrected file")
    blob = json.dumps(state, ensure_ascii=False)
    for banned in ("correctcode", "corrected_code", "fixed_code", "diff", "patch"):
        # A banned *key*, not the bare word: "different" must not trip this.
        if f'"{banned}"' in blob:
            findings.append(f"state carries a banned key: {banned}")
    if '"fix.sv"' in blob:
        findings.append("state carries a banned key: fix.sv")
    return findings


def _pr_line(bug: str) -> str:
    """The first line of a description, which here is the upstream PR title."""
    stripped = (bug or "").strip()
    if not stripped:
        return ""
    return re.sub(r"\s+", " ", stripped.splitlines()[0].strip().lower())[:120]


def _norm_testbench(testbench: str) -> str:
    """Harness text with comments, digits and whitespace normalised away."""
    text = re.sub(r"//.*", "", testbench or "")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"\d+", "N", text))


def _design_lines(source: str) -> set[str]:
    """Distinct, non-trivial lines of a design file, for overlap comparison."""
    return {line.strip() for line in (source or "").splitlines() if len(line.strip()) > 8}


def lineage_families(cases: list[dict]) -> dict:
    """Group cases that share a design, a harness, or an upstream pull request.

    Fixbench is not 100 independent decisions: its cases were cut from a much
    smaller set of upstream PRs and designs. Nine families cover 30 of the 100
    cases, so a row-wise train/test split would put the same design on both
    sides. This grouping is the unit a split should use.

    Four signals, each structural and recheckable:

    - `same_testbench` — identical harness after stripping comments, digits and
      whitespace;
    - `shared_module_name` — the design file declares the same module;
    - `same_upstream_pr_line` — the first line of the description, which in this
      dataset is verbatim the upstream PR title, is identical;
    - `same_design_code` — symmetric line Jaccard >= 0.4 between the two buggy
      revisions. Symmetric on purpose: a min()-normalised overlap lets a short
      fragment match half the corpus and merge unrelated families.
    """
    n = len(cases)
    record_ids = [f"fixbench:{stable_case_id(i, c)}" for i, c in enumerate(cases)]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    testbenches: dict[str, list[int]] = defaultdict(list)
    modules: dict[str, list[int]] = defaultdict(list)
    pr_lines: dict[str, list[int]] = defaultdict(list)
    for i, case in enumerate(cases):
        testbenches[hashlib.sha256(
            _norm_testbench(case.get("testbench") or "").encode()).hexdigest()[:18]].append(i)
        for module in re.findall(r"^\s*module\s+([A-Za-z_]\w*)",
                                 case.get("buggycode") or "", re.M):
            modules[module].append(i)
        pr_lines[_pr_line(case.get("bug") or "")].append(i)

    signals: dict[str, list[list[str]]] = defaultdict(list)
    for name, buckets in (("same_testbench", testbenches),
                          ("shared_module_name", modules),
                          ("same_upstream_pr_line", pr_lines)):
        for members in buckets.values():
            if len(members) < 2:
                continue
            signals[name].append(sorted(record_ids[i] for i in members))
            for extra in members[1:]:
                union(members[0], extra)

    code_pairs: list[dict] = []
    lines = [_design_lines(case.get("buggycode") or "") for case in cases]
    for i in range(n):
        for j in range(i + 1, n):
            if not lines[i] or not lines[j]:
                continue
            overlap = len(lines[i] & lines[j]) / len(lines[i] | lines[j])
            if overlap >= 0.4:
                union(i, j)
                code_pairs.append({"cases": [record_ids[i], record_ids[j]],
                                   "line_jaccard": round(overlap, 3)})

    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(i)

    signal_buckets = {name: [set(group) for group in groups]
                      for name, groups in signals.items()}
    code_pairs_by_set = [(set(pair["cases"]), pair["cases"]) for pair in code_pairs]

    families, case_group = [], {}
    for members in sorted(buckets.values(), key=lambda m: (-len(m), m[0])):
        key = f"group_{members[0]:03d}"
        for i in members:
            case_group[record_ids[i]] = key
        if len(members) == 1:
            continue
        member_ids = sorted(record_ids[i] for i in members)
        member_set = set(member_ids)
        # A signal that covers the whole family is reported as uniform; one that
        # only holds inside a sub-block is reported with that block, so a reader
        # never sees a claim that is true of part of the family stated as if it
        # were true of all of it.
        uniform: list[str] = []
        partial: dict[str, list[list[str]]] = {}
        for name, groups in signal_buckets.items():
            hits = [group for group in groups if group & member_set]
            if not hits:
                continue
            if any(group >= member_set for group in hits):
                uniform.append(name)
                continue
            partial[name] = sorted(sorted(group) for group in hits)
        for pair_set, pair in code_pairs_by_set:
            if pair_set <= member_set:
                partial.setdefault("same_design_code", []).append(pair)
        for name in partial:
            partial[name] = sorted(partial[name])
        if "same_upstream_pr_line" in uniform:
            label = next(_pr_line(cases[i].get("bug") or "") for i in members
                         if _pr_line(cases[i].get("bug") or ""))
        else:
            shared_modules: set[str] = set()
            member_modules = [set(re.findall(r"^\s*module\s+([A-Za-z_]\w*)",
                                             cases[i].get("buggycode") or "", re.M))
                              for i in members]
            if member_modules:
                shared_modules = set.intersection(*member_modules)
            label = (f"shared module {', '.join(sorted(shared_modules))}" if shared_modules
                     else "identical testbench" if "same_testbench" in uniform
                     else _pr_line(cases[members[0]].get("bug") or "")
                     or "near-identical design code")
        families.append({"group": key, "size": len(members), "members": member_ids,
                         "signals_uniform": sorted(uniform),
                         "signals_partial": partial,
                         "shared_label": label[:90]})

    return {
        "why": ("Fixbench cases are cut from a small number of upstream pull "
                "requests and designs; the families below are near-duplicates by "
                "harness, design code, module name, or PR title. Split by group, "
                "never by row."),
        "signals": {
            "same_testbench": "identical harness (comments, digits and whitespace stripped)",
            "shared_module_name": "both design files declare the same module",
            "same_upstream_pr_line": "identical first line of the description (the upstream PR title)",
            "same_design_code": "buggy revisions share >= 40% of their distinct lines (symmetric Jaccard)",
        },
        "case_group": case_group,
        "groups_total": len(buckets),
        "families": families,
        "cases_in_families": sum(f["size"] for f in families),
        "largest_family": max((f["size"] for f in families), default=1),
        "design_code_pairs": code_pairs,
        "advisory": ("Cases in different families can still resemble each other: the "
                     "descriptions in this dataset were written from a common template, "
                     "so text similarity alone is not used as a grouping signal."),
    }


def suggested_split(case_group: dict[str, str], trusted_ids: list[str]) -> dict:
    """A deterministic, family-disjoint train/test sketch over the trusted set.

    Groups are ordered by a hash of their key, so the split does not depend on
    record order, and whole groups move together, so no design appears on both
    sides. It is a starting point for a small experiment, not a finished split.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for cid in sorted(trusted_ids):
        grouped[case_group.get(cid, cid)].append(cid)
    ordered = sorted(grouped, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    target = (len(trusted_ids) + 3) // 4            # about a quarter held out
    test_groups: list[str] = []
    for group in ordered:
        if sum(len(grouped[g]) for g in test_groups) >= target:
            break
        test_groups.append(group)
    test = sorted(cid for group in test_groups for cid in grouped[group])
    train = sorted(set(trusted_ids) - set(test))
    return {
        "basis": "usage_sets.json:replay_verified_3q",
        "train": train,
        "test": test,
        "test_groups": sorted(test_groups),
        "family_disjoint": not any(set(grouped[g]) & set(train) for g in test_groups),
        "rule": ("groups ordered by sha256(group key); whole groups are assigned to "
                 "test until it holds about a quarter of the trusted cases"),
        "caveat": ("35 trusted cases across 30 independent groups is too small for a "
                   "confident split; treat any score from it as a smoke test"),
    }


def provenance_sheet(records: list[dict], audits_by_id: dict[str, dict],
                     trusted_ids: list[str], case_group: dict[str, str]) -> str:
    """A per-question provenance sheet for the strictest replay set.

    The point is to make it possible to judge a label *before* calling it trusted
    supervision, rather than reading a count off the report.
    """
    by_id = {record["id"]: record for record in records}
    lines = [
        "# Trusted-set label provenance",
        "",
        f"Every case in `usage_sets.json` → `replay_verified_3q` ({len(trusted_ids)} cases), with each",
        "question's winning label, the label source, the evidence channel that drove",
        "it, and the rules that fired. Generated by `training/convert_fixbench_rtl.py`.",
        "",
        "How to read it: `verified_repair` means the replay and the source evidence",
        "support that label — not that it is the only defensible judgement.",
        "`manual_bug_label` means a human-written description states it. The two choice",
        "questions stay readings of the evidence even here, so inspect the traces",
        "before treating any of this as ground truth.",
        "",
        "| id | next_action | root_cause_type | evidence_sufficient | channel | replay class | family |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cid in sorted(trusted_ids):
        audit = audits_by_id[cid]
        gold = json.loads(by_id[cid]["gold"])

        def cell(question: str, winner: str) -> str:
            shape = audit["root_shape"] if question == "root_cause_type" else audit["action_shape"]
            mark = " · one-hot" if shape == "one_hot" else ""
            return f"`{winner}`{mark} ({gold[question]['label_source']})"

        evidence = "true" if audit["evidence_true"] else "false"
        lines.append(
            f"| `{cid}` | {cell('next_action', audit['action'])} "
            f"| {cell('root_cause_type', audit['root_cause'])} "
            f"| `{evidence}` ({gold['evidence_sufficient']['label_source']}) "
            f"| {audit['evidence_channel']} | {audit['replay_class']} "
            f"| {case_group.get(cid, '?')} |")

    lines += ["", "## Rules that fired", ""]
    for cid in sorted(trusted_ids):
        record = by_id[cid]
        audit = audits_by_id[cid]
        traces = record["provenance"]["rule_traces"]
        root_trace = "; ".join(traces["root_cause"][:3]) or "none"
        action_trace = "; ".join(traces["next_action"][:3]) or "none"
        lines.append(
            f"- **{cid}** — `{audit['replay_class']}` decided by "
            f"{audit['deciding_backend'] or 'no backend'}; buggy "
            f"`{audit['buggy_verdict']}` → corrected `{audit['correct_verdict']}`.")
        lines.append(f"  - root: {root_trace}")
        lines.append(f"  - action: {action_trace}")
    lines += [
        "",
        "## Cautions",
        "",
        f"- {len(trusted_ids)} cases across {len({case_group.get(c, c) for c in trusted_ids})} lineage groups: a score from any split of this set is a smoke",
        "  test, not a measurement.",
        "- `replay_verified` compares two revisions under the supplied testbench. It does",
        "  not certify the choice labels; see the label sources above.",
        "- The score questions (risk, urgency) are absent here on purpose: none of these",
        "  cases supports them. See `records_5q_pseudo_unverified.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.raw.is_file():
        sys.exit(f"missing raw dataset: {args.raw}. Download it from the public "
                 f"Hugging Face dataset KSU-HW-SEC/Fixbench-RTL; do not synthesise it.")
    cases = json.loads(args.raw.read_text())
    raw_sha = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    raw_rel = str(args.raw.relative_to(REPO_ROOT))

    replay_payload = json.loads(args.replay.read_text()) if args.replay.is_file() else None
    replay_by_index = {}
    tool_versions: dict = {}
    if replay_payload:
        if replay_payload.get("raw_sha256") != raw_sha:
            sys.exit("replay cache was built from a different Fixbench-RTL.json "
                     f"({replay_payload.get('raw_sha256')} != {raw_sha}); rebuild it")
        replay_by_index = {r["index"]: r for r in replay_payload["results"]}
        tool_versions = replay_payload.get("tool_versions") or {}

    args.out.mkdir(parents=True, exist_ok=True)
    records, audits, gold3_by_id = [], [], {}
    for index, case in enumerate(cases):
        record, audit, gold3 = convert_case(index, case, replay_by_index.get(index),
                                            tool_versions, raw_rel, raw_sha)
        records.append(record)
        audits.append(audit)
        gold3_by_id[record["id"]] = gold3

    def root_of(record: dict) -> str:
        return max(json.loads(record["gold"])["root_cause_type"]["probabilities"].items(),
                   key=lambda kv: kv[1])[0]

    def is_pseudo(record: dict, qids=BASE_QUESTION_IDS) -> bool:
        gold = json.loads(record["gold"])
        return any(gold[q]["label_source"] == "codex_pseudo_unverified" for q in qids)

    weak3 = [r for r in records if is_pseudo(r)]
    strong3 = [r for r in records if r not in weak3]

    # --- five-question records -------------------------------------------------
    # A record earns a place in records_5q.jsonl only when risk *and* urgency are
    # genuinely supported by the source. Nothing in Fixbench states a deadline or
    # a production/user impact, and the risk rubric scores the recommended action,
    # which is local read-only inspection plus routine simulation for every case.
    # So the defensible file stays empty and every rubric estimate goes to the
    # pseudo file, marked as such on both score labels.
    defensible_5q, pseudo_5q = [], []
    for record in records:
        gold = json.loads(record["gold"])
        prov = record["provenance"]
        risk_level = prov["risk"]["level"]
        urgency_level = prov["urgency"]["level"]
        risk_defensible = prov["risk"]["defensible"]
        urgency_defensible = prov["urgency"]["defensible"]
        five = dict(gold)
        five["risk"] = score_gold(risk_level, prov["risk"]["label_source"])
        five["urgency"] = score_gold(urgency_level, prov["urgency"]["label_source"])
        row = dict(record)
        row["questions"] = json.dumps(build_questions(5), ensure_ascii=False, sort_keys=True)
        row["gold"] = json.dumps(five, ensure_ascii=False, sort_keys=True)
        row["provenance"] = dict(prov)
        defensible = bool(risk_defensible and urgency_defensible)
        row["provenance"]["score_tier"] = "5q_defensible" if defensible else "5q_pseudo"
        (defensible_5q if defensible else pseudo_5q).append(row)

    # Cases whose evaluation is compromised are listed, not silently dropped:
    # records_3q.jsonl still holds them, and the id list here is what to exclude.
    exclusions = _evaluation_exclusions(records, replay_by_index)

    def write(name: str, rows: list[dict]) -> None:
        path = args.out / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # records_3q.jsonl carries every converted case, with the honest per-question
    # label source. The filtered views that training should actually use are the
    # id lists in usage_sets.json, so no record is duplicated across files.
    write("records_3q.jsonl", records)
    write("records_5q.jsonl", defensible_5q)
    write("records_5q_pseudo_unverified.jsonl", pseudo_5q)

    strong_ids = {r["id"] for r in strong3}
    verified_ids = {r["id"] for r in records if r["provenance"]["repair_evidence_verified"]}
    unvalidated_ids = {r["id"] for r in records if r["provenance"]["quality"] == "unvalidated"}
    flagged_ids = {r["id"] for r in records if r["provenance"]["trust_flags"]}
    by_class: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_class[r["provenance"]["replay_class"]].append(r["id"])
    usage_sets = {
        # Three nested gates, so the reason a case is left out is visible:
        # verified -> unflagged -> no pseudo label among the three questions.
        "repair_evidence_verified_3q": sorted(verified_ids),
        "replay_verified_unflagged_3q": sorted(verified_ids - flagged_ids),
        "replay_verified_3q": sorted(strong_ids & verified_ids - flagged_ids),
        "unverified_label_3q": sorted({r["id"] for r in records} - strong_ids),
        "unvalidated_no_replay": sorted(unvalidated_ids),
        "flagged_for_review": sorted(flagged_ids),
        "flag_reasons": {r["id"]: r["provenance"]["trust_flags"]
                         for r in records if r["provenance"]["trust_flags"]},
        "replay_class_3q": {k: sorted(v) for k, v in sorted(by_class.items())},
        "pseudo_5q": sorted(r["id"] for r in pseudo_5q),
        "defensible_5q": sorted(r["id"] for r in defensible_5q),
        "policy": (
            "records_3q.jsonl holds all converted cases with the honest per-question "
            "label source. The three gates nest: 'repair_evidence_verified_3q' is "
            "every case whose buggy revision demonstrably fails the supplied testbench "
            "under a simulator in which the corrected revision passes it "
            "(provenance.repair_evidence_verified); 'replay_verified_unflagged_3q' "
            "additionally drops cases carrying a trust flag; 'replay_verified_3q' "
            "additionally requires all three labels to be non-pseudo. 'verified_repair' "
            "means the replay and the source evidence support that label; it is not a "
            "claim that the judgement is uniquely correct, and next_action / "
            "root_cause_type remain readings even in the trusted set. The score "
            "questions in records_5q_pseudo_unverified.jsonl are rubric estimates and "
            "must not be used as gold. Nothing here is merged into dataset/normalized/."
        ),
    }
    (args.out / "usage_sets.json").write_text(json.dumps(usage_sets, indent=2))

    # --- lineage grouping and trusted-set provenance --------------------------
    # Fixbench cases are not independent draws: nine families cover 30 of them.
    # The grouping is what a split must use, and the provenance sheet is what to
    # read before calling the strictest set trusted supervision.
    lineage = lineage_families(cases)
    lineage["suggested_split"] = suggested_split(lineage["case_group"],
                                                usage_sets["replay_verified_3q"])
    (args.out / "split_groups.json").write_text(json.dumps(lineage, indent=2))
    audits_by_id = {record["id"]: audit for record, audit in zip(records, audits)}
    (args.out / "trusted_label_provenance.md").write_text(
        provenance_sheet(records, audits_by_id, usage_sets["replay_verified_3q"],
                         lineage["case_group"]))

    # --- report ---------------------------------------------------------------
    replay_summary = {
        "cache": (str(args.replay.relative_to(REPO_ROOT))
                  if args.replay.is_file() and args.replay.is_relative_to(REPO_ROOT)
                  else str(args.replay)),
        "tool_versions": tool_versions,
        "procedure": ("for each case the supplied testbench was run against the buggy "
                      "revision and, separately, against the corrected revision, in a "
                      "scratch directory per revision with its own work library; the "
                      "testbench was compiled after the design file because 13 "
                      "testbenches import a package declared in the design"),
        "buggy_verdicts": dict(sorted(Counter(a["buggy_verdict"] for a in audits).items())),
        "corrected_verdicts": dict(sorted(Counter(a["correct_verdict"] for a in audits).items())),
        "buggy_verdicts_by_backend": {
            backend: dict(sorted(Counter(
                a["buggy_verdicts_by_backend"].get(backend, "not_attempted")
                for a in audits).items()))
            for backend in sorted({b for a in audits
                                   for b in a["buggy_verdicts_by_backend"]})},
        "replay_classes": dict(sorted(Counter(a["replay_class"] for a in audits).items())),
        "repair_evidence_verified": sum(1 for a in audits if a["replay_verified"]),
        "testbench_passed_buggy_on_a_backend": sum(
            1 for a in audits if a["testbench_passed_buggy_on"]),
        "deciding_backend": dict(sorted(Counter(
            a["deciding_backend"] for a in audits if a["deciding_backend"]).items())),
        "not_replayed": sum(1 for a in audits if a["buggy_verdict"] == "not_replayed"),
        "policy": (
            "A case counts as repair-evidence-verified only when, under the *same* "
            "simulator, the buggy revision fails (compile or run) and the corrected "
            "revision passes. Pairs are formed per simulator because a revision can "
            "fail to build under one tool and run fine under another. Cases where the "
            "testbench passes the buggy revision, where the pair cannot be built at "
            "all, where the failure survives the repair, and where no verdict line "
            "appears are recorded as their own classes and are not verification."
        ),
    }
    report = {
        "source": SOURCE,
        "source_file": raw_rel,
        "source_sha256": raw_sha,
        "generated_by": "training/convert_fixbench_rtl.py",
        "license": LICENSE,
        "total_cases_found": len(cases),
        "converted": len(records),
        "skipped": [],
        "replay": replay_summary,
        "record_counts": {
            "records_3q.jsonl": len(records),
            "records_3q_non_pseudo_labels": len(strong3),
            "records_5q.jsonl": len(defensible_5q),
            "records_5q_pseudo_unverified.jsonl": len(pseudo_5q),
            "three_question_only": len(records) - len(defensible_5q) - len(pseudo_5q),
            "evaluation_excluded": len(exclusions),
        },
        "evaluation_exclusions": exclusions,
        "root_cause_distribution": dict(sorted(Counter(root_of(r) for r in records).items())),
        "next_action_distribution": dict(sorted(Counter(
            max(json.loads(r["gold"])["next_action"]["probabilities"].items(),
                key=lambda kv: kv[1])[0] for r in records).items())),
        "probability_shapes": {
            "root_cause": dict(sorted(Counter(a["root_shape"] for a in audits).items())),
            "next_action": dict(sorted(Counter(a["action_shape"] for a in audits).items())),
        },
        "label_source_distribution": {
            q: dict(sorted(Counter(json.loads(r["gold"])[q]["label_source"]
                                   for r in records).items()))
            for q in BASE_QUESTION_IDS
        },
        "evidence_channel": dict(sorted(Counter(a["evidence_channel"] if "evidence_channel" in a
                                                else "none" for a in audits).items())),
        "trust_flags": dict(sorted(Counter(f for a in audits for f in a["flags"]).items())),
        "cases_flagging_trust": {a["case_id"]: a["flags"] for a in audits if a["flags"]},
        "score_question_policy": {
            "defensible_5q": ("risk and urgency both anchored in source evidence; "
                              "empty for Fixbench because no case states a deadline or a "
                              "production/user impact and the risk rubric scores the action, "
                              "which is local inspection plus routine simulation throughout"),
            "pseudo_5q": ("transparent rubric estimate; both score labels are "
                          "codex_pseudo_unverified and must not be treated as gold"),
            "attestation": "no pseudo risk or urgency is presented as gold anywhere",
        },
        "leakage": {
            "cases_with_findings": {a["case_id"]: a["leakage"] for a in audits if a["leakage"]},
            "findings_total": sum(len(a["leakage"]) for a in audits),
            "checks": [
                "every line of state.buggy_rtl is a line of the buggy revision",
                "every line of state.testbench is a line of the supplied testbench",
                "no repaired-only line appears as a state line (line-exact, so a "
                "repaired line that is a prefix of a buggy line is not counted)",
                "the whole correctcode text is absent from the state",
                "no replay command in the state references the corrected file",
                "no banned keys (correctcode/diff/fix.sv)",
                "state.replay carries buggy-side commands and output only"],
        },
        "post_write_audit": {},
        "audit_index": audits,
    }

    expected_questions = {"records_3q.jsonl": 3, "records_5q.jsonl": 5,
                          "records_5q_pseudo_unverified.jsonl": 5}
    for name, n_questions in expected_questions.items():
        ids, problems = [], []
        for line in (args.out / name).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(row["id"])
            qs, gold = json.loads(row["questions"]), json.loads(row["gold"])
            if len(qs) != n_questions:
                problems.append(f"{row['id']}: {len(qs)} questions, expected {n_questions}")
            if set(qs) != set(gold):
                problems.append(f"{row['id']}: question/gold id mismatch")
            if row["source"] != SOURCE or row["source_group"] != GROUP:
                problems.append(f"{row['id']}: bad source/source_group")
            if not row["provenance"].get("label_source"):
                problems.append(f"{row['id']}: no label_source")
        report["post_write_audit"][name] = {
            "records": len(ids), "unique_ids": len(set(ids)),
            "questions_per_record": n_questions, "problems": problems[:20]}
    (args.out / "conversion_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "total_cases_found": len(cases),
        "records_3q": len(records),
        "records_3q_non_pseudo_labels": len(strong3),
        "records_5q_defensible": len(defensible_5q),
        "records_5q_pseudo": len(pseudo_5q),
        "replay_classes": replay_summary["replay_classes"],
        "repair_evidence_verified": replay_summary["repair_evidence_verified"],
        "buggy_verdicts": replay_summary["buggy_verdicts"],
        "label_sources": report["label_source_distribution"],
        "root_cause_distribution": report["root_cause_distribution"],
        "next_action_distribution": report["next_action_distribution"],
        "leakage_findings": report["leakage"]["findings_total"],
    }, indent=2))


if __name__ == "__main__":
    main()
