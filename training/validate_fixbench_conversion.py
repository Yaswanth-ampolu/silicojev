#!/usr/bin/env python3
"""Independent validation of the Fixbench-RTL conversion.

Reads only the files on disk — the raw dataset, the replay cache and the emitted
records — and re-derives every checkable fact rather than trusting the converter:

* record ids are recomputed from the four source fields;
* the state's RTL and testbench are checked against the raw file line by line;
* the state's replay block is checked against the *buggy* side of the replay
  cache, so a corrected-side transcript or command cannot pass unnoticed;
* question and gold structure is checked against the canonical criteria in
  `training/prepare_dataset.py` and against the brief's own wording;
* probability vectors are checked for range, ordering and normalisation;
* every label source is checked against the replay evidence that is claimed.

Exit code is non-zero if any check fails. Nothing here writes to the conversion
directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "dataset/raw/hf/Fixbench-RTL/Fixbench-RTL.json"
DEFAULT_REPLAY = Path(__file__).resolve().parent / "fixbench_replay_cache/replay_results.json"
DEFAULT_DIR = REPO_ROOT / "dataset/converted/fixbench_rtl"
PREPARE_DATASET = Path(__file__).resolve().parent / "prepare_dataset.py"

SOURCE = "Fixbench-RTL"
GROUP = "fixbench_rtl"
BASE_QUESTIONS = ("next_action", "root_cause_type", "evidence_sufficient")
SCORE_QUESTIONS = ("risk", "urgency")
# Verbatim from the brief.
EVIDENCE_CRITERIA = {
    "false": "Evidence is insufficient or unvalidated",
    "true": "Evidence is sufficient and supported by a tool or trusted label",
}
# The canonical score rubric, imported from the project's own score auditor so
# this check cannot drift from the merged 6,248-case set.
REVIEW_SCORES = Path(__file__).resolve().parent / "review_pseudo_scores.py"
LABEL_SOURCES = {"verified_repair", "manual_bug_label", "codex_pseudo_unverified"}
FILES = {"records_3q.jsonl": 3, "records_5q.jsonl": 5,
         "records_5q_pseudo_unverified.jsonl": 5}
PROB_TOLERANCE = 0.002
TRUNCATION_MARK = "...[truncated]"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_criteria() -> tuple[dict, dict, list, list]:
    """Canonical criteria, imported from the training-time definitions."""
    prepare = load_module(PREPARE_DATASET, "_silicojev_prepare")
    scores = load_module(REVIEW_SCORES, "_silicojev_scores")
    return (prepare.ACTION_CRITERIA, prepare.ROOT_CAUSE_CRITERIA,
            scores.RISK_CRITERIA, scores.URGENCY_CRITERIA)


def stripped_lines(text: str, drop_clipped_tail: bool = True) -> list[str]:
    rows = (text or "").splitlines()
    if drop_clipped_tail and rows and rows[-1].endswith(TRUNCATION_MARK):
        rows = rows[:-1]
    return [row.strip() for row in rows if row.strip()]


def stable_case_id(index: int, case: dict) -> str:
    """Recomputed exactly as the brief's contract requires."""
    blob = json.dumps({k: case.get(k) for k in ("bug", "buggycode", "correctcode", "testbench")},
                      sort_keys=True, ensure_ascii=False).encode()
    return f"case_{index:03d}_{hashlib.sha256(blob).hexdigest()[:8]}"


def step_streams(step: dict | None) -> str:
    """Both streams of one tool step, labelled, stdout first — as the state stores them."""
    parts = []
    for name in ("stdout", "stderr"):
        text = ((step or {}).get(name) or "").strip()
        if text:
            parts.append(f"[{name}]\n{text}")
    return "\n".join(parts)


def attempt_stream(attempt: dict | None) -> str:
    """Everything a single replay attempt printed, compile and run combined."""
    if not attempt:
        return ""
    return "\n".join(filter(None, [step_streams(attempt.get("compile")),
                                   step_streams(attempt.get("run"))]))


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    raw_path, replay_path, out_dir = DEFAULT_RAW, DEFAULT_REPLAY, DEFAULT_DIR
    for path in (raw_path, replay_path, PREPARE_DATASET):
        if not path.is_file():
            print(f"missing required input: {path}", file=sys.stderr)
            return 2

    raw = json.loads(raw_path.read_text())
    replay_payload = json.loads(replay_path.read_text())
    replay = {r["index"]: r for r in replay_payload["results"]}
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    action_criteria, root_criteria, risk_criteria, urgency_criteria = load_criteria()

    check(replay_payload.get("raw_sha256") == raw_sha,
          "replay cache was built from a different raw file")
    check(len(raw) == 100, f"raw dataset holds {len(raw)} cases, expected 100")

    # ---------------------------------------------------------------- files
    rows_by_file: dict[str, list[dict]] = {}
    for name, n_questions in FILES.items():
        path = out_dir / name
        check(path.is_file(), f"{name} is missing")
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows_by_file[name] = rows
        ids = [row["id"] for row in rows]
        duplicates = [i for i, n in Counter(ids).items() if n > 1]
        check(not duplicates, f"{name}: duplicate ids within the file: {duplicates[:5]}")
        for row in rows:
            rid = row["id"]
            if not set(row) >= {"id", "source", "source_group", "state", "questions",
                                "gold", "outcome", "provenance"}:
                problems.append(f"{name}/{rid}: missing top-level keys")
                continue
            check(row["source"] == SOURCE, f"{name}/{rid}: bad source {row['source']!r}")
            check(row["source_group"] == GROUP,
                  f"{name}/{rid}: bad source_group {row['source_group']!r}")
            check(isinstance(row["outcome"], dict) and isinstance(row["provenance"], dict),
                  f"{name}/{rid}: outcome/provenance must be objects")
            try:
                state = json.loads(row["state"])
                questions = json.loads(row["questions"])
                gold = json.loads(row["gold"])
            except json.JSONDecodeError as exc:
                problems.append(f"{name}/{rid}: state/questions/gold is not JSON: {exc}")
                continue
            check(isinstance(state, dict) and isinstance(questions, dict) and isinstance(gold, dict),
                  f"{name}/{rid}: state/questions/gold must be JSON objects")
            check(len(questions) == n_questions,
                  f"{name}/{rid}: {len(questions)} questions, expected {n_questions}")
            check(set(questions) == set(gold),
                  f"{name}/{rid}: question ids {sorted(questions)} != gold ids {sorted(gold)}")
            expected = set(BASE_QUESTIONS) | (set(SCORE_QUESTIONS) if n_questions == 5 else set())
            check(set(questions) == expected,
                  f"{name}/{rid}: question set {sorted(set(questions) ^ expected)} differs")

            # ---- question definitions
            for qid, question in questions.items():
                criteria = question.get("criteria")
                if question.get("type") == "score":
                    expected_levels = risk_criteria if qid == "risk" else urgency_criteria
                    check(isinstance(criteria, list) and len(criteria) == 4,
                          f"{name}/{rid}/{qid}: score criteria must be 4 ordered levels")
                    check(list(criteria or []) == list(expected_levels),
                          f"{name}/{rid}/{qid}: score levels are not the required wording")
                    check(question.get("instructions"),
                          f"{name}/{rid}/{qid}: no instructions")
                else:
                    check(isinstance(criteria, dict),
                          f"{name}/{rid}/{qid}: criteria must be a dictionary")
                    if qid == "next_action":
                        check(criteria == action_criteria,
                              f"{name}/{rid}/{qid}: criteria differ from the canonical set")
                    elif qid == "root_cause_type":
                        check(criteria == root_criteria,
                              f"{name}/{rid}/{qid}: criteria differ from the canonical set")
                    elif qid == "evidence_sufficient":
                        check(criteria == EVIDENCE_CRITERIA,
                              f"{name}/{rid}/{qid}: criteria are not the required wording")

            # ---- gold
            for qid, entry in gold.items():
                probs = entry.get("probabilities")
                check(isinstance(probs, dict),
                      f"{name}/{rid}/{qid}: probabilities must be a mapping")
                if not isinstance(probs, dict):
                    continue
                expected_keys = ([str(i) for i in range(4)]
                                 if questions[qid].get("type") == "score"
                                 else set(questions[qid]["criteria"]))
                check(set(probs) == set(expected_keys),
                      f"{name}/{rid}/{qid}: probability keys do not match the question")
                values = list(probs.values())
                check(all(isinstance(v, (int, float)) and v >= 0 for v in values),
                      f"{name}/{rid}/{qid}: negative or non-numeric probability")
                total = sum(v for v in values if isinstance(v, (int, float)))
                check(abs(total - 1.0) <= PROB_TOLERANCE,
                      f"{name}/{rid}/{qid}: probabilities sum to {total}")
                check(entry.get("label_source") in LABEL_SOURCES,
                      f"{name}/{rid}/{qid}: unknown label_source {entry.get('label_source')!r}")
                if qid in SCORE_QUESTIONS and "pseudo" in name:
                    check(entry.get("label_source") == "codex_pseudo_unverified",
                          f"{name}/{rid}/{qid}: a pseudo score file must say "
                          f"codex_pseudo_unverified")
                if questions[qid].get("type") == "score":
                    # same objective checks review_pseudo_scores.py applies
                    expectation = sum(int(k) * v for k, v in probs.items())
                    check(abs(float(entry.get("score", -1)) - expectation) <= 1e-5,
                          f"{name}/{rid}/{qid}: score is not the expectation of the vector")
                    argmax = max(probs, key=lambda k: (probs[k], -int(k)))
                    check(str(entry.get("label")) == argmax,
                          f"{name}/{rid}/{qid}: label is not the argmax of the vector")
                    confidence = entry.get("confidence")
                    check(isinstance(confidence, (int, float))
                          and 0.0 <= float(confidence) <= 1.0,
                          f"{name}/{rid}/{qid}: confidence is not a probability")

            # ---- provenance must not present pseudo scores as gold
            prov = row["provenance"]
            for qid in SCORE_QUESTIONS:
                if qid in prov:
                    check(prov[qid].get("defensible") is False or "pseudo" not in name,
                          f"{name}/{rid}/{qid}: marked defensible in a pseudo file")
                    if prov[qid].get("defensible") is False:
                        check(prov[qid].get("label_source") == "codex_pseudo_unverified",
                              f"{name}/{rid}/{qid}: indefensible score with a non-pseudo source")

            # ---- state, ids and evidence
            index = prov.get("source_index")
            check(isinstance(index, int) and 0 <= index < len(raw),
                  f"{name}/{rid}: source_index {index!r} out of range")
            if not isinstance(index, int) or not 0 <= index < len(raw):
                continue
            case = raw[index]
            expected_id = f"fixbench:{stable_case_id(index, case)}"
            check(rid == expected_id,
                  f"{name}/{rid}: id is not re-derivable from source case {index} "
                  f"(expected {expected_id})")
            check(state.get("case_id") == stable_case_id(index, case),
                  f"{name}/{rid}: state.case_id does not match the source case")
            check(state.get("previous_actions") == [],
                  f"{name}/{rid}: previous_actions must be an empty list")
            check(list(state) != [] and "buggy_rtl" in state and "testbench" in state,
                  f"{name}/{rid}: state is missing buggy_rtl/testbench")

            # the state's RTL and testbench must be the source revisions, line for line
            for field, source in (("buggy_rtl", case["buggycode"]), ("testbench", case["testbench"])):
                state_lines = stripped_lines(state.get(field) or "")
                source_lines = stripped_lines(source, drop_clipped_tail=False)
                check(state_lines == source_lines[: len(state_lines)],
                      f"{name}/{rid}: state.{field} is not a prefix of the source revision")

            blob = json.dumps(state, ensure_ascii=False)
            check(case["correctcode"].strip() not in blob,
                  f"{name}/{rid}: correctcode text found in the state")
            repaired_only = {l.strip() for l in case["correctcode"].splitlines()
                             if len(l.strip()) >= 12 and not l.strip().startswith("//")}
            repaired_only -= {l.strip() for l in case["buggycode"].splitlines()}
            repaired_only -= {l.strip() for l in case["testbench"].splitlines()}
            for field in ("buggy_rtl", "testbench"):
                leaked = set(stripped_lines(state.get(field) or "")) & repaired_only
                check(not leaked,
                      f"{name}/{rid}: repaired-only line in state.{field}: "
                      f"{sorted(leaked)[:2]}")
            for qid in ("compiler_output", "simulation_output"):
                leaked = set(stripped_lines(state["replay"].get(qid) or "")) & repaired_only
                check(not leaked,
                      f"{name}/{rid}: repaired-only line in replay.{qid}: {sorted(leaked)[:2]}")
            for banned in ("correctcode", "corrected_code", "fixed_code", "diff", "patch"):
                check(f'"{banned}"' not in blob, f"{name}/{rid}: banned key {banned!r} in state")
            check('"fix.sv"' not in blob, f"{name}/{rid}: corrected file referenced in state")

            # ---- the state's replay block must be the buggy side of the cache
            cached = replay.get(index)
            check(cached is not None, f"{name}/{rid}: no replay result for case {index}")
            if cached is None:
                continue
            block = state["replay"]
            chosen = cached["buggy"]
            check(block.get("status") == chosen["verdict"],
                  f"{name}/{rid}: state replay status {block.get('status')!r} != cache "
                  f"buggy verdict {chosen['verdict']!r}")
            for field, step_key, cache_key in (
                    ("compile_command", "compile", "cmd"), ("run_command", "run", "cmd"),
                    ("compile_returncode", "compile", "returncode"),
                    ("run_returncode", "run", "returncode")):
                step = chosen.get(step_key) or {}
                cached_value = " ".join(step.get(cache_key) or []) if cache_key == "cmd" \
                    else step.get(cache_key)
                check(block.get(field) == (cached_value if cached_value not in (None, "") else None),
                      f"{name}/{rid}: state replay {field} {block.get(field)!r} != cache "
                      f"{cached_value!r}")
            for field, step_key in (("compiler_output", "compile"), ("simulation_output", "run")):
                stream = step_streams(chosen.get(step_key))
                state_value = (block.get(field) or "").rstrip(TRUNCATION_MARK)
                check(stream.startswith(state_value) or state_value.startswith(stream),
                      f"{name}/{rid}: state replay {field} is not the cached buggy stream")
            # No line that appears *only* on the corrected side may reach the state.
            # Lines shared with the buggy side are the testbench's own output and
            # are legitimately present on both sides.
            buggy_lines = set(stripped_lines(attempt_stream(chosen), drop_clipped_tail=False))
            corrected_lines = set()
            for attempt in (cached.get("correct_by_backend") or {}).values():
                corrected_lines |= set(stripped_lines(attempt_stream(attempt),
                                                      drop_clipped_tail=False))
            corrected_only = {line for line in corrected_lines - buggy_lines if len(line) >= 20}
            for line in sorted(corrected_only & set(stripped_lines(blob, drop_clipped_tail=False))):
                problems.append(f"{name}/{rid}: corrected-only transcript line in state: {line[:60]}")

            # ---- label semantics against the replay evidence
            prov_verified = prov.get("repair_evidence_verified")
            check(prov_verified == cached["failure_attributable_to_bug"],
                  f"{name}/{rid}: repair_evidence_verified disagrees with the replay cache")
            check(prov.get("replay_class") == cached["attribution"],
                  f"{name}/{rid}: replay_class disagrees with the replay cache")
            if not cached["failure_attributable_to_bug"]:
                for qid, entry in gold.items():
                    check(entry.get("label_source") != "verified_repair",
                          f"{name}/{rid}/{qid}: verified_repair on a case whose failure is "
                          f"not attributable ({cached['attribution']})")
            if cached["buggy_passes_on_backends"]:
                check("testbench_does_not_detect_the_bug" in (prov.get("trust_flags") or []),
                      f"{name}/{rid}: testbench passed the buggy revision but the flag is absent")

    # ---------------------------------------------- coverage of the source cases
    three = rows_by_file.get("records_3q.jsonl", [])
    indices = sorted(r["provenance"]["source_index"] for r in three)
    check(indices == list(range(len(raw))),
          "records_3q.jsonl does not cover every source case exactly once")
    five = rows_by_file.get("records_5q.jsonl", []) + \
        rows_by_file.get("records_5q_pseudo_unverified.jsonl", [])
    five_indices = sorted(r["provenance"]["source_index"] for r in five)
    check(five_indices == list(range(len(raw))),
          "the 5q files together do not cover every source case exactly once")

    # 5q records must be the five-question extension of the same 3q record
    three_by_id = {r["id"]: r for r in three}
    for row in five:
        base = three_by_id.get(row["id"])
        check(base is not None, f"{row['id']}: 5q record has no 3q counterpart")
        if base is None:
            continue
        five_gold = json.loads(row["gold"])
        base_gold = json.loads(base["gold"])
        for qid in BASE_QUESTIONS:
            check(five_gold.get(qid) == base_gold.get(qid),
                  f"{row['id']}/{qid}: the three-question projection differs from the 3q record")
        check(json.loads(row["state"]) == json.loads(base["state"]),
              f"{row['id']}: the 5q state differs from the 3q state")

    # ------------------------------------------------------------- usage sets
    usage: dict = {}
    trusted: set[str] = set()
    verified_ids: set[str] = set()
    usage_path = out_dir / "usage_sets.json"
    check(usage_path.is_file(), "usage_sets.json is missing")
    if usage_path.is_file():
        usage = json.loads(usage_path.read_text())
        all_ids = {r["id"] for r in three}
        trusted = set(usage.get("replay_verified_3q", []))
        trusted_any = set(usage.get("replay_verified_unflagged_3q", []))
        all_verified = set(usage.get("repair_evidence_verified_3q", []))
        flagged = set(usage.get("flagged_for_review", []))
        check(trusted <= all_ids, "usage_sets: a trusted id is not in records_3q.jsonl")
        check(trusted <= trusted_any,
              "usage_sets: replay_verified_3q is not a subset of the any-label set")
        check(not (trusted & flagged),
              f"usage_sets: {len(trusted & flagged)} flagged ids appear in the trusted set")
        verified_ids = {r["id"] for r in three if r["provenance"]["repair_evidence_verified"]}
        check(trusted_any <= verified_ids,
              "usage_sets: an unflagged trusted id is not repair-evidence verified")
        check(trusted_any <= all_verified,
              "usage_sets: the unflagged set is not a subset of the verified set")
        check(all_verified == verified_ids,
              "usage_sets: repair_evidence_verified_3q is not exactly the verified set")
        check(not (trusted_any & flagged),
              "usage_sets: flagged ids appear in the unflagged set")
        nonpseudo = {r["id"] for r in three
                     if all(json.loads(r["gold"])[q]["label_source"] != "codex_pseudo_unverified"
                            for q in BASE_QUESTIONS)}
        check(trusted == trusted_any & nonpseudo,
              "usage_sets: replay_verified_3q is not the unflagged non-pseudo set")
        check(set(usage.get("defensible_5q", [])) <= all_ids,
              "usage_sets: a defensible 5q id is not in records_3q.jsonl")
        by_class = usage.get("replay_class_3q", {})
        flat = [i for ids in by_class.values() for i in ids]
        check(sorted(flat) == sorted(all_ids),
              "usage_sets: replay_class_3q does not partition the 3q ids")
        notes.append(f"usage_sets: verified {len(all_verified)}, unflagged "
                     f"{len(trusted_any)}, trusted {len(trusted)}, flagged {len(flagged)}")

    # ------------------------------------------------- lineage grouping
    # Independent re-derivation of the family claims: each claimed signal must
    # actually hold for the members, and no pair of cases sharing a harness may
    # be left in different groups.
    split_path = out_dir / "split_groups.json"
    check(split_path.is_file(), "split_groups.json is missing")
    if split_path.is_file() and usage_path.is_file():
        split = json.loads(split_path.read_text())
        all_ids = {r["id"] for r in three}
        by_id = {r["id"]: r for r in three}
        case_group = split.get("case_group") or {}
        check(set(case_group) == all_ids,
              "split_groups: case_group does not cover exactly the 3q ids")

        def norm_tb(text: str) -> str:
            text = re.sub(r"//.*", "", text or "")
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
            return re.sub(r"\s+", " ", re.sub(r"\d+", "N", text))

        def pr_line(text: str) -> str:
            stripped = (text or "").strip()
            return re.sub(r"\s+", " ", stripped.splitlines()[0].strip().lower())[:120] \
                if stripped else ""

        def dlines(text: str) -> set[str]:
            return {l.strip() for l in (text or "").splitlines() if len(l.strip()) > 8}

        def case_rtl(rid: str) -> dict:
            # Unknown ids are a problem the checks report, not a crash.
            entry = by_id.get(rid)
            if entry is None:
                return {"testbench": "", "bug": "", "buggycode": ""}
            return raw[entry["provenance"]["source_index"]]

        def signal_holds(name: str, members: list[str]) -> bool:
            if name == "same_testbench":
                return len({norm_tb(case_rtl(m)["testbench"]) for m in members}) == 1
            if name == "same_upstream_pr_line":
                return len({pr_line(case_rtl(m)["bug"]) for m in members}) == 1
            if name == "shared_module_name":
                sets = [set(re.findall(r"^\s*module\s+([A-Za-z_]\w*)",
                                       case_rtl(m)["buggycode"], re.M)) for m in members]
                return bool(set.intersection(*sets)) if sets else False
            if name == "same_design_code":
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = dlines(case_rtl(members[i])["buggycode"]), \
                            dlines(case_rtl(members[j])["buggycode"])
                        if a and b and len(a & b) / len(a | b) >= 0.4:
                            return True
                return False
            return False

        families = split.get("families") or []
        seen: dict[str, str] = {}
        for family in families:
            members = family.get("members") or []
            check(len(members) == family.get("size"),
                  f"split_groups/{family.get('group')}: size does not match members")
            check(len(members) > 1, f"split_groups/{family.get('group')}: family of one")
            check(all(m in all_ids for m in members),
                  f"split_groups/{family.get('group')}: member outside the 3q ids")
            uniform = family.get("signals_uniform") or []
            partial = family.get("signals_partial") or {}
            if isinstance(partial, str):        # guard the older flat shape
                problems.append(f"split_groups/{family.get('group')}: signals are not "
                                f"split into uniform and partial")
                partial = {}
            check(bool(uniform or partial),
                  f"split_groups/{family.get('group')}: no signal recorded")
            for name in uniform:
                check(name in split.get("signals", {}),
                      f"split_groups/{family.get('group')}: unexplained signal {name!r}")
                check(signal_holds(name, members),
                      f"split_groups/{family.get('group')}: signal {name!r} is claimed for "
                      f"the whole family but does not hold for all its members")
            for name, subsets in partial.items():
                check(name in split.get("signals", {}),
                      f"split_groups/{family.get('group')}: unexplained signal {name!r}")
                for subset in subsets:
                    check(len(subset) >= 2 and set(subset) <= set(members),
                          f"split_groups/{family.get('group')}: partial block {subset} is "
                          f"not a subset of the family")
                    check(signal_holds(name, subset),
                          f"split_groups/{family.get('group')}: signal {name!r} does not "
                          f"hold within its claimed block {subset}")
            for member in members:
                check(case_group.get(member) == family.get("group"),
                      f"split_groups/{member}: case_group disagrees with the family list")
                if member in seen:
                    problems.append(f"split_groups/{member}: in two families "
                                    f"({seen[member]}, {family.get('group')})")
                seen[member] = family.get("group")
        grouped = {g: sorted(m for m, gg in case_group.items() if gg == g)
                   for g in set(case_group.values())}
        check(sorted(m for members in grouped.values() if len(members) > 1
                     for m in members) == sorted(seen),
              "split_groups: the family list does not cover every multi-case group")

        # Same harness is the strongest signal: a missed pair would be a leak.
        by_tb: dict[str, set[str]] = {}
        for rid in all_ids:
            by_tb.setdefault(norm_tb(case_rtl(rid)["testbench"]), set()).add(rid)
        for key, members in by_tb.items():
            if len(members) > 1:
                check(len({case_group.get(m, m) for m in members}) == 1,
                      f"split_groups: cases sharing a harness are split across groups: "
                      f"{sorted(members)}")

        suggested = split.get("suggested_split") or {}
        train = set(suggested.get("train") or [])
        test = set(suggested.get("test") or [])
        check(train and test, "split_groups: the suggested split has an empty side")
        check(not (train & test), "split_groups: train and test overlap")
        check(train | test == trusted,
              "split_groups: the suggested split is not exactly the trusted set")
        shared_groups = ({case_group.get(m, m) for m in train}
                         & {case_group.get(m, m) for m in test})
        check(not shared_groups,
              f"split_groups: a lineage family spans train and test: {sorted(shared_groups)}")
        check(suggested.get("family_disjoint") is True,
              "split_groups: suggested_split does not report family disjointness")
        notes.append(f"split_groups: {len(families)} families covering {len(seen)} cases, "
                     f"{len(grouped)} groups; suggested split train {len(train)} / test {len(test)}")

    # ------------------------------------------------- trusted-set provenance
    sheet_path = out_dir / "trusted_label_provenance.md"
    check(sheet_path.is_file(), "trusted_label_provenance.md is missing")
    if sheet_path.is_file() and usage_path.is_file():
        sheet = sheet_path.read_text()
        listed = set(re.findall(r"^\| `(fixbench:[^`]+)`", sheet, re.M))
        detailed = set(re.findall(r"^- \*\*(fixbench:[^`*]+)\*\*", sheet, re.M))
        check(listed == trusted,
              "provenance sheet: the table does not list exactly the trusted set "
              f"(missing {sorted(trusted - listed)[:3]}, extra {sorted(listed - trusted)[:3]})")
        check(detailed == trusted,
              "provenance sheet: the trace section does not cover the trusted set")
        notes.append(f"provenance sheet: {len(listed)} trusted cases documented")

    # ------------------------------------------------------ pseudo quarantine
    quarantine_hits: list[str] = []
    normalised = REPO_ROOT / "dataset/normalized"
    for path in sorted(normalised.rglob("*.jsonl")):
        text = path.read_text()
        for rid in re.findall(r"fixbench:[A-Za-z0-9_]+", text):
            quarantine_hits.append(f"{path.name}:{rid}")
    check(not quarantine_hits,
          f"pseudo quarantine: Fixbench ids reached the normalized dataset: {quarantine_hits[:3]}")
    pseudo_rows = rows_by_file.get("records_5q_pseudo_unverified.jsonl", [])
    defensible_rows = rows_by_file.get("records_5q.jsonl", [])
    check(len(pseudo_rows) > 0,
          "the pseudo score file is empty; the quarantine note would be misleading")
    check((usage.get("defensible_5q") or []) == [r["id"] for r in defensible_rows],
          "usage_sets: defensible_5q does not match records_5q.jsonl")
    check(not ({r["id"] for r in pseudo_rows} & {r["id"] for r in defensible_rows}),
          "a case is in both the defensible and the pseudo five-question file")
    check(not ({r["id"] for r in pseudo_rows} - {r["id"] for r in three}),
          "a pseudo five-question record has no three-question counterpart")
    # A pseudo risk/urgency estimate must never be presented as gold: both score
    # labels carry the pseudo source and the defensible flag is false.
    for row in pseudo_rows:
        prov = row["provenance"]
        for qid in SCORE_QUESTIONS:
            check(prov.get(qid, {}).get("label_source") == "codex_pseudo_unverified",
                  f"{row['id']}/{qid}: a pseudo record does not declare its source")
            check(prov.get(qid, {}).get("defensible") is False,
                  f"{row['id']}/{qid}: a pseudo estimate is marked defensible")
    urgency_levels = {json.loads(r["gold"])["urgency"]["label"] for r in pseudo_rows}
    risk_levels = {json.loads(r["gold"])["risk"]["label"] for r in pseudo_rows}
    notes.append(f"quarantine: pseudo 5q ids absent from dataset/normalized; "
                 f"risk levels {sorted(risk_levels)}, urgency levels {sorted(urgency_levels)}")

    # ---------------------------------------------------------------- report
    report_path = out_dir / "conversion_report.json"
    check(report_path.is_file(), "conversion_report.json is missing")
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        check(report.get("total_cases_found") == len(raw),
              "report: total_cases_found does not match the raw file")
        check(report.get("source_sha256") == raw_sha,
              "report: source_sha256 does not match the raw file")
        counts = report.get("record_counts", {})
        for name, rows in rows_by_file.items():
            check(counts.get(name) == len(rows),
                  f"report: record_counts[{name}]={counts.get(name)} but the file holds {len(rows)}")
        classes = Counter(r["provenance"]["replay_class"] for r in three)
        check(report.get("replay", {}).get("replay_classes") == dict(sorted(classes.items())),
              "report: replay_classes does not match the records")
        verified = sum(1 for r in three if r["provenance"]["repair_evidence_verified"])
        check(report.get("replay", {}).get("repair_evidence_verified") == verified,
              "report: repair_evidence_verified does not match the records")
        leak_total = sum(len(v) for v in report.get("leakage", {})
                         .get("cases_with_findings", {}).values())
        check(leak_total == 0, f"report: {leak_total} leakage findings recorded")
        notes.append(f"report: {json.dumps(report.get('replay', {}).get('replay_classes', {}))}")

    print(json.dumps({
        "checked_files": {name: len(rows) for name, rows in rows_by_file.items()},
        "problems": len(problems),
        "problem_detail": problems[:40],
        "notes": notes,
    }, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
