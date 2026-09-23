# Fixbench-RTL → SilicoJev conversion

Conversion of all 100 records of **Fixbench-RTL** into SilicoJev/Laya-compatible
decision records, with a real replay of every case against both the buggy and the
corrected revision.

This directory is a **standalone audit artefact**. It is *not* merged into
`dataset/normalized/` (the 6,248-case SilicoJev set) and has not been fed to Laya.

| File | Records | Questions |
|---|---|---|
| `records_3q.jsonl` | 100 | 3 (`next_action`, `root_cause_type`, `evidence_sufficient`) |
| `records_5q.jsonl` | 0 | 5 (adds `risk`, `urgency`) — empty on purpose, see below |
| `records_5q_pseudo_unverified.jsonl` | 100 | 5, rubric-script scores marked `codex_pseudo_unverified` |
| `records_5q_distilled_pseudo_unverified.jsonl` | 100 | 5, both scores model-judged and marked `codex_pseudo_unverified` |
| `distillation_report.json` | — | per-batch counts, replay/trust tiers, score distributions, prior-label disagreements, validation |
| `distillation_progress.jsonl` | 100 | crash-safe progress; same rows as the distilled output |
| `usage_sets.json` | — | the recommended id lists (trusted vs review vs experiment) |
| `split_groups.json` | — | lineage families (near-duplicates) and a family-disjoint split sketch |
| `trusted_label_provenance.md` | — | per-question provenance for the 35 trusted cases |
| `conversion_report.json` | — | counts, replay classes, rule traces, post-write audit |
| `README.md` | — | this file |

`records_3q.jsonl` carries all 100 converted cases with an honest per-question
label source; the filtered views that training should use are the id lists in
`usage_sets.json`, so no record is duplicated across files.

### Recommended use

The three gates nest, so the reason a case is left out is always visible:

| Set | Records | Where |
|---|---|---|
| repair evidence verified | **54** | `usage_sets.repair_evidence_verified_3q` |
| … and unflagged | 48 | `usage_sets.replay_verified_unflagged_3q` |
| … and no pseudo label | **35** | `usage_sets.replay_verified_3q` ← the trusted set |
| at least one label is a guess | 55 | `usage_sets.unverified_label_3q` |
| flagged for review | 52 | `usage_sets.flagged_for_review` |
| separate experiment (5q, pseudo scores) | 100 | `records_5q_pseudo_unverified.jsonl` |

"Trusted" means: the supplied testbench fails the buggy revision under a
simulator in which the corrected revision passes it, none of the three labels is
a full guess, and no trust flag applies. It does **not** mean the judgement labels
are certainly right — `next_action` and `root_cause_type` remain readings of the
description and the replay even in the trusted set. The exact composition:

```
100 converted
├─ 54  repair evidence verified (the failure tracks the defect)
│   │      6 excluded by a trust flag ................ -> 48 unflagged
│   │      61, 63, 83, 87, 89, 90 carry bug_text_is_upstream_pr_text,
│   │      security_relevant_content or testbench_does_not_detect_the_bug
│   └─ 48  verified and unflagged
│          13 excluded for a pseudo label ............ -> 35 trusted
└─ 46  not verified (21 testbench does not detect the bug, 13 cannot build,
       11 failure survives the repair, 1 no usable verdict)
```

### How to use this without fooling yourself

Four rules, in the order they bite:

1. **Select by id, never by file.** `records_3q.jsonl` holds all 100 cases so
   every conversion is auditable in one place, but only 35 meet the strictest
   gate. Reading the file as a 100-case training set treats 65 cases as
   supervision they cannot carry: 46 have no verified repair evidence and 19
   more carry a trust flag or a pseudo label. Use
   `usage_sets.json → replay_verified_3q` for anything you intend to score.
2. **"Replay verified" is about the repair, not about the judgement.**
   `repair_evidence_verified` means the buggy revision fails the supplied
   testbench under a simulator in which the corrected revision passes it. It says
   nothing about whether `next_action` or `root_cause_type` is the only right
   reading. Inside the trusted 35, `next_action` is `verified_repair` for only 3
   cases and `manual_bug_label` for the other 32; `root_cause_type` is
   `verified_repair` for 24 and `manual_bug_label` for 11. Inspect
   `trusted_label_provenance.md` — it lists each trusted case's winning label,
   its source, the evidence channel that drove it, and the rules that fired —
   before calling any of it ground truth.
3. **Keep the pseudo score file quarantined.** Every record in
   `records_5q_pseudo_unverified.jsonl` carries the same rubric defaults (risk 1,
   urgency 0) because no case states a deadline or a production impact. Training
   on that file teaches two constants and would misstate a rubric application as
   supervision. It is for inspecting the rubric, not for fitting. The file's
   score labels are all `codex_pseudo_unverified`, `defensible` is `false`, and
   the validator re-checks that its ids never reach `dataset/normalized/`.
4. **Split by lineage, not by row.** The 100 cases are not 100 independent draws.
   `split_groups.json` records nine families covering 30 cases — identical
   harnesses, near-identical designs, or the same upstream PR text — plus the
   signals behind each family (uniform vs partial, so a claim that holds only for
   a sub-block is never stated as if it held for the whole family). A row-wise
   split puts the same design on both sides. The suggested split there is
   deterministic and family-disjoint over the trusted 35 (26 train / 9 test), and
   is a smoke test, not a measurement: 35 cases across 35 groups is far too
   little to estimate anything.

Family grouping happens to be moot for the trusted 35 today — 0 of the 30 family
cases are in that set (29 of them are flagged for review; the one unflagged
member, `case_091`, still fails the non-pseudo gate) — but it is not moot for the
wider corpus: 29 of the flagged 52 sit in families, so any evaluation over the
100 records or the 48-case unflagged set must group them.

## Source and provenance

- Dataset: <https://huggingface.co/datasets/KSU-HW-SEC/Fixbench-RTL> (public, not
  gated), downloaded to `dataset/raw/hf/Fixbench-RTL/Fixbench-RTL.json`.
- sha256 `a0c6543afacb4882a8792fabb3fa262d6351401a8c6945363c5d35173733d524`,
  1,359,485 bytes, a JSON list of exactly 100 objects, each with exactly the keys
  `bug`, `buggycode`, `correctcode`, `testbench`.
- Licence: CC-BY-4.0 (from the dataset card); upstream source licences also apply
  to the RTL, which comes from real repositories (CVA6, ibex, cv32e40p, OpenTitan,
  Wally, Trivium/Bivium primitives and others).
- **The raw file was not modified.** The converter only reads it; the replay wrote
  its scratch copies under `training/fixbench_replay_cache/`.
- `provenance.synthetic` is `false`: these are real bugs, most of them injected or
  transcribed against real designs rather than generated mutations.

### Record identity

Fixbench ships **no per-case identifier**, so one is derived:

```
case_<index:03d>_<sha256(canonical json of {bug, buggycode, correctcode, testbench})[:8]>
```

The record id is `fixbench:<that>`. The hash makes it verifiable that the id
refers to those exact four fields, and the index keeps it readable. The
independent validator recomputes every id from the raw file and fails if it does
not match, so the mapping is checked rather than asserted.

## State contract

The state is a JSON object with `case_id`, `source_index`, `bug_description`,
`buggy_rtl`, `testbench`, `replay`, `source_metadata` and `previous_actions: []`.

Permitted content, and nothing else:

- the case id, index and source metadata (dataset, file, sha256, licence,
  detected top module, toolchain versions);
- the bug description, verbatim;
- the buggy revision's RTL and the supplied testbench, verbatim;
- the **buggy-side** replay: status, per-simulator verdicts, the exact compile and
  run commands, return codes, timeouts, and both output streams;
- `previous_actions: []`.

Excluded by construction: `correctcode`, any diff against it, the corrected
revision's commands, its log or its verdict, any statement that a corrected
version builds or passes, and the replay attribution class (which is derived by
comparing the two revisions and therefore lives in `provenance` only).

`correctcode` is used for exactly three things: as the repair target in `outcome`
(with its sha256), as the comparison that proves no repaired content reached the
state, and as the second half of the replay pair that decides whether a failure
tracks the defect.

## Replay procedure

Every case was replayed. Two simulators, tried in this order per revision:

```
iverilog -g2012 -o <side>.vvp <side>.sv tb.sv      # Icarus Verilog 12.0
vvp <side>.vvp

vlib work                                          # Questa Sim 2021.2_1
vlog -sv -quiet -work work <side>.sv tb.sv
vsim -c -quiet -do "run -all; quit -f" <top>
```

- Each revision gets its own scratch directory with its own `work` library, so a
  stale compiled model can never be loaded by the other side.
- The **design file is compiled before the testbench**, because 13 supplied
  testbenches `import` a package that is declared in the design file and both
  simulators resolve package imports while parsing. Compiling the testbench first
  produced spurious `tb.sv: syntax error` results and hid a genuine failure in
  case 61.
- The top module is detected from the testbench (modules it declares but does not
  instantiate, conventional names first) instead of assuming `tb`. Testbenches in
  this dataset declare `testbench`, `bus`, `fpgaTop` and others.
- `LM_LICENSE_FILE=/usr/local/questasim/license.dat`.

Verdict vocabulary per revision: `passed`, `failed`, `compile_failed`, `timeout`,
`tool_error`, `unknown`, `tool_missing`. A tool failure ("Error loading design",
missing licence) is **not** a verdict about the design and never counts as a
failure; it is recorded as `tool_error` and retried with the other simulator.
Signal parsing reads the testbench's own verdict forms: `=====Passed=====`,
`=====Failed: <reason>=====`, `=====Test completed with N / M failures=====`
(and the denominator-less form), plus bare `ERROR:`/`Error:` lines and Questa's
`# Errors: N` summary. The last marker in a transcript wins, and both signals are
kept, so a transcript that says both things is visible rather than silently
resolved.

### Replay classes

A failure is only evidence about the defect if the repair target removes it, so
classes are decided by **pairs of runs under the same simulator** — a revision can
fail to build under one tool and run fine under the other, and only a within-tool
pair isolates the defect.

| Class | Cases | Meaning |
|---|---|---|
| `verified_failure_at_base` | **54** | some simulator fails the buggy revision and passes the corrected one |
| `testbench_passes_on_buggy_code` | 21 | the supplied testbench does not detect the defect at all |
| `cannot_build_either_revision` | 13 | neither revision elaborates (external package, unsupported construct, testbench that does not fit the design) |
| `failure_survives_repair` | 11 | the testbench still fails with the corrected revision |
| `no_usable_verdict` | 1 | no verdict line from either simulator |

Deciding simulator: iverilog for 52 cases, Questa for 2 (4, 21 — where the buggy
revision does not build under Questa, or fails its run, and the corrected revision
passes). Cases of note:

- `cannot_build_either_revision` (65, 70, 71, 72, 77, 79, 81, 86, 88, 94, 95, 97,
  98): the design depends on a package that is not among the supplied files
  (`config_pkg`, `cva6_config_pkg`, `rvfi_pkg`, `wt_cache_pkg`, `ibex_pkg`) or the
  testbench calls a task that no supplied file defines (`cva6_cfg_empty`). Not
  attributable to the bug, so the failure is shown in the state but the labels
  never claim verification.
- `failure_survives_repair` (12, 13, 22, 30, 39, 45, 59, 60, 68, 69, 82): six of
  these are the fault-injection/Hamming-distance testbenches, where the supplied
  corrected revision still violates the metric the testbench checks.
- `no_usable_verdict` (16): the testbench prints a literal placeholder,
  `=====Test completed with x/20 failures=====`, and never fills it in.
- 22 cases have the testbench passing the buggy revision on at least one
  simulator — including 87, where iverilog's parse failure does track the defect
  while Questa runs the testbench green. Such a case is verified *and* flagged:
  the tool evidence in the state does not point at the defect.

## How labels were derived

Each question carries a `label_source`, and the distinction the brief asks for is
kept strictly: **a verified repair is evidence about the repair, not a verified
decision label.**

| `label_source` | Meaning |
|---|---|
| `verified_repair` | the replay showed the failure tracks the defect, *and* the winning category is stated by the human description or by the buggy-side tool output, *and* the category is decisive |
| `manual_bug_label` | the winning category is stated by the bug description itself |
| `codex_pseudo_unverified` | the rule engine's reading of wording alone; a guess |

The source follows from **which channel drove the winning category**, not from a
second keyword list: a `log`- or `bug`-derived winner can be `verified_repair`
when the pair verified it, a `bug`-derived winner is `manual_bug_label`, and a
winner that only emerged from generic wording is a guess. The per-case rule trace
(`provenance.rule_traces`) records every rule that fired, its weight, the channel
and the matched text.

Evidence channels:

- `log` — the **buggy** side's tool output under every simulator that ran it. It is
  pre-repair evidence about the buggy revision whether or not the corrected
  revision confirms the failure; confirmation is a separate question handled by
  the label-source gate, not by hiding evidence. 35 cases take their winning
  category from the log.
- `bug` — the human description, with `Signed-off-by:`-style trailer lines removed
  from the rule input. These lines were firing the signedness rule
  (`Signed-off-by` matched `signed`) before the filter was added. The legal
  boilerplate echoed in Questa transcripts ("…information that is the property of
  …") was likewise firing the assertion rule. 57 cases take their winning
  category from the description.
- `fix` — **not used for labels.** A diff shows what changed, not what class of
  defect the change belongs to, and the patterns that can be written against it
  fire on nearly every case, adding a constant offset that only flattens the
  distribution. `correctcode` is still used to prove no repaired content reached
  the state and to record the outcome.

Two rules come from the replay rather than from wording: an attributable **build**
failure of the buggy revision is syntax/elaboration evidence by construction
(whatever the compiler's phrasing), and a case that builds and fails at run time
is explicitly *not* credited with `syntax_compile`.

### Label distributions

`root_cause_type` (dominant category):

| Category | Cases | | Category | Cases |
|---|---|---|---|---|
| `syntax_compile` | 27 | | `security` | 5 |
| `sequential_assignment` | 14 | | `timing_protocol` | 5 |
| `combinational_logic` | 13 | | `type_width` | 5 |
| `state_machine` | 8 | | `reset_initialization` | 2 |
| `unknown` | 20 | | `formal_property` | 1 |

`next_action` (dominant action): `rtl` 72, `testbench` 14, `simulation` 8,
`waveform` 5, `constraints` 1.

`unknown` dominating means *no category cleared the reserve*, not that nothing is
known: the vector keeps 12–30% on `unknown` whenever the evidence is flat or the
failure was not shown to track the defect. 20 cases have `unknown` as the single
largest entry, at 0.30.

**Why `rtl` is the dominant action in 72 cases, and why that is not the reflex the
brief warns against.** Fixbench's `bug` field is an unusually explicit defect
description — it typically names the faulty construct ("the partial product shift
amount uses `i` instead of `(i-1)`", "external `Cin` feeds the high byte adder").
52 of the 72 winners carry an identifier in the description; the other 20 name a
construct without an identifier. The label is never derived from the patch: the
`fix` channel is retired, and 0 of the 72 take their direction from the corrected
code. Where the transcript also carries cycle-level values the distribution stays
soft — `next_action` is one-hot in only 8 of 100 cases (92 are soft), and
`root_cause_type` is one-hot in 17 — so "look at the RTL or look at the waveform"
is usually expressed as a split rather than a choice. A bare `Failed` marker no
longer counts as waveform evidence: it produced confident `waveform` labels on
cases whose description named the defect outright, so cycle-level evidence now
requires actual values or times in the transcript.

`evidence_sufficient` is graded rather than binary in truth: `{false 0.05, true
0.95}` when the replay verified the failure and a trusted label agrees,
`{0.2, 0.8}` when only the replay verified, `{0.25, 0.75}` when the description
alone states the category, `{0.85, 0.15}` when the pair cannot be built,
`{0.9, 0.1}` when the testbench passes the buggy revision (positive evidence that
the tool output does not localise the defect), and `{1.0, 0.0}` if a case were
ever unreplayed (none are).

### Cross-dataset labelling

These records use the same question wording as the rest of the project: the
`next_action` and `root_cause_type` criteria are imported verbatim from
`training/prepare_dataset.py`, the `risk`/`urgency` rubric text from
`training/review_pseudo_scores.py`, and the `evidence_sufficient` wording is the
brief's. The independent validator compares each record's criteria against those
two modules, so the wording cannot drift.

## Why `records_5q.jsonl` is empty

A case earns a place in the defensible five-question file only when **both**
`risk` and `urgency` are genuinely supported by the source. For Fixbench, neither
is:

- **Risk** scores the *recommended action*. Every case's action is local
  read-only diagnosis — read the buggy RTL, run the supplied testbench in a
  scratch directory — which the rubric places at level 1 (Low) and never at
  level 2+, because no case asks anyone to modify shared RTL, constraints or
  configuration. A rubric application is not an observation, so it is not gold.
- **Urgency** has nothing to stand on: no record states a deadline, a release, a
  customer, or a production impact, so every case is level 0 (No time pressure).
  The project's own score auditor flags exactly this — a level-0 urgency is
  recorded with the flag `urgency_no_time_pressure_not_proven`, because the
  absence of a deadline does not prove that an active bug can wait indefinitely.
  Difficulty and technical severity were deliberately not treated as urgency.

All 100 rubric estimates therefore live in
`records_5q_pseudo_unverified.jsonl`, with **both** score labels marked
`codex_pseudo_unverified`, `provenance.risk.defensible`/`urgency.defensible` set to
`false`, and `score_tier: 5q_pseudo`. No pseudo risk or urgency is presented as
gold anywhere in this directory.

## Cases to review before any use

52 records carry at least one trust flag (`usage_sets.flag_reasons` has the exact
lists):

| Flag | Cases |
|---|---|
| `bug_text_is_upstream_pr_text` | 24 |
| `testbench_does_not_detect_the_bug` | 22 |
| `cannot_build_either_revision` | 13 |
| `failure_survives_repair` | 11 |
| `security_relevant_content` | 6 (45, 61, 83, 84, 88, 89) |
| `no_usable_verdict` | 1 (16) |

`security_relevant_content` marks cases whose content is about fault injection,
Hamming-distance metrics, one-hot encoding integrity or similar hardening work.
They are not wrong, but they deserve a human read before being used to train a
risk judgement.

## Validation

`training/validate_fixbench_conversion.py` re-derives everything from the files on
disk — it does not import the converter. Result: **0 problems**, over

- ids recomputed from the raw file and matched (100/100);
- every source index 0–99 present exactly once in the 3q file, and once across the
  5q files; each 5q record's three-question projection identical to its 3q
  counterpart;
- question ids equal to gold ids; 3 questions in the 3q file, 5 in the 5q files;
  choice criteria are dictionaries and equal the canonical sets;
  `evidence_sufficient` equals the brief's wording; score criteria are exactly the
  four ordered canonical levels;
- probabilities non-negative, keys equal to the criteria keys, summing to 1 ±
  0.002; score questions additionally re-checked the way
  `review_pseudo_scores.py` checks them (`label` = argmax, `score` = expectation,
  `confidence` ∈ [0, 1]);
- state RTL and testbench are line-for-line prefixes of the source revisions;
- no `correctcode` text, no repaired-only line (line-exact), no corrected-side
  transcript line, no corrected-file reference and no banned key in any state;
- the state's replay block equals the **buggy** side of the replay cache: same
  verdict, same commands, same return codes, same output streams;
- `repair_evidence_verified` and `replay_class` agree with the replay cache, and
  **no** record claims `verified_repair` where the failure is not attributable;
- `usage_sets.json` consistency: trusted ⊆ any-label ⊆ verified, no flagged id in
  the trusted set, the class lists partition the 3q ids;
- `split_groups.json` is re-derived independently: every claimed family signal is
  tested against the raw file (uniform claims for all members, partial claims
  within their stated sub-block), no pair of cases sharing a harness may sit in
  different groups, and the suggested split must be family-disjoint, non-empty
  and exactly the trusted set;
- `trusted_label_provenance.md` lists exactly the trusted set, in both its table
  and its trace section;
- the pseudo score file is quarantined: no Fixbench id appears anywhere under
  `dataset/normalized/`, no record is in both 5q files, and every pseudo score
  label is `codex_pseudo_unverified` with `defensible: false`;
- the report's counts equal the files on disk, and the raw file's sha256 is
  unchanged.

Leakage findings: **0**. The five checks above were each tested against a
deliberately corrupted copy of this directory (a family split across train and
test, a false signal claim, a dropped family, a broken provenance row, an
overlapping split) and each one failed the validator, so a zero here is a
measurement rather than a default.

## Merging status

Not merged into the main corpus. Nothing in `dataset/normalized/` references
Fixbench, and this conversion should be treated as a standalone experiment until
its labels have been reviewed, in line with the same policy applied to
`dataset/converted/rtl_benchls/`. The recommended first use is a small experiment
over `usage_sets.json → replay_verified_3q` only, after reading each of those
cases' provenance in `trusted_label_provenance.md`.

### Merged five-question dataset

`records_5q_distilled_pseudo_unverified.jsonl` is the full five-question dataset
for this source: each of the 100 rows carries the description-bearing `state`
unchanged plus all five questions (`next_action`, `root_cause_type`,
`evidence_sufficient`, `risk`, `urgency`). It is combined with the RTL-BenchLS
five-question set in `dataset/converted/merged_5q/` (`all.jsonl`, 208 rows), built
by `training/merge_converted_5q.py`, which re-checks the base-questions-and-state
invariant before writing. That combined directory is a standalone artefact too;
it is not the `dataset/normalized/merged_5q/` six-thousand-case corpus. The empty
`records_5q.jsonl` stays empty: the merged file keeps both score labels
`codex_pseudo_unverified` with `defensible: false`, so the merged set is for
exploratory training, not evaluation truth.

## Caveats

1. **The trusted set is small (35 of 100) and it is small for honest reasons.**
   46 cases are not verified: 21 where the testbench does not detect the bug, 13
   where the file pair cannot be built offline, 11 where the failure survives the
   repair, 1 with no verdict. None of those were converted into confident labels.
2. **`next_action` and `root_cause_type` are readings, not measurements,** even
   where `verified_repair` is the source. The replay verifies that a failure
   tracks the defect; it does not verify that the chosen category is the one a
   human would name.
3. **`unknown` is a real answer in 20 cases** and is kept as the largest entry of
   the vector rather than being squeezed into a category.
4. **The 13 `cannot_build_either_revision` cases are a property of the dataset, not
   of this conversion:** their testbenches reference packages or tasks that the
   four fields do not supply, so no offline replay can confirm them. Their buggy
   compile output is still in the state, since that is what a real agent would see.
5. **Six fault-injection/Hamming-distance testbenches fail even against their own
   corrected revision** (part of the 11 `failure_survives_repair`). Whether the
   testbench or the corrected code is at fault cannot be settled from the record.
6. **`bug_text_is_upstream_pr_text` (24 cases) means the description reads as PR or
   commit prose**, sometimes describing the change rather than the defect as it was
   found. Those labels are anchored in text about the fix, so their `manual_bug_label`
   source should be read as "the human text says so", not "the defect report says so".
7. **Risk/urgency for this dataset is not gold** and never will be without
   external information about deployment and deadlines. Treat
   `records_5q_pseudo_unverified.jsonl` as a starting point for a rubric
   experiment, not as supervision.
8. **`no_usable_verdict` (case 16) is unvalidated by construction** — its testbench
   never states a pass or fail count.
9. Only iverilog and Questa were available; a third simulator or a lint/synthesis
   tool would likely resolve some of the 13 non-buildable pairs and might change
   the phase attribution of a few cases.
10. **One behavioural change was made after seeing the numbers, and it is
    disclosed here.** The `rtl` description weight was raised above the waveform
    log weight (1.6 vs 1.5) so that a description naming a construct outranks a log
    that only reports wrong behaviour. Both readings were defensible; the choice
    determines whether the argmax of a two-way split is `rtl` or `waveform` — in
    92 of 100 cases the distribution is soft either way, so the change moves
    argmax labels, not confidence. Nothing else was tuned against the resulting
    distribution: the rule changes made after the first run were all corrections of
    demonstrated mis-fires (the Questa licence banner, `Signed-off-by`, a bare
    `Failed` marker, greedy patterns such as `type`/`valid`/`ack`, and the
    retired `fix` channel), each of which had a concrete wrong label behind it.
