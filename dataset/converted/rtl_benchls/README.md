# RTL-BenchLS Task 3 → SilicoJev conversion

Source-specific conversion of the 108 repository-issue cases of RTL-BenchLS
**Task 3** into SilicoJev/Laya-compatible decision records.

This directory is a **standalone audit artefact**. It is *not* merged into
`dataset/normalized/` (the 6,248-case SilicoJev set) and not fed to Laya.

| File | Records | Questions |
|---|---|---|
| `records_3q.jsonl` | 36 | 3 (`next_action`, `root_cause_type`, `evidence_sufficient`) |
| `records_3q_unverified.jsonl` | 72 | 3 |
| `records_5q.jsonl` | 13 | 5 (adds `risk`, `urgency`) |
| `records_5q_pseudo_unverified.jsonl` | 75 | 5, both scores marked `codex_pseudo_unverified` |
| `records_5q_distilled_pseudo_unverified.jsonl` | 108 | 5, both scores model-judged and marked `codex_pseudo_unverified` |
| `distillation_report.json` | — | per-batch counts, tier breakdown, score distributions, prior-label disagreements, validation |
| `distillation_progress.jsonl` | 108 | crash-safe progress; same rows as the distilled output |
| `usage_sets.json` | — | the recommended per-file usage sets (trusted vs separate experiment) |
| `conversion_report.json` | — | counts, splits, rule traces, post-write audit |
| `README.md` | — | this file |

### Model-judged risk/urgency (`records_5q_distilled_pseudo_unverified.jsonl`)

A second five-question pass over **all 108** cases. Unlike
`records_5q_pseudo_unverified.jsonl` (whose scores came from a rubric script), the
two scores here were decided by a model reading each record's case evidence, in
36 ordered batches of three; `training/annotate_scores_model.py` is only plumbing
(it selects the next batch, validates schema/ids/probability sums, and writes
JSONL — it contains no rule, regex, template or default that can choose a label
or a probability). Every score stays `codex_pseudo_unverified` with
`defensible: false`; the source rows are unchanged and the three original
questions and golds are byte-identical. A rationale and the evidence considered
are recorded per score in `score_annotation`.

The two 3q files partition the 108 converted cases exactly once. The 5q files are
supersets: the three-question projection of every 5q record is identical to its
3q counterpart, and record ids intentionally repeat across the 3q/5q pair — the
same convention the existing SilicoJev files use (`normalized/all.jsonl` and
`normalized/merged_5q/all.jsonl` share all 6,248 ids). Ids never repeat *within*
a file.

### Recommended use

| Set | Definition | Records |
|---|---|---|
| trusted exploratory 3q | `records_3q.jsonl` minus the evaluation-excluded cases it contains | 34 |
| trusted exploratory 5q | `records_5q.jsonl` ∩ trusted 3q ids | 2 |
| separate experiment 3q | `records_3q_unverified.jsonl` | 72 |
| separate experiment 5q | `records_5q_pseudo_unverified.jsonl` | 75 |

The exact id lists are in `usage_sets.json`. "Trusted" means the repair evidence is
objectively verified and no base-question label is a full pseudo guess. It does
**not** mean the judgement labels are correct: `next_action` and
`root_cause_type` remain inferred readings of the repair even in the trusted set
(see [Label semantics](#label-semantics-repair-evidence-vs-decision-labels)).

The trusted 5q set is small (2 records) for a specific reason: 11 of the 13
verified-5q cases carry a pseudo `root_cause_type`, so they fail the 3q gate even
though their `risk`/`urgency` labels are issue-anchored. The bottleneck is
root-cause attribution, not the score questions.

## Source and provenance

- Benchmark: <https://github.com/hkust-zhiyao/RTL-BenchLS> (Task 3), cloned to
  `dataset/raw/github/RTL-BenchLS` at upstream commit `5a8d15a`.
- Cases: `data/repo_issue_108_cases.json` — 108 real reported issues, each fixed
  by a real merged PR, across 9 upstream repositories (ibex 36, opentitan 31,
  cv32e40p 31, cvfpu 3, hdl 2, mor1kx 2, picorv32 1, black-parrot 1,
  common_cells 1). All 108 are `lec_status: useful_pass`.
- License: RTL-BenchLS is CC-BY-4.0; each upstream repository's license also
  applies to its RTL.
- `provenance.synthetic` is `false` for every record: these are historical
  repository bugs, not generated mutations.

The brief assumed `repo_cache/` already existed alongside the clone. It does not
— upstream `.gitignore`s it and ships only `scripts/clone_repos.py`. It was
rebuilt with `training/build_rtlbenchls_repo_cache.py`, which fetches the 205
referenced base/head commits from the 9 upstream repositories using shallow +
blobless fetches (≈64 MB on disk instead of multi-GB full clones) and verifies
every commit object is present. Base-revision blobs are then read with
`git show <base_commit>:<path>`. Re-running the builder is idempotent and
`--verify-only` re-checks all 205 commits.

## Reproducing

```bash
python3 training/build_rtlbenchls_repo_cache.py     # once; needs network
python3 training/convert_rtl_benchls.py             # writes this directory
python3 training/validate_rtl_benchls_conversion.py # independent re-check
```

The converter takes `--rtl-file-limit` / `--rtl-total-limit` (defaults 40,000 /
100,000 characters per file and per case) if a different context budget is
wanted; a file that does not fit the budget is delivered as bounded windows and
flagged per file.

## State contract

`state` is a JSON string built **only** from evidence a developer had before the
fix. Contents:

| Field | Why it is pre-repair |
|---|---|
| `repository`, `task_id`, `base_commit`, `base_ref` | revision identity |
| `issue_title`, `issue_body`, `issue_labels` | the report as filed |
| `affected_files` | file list published by the benchmark task |
| `source_provenance` (`source_file`, `line_number`) | benchmark provenance |
| `rtl_context[]` | `git show <base_commit>:<path>` only, with `mode`, `windows`, `truncated`, `lines`, `total_chars` |
| `failure_context` | states that no tool log exists for these cases |
| `previous_actions` | always `[]` |

Explicitly **excluded**, and recorded per record in
`provenance.state_redactions`:

| Excluded | Reason |
|---|---|
| `patches` | contains the fix |
| `head_commit` | the repaired revision |
| corrected RTL / head blobs | the answer |
| `pr_info.title`, `pr_info.body` | describe the completed fix |
| `issue_info.labels` matching `^Status:` | resolution bookkeeping added during/after the fix |
| `lec_status`, `source_info.verified` | post-repair triage, not pre-repair evidence |
| `additions`, `deletions` | reveal the size/shape of the repair |

Repair and verification facts live in `outcome` (head commit, patched files,
additions/deletions, `lec_status`, PR/issue URLs, merged timestamp, per-file
patch verification) and in `provenance`.

Every `rtl_context` entry records the exact line ranges it contains, and the
validator re-derives each entry's text from the real base blob with the same
renderer the converter used (`render_windows`), so only base-revision lines can
appear in a state. Head-revision text cannot reach the state by construction:
window ranges must lie inside the base blob, must not overlap, and must
reproduce the stored text byte-for-byte. Result: **0 leakage findings**.

## RTL context windows

The earlier policy took a top-anchored 40,000-character prefix of each file,
which has two failure modes: a large file can consume the whole per-case budget
before later files are reached, and the defect site simply is not in the prefix
when it sits deep in a file.

The current policy:

- a file whose full text fits the remaining budget is included **whole** (130 file
  entries across the corpus);
- otherwise the file is delivered as **bounded windows** (94 file entries), never
  exceeding the per-file or per-case character budget;
- window priority is the **top of the file** (module declaration, ports,
  parameters, localparams — one window deep, at least 60 lines), then an **even
  spread** across the file, then the **sites of identifiers the issue names**.
  With a limited budget the least specific windows are dropped first.

Everything that decides window placement is pre-repair evidence. The patch
decides nothing — it is used afterwards, in the audit, only to ask whether the
repaired region happened to be visible.

**`source_info.line_number` could not be used as the anchor.** The brief
suggested a bounded window around it. It indexes the benchmark's own per-case
record, not RTL: `source_info.source_file` takes exactly 9 distinct values across
the 108 cases, one per repository, and only 22 of the 108 `line_number` values
land within 30 lines of a real patch hunk in the files that case lists — about
what random placement would give. Anchoring on it would have placed windows at
arbitrary RTL lines while looking principled.

An ablation over the 108 cases shows the issue-named anchors do not localise
better than a plain spread (526 vs 534 of 768 repair hunks visible), so they are
given only the capacity a spread leaves over — a bug report is typically wrong
about where the defect lives.

The audit (`conversion_report.json` → `context_window_coverage`) compares the
delivered windows against a faithful reconstruction of the replaced prefix slice
(same per-file/per-case budget, same file order). The denominators below are the
224 listed files that carry a repair hunk (768 hunks, 9,519 lines):

| Metric | Windowed context | Old top-anchored slice |
|---|---|---|
| Repair hunks visible | 538 / 768 (70.1%) | 534 / 768 (69.5%) |
| Patched lines visible | 6,550 / 9,519 (68.8%) | 6,426 / 9,519 (67.5%) |
| Patched files with their first hunk visible | 194 / 224 | 181 / 224 |
| Patched files fully visible | 145 / 224 | 141 / 224 |
| No file left without context | 108 / 108 cases | not guaranteed |

The structural difference is larger than the percentages suggest: 234 hunks sit
outside anything the old prefix could ever show, and the windowed context shows
100 of them. Neither policy shows the remaining 134, so about 30% of repair hunks
are absent from the model's context under either policy — that is a property of
the context budget, not of the window shape. Raising `--rtl-total-limit` is the
only lever that moves it.

Two behaviours were added because the audit exposed them:

- The prefix slice spent budget greedily in file order, so a large file early
  could leave a later **patched** file with no context at all (5 file entries
  across 4 cases). Each file now holds back `RTL_MIN_FILE_CHARS` (4,000) for the
  files still to come, so no case hides its repaired file. `omitted_files` in the
  report is now empty.
- An ablated variant that capped every file at an equal share of the budget
  removed the starvation too, but forced windows on files that would otherwise
  have been included whole and lost 6 points of context coverage. The reserve
  model keeps whole-file inclusion and the floor at the same time.

**Read this audit as evidence about one corpus, not as a tuned result.** Every
rule in the policy has a structural rationale — include whole what fits, start at
the declarations, spread the rest, then honour the reporter's wording — but three
details were settled *after* seeing these numbers:

1. the even spread outranks the issue-named anchors (the ablation above);
2. the top window covers a full window rather than 60 lines (first-hunk
   visibility dropped from 194 to 174 files with the shallow top);
3. the reserve model was preferred over an equal-share cap (6 points of coverage).

Nothing here used the patch to *place* a window, and the audit cannot affect what
a record contains once written — but the policy is lightly fitted to this corpus,
so the 70% repair-hunk figure should not be read as a guarantee on other RTL.

## How labels were derived

Objective verification available for these cases: the base and head commits, the
supplied patch, and the real `git diff base..head`. The converter checks that
every patch `-` line exists in the base blob, every patch `+` line exists in the
head blob, and the patch is a whitespace-tolerant subset of the actual diff
(some PRs bundle extra edits, so a strict equality test would be wrong).

### Label semantics: repair evidence vs decision labels

**A verified repair is evidence about the repair, not a verified decision label.**
Each record therefore separates the two:

- `provenance.repair_evidence_verified` (`true`/`false`) and
  `provenance.repair_evidence` (`verified_repair_evidence` /
  `unverified_patch`) describe the *evidence*: the base and head commits exist,
  the affected file is readable at the base revision, and the supplied patch is
  reproduced by the real diff.
- `label_source` on each question describes the *judgement*, and is one of:
  - `inferred_from_verified_repair` — the label is an inference drawn from a
    repair that is objectively verified. It is still an inference: the repair
    shows what changed, not what a developer should have investigated next.
  - `manual_issue_label` — the issue metadata itself states the label (e.g. a
    `Component:DV` or `Type:Spec-Compliance` tracker label, or an explicit
    location in the report).
  - `codex_pseudo_unverified` — interpreted, not supported by verified repair or
    issue metadata.

The earlier vocabulary called the first case `verified_repair`, which read as if
the *decision* were verified. It was renamed to `inferred_from_verified_repair`
for that reason. `next_action` and `root_cause_type` are inferred judgements even
on a perfectly verified repair, and are marked as such.

| Question | Evidence used | Label source |
|---|---|---|
| `next_action` | issue title/body (spec citations, DV, waveform, formal, constraints wording) + tracker labels (`Component:RTL`, `Component:DV`, `Type:Spec-Compliance`, `Type:Question`) + the fact that the merged fix edits RTL at the base commit | `manual_issue_label` for a clear tracker label; `inferred_from_verified_repair` when the verified repair corroborates the dominant direction (all 108 dominant = `rtl`); else `codex_pseudo_unverified` |
| `root_cause_type` | semantics of the verified repair (patch `+`/`-` lines), then the reporter's issue title, then the issue body/labels, scored with explicit per-category regex rules | `inferred_from_verified_repair` only when a patch-anchored rule reached the strong threshold and one category reaches ≥0.60; else `codex_pseudo_unverified` with mass reserved for `unknown` |
| `evidence_sufficient` | whether the issue names a file/symbol/line that is actually present in the included base RTL, and whether the cited RTL had to be windowed | `manual_issue_label` when the issue states a location, `inferred_from_verified_repair` when it names a symbol present in the included RTL, else `codex_pseudo_unverified` |
| `risk`, `urgency` (5q only) | issue-anchored statements (`U_*`/`R_*` rules below) | `manual_issue_label` in `records_5q.jsonl`; `codex_pseudo_unverified` for both in `records_5q_pseudo_unverified.jsonl` |

Across the three base questions in the two 3q files: `inferred_from_verified_repair`
155, `manual_issue_label` 96, `codex_pseudo_unverified` 73 (of 324 labels).

The per-record `provenance.rule_traces` names every rule that fired and why,
`provenance.label_rationale` carries a one-line justification for all five
questions, and `provenance.pseudo_questions` lists every question whose label is
a pseudo guess. No rationale text is stored in `state`.

Guards against over-confident labels:

- `next_action` keeps real mass on the directions the issue text points at
  (specification, testbench, formal, waveform, …) rather than forcing a one-hot
  `rtl` for all 108 cases. The RTL direction is *corroborated* by the repair, not
  derived from it alone.
- `root_cause_type` reserves probability mass for `unknown` (0.05 for an
  unambiguous patch-anchored rule, 0.15 for a weaker one, 0.30 for report-only
  evidence) and hands the plurality to `unknown` when no category reaches 0.40.
  30 of 108 cases have `unknown` as the dominant root cause.
- `evidence_sufficient` is never set true merely because the case was eventually
  fixed; it depends on what is visible in the constructed state.

## Risk and urgency: the verified/pseudo split

Risk and urgency were **only** written as defensible labels when the issue itself
carries the evidence. Every rule is named in `provenance.rule_traces.risk_rule` /
`urgency_rule` and tallied in `conversion_report.json`.

`records_5q.jsonl` (13 records) requires **both**:

- **urgency anchored** in an explicit tracker or issue statement —
  `U_PRIORITY_P1` (6), `U_PRIORITY_P2` (8), `U_PRODUCTION_LABEL` (4),
  `U_MILESTONE_LABEL` (2), `U_DEFERRED_LABEL` (1), `U_DESTRUCTIVE_IMPACT` (8),
  `U_ACCESS_CONTROL` (3); and
- **risk anchored** in explicit issue content — `R_DESTRUCTIVE_IMPACT` (10),
  `R_ACCESS_CONTROL` (3).

`records_5q_pseudo_unverified.jsonl` (75 records) holds cases where a transparent
rubric estimate exists but is not source-anchored. Both score labels there are
`codex_pseudo_unverified`. The two rules involved are:

- `R_READONLY_ACTION` (68 cases): the published risk criteria are applied to a
  repair-pinned read-only action (`rtl`/`waveform`/`specification` inspection →
  benign/low). This is a *criteria application*, not an independent observation,
  which is exactly why it is pseudo rather than verified.
- `U_NONE` (76 cases): no priority, milestone or impact statement exists, so
  urgency falls back to a backlog baseline.

Severity is deliberately tiered rather than maximised. Genuine vulnerability
wording (`U_SECURITY_VULNERABILITY`, `R_SECURITY_VULNERABILITY`) matched **no**
case in this corpus, while access-control correctness defects
("illegal write to read-only field", "CSR access to non-existing CSR") score
Elevated urgency / Moderate risk rather than Critical / High, because no issue
demonstrates an exploit.

The remaining 20 cases have neither anchored nor rubric evidence and keep three
questions only (`three_question_only` in the report) rather than inventing score
labels.

The `urgency` criteria text follows the task brief wording
("No time pressure; can wait indefinitely." etc.); note the existing
`dataset/normalized/merged_5q` files use a slightly longer variant of the same
four levels.

**These score labels are not ground truth.** They are rubric-derived and must not
be used as final evaluation truth, matching the standing policy in
`dataset/EXTERNAL_DATASETS.md` rule 6.

## Evaluation exclusions

Seven cases carry a `trust_flags` entry and are marked `eval_excluded: true` in
`provenance`. They stay in the output files (so the corpus stays complete and
auditable) but must not be trained or evaluated on. All seven are listed in
`usage_sets.json` → `eval_excluded`, with reasons and file membership.

| Flag | Cases | Why it excludes the case |
|---|---|---|
| `self_answering_issue_body` | 6 | The issue body already quotes lines the merged fix adds, so the answer is in the input state. This is genuine pre-repair evidence (a reporter pasting a suggested patch), which is why it is kept in `state` and the case is excluded from *use* instead. |
| `patch_inconsistent_with_git_diff` | 1 | `openhwgroup_cv32e40p_327_277`: the supplied patch claims to add `parameter PULP_HWLP = 0,` which already exists at the base commit, so the patch is not reproduced by the real diff and the repair evidence is unreliable. |

The self-answering cases: `lowRISC_ibex_48_46`,
`lowRISC_opentitan_10696_10680`, `lowRISC_opentitan_12218_12204`,
`openhwgroup_cv32e40p_145_143`, `openhwgroup_cv32e40p_165_150`,
`openrisc_mor1kx_110_102`; plus `openhwgroup_cv32e40p_327_277` for the patch
inconsistency. Two of the seven (`lowRISC_ibex_48_46`,
`openrisc_mor1kx_110_102`) were in `records_3q.jsonl` and are the reason the
trusted 3q set is 34 rather than 36.

## Splits

Repository-disjoint, written to `conversion_report.json` and to
`provenance.split` on every record. No repository appears in two splits, and the
openhwgroup cores plus the pulp-platform cells they instantiate are treated as
one bug lineage.

| Split | Repositories | Records |
|---|---|---|
| train | `lowRISC/ibex`, `analogdevicesinc/hdl`, `openrisc/mor1kx`, `YosysHQ/picorv32`, `black-parrot/black-parrot` | 42 |
| validation | `openhwgroup/cv32e40p`, `openhwgroup/cvfpu`, `pulp-platform/common_cells` | 35 |
| test | `lowRISC/opentitan` | 31 |

## Audit results

From `conversion_report.json` and the independent validator:

| Check | Result |
|---|---|
| Source cases | 108 |
| Converted | 108 (0 skipped, 0 invalid) |
| Missing base files | 0 |
| Checkout / `git show` failures | 0 |
| State-leakage findings | 0 |
| Patch verified against base+head git objects | 107 (106 exact matches) |
| File entries included whole / as windows / omitted | 130 / 94 / 0 |
| Repair hunks visible in the delivered context | 538 / 768 (70.1%) |
| `label_source` across the 3 base questions (3q files) | `inferred_from_verified_repair` 155, `manual_issue_label` 96, `codex_pseudo_unverified` 73 |
| Evaluation-excluded cases | 7 (6 self-answering, 1 inconsistent patch) |
| Trusted / separate-experiment records (3q) | 34 / 72 |
| Trusted / separate-experiment records (5q) | 2 / 75 |
| Schema, question-structure, projection, usage-set problems | 0 |

## Known caveats

1. **One case has unreliable repair evidence.**
   `openhwgroup_cv32e40p_327_277` has a source patch that claims to add a line
   (`parameter PULP_HWLP = 0,`) already present at the base commit, so it is not a
   subset of the real diff. Its labels are all `codex_pseudo_unverified`, it lands
   in the unverified 3q file, and it is flagged `eval_excluded`.
2. **Two PRs bundle extra edits.** `lowRISC_ibex_83_80` and
   `openhwgroup_cv32e40p_327_277` have head commits containing more than the
   issue patch; the subset check handles this, and
   `head_contains_extra_changes` lists them. `lowRISC_ibex_83_80` is still
   trustworthy: its patch is fully reproduced, the head commit just carries more.
3. **Six issue bodies already quote the repair** (e.g. the reporter pastes a
   suggested patch) and are self-answering. They are flagged
   `self_answering_issue_body`, marked `eval_excluded`, and listed in
   `usage_sets.json`; they are also tallied under
   `issue_body_quotes_patch_lines` in the report.
4. **Context windows still omit about 30% of repair hunks.** 538 of 768 hunks are
   visible under the delivered policy; the old top-anchored slice managed 534 but
   could not see 234 hunks at all, 100 of which the windows now show. The misses
   are a budget limit (40,000 characters per file, 100,000 per case), not a
   placement bug — no pre-repair signal (including `source_info.line_number`)
   localises the defect better, and 45 cases still need windowing at the default
   budget. Truncation never uses the patch, so it cannot hint at the answer.
5. **`root_cause_type` is the weakest question and gates the trusted 5q set.** 71
   of the 108 cases have a pseudo root-cause label (against 1 for `next_action`
   and 1 for `evidence_sufficient`), because many repairs are small edits whose
   semantics do not clearly select one canonical category (`unknown` dominates for
   30 cases by design). Because the trusted 5q set requires the three base labels
   to be non-pseudo, 11 of the 13 verified-5q cases fall out and only 2 remain.
   That scarcity is honest: their score labels are issue-anchored, but their root
   causes are guesses.
6. **`risk` is near-degenerate on this corpus.** All 108 cases are diagnosis
   tasks on a base revision, so the recommended action is almost always
   read-only inspection; only 13 cases carry issue-level risk evidence. Do not
   read the risk distribution as a property of RTL debugging in general.
7. **Trusted does not mean correct.** The trusted sets exclude pseudo *guesses*,
   not inferences. `next_action` and `root_cause_type` are inferred readings of a
   verified repair in every record, including the trusted ones.
8. **Not merged into the main corpus.** Nothing here is written into
   `dataset/normalized/` (the 6,248-case SilicoJev set) and nothing is fed to
   Laya. Jev/API outputs were not used as labels at any point. Per the
   recommendation, this stays a standalone RTL-BenchLS source for a controlled
   training/evaluation addition.

## Merged five-question dataset

`records_5q_distilled_pseudo_unverified.jsonl` is already the full five-question
dataset for this source: each of the 108 rows carries the description-bearing
`state` unchanged plus all five questions (`next_action`, `root_cause_type`,
`evidence_sufficient`, `risk`, `urgency`). It is combined with the Fixbench-RTL
five-question set in `dataset/converted/merged_5q/` (`all.jsonl`, 208 rows), built
by `training/merge_converted_5q.py`, which re-checks the base-questions-and-state
invariant before writing. That combined directory is a standalone artefact too;
it is not the `dataset/normalized/merged_5q/` six-thousand-case corpus.
