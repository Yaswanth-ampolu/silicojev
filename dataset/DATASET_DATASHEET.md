# SilicoJev dataset datasheet

**How this dataset is built, and why.** This document is the construction ledger for the
SilicoJev decision corpus. It follows the seven-category *Datasheets for Datasets* structure
(motivation, composition, collection, preprocessing/labeling, uses, distribution,
maintenance) so that a reader can tell, for any row, where it came from, how each label was
produced, and how much to trust it.

Every number below was measured, and the command that produced it is given in
[§9 Reproducing the numbers](#9-reproducing-the-numbers). Nothing here is an estimate.

> **Scope note.** This file documents the corpus. It is not itself a training artifact and is
> not read by any loader. It does not modify `dataset/normalized/` or any gold file.

---

## 1. Motivation — *why this dataset exists*

SilicoJev is the RTL/EDA specialization of **Laya**, a *typed-decision model*. It is not a text
generator: it takes a `state` plus typed `questions` and returns **probability distributions**,
not prose. There are exactly three head types — `choice`, `noul` (boolean as a probability),
`score` (ordered 0–3).

The dataset exists to teach a **calibrated, abstaining decision policy for hardware
debugging: a controller, not a code generator.** The payoff is that a router can gate real
actions — automatically run a benign step when confidence is high, escalate to a human when
evidence is insufficient or the action is risky.

Five design decisions follow from that goal, and they explain most of the corpus's shape:

| decision | why |
|---|---|
| **Typed distributions, not text** | There is no JSON to repair and no parsing step; a confidence number becomes something you can *threshold on* rather than a vibe. |
| **These five questions** | They are the minimal set a router needs: `next_action` (what to do), `root_cause_type` (what it probably is), `evidence_sufficient` (should it act at all), `risk` (what does the step cost), `urgency` (how fast does it matter). |
| **Pre-repair `state` only** | At decision time the fix does not exist. Putting the patch, corrected RTL, or hidden root cause in `state` converts a diagnosis task into memorization, and inflates every metric. Those belong in `outcome`/`provenance`. |
| **Soft gold distributions** | The model is trained RLCD-style against distributions and scored with proper scoring rules. A one-hot target rewards confident guessing; a distribution rewards *being calibrated about uncertainty* — which is the entire product claim. |
| **`abstain` and `ask_human` are first-class options** | The value of the system is knowing *when it doesn't know*. If abstention were not representable in the label space, the model could never learn it. |

A sixth decision is a deliberate *non*-use, documented in §6: **`score` stays disabled in
evaluation** until a source with defensible ordered labels exists. Printing an RPS over
fabricated targets would be misleading, so the evaluator says so instead.

---

## 2. The record contract — *what a row looks like*

One case = one JSONL row. The schema is `dataset/SILICOJEV_SCHEMA.json`.

**The single most important thing to know:** `state`, `questions`, and `gold` are **JSON
strings**, not nested objects, while `outcome` and `provenance` **are** nested objects. This is
the established convention. It is not a bug, and "fixing" it would break every loader, the
converter, the trainer, and the evaluator at once.

```jsonc
{
  "id":            "hwe:2233907056",
  "source":        "hwe-bench",
  "source_group":  "hwe:lowRISC/ibex",     // the lineage key used for splitting
  "state":         "{...}",                 // JSON STRING — pre-decision evidence only
  "questions":     "{...}",                 // JSON STRING — head types + criteria
  "gold":          "{...}",                 // JSON STRING — distributions + label_source
  "outcome":       { ... },                 // nested object — post-decision truth
  "provenance":    { ... }                  // nested object — licence, quality, synthetic
}
```

### 2.1 The five heads

| question | type | instruction | option space |
|---|---|---|---|
| `next_action` | `choice` | What diagnostic direction should be investigated next? | 9 inspection/run verbs: `rtl`, `waveform`, `simulation`, `constraints`, `formal`, `testbench`, `specification`, `ask_human`, `abstain` |
| `root_cause_type` | `choice` | Which root-cause category best describes the available evidence? | 10 classes: `combinational_logic`, `sequential_assignment`, `reset_initialization`, `state_machine`, `type_width`, `timing_protocol`, `syntax_compile`, `formal_property`, `security`, `unknown` |
| `evidence_sufficient` | `noul` | Is there enough evidence to choose a likely root cause? | `true` / `false` |
| `risk` | `score` | How risky would the recommended diagnostic or agent action be? | 0–3: Benign / Low / Moderate / High |
| `urgency` | `score` | How quickly does this hardware issue require attention? | 0–3: No time pressure / Routine / Elevated / Critical |

`next_action` options are all **inspection or run** verbs. `rtl` means *inspect* the RTL — a
read-only act at risk level 0 — even though a *repair* would edit RTL. Conflating the two is
the easiest way to mis-label `risk`.

### 2.2 3q vs 5q — the exact structural difference

The corpus ships in two shapes. **3q is the base** (source-derived); **5q = 3q + two
model-judged score heads**. Measured invariants across every corpus file:

| | 3q main | 5q main | 5q converted |
|---|---|---|---|
| file | `dataset/normalized/all.jsonl` | `dataset/normalized/merged_5q/all.jsonl` | `dataset/converted/merged_5q/all.jsonl` |
| rows | **6,248** | **6,248** | **208** |
| heads | 3 | 5 | 5 |
| extra top-level keys | — | `pseudo_annotation_notes`, `pseudo_annotation_review` | `score_annotation` |
| `gold` fields per `choice`/`noul` head | `label_source`, `probabilities` | same | same |
| `gold` fields per `score` head | *n/a* | `confidence`, `label`, `label_source`, `probabilities`, `score` | **+ `defensible`** |

Three consequences worth stating plainly:

1. **Score heads carry a richer gold shape.** Only `score` golds have `confidence`, `label`
   (the argmax level as a string) and `score` (the expected value, e.g. `1.84`). A loader that
   assumes `{label_source, probabilities}` for every head will break on 5q.
2. **`defensible` exists only in the converted files.** `dataset/converted/merged_5q/` adds the
   boolean `defensible` to every score gold. In the main 5q corpus the equivalent signal lives
   in `provenance.pseudo_label_status` instead. Two mechanisms for one idea — worth unifying.
3. **The 5q merge is additive and non-destructive.** The merge preserves the base record's
   `state` byte-identically and leaves the three source-derived questions and golds unchanged;
   it only *adds* `risk` and `urgency` plus the two `pseudo_annotation_*` fields.

### 2.3 Where answers are allowed to live

| field | may contain | may **never** contain |
|---|---|---|
| `state` | buggy RTL, testbench, tool logs, waveform summary, issue text, previous actions | the patch, corrected RTL, hidden root cause, fault location, post-fix result |
| `questions` | head type, instruction, option/level criteria | any case-specific answer |
| `gold` | distributions + `label_source` | unverified output promoted to a trusted source |
| `outcome` | resolution, verification verdict, commits, line counts, replay verdicts | — |
| `provenance` | licence, quality tier, synthetic flag, pseudo-label status, family | — |

The rule is one sentence: **`state` describes the world before the decision; everything that
reveals the decision's answer goes in `outcome` or `provenance`.**

---

## 3. Collection — *where the rows come from*

```
6,456 cases total  =  6,248 main corpus  +  208 converted
```

### 3.1 Main corpus — 6,248 rows in `dataset/normalized/`

| source | rows | quality | label_source | what it is |
|---|---|---|---|---|
| OriGen | 5,444 | `bronze` | `repair_pair_unvalidated` | templated generator; one shared prompt prefix |
| hwe-bench | 417 | `gold` | `verified_repair` | real fixes across ibex, OpenTitan, CVA6, rocket-chip, XiangShan, caliptra; **fail-to-pass verified end-to-end** |
| ChipBench | 178 | `silver_benchmark` | `benchmark_bug_family` | benchmark artifacts |
| RTL-augmented | 145 | `silver_validated` | `validated_mutation` | injected and validated mutations |
| RootCause-Bench | 36 | `silver` | `manual_bug_label` | manually labelled root causes |
| HierSVA | 21 | `silver_formal_context` | `synthetic_bug_pattern` | formal/assertion context |
| HierSVA (historical) | 7 | `silver_formal_context` | `historical_bug` | historical bug records |

### 3.2 Converted sources — 208 rows in `dataset/converted/`

Standalone audit artefacts. **Deliberately not merged into `dataset/normalized/`.**

| source | records | pipeline |
|---|---|---|
| RTL-BenchLS Task 3 | 108 | real repository issues, base/head commits, patch-equality verified, LEC status |
| Fixbench-RTL | 100 | buggy RTL + corrected RTL + bug description + testbench, replay-verified |
| merged | 208 | per-source subsets, `all.jsonl` |

---

## 4. Preprocessing & labeling — *how a record is actually built*

This is the section that answers "how". It is also where the honest caveats live, because
**the five labels are not produced the same way, and three of them are derived by rule.**

### Stage 1 — 3q conversion · `training/prepare_dataset.py`

For each source it emits `state`, `questions`, and three golds. The three base labels are
**derived, not observed**:

| head | mechanism | honest description |
|---|---|---|
| `next_action` | `distribution(ACTION_CRITERIA, active_actions)` | A *rubric application*. `active_actions` is hard-coded to `["rtl"]` for RootCause-Bench, OriGen, RTL-augmented, HierSVA and ChipBench. The label is therefore near-constant for most of the corpus. |
| `root_cause_type` | `root_cause_key(text)` | A **regex keyword matcher** over the text. Not a reading of the evidence. |
| `evidence_sufficient` | boolean heuristic | Not a reading of the evidence. |

Splitting happens here too, group-hashed by `sha1(source_group)`. One weakness is structural:
**OriGen's group is `sha1(instruction)`, unique per row**, so OriGen is effectively split at
random rather than by lineage. 87% of the corpus is OriGen, so the split is weaker than the
group-hashing makes it appear.

### Stage 2 — teacher pass for `risk` / `urgency` · `training/review_pseudo_scores.py`

Adds the two score heads as soft distributions, written to
`dataset/normalized/pseudo_scores/`. These are **model judgments**, recorded as
`codex_pseudo_unverified`.

### Stage 3 — the 5q merge · `training/merge_silicojev_questions.py`

Joins the two score heads onto the description-bearing base records, producing
`dataset/normalized/merged_5q/`. Additive: base `state` preserved byte-identically.

### Stage 4 — converted sources · a different, stricter pipeline

For RTL-BenchLS and Fixbench the `risk`/`urgency` scores were assigned by **per-record model
judgment in batches of three**, with resumable progress, then checked by an independent
validator (`training/validate_score_outputs.py`, 11 checks). The guardrail governing this
stage: **no script, regex, keyword map or default may select a label or a probability.** Python
was plumbing only — selecting the next batch, reading/writing JSONL, checking schema, ids,
probability sums and completeness. Nothing else.

Result: every added score is `label_source: "codex_pseudo_unverified"`, `defensible: false`.

### Stage 5 — tool-backed replay · `training/build_fixbench_replay.py`

The only stage that produces **objective, executable evidence**. It replays each Fixbench case
on both the buggy and corrected revision and records verdicts per backend.

It already drives **two** backends — `--backends iverilog,questa` — including a full Questa
path (`vlog` + `vsim`, `modelsim.ini`, license discovery under `/usr/local/questa/`). This
matters: the project's own tooling is the only place Questa data can come from, because no
public dataset may redistribute it. What is missing is not the capability but the **logging**:
the `(state, action, command, stdout, outcome)` tuples are executed and discarded rather than
retained as trajectory data.

---

## 5. Composition & label strength — *how much of this is trustworthy*

### 5.1 Label strength by head (main corpus, 6,248 rows)

| question | total | strong labels | weak / pseudo |
|---|---|---|---|
| `next_action` | 6,248 | 783 | 5,465 |
| `root_cause_type` | 6,248 | 783 | 5,465 |
| `evidence_sufficient` | 6,248 | 783 | 5,465 |
| `risk` | 6,248 | **0** | 6,248 |
| `urgency` | 6,248 | **0** | 6,248 |

Measured `label_source` per head confirms this exactly: `risk` and `urgency` are
`codex_pseudo_unverified` for **6,248 / 6,248** rows, while the three base heads are split
across the seven source-derived provenance values.

**The three-tier vocabulary used throughout this project:**

- **strong** — observed from an executable verdict or a trusted manual label
  (`verified_repair`, `validated_mutation`, `manual_bug_label`).
- **inferred** — derived from a verified artifact but still a *reading*
  (`inferred_from_verified_repair`, `benchmark_bug_family`, `historical_bug`).
- **pseudo** — model-generated or rule-generated, unverified
  (`codex_pseudo_unverified`, `repair_pair_unvalidated`, `synthetic_bug_pattern`).

### 5.2 Provenance distribution (main corpus)

| rows | quality | label_source |
|---|---|---|
| 5,444 | `bronze` | `repair_pair_unvalidated` |
| 417 | `gold` | `verified_repair` |
| 178 | `silver_benchmark` | `benchmark_bug_family` |
| 145 | `silver_validated` | `validated_mutation` |
| 36 | `silver` | `manual_bug_label` |
| 21 | `silver_formal_context` | `synthetic_bug_pattern` |
| 7 | `silver_formal_context` | `historical_bug` |

Across all decisions, roughly **92% of the corpus is unvalidated or pseudo**, and **87% comes
from a single templated generator** (OriGen, all sharing one identical prompt prefix). Effective
diversity is closer to 1–2k distinct scenarios than 6,248.

**A verified repair is evidence about the repair, not about the judgement label.**
`next_action` and `root_cause_type` remain *readings* even where
`outcome.verification == "verified_repair"`. This distinction is the reason the strong-label
count is 783 and not 6,248.

---

## 6. Known limitations and open defects

Stated here rather than discovered later by a user.

### 6.1 `score` heads have no defensible ordered source

`risk` and `urgency` are 100% pseudo. The converted sources are worse than the summary
suggests: on the 208 added rows, `risk` is level 0 for **108/108** RTL-BenchLS rows and 95/100
Fixbench rows, and is near-**collinear with `next_action`** (risk 1 appears only where the
action is `simulation`). `urgency` is a near-constant prior (199/208 at level 0) with mean
confidence only 0.44–0.53, and 205/208 rows flagged as having no supporting evidence.

Trusted five-question rows: **RTL-BenchLS 13, Fixbench 0.**

Also measured on HWE-bench: its discarded `priority_score` (present 372/417, range 4–22) is
**not** an urgency signal — its README defines it as *"Candidate ranking score used during case
selection."* It measures benchmark usefulness, not how fast an issue needs attention. Training
`urgency` on it would be a category error.

**Consequence:** `score` remains disabled in evaluation.

### 6.2 Answer leakage into `state` — 100 rows need an audit

`state.bug_description` appears in **100 / 208** converted rows. For those same 100 rows,
`provenance.leakage_findings` is an **empty list**. A verified example:

> `"Module port list uses only names (Verilog-1995 style) without directions or types: data,
> clk, and rst must be inputs, q must be an output reg because it is assigned inside always
> blocks."`

That is not a symptom report — it states the required change, i.e. the repair.

A heuristic screen flags **29 of the 100**. The exact pattern, which is the one used in the
reproduction script in §9, is:

```python
pat = re.compile(r'\b(must be|should be|instead of|replace|needs? to be)\b', re.I)
```

**This is a screen, not a verdict, and the number is sensitivity-dependent — which is exactly
why it must not be quoted as a finding.** Widening the same pattern to also match `changed` and
`fixed by` raises the count from 29 to **38** on unchanged data, a 31% swing from a one-line
edit. So: 29 rows match this specific pattern, the true count of answer-leaking rows is not
established by any regex, and all **100** rows need human review. None of the 100 should be
treated as leakage-free on the strength of the empty `leakage_findings` field alone. This is a
concrete, bounded audit task — 100 rows, one reviewer-session, not a corpus-wide re-audit.

### 6.3 Absolute paths baked into provenance

**6,248 / 6,248** rows carry a `provenance.pseudo_source_file` absolute path rooted at
`/home/jovyan/...` — a path from the notebook environment where the merge ran. The field is not
portable and will not resolve on any other machine. It is harmless for training but it should
be relativised or dropped before the corpus is published or moved.

### 6.4 Split integrity

No exact duplicate leakage across splits was found, but the OriGen group key is
`sha1(instruction)` — unique per row — so OriGen cannot be split by lineage. Any split
containing OriGen is effectively random with respect to that source. New sources must be split
by **project/family**, not by row.

### 6.5 Licensing

Not every source is licence-clean. `dataset/EXTERNAL_DATASETS.md` records URL, commit, sha256,
byte size and licence for each acquisition, and lists what was deliberately **not** acquired.
Two standing items: **VeriBugBench**'s bundled RTL is *not* covered by its MIT licence
(`license_scope` reads "upstream project license" for all 45 projects), and any source with
unstated licensing is excluded from training use until resolved.

---

## 7. Uses

**Intended:**
- Fine-tune the `silicojev` checkpoint from the `laya-typed-decisions` base, preserving Laya's
  output contract so the existing runtime evaluates both and comparisons stay meaningful.
- Evaluate per head with exact accuracy, soft accuracy, Brier, KL, TV, ECE and p50/p95 latency —
  and **per source**, via `*_by_source.json`, so validated and weakly-labelled sources are
  visible separately rather than averaged into one flattering number.
- Fit one validation-set temperature per head (`choice`, `score`, `noul`) with LBFGS on NLL. A
  checkpoint without a fitted temperature is reported as **uncalibrated**, never silently
  treated as calibrated.

**Explicit non-uses:**
- Do not train on `patch` / `output` and then evaluate on the same cases.
- Do not use Jev or any API output as a label or distillation target.
- Do not force weak labels into a gold file to inflate the training set.
- Do not enable `score` metrics while `risk`/`urgency` remain pseudo.
- Do not describe a rubric application (`distribution(ACTION_CRITERIA, ["rtl"])`) as an
  observation of what an engineer did.

---

## 8. Distribution & maintenance

- Raw inputs live under `dataset/raw/<source>/` and are **git-ignored** (`.gitignore`
  `dataset/raw/`), so multi-gigabyte data is never committed. Only converted records and
  metadata are tracked.
- Converted source directories are **standalone audit artefacts**: not merged into
  `dataset/normalized/`, not fed to Laya.
- Training must be resumable: model, optimizer, scheduler, scaler, RNG, epoch, batch position,
  dataset fingerprint and configuration in every checkpoint. **Never resume a checkpoint
  against a changed dataset without recording the new dataset version.**
- To add a source: acquire into `dataset/raw/<source>/`, record URL + sha256 + byte size +
  licence in `EXTERNAL_DATASETS.md`, write an adapter, validate, then split by lineage.

---

## 9. Reproducing the numbers

```bash
cd /silicogenplayground/projects/silicogen/silicojev

# corpus sizes
wc -l dataset/normalized/merged_5q/all.jsonl \
      dataset/normalized/all.jsonl \
      dataset/converted/merged_5q/all.jsonl \
      dataset/converted/rtl_benchls/records_3q.jsonl

# structural invariants: heads, head types, gold field shapes per file
python3 - <<'PY'
import json, collections
for p in ["dataset/normalized/all.jsonl",
          "dataset/normalized/merged_5q/all.jsonl",
          "dataset/converted/merged_5q/all.jsonl"]:
    rows=[json.loads(l) for l in open(p) if l.strip()]
    typ, shape = {}, collections.defaultdict(collections.Counter)
    heads=collections.Counter()
    for r in rows:
        g=json.loads(r["gold"]); q=json.loads(r["questions"])
        for h,qv in q.items(): typ[h]=qv.get("type")
        for h,gv in g.items():
            heads[h]+=1; shape[h][tuple(sorted(gv))]+=1
    print(f"{p}: {len(rows)} rows")
    for h in sorted(shape):
        print(f"   {h:20s} type={typ[h]:7s} n={heads[h]} gold={dict(shape[h])}")
PY

# label_source distribution per head
python3 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("dataset/normalized/merged_5q/all.jsonl") if l.strip()]
ls=collections.defaultdict(collections.Counter)
for r in rows:
    for h,gv in json.loads(r["gold"]).items(): ls[h][gv.get("label_source")]+=1
for h in sorted(ls): print(h, dict(ls[h]))
PY

# leakage screen (heuristic) + portability check
python3 - <<'PY'
import json, re
rows=[json.loads(l) for l in open("dataset/converted/merged_5q/all.jsonl") if l.strip()]
pat=re.compile(r'\b(must be|should be|instead of|replace|needs? to be)\b', re.I)
has=sum(1 for r in rows if "bug_description" in json.loads(r["state"]))
dir_=sum(1 for r in rows if "bug_description" in (s:=json.loads(r["state"]))
         and pat.search(s["bug_description"] or ""))
empty=sum(1 for r in rows if r["provenance"].get("leakage_findings")==[])
print(f"bug_description={has}/{len(rows)}  directive(heuristic)={dir_}  leakage_findings==[]={empty}")

m=[json.loads(l) for l in open("dataset/normalized/merged_5q/all.jsonl") if l.strip()]
abs_=sum(1 for r in m if str(r["provenance"].get("pseudo_source_file","")).startswith("/"))
print(f"absolute pseudo_source_file = {abs_}/{len(m)}")
PY
```

---

## 10. Citation

If this corpus is used, cite the underlying sources — HWE-bench, RTL-BenchLS, Fixbench-RTL,
ChipBench, RootCause-Bench, HierSVA, OriGen, and the remainder listed in
`dataset/EXTERNAL_DATASETS.md` — alongside SilicoJev. Licences are recorded per source; the
`provenance.license` field on each row states the licence scope for that record.
