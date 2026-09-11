# SilicoJev

SilicoJev is the RTL/EDA specialization of Laya. The first version should
preserve Laya's architecture and output contract (`choice`, `noul`, and
`score`) while replacing the general workflow training data with validated
hardware-debugging decisions.

## Current layout

- `dataset/raw/`: curated copies of the selected upstream sources.
- `dataset/extracted/`: archives unpacked for inspection and conversion.
- `dataset/CONVERSION_PLAN.md`: source-by-source conversion plan.
- `dataset/SILICOJEV_SCHEMA.json`: normalized record contract.
- `dataset/EXTERNAL_DATASETS.md`: acquired RTL-BenchLS, Fixbench-RTL, and
  CVDP data, licenses, and curation rules.
- `reference/jev/`: Laya/Laya.cpp, official TypeSafe clients/skills, and
  community routing/evaluation references.

The original downloads under `../data/` are intentionally preserved. The
curated corpus is a working copy so source data can be re-audited.

The notebook keeps the newly acquired external corpora raw until a
source-specific adapter, replay check, and repository-disjoint split are in
place. Unreplayed OriGen repair pairs are excluded from the notebook's default
conversion path.

## Model strategy

Start from the downloaded `laya-typed-decisions` checkpoint and fine-tune it
as `silicojev`. Keep the Laya output shape initially so the existing runtime
can evaluate SilicoJev and comparisons remain meaningful. The model name,
training metadata, and Hugging Face repository can change without changing the
decision primitives.

Training must be resumable: save model, optimizer, scheduler, scaler, RNG,
epoch, batch position, dataset fingerprint, and configuration in every
checkpoint. Never resume a checkpoint against a changed dataset without
explicitly recording the new dataset version.

## Evaluation contract

`training/evaluate_silicojev.py` follows the original Laya benchmark shape. It
reports exact accuracy, soft accuracy (the predicted probability assigned to
the soft gold distribution), Brier score, KL divergence, total variation, ECE,
and inference latency at p50/p95. It also writes `*_by_source.json` with the
same metrics grouped by source (`hwe-bench`, `RootCause-Bench`, and so on), so
performance on validated and weakly labeled sources is visible separately.

The final training step fits one validation-set temperature per typed head
(`choice`, `score`, `noul`) with LBFGS on negative log likelihood and stores
the result in `rl_agent_config.json`. A checkpoint without a fitted
temperature is reported as uncalibrated rather than silently treated as
calibrated.

Score is explicitly disabled in the current normalized release. The selected
hardware sources do not supply a trustworthy ordered urgency/priority label,
so adding a fabricated score target would make RPS, score MAE, and
within-one-level numbers misleading. The evaluator will enable those metrics
only after a source with defensible ordered labels is added.

## Pseudo-score merge (training experiments only)

The ten-worker annotation pass has been structurally audited and merged into:

- `dataset/normalized/pseudo_scores/all_with_pseudo_scores_reviewed.jsonl`
- `dataset/normalized/pseudo_scores/train_with_pseudo_scores.jsonl`
- `dataset/normalized/pseudo_scores/validation_with_pseudo_scores.jsonl`
- `dataset/normalized/pseudo_scores/test_with_pseudo_scores.jsonl`
- `dataset/pseudo_annotation/review_report.json`

The merge preserves the original records and adds `risk` and `urgency` score
questions. It corrected no semantic judgments automatically: all 12,496 score
labels remain `codex_pseudo_unverified`. The audit found 617 low-confidence
records and 631 records where a `no time pressure` urgency label was not proven
by the evidence. The pseudo-score splits may be used for exploratory training,
but must not replace the independently validated test set.
