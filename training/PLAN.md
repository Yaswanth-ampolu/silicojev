# SilicoJev A100 training plan

## Starting checkpoint

Use `/home/jovyan/shared/jevlikeresearch/models/laya-typed-decisions` as the
initial checkpoint. Keep the original Laya checkpoint immutable and write all
SilicoJev outputs below `silicojev/checkpoints/`.

The first SilicoJev release should keep Laya's output contract. That means the
model still returns `choice`, `noul`, and later `score`; only the learned domain
behavior and metadata name change.

## First run

- One A100-SXM4 40 GB.
- Mixed precision, preferably BF16 on A100.
- Gradient checkpointing enabled.
- `max_len=1024` and `head_max_len=256` initially.
- Begin with one epoch and a small validated subset as a smoke test.
- Continue with 3–4 epochs only after validation and loss checks pass.
- Keep the first action space below 10 options.

The official Laya recipe uses 4 epochs and an effective batch of about 64
question sequences. The A100 gives enough memory to use one process and tune
the micro-batch upward, but we should measure throughput rather than assume a
batch size.

## Resume/checkpoint requirements

Save an atomic checkpoint at the end of every epoch and periodically by
optimizer step. Each checkpoint must contain:

- model weights,
- optimizer state,
- scheduler state,
- AMP scaler state when used,
- epoch and optimizer-step counters,
- Python, NumPy, and Torch RNG states,
- dataset fingerprint and split name,
- complete training configuration,
- validation metrics and git commit.

Resume must restore all of these before reading the next batch. If the dataset
fingerprint or configuration differs, require an explicit new run name instead
of silently continuing.

## Data order

1. Build and validate normalized records.
2. Deduplicate and split by repository/bug lineage.
3. Generate tokenized training items and record their fingerprint.
4. Run a forward-only validation pass.
5. Train a one-epoch smoke checkpoint.
6. Evaluate on unseen projects.
7. Run the longer fine-tune only if the smoke run is healthy.

No full training run should begin on raw or unvalidated records.

## Metrics and calibration

The evaluation report mirrors the original Laya notebook: exact and soft
accuracy, Brier score, KL divergence, total variation, ECE, and latency p50/p95
are reported overall, by question type, and by source. A separate
`<report>_by_source.json` artifact is written for source-level comparisons.

After the final validation pass, the trainer fits temperatures for each typed
head with validation negative log likelihood and LBFGS, then writes them into
the final checkpoint configuration. The calibration metadata records the
split, method, and item count for each type.

Score/RPS, score MAE, and within-one-level are currently disabled. The
normalized corpus has no defensible ordered score labels; the report states
this explicitly. Do not add score targets until an ordered hardware-debugging
rubric is sourced and independently validated.

The current ten-worker teacher pass is available as a separate experimental
merge under `dataset/normalized/pseudo_scores/`. It adds `risk` and `urgency`
with soft distributions, but all labels remain unverified. Structural review
confirmed complete shard coverage, preserved original fields, valid score
probabilities, and stable splits. It also flags low-confidence records and
cases where the evidence does not prove the lowest urgency level. These files
must not be used as final evaluation truth.
