# SilicoJev

SilicoJev adapts Laya's typed decision model to RTL/chip-design and design-verification tasks. Given a task state, it returns calibrated distributions for five independent questions: `next_action`, `root_cause_type`, `evidence_sufficient`, `risk`, and `urgency`. It is a specialized decision model, not a free-form code-generating LLM.

## Fresh clone and training

Clone the repository and launch the new notebook from the repository root:

```bash
git clone https://github.com/Yaswanth-ampolu/silicojev.git
cd silicojev
jupyter lab silicojev_training_hf.ipynb
```

Run the notebook in order. It:

1. Finds the repository root without machine-specific paths and checks CUDA/BF16/VRAM.
2. Installs notebook dependencies without replacing CUDA-enabled PyTorch.
3. Authenticates through Hugging Face's secure interactive login prompt if needed; tokens are not put in the notebook or repository.
4. Downloads the four 5-question JSONL files from the public `Yaswanth-ampolu/silicojev` bucket into ignored `dataset/cache/hf_bucket/`.
5. Validates the typed-question schema, gold probability distributions, duplicate IDs, split integrity, and family/project leakage. It writes SHA-256 hashes and split/source counts to ignored `dataset/prepared/hf_bucket_v1/manifest.json`.
6. Clones the pinned Laya source into ignored `external/laya/` and downloads the public base checkpoint into ignored `models/laya-typed-decisions/`.
7. Runs a one-optimizer-step smoke test into a separate checkpoint directory. Full training stays disabled until the smoke results are reviewed.
8. Fine-tunes with the existing Laya typed-decision architecture and RLCD-style proper-scoring plus soft cross-entropy objective. It saves resumable checkpoints under `checkpoints/hf_bucket_v1/` and evaluates the held-out test split.
9. Optionally uploads only inference artifacts to a Hugging Face model repository. Upload is disabled by default, and the default target is private; optimizer/RNG state is excluded.

The notebook's first Hugging Face login uses `huggingface_hub.login()`'s interactive prompt. You may instead log in from a terminal before opening Jupyter with `hf auth login`; the notebook checks the local Hugging Face credential store. Never paste access tokens into notebook cells.

## Data and label policy

The bucket inputs are `silicojev_5q.jsonl`, `fixbench_rtl_5q.jsonl`, `rtl_benchls_5q.jsonl`, and `veribugbench_5q.jsonl`. They already use the SilicoJev/Laya typed record format, so the pipeline does not rewrite task state, questions, or gold probabilities. Preparation writes four group-disjoint splits: `train`, `validation` (dev/checkpoint selection), `calibration` (temperature fitting only), and `test` (final-only evaluation). Existing test assignments are preserved; existing validation groups are deterministically separated into validation and calibration groups. RTL-BenchLS remains repository-disjoint, Fixbench-RTL is split by audited lineage families, and VeriBugBench by project. Rows source-flagged `eval_excluded` are recorded in the manifest and excluded. The manifest records split counts, decision/source/label-source distributions, file/content fingerprints, and weighted training mass by source and label source.

Some targets are verified or benchmark/manual labels, while others are inferred, teacher-judged, synthetic, or pseudo/unverified. The preparation script adds explicit per-question `training_weights` without promoting or modifying gold labels. The initial policy assigns lower weights to weak labels; it is an experimental setting and is documented in `training/prepare_hf_bucket_data.py`. The evaluation data and provenance remain available for auditing; pseudo-score results must not be represented as validated quality.

Bucket contents can be replaced independently of a Git commit. The generated data manifest records local SHA-256 values so a training run can be tied to exact downloaded bytes. For a fully immutable data source, publish the files in a versioned Hugging Face Dataset repository and pin its commit revision.

## Model, code, and artifact locations

- Base checkpoint: public `convaiinnovations/laya-typed-decisions`, downloaded at the pinned revision configured in the notebook.
- Laya source: cloned at a pinned commit into `external/laya/`.
- Dataset cache and prepared split files: `dataset/cache/` and `dataset/prepared/`.
- Fine-tuning checkpoints, including optimizer and RNG state: `checkpoints/`.
- Evaluation reports: `evaluation/hf_bucket_v1/`.
- Optional published inference model: selected Hugging Face model repository.

Model weights, caches, and training checkpoints are deliberately ignored by Git; do not push these large artifacts to the source repository. The notebook uploads model weights to Hugging Face only when `UPLOAD_MODEL_TO_HF = True` is explicitly enabled.

## Training implementation

`training/train_silicojev.py` is the authoritative trainer. It retains Laya's encoder, heads, and RLCD/proper-scoring plus soft cross-entropy loss. Validation selects `best/` by Brier score (minimum by default); the final epoch is saved separately in `final/`. Temperature fitting uses only `calibration.jsonl`, and removes inherited `temperature_by_options` because Laya gives those per-option values precedence. Test data is loaded only for final evaluation. Step checkpoints are atomically written and pruned; `latest` resumes training while `best/` and `final/` are retained. Resume validates content-based data/model fingerprints and semantic training settings, allows epoch/max-step extension, and ignores harmless loader-worker changes.

Preparation and training CLI sequence (after downloading the four bucket files into `dataset/cache/hf_bucket/` and preparing a Laya checkpoint under `models/laya-typed-decisions/`):

```bash
python training/prepare_hf_bucket_data.py \
  --input-dir dataset/cache/hf_bucket \
  --output-dir dataset/prepared/hf_bucket_v1

python training/train_silicojev.py \
  --model-dir models/laya-typed-decisions \
  --train dataset/prepared/hf_bucket_v1/train.jsonl \
  --validation dataset/prepared/hf_bucket_v1/validation.jsonl \
  --calibration dataset/prepared/hf_bucket_v1/calibration.jsonl \
  --test dataset/prepared/hf_bucket_v1/test.jsonl \
  --output-dir checkpoints/hf_bucket_v1 \
  --dtype bf16 --epochs 4 --micro-batch-size 32 --grad-accum-steps 2 \
  --encoder-lr 2.5e-5 --head-lr 1e-4 --warmup-ratio 0.05 \
  --checkpoint-every 250 --keep-checkpoints 2 --resume auto
```

Suggested controlled A100/H100 starting point: BF16, 4 epochs, effective batch 64 (H100/A100-80GB `32×2`; adjust explicitly for A100-40GB), encoder/head LR `2.5e-5`/`1e-4`, max/head lengths 1024/256, 5% warmup, cosine decay, gradient checkpointing `auto`, fused AdamW `auto`, and option permutation disabled (`0.0`). `--attention-backend auto` selects FlashAttention 2 only when installed/initializable, otherwise SDPA. These are reference settings, not claims of optimality. Tune one experimental setting at a time.

For a base-model A/B run, download each candidate at its own pinned Hub revision into a separate model directory and pass the matching pair with `--base-model-id` and `--base-model-revision`; use separate output directories so their checkpoints, metrics, and resume identities cannot mix.

The trainer reports GPU, CUDA, VRAM, precision, throughput, token lengths/truncation, validation metrics, and writes append-only `metrics.jsonl` plus optional TensorBoard events. This repository runtime has no CUDA device, so only CPU-side tests can be run here; perform the required BF16 forward/backward/checkpoint/resume smoke test on the target A100/H100 before launching the full run.
