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

The bucket inputs are `silicojev_5q.jsonl`, `fixbench_rtl_5q.jsonl`, `rtl_benchls_5q.jsonl`, and `veribugbench_5q.jsonl`. They already use the SilicoJev/Laya typed record format, so the pipeline does not rewrite the task state, questions, or gold probabilities. The base SilicoJev split is retained; RTL-BenchLS keeps its repository-disjoint split; Fixbench-RTL is split by its audited lineage groups; VeriBugBench is split by project. Rows source-flagged `eval_excluded` are recorded in the manifest and held out of all generated splits.

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

`training/train_silicojev.py` is the authoritative trainer. It retains Laya's encoder and typed heads, uses BF16 where supported, enables gradient checkpointing, and periodically saves resumable checkpoints. `training/prepare_hf_bucket_data.py` validates and combines the bucket datasets while keeping splits group-disjoint. `training/evaluate_silicojev.py` reports exact/soft accuracy, Brier score, KL divergence, total variation, ECE, latency, and metrics by source, question type, and label source; score metrics must be interpreted in light of each label's provenance.

The notebook defaults are aimed at a single 80 GB H100 (BF16, micro-batch 32, accumulation 2) and reduce the batch for lower-memory GPUs. This environment does not have the H100, so the actual GPU smoke test must be run in the target Jupyter instance before full training.
