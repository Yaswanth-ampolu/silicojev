# External RTL/DV datasets

Acquired on 2026-09-22. These files are raw inputs only; they are not merged
into the normalized SilicoJev training split automatically.

## RTL-BenchLS

- Source: <https://github.com/hkust-zhiyao/RTL-BenchLS>
- Local checkout: `dataset/raw/github/RTL-BenchLS`
- Data: 9,698 formally verified Task 1 designs, 425 masked-content records,
  and 108 repository-issue cases.
- Task 3 upstream repositories are cached under
  `dataset/raw/github/RTL-BenchLS/repo_cache/`; the exact base/head commit
  objects for all 108 cases were checked and are available. The
  `analogdevicesinc/hdl` working tree has missing Git-LFS checkout files, but
  its required commit objects are available for commit-based extraction.
- Use first: the 108 repository-issue cases. They contain real issue/PR
  metadata, base/head commits, patches, and formal-verification status.
- Do not put the golden patch or head-commit RTL in the model state; retain it
  as a target/evaluation artifact.
- Converted Task 3 decision records live in `dataset/converted/rtl_benchls/` as a
  standalone source: 34 trusted 3-question records, 2 trusted 5-question records,
  and unverified/pseudo files for separate experiments. Read that directory's
  `README.md` and `usage_sets.json` before use; those files are deliberately
  *not* merged into `dataset/normalized/`.
- Dataset license: CC-BY-4.0; upstream source licenses remain applicable.

## Fixbench-RTL

- Source: <https://huggingface.co/datasets/KSU-HW-SEC/Fixbench-RTL>
- Local checkout: `dataset/raw/hf/Fixbench-RTL`
- Data: 100 records containing buggy RTL, corrected RTL, bug description, and
  a testbench.
- Use: compact bug-repair training/evaluation after replaying the supplied
  testbenches. `correctcode` is a target, not input state.
- Converted decision records live in `dataset/converted/fixbench_rtl/` as a
  standalone source: 100 three-question records (35 in the trusted nested-gate
  set, 54 with replay-verified repair evidence, 52 flagged for review, and 46
  not replay-verified: 21 where the supplied testbench passes on the buggy code,
  13 that build under neither revision, 11 whose failure survives the repair,
  1 with no usable verdict), 0 defensible 5-question records, and a rubric-only
  pseudo 5-question file. Read that directory's `README.md` and
  `usage_sets.json` before use; those files are deliberately *not* merged into
  `dataset/normalized/`.
- Dataset license: CC-BY-4.0.

## CVDP v1.1.0

- Source: <https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset>
- Local checkout: `dataset/raw/hf/CVDP`
- Downloaded: non-commercial agentic generation, heavy-task metadata,
  non-agentic generation, and code-comprehension files, plus LICENSE/NOTICE.
- Downloaded lightweight records: 592. The full release includes additional
  commercial-use-restricted records and large public project bundles; those
  were not copied into this first acquisition.
- Use: primarily held-out DV evaluation and tool/verification workflow
  analysis. Do not train on `patch`/`output` and then evaluate on the same
  cases.
- Licensing: mixed; follow `LICENSE` and `NOTICE` plus each upstream
  repository's license.

## VeriBugBench v1.0

- Source: <https://github.com/wndif/VeriBugBench>
- Local checkout: `dataset/raw/github/VeriBugBench`
- Commit: `463d490e261dff247eb6ef82b07b6e924fee19e8` (committed 2026-08-31).
- Data: 45 projects and 2,608 selected single-fault instances. The frozen manifest
  (`dataset/v1.0/manifest.yaml`) states `projects: 45` and `instances: 2608`, and this
  was independently confirmed: `metadata/instances.csv` has 2,609 lines = 2,608 data
  rows, and 45 `veribugbench_v_1_0.yaml` project manifests are present.
- Size: 854,155,898 bytes (checkout on disk including `.git`: ~992 MB).
- sha256: `LICENSE` = `25f69cbc8e3172f3044990a7d39e68c500ff358a26b6ff1b4a90413943af8455`;
  `dataset/v1.0/manifest.yaml` = `f54dd2a5671c98aa6e80977710c22d95767210dd1a6769f36f1fd5fc0ff59f56`;
  `metadata/instances.csv` = `8cf4fa019e3687f63351af210d4a83d9a0e9ac0e6af3e0ee87c641544f29fc98`;
  `dataset/v1.0/project_paths.json` = `ccda7e5327324ceced0258f0c54658a7352f88700730eb4c0355d5461699192b`.
- Licence: **MIT covers VeriBugBench software only.** `LICENSE` reads "MIT License".
  This is *not* a licence-clean RTL corpus: `metadata/projects.csv` carries a
  `license_scope` column whose value is "upstream project license; not covered by
  repository MIT license" for **all 45** projects, and `licenses/` contains only
  `MANTRA-MIT.txt`. `THIRD_PARTY_NOTICES.md` lists Mantra (MIT) and PyVerilog
  (Apache-2.0) as third-party software, and states that the repository-level MIT
  licence "does not replace the licenses of bundled RTL projects". Per-project upstream
  licences (source groups: CirFix 11, Native/opencores 11, RTLLM 23) must be enumerated
  and resolved before any training use.
- Verification: the frozen release was executed with **Synopsys VCS** as the primary
  backend and **URG** for time-windowed coverage. `iverilog`/`vvp` is provided only as a
  lightweight compatibility backend and is "not guaranteed to support the complete
  benchmark". The published reference/mutant verdicts are therefore VCS-derived;
  reproducing them locally requires a licensed VCS installation.
- Use: 2,608 instance IDs, each with mutation operator and implementation schema. Keep
  `mantra/` mutant sources, `buggy_versions/` assets, the fault site, the corrected RTL
  and golden outputs out of `state` — these are answers. Golden values come from
  executing the reference RTL, not from publication statistics.
- Lineage warning: source groups overlap sources SilicoJev already vets or uses. Do not
  add the 2,608 blind; de-duplicate by project lineage and split by project, per the
  curation rules below.

## svabench

- Source: <https://github.com/deebakkarthi/svabench>
- Local checkout: `dataset/raw/github/svabench`
- Commit: `9d706e20c23a0d7daca741aa993cc82b8e335729`
- Size: 1,579,106 bytes across 83 tracked files.
- sha256: `LICENSE` = `47596612a02e8749458fb1ff3f0cc7789436cbf858cc065967f498afec8ed827`;
  `README.md` = `47d37b0987a36e18d4cd5c0114bef4771daf123e155fc4344009a62db7bb72bd`.
- Licence: MIT (`LICENSE` reads "MIT License", Copyright (c) 2026 Deebakkarthi
  Chinnasame Rani).
- Use: a SystemVerilog assertion *generation* benchmark. This is an assertion-authoring
  task, not a bug corpus, and carries no repair verdict. At most it can supply
  `next_action: formal` / `specification` readings; it is not a source of debugging cases.

## eda-log-dataset-for-ic-debugging

- Source: <https://huggingface.co/datasets/innofacisteven/eda-log-dataset-for-ic-debugging>
- Local checkout: `dataset/raw/hf/eda-log-dataset-for-ic-debugging`
- Size: 28,898 bytes. This is a **sample pack, not a corpus** — the published files are
  explicitly prefixed `sample_`.
- sha256: `sample_eda_logs.jsonl` = `4addc9a77396a822b1de8c32af61a4b4213e3f22ef8f958e21a80da718cfe32e`;
  `sample_guardrail_cases.jsonl` = `1d9bbe5e14d37fcd49f11c5e3f1d23f6a3c2527cf4defcd3fea228499b1de39c`;
  `sample_rtl_bugs.jsonl` = `fe1b6c49a11bb56485aa0b95d2cb49c7071ae3996800f9f02006191a29837fd6`.
- Licence: MIT (HF dataset card `license: mit`).
- Use: OpenROAD-flow log text — WNS/TNS timing violations, physical-design rules such as
  multi-driven net and latch inferred — plus regex-extracted structured metadata including
  a Severity field. It records the *shape* of EDA log output and carries no executable
  verdict. The Severity field is regex-derived from log lines, not a judged severity
  label, so it must never be used as a score target. Suitable only as a `root_cause_type`
  and `evidence_sufficient` format reference.

## BuggyVerilog

- Source: <https://huggingface.co/datasets/LLM-EDA/BuggyVerilog>
- Local checkout: `dataset/raw/hf/BuggyVerilog`
- Size: 719,196,326 bytes.
- Rows: `generative.jsonl` = 34,565 and `embedding.jsonl` = 74,262 (measured with `wc -l`).
- sha256: `generative.jsonl` = `6398adae35e162100d19bb396abadbffd9f6f9a74f2009b3526ccaa7d822c4da`;
  `embedding.jsonl` = `9121cee05c9faab5ae1a6e0d73781a2dfe8dd5948eb5c2b609e9e2780fe15464`;
  `CARD-README.md` = `1c465fbd8d243a9fd5d634e73fded7034e6aad0617851f208e00a4bbfc23cac9`.
- Licence: Apache-2.0. No `LICENSE` file ships in the repository itself, so the HF dataset
  card (`license: apache-2.0`, saved locally as `CARD-README.md`) is the licence evidence.
  The card's only usage pointer is <https://github.com/CatIIIIIIII/VeriDebug>.
- Use: a buggy-Verilog corpus for repair-style training. The files do not record per-row
  provenance (real upstream bug versus synthetic injection), so row-level provenance is
  unverified until audited. Do not promote any row to a trusted label on the strength of
  membership in this corpus.

## Considered and deliberately not acquired

Sizes measured from the HuggingFace blobs API on 2026-09-24. None of these are in
`dataset/raw/`.

| source | size | licence | why not acquired |
|---|---|---|---|
| `ytliu99/TroLL-Logic-Locking-based-Hardware-Trojans` | 15,437,390,315 B (14.4 GB) | cc-by-4.0 | Oversized; Trojan/logic-locking netlists, not debugging cases with repair verdicts |
| `vkenbeek/verilog-wavedrom` | 5,195,237,764 B (4.8 GB) | MIT | Oversized; synthetic Verilog/WaveDrom pairs, no bugs and no verdicts |
| `hasankursun/soc-builder-rtl-v1` | 1,740,900,587 B (1.6 GB) | Apache-2.0 | Generation-only SoC-builder designs, no bugs and no verdicts |
| `LLM-EDA/pyra` + `pyra_medium` + `pyra_tb` | 237,090,826 B (226 MB) | Apache-2.0 | Generation/SFT data ("Filtered dataset sourced from `bnadimi/PyraNet-Verilog` for SFT"), no bugs |
| `AnandMenon12/VERT` | 37,567 KB repo | **unstated** | Licence unstated; also synthetic assertion augmentation, no verdicts |
| `architect-ubc-capstone/rtl-bug-fix-sft` | 428 rows measured (`exp2_*` are 0 bytes) | **unstated** | Licence unstated; same organisation as the in-corpus `rtl-augmented` source |
| `KSU-HW-SEC/5W1HHardwareDebug` | unmeasurable | MIT | `gated=auto`; row count and contents unverifiable without authentication |
| `KSU-HW-SEC/LLM4SecHW-OSHD` | unmeasurable | Apache-2.0 | `gated=auto`; row count and contents unverifiable without authentication |
| `KSU-HW-SEC/issue_benchmark` | unmeasurable | **unstated** | Licence unstated; also contains LLM outputs, which are not verdicts |
| `tieuandpepper/eda-log-dataset` | 2,189,792,684 B (2.19 GB) | **unstated** | `gated=manual` and licence unstated |
| `swordboom/rtl-verification` | n/a | unstated | **Actively excluded.** Advertises "priority_score (0-100)" and 10k-50k synthetic RTL failures, but its own README documents labels that are `estimated_from_severity` (proxy) or `confidence_based_estimate` (no labels). Generated labels: forbidden by the curation rules. |

Corpus-wide note: no public Questa/vsim transcript corpus was found. HuggingFace search
returns **0** datasets for `modelsim`, `vsim log`, `iverilog`, `verilator`, `cocotb` and
`yosys`. Tool command lines exist only inside benchmarks (CVDP drives `iverilog`/`vvp`;
VeriBugBench drives VCS) and never as a standalone dataset. Tool-command data may inform
`state` and may *derive* an `next_action` reading from the tool actually invoked, but it
is never a label target: `next_action` is a `choice` head over nine fixed verbs, not over
command strings.

Housekeeping note: the CVDP section above records a local checkout at
`dataset/raw/hf/CVDP`, but that directory is **not present** — `dataset/raw/hf/` contains
only `Fixbench-RTL`, `BuggyVerilog` and `eda-log-dataset-for-ic-debugging`, and both
`../data/` and `dataset/extracted/` are empty. CVDP was not re-downloaded here.

## Curation rules

1. Preserve source, repository, commit, license, tool, harness, and replay
   status on every converted record.
2. Keep buggy input separate from corrected RTL, patches, and reference
   answers.
3. A process exit code is not sufficient evidence of a passing design; require
   an assertion/test result or formal-equivalence result.
4. Split by repository/project or bug lineage, not by random individual row.
5. Do not use Jev API outputs as labels or distillation targets. Use objective
   EDA outcomes and independently reviewed decisions.
6. Add `risk` and `urgency` only after a real rubric and independent labels
   exist. The current pseudo-score merge remains experimental.
