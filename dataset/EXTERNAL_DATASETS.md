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
- Dataset license: CC-BY-4.0; upstream source licenses remain applicable.

## Fixbench-RTL

- Source: <https://huggingface.co/datasets/KSU-HW-SEC/Fixbench-RTL>
- Local checkout: `dataset/raw/hf/Fixbench-RTL`
- Data: 100 records containing buggy RTL, corrected RTL, bug description, and
  a testbench.
- Use: compact bug-repair training/evaluation after replaying the supplied
  testbenches. `correctcode` is a target, not input state.
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
