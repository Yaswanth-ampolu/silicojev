# Merged five-question dataset (converted sources)

The two converted sources — RTL-BenchLS Task 3 and Fixbench-RTL — merged onto their description-bearing base records, so every row carries all five SilicoJev questions: the three source-derived base questions (`next_action`, `root_cause_type`, `evidence_sufficient`) plus the distilled `risk` and `urgency` scores.

`all.jsonl` is the combined set; `rtl_benchls.jsonl` and `fixbench_rtl.jsonl` are the per-source subsets. Each row preserves the base record's `state` (the case description) byte-identically, and the three base questions and golds are verified unchanged.

The two added score labels are model judgments marked `codex_pseudo_unverified` with `defensible: false`. They are for exploratory training only and must not be used as final evaluation truth. This directory is a standalone artefact: it is not merged into `dataset/normalized/` and has not been fed to Laya.
