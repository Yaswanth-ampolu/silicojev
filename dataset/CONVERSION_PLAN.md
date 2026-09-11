# SilicoJev dataset conversion plan

## Goal

Convert heterogeneous RTL/DV bug and benchmark records into Laya-compatible
decision cases. Raw RTL, repaired code, and benchmark prompts are not labels by
themselves. Each normalized case needs a pre-decision state, typed questions,
and a gold probability distribution.

## Normalized case

One case can contain several independent questions. The state must contain only
evidence available before the decision. The repaired RTL, patch, hidden root
cause, and post-fix result stay in metadata or the gold/outcome fields, never in
the input state.

```json
{
  "id": "stable-source-case-id",
  "source": "hwe-bench",
  "state": {
    "repository": "lowRISC/ibex",
    "commit": "buggy-commit",
    "rtl_context": "relevant RTL or a bounded summary",
    "testbench_context": "relevant test or verification context",
    "tool": "verilator",
    "logs": "failure output",
    "waveform_summary": null,
    "previous_actions": []
  },
  "questions": {
    "next_action": {
      "type": "choice",
      "instructions": "What diagnostic direction should be investigated next?",
      "criteria": {
        "rtl": "Inspect RTL implementation",
        "testbench": "Inspect testbench or stimulus",
        "waveform": "Inspect waveform or cycle behavior",
        "formal": "Run formal verification",
        "specification": "Check protocol or design specification",
        "simulation": "Run a targeted simulation",
        "ask_human": "Request human review",
        "abstain": "Evidence is insufficient"
      }
    },
    "evidence_sufficient": {
      "type": "noul",
      "instructions": "Is there enough evidence to select a likely root cause?",
      "criteria": {
        "false": "Evidence is insufficient",
        "true": "Evidence is sufficient"
      }
    }
  },
  "gold": {
    "next_action": {
      "probabilities": {
        "rtl": 0.8,
        "testbench": 0.05,
        "waveform": 0.1,
        "formal": 0.02,
        "specification": 0.01,
        "simulation": 0.01,
        "ask_human": 0.005,
        "abstain": 0.005
      },
      "label_source": "verified_repair"
    },
    "evidence_sufficient": {
      "probabilities": {"false": 0.1, "true": 0.9},
      "label_source": "expert_annotation"
    }
  },
  "outcome": {
    "resolved": true,
    "verification": "simulation",
    "time_seconds": null
  },
  "provenance": {
    "license": "source-specific",
    "synthetic": false,
    "bug_family": "protocol"
  }
}
```

The Hugging Face training files can store `state`, `questions`, and `gold` as
JSON strings in Parquet. The extra metadata columns are retained for auditing.

## Source adapters

| Source | Initial extraction | Label confidence |
|---|---|---|
| HWE-Bench | PR metadata, buggy baseline, failure/reproducer, modified files, fix patch, pass result | High for repair outcome; action category needs mapping | 
| HierSVA | buggy RTL, hierarchy, specification, failing/assertion context, synthetic or historical bug tag | High when formal result is available; mark synthetic/history | 
| RootCause-Bench | RTL/testbench, synthesis/runtime error, manually assigned bug type | High but only 34 cases | 
| Inspect-Eval-ChipBench | debug prompt, buggy RTL, optional waveform, simulator-confirmed result | High for the benchmark result; preserve bug type | 
| OriGen debug | instruction containing original RTL/error plus corrected response | Medium; parse and validate the correction before labeling | 
| RTL-augmented | pre-mutation RTL, mutation type, simulator status, generated case | Medium to high only for `sim_ok` entries | 
| CircuitNet mutation artifacts | source RTL, mutation, tool output, timing/power result | Use for PPA decisions; do not treat all rows as debug labels | 

## Validation gates

1. Parse the record and confirm required fields.
2. Hash repository, commit, bug family, and normalized RTL to remove duplicates.
3. Ensure the input excludes the patch, fixed code, and hidden label.
4. For synthetic mutations, require compile/simulation/formal success for the
   expected failure and a passing result after repair where available.
5. Store whether the label is expert, tool-derived, historical, or generated.
6. Use one-hot targets only when there is one observed action; use soft targets
   for multiple expert judgments or genuinely ambiguous next actions.
7. Split by repository and bug lineage, not random rows.
8. Keep a held-out project set for transfer testing.

## First training slice

Start with validated cases from HWE-Bench, HierSVA's 28 deep cases,
RootCause-Bench, ChipBench debugging, and `sim_ok` RTL augmentations. This
will produce a small but defensible decision set. Add OriGen and broad code
corpora later for context generation and retrieval, not as direct decision
labels.
