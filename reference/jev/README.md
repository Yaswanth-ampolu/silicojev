# Jev/Laya reference repositories

These checkouts are reference material for interface design, calibration,
serving, and evaluation. None contains Jev's private weights or training
corpus.

| Local path | Upstream | What it contributes |
|---|---|---|
| `../../../laya` | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | Laya architecture, RLCD-style proper-scoring training, calibration, and the original notebook. |
| `jev/laya.cpp` | [lkarlslund/laya.cpp](https://github.com/lkarlslund/laya.cpp) | Native ggml/CUDA inference, checkpoint validation, output compatibility, and serving. |
| `jev/typesafe-ai-skills` | [typesafe-ai/skills](https://github.com/typesafe-ai/skills) | Official question/state design guidance for System One API use. |
| `jev/typesafe-sdk-python` | [typesafe-ai/typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python) | Official Python API client; integration only. |
| `jev/system-one-adapter-python` | [typesafe-ai/system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) | LLM-backed compatibility adapter; useful for baseline comparisons, not Jev training. |
| `jev/typesafe-router` | [TypeSafe router example](https://github.com/TypeSafeAI/typesafe-router) | Choice validation, confidence thresholds, deterministic fallback, and executor separation. |
| `jev/tenbin` | [simota/tenbin](https://github.com/simota/tenbin) | Question linting, labeled evaluation, confidence-band analysis, and threshold measurement. |
| `jev/typesafe-arena` | [DeepBlueDynamics/typesafe-arena](https://github.com/DeepBlueDynamics/typesafe-arena) | Community documentation and workflow patterns; official docs remain authoritative. |

The training implication is clear: reuse Laya's encoder/head and typed output
format for SilicoJev. Apply the router/tenbin lessons to question design,
thresholds, and evaluation. Use Unsloth only for a separate decoder-model
baseline such as Qwen or DeepSeek; it does not replace Laya's custom decision
head or RLCD objective.
