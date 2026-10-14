# Kimi-VL-A3B-Instruct reference result

This page records a completed local measurement for Kimi-VL-A3B-Instruct on an
NVIDIA A800-SXM4-80GB. The repository does not redistribute model weights,
evaluation data, raw answers or the local result tree.

## Final result

The selected serving profile combines physical structured pruning, a freshly
quantized Parent, post-encoder visual Token selection, Runtime compilation and
automatic KV-cache selection. It retains **95.8906%** of the model's own Dense
quality across five equally weighted tasks and reaches **4.5465x** matched
batch-1 end-to-end acceleration. The final outcome is `goal_achieved` and the
quality freeze is `lmms_qualified`.

## System and serving setting

| Setting | Selected value |
| --- | --- |
| Model | Kimi-VL-A3B-Instruct, native Instruct/non-thinking protocol |
| GPU and driver | NVIDIA A800-SXM4-80GB, driver 580.126.09 |
| Dense reference | Transformers, BF16, native eager attention |
| Structured pruning | `sp-0070`; 23.6237% stored decoder FFN-parameter removal |
| Parent quantization | GPTQModel W4A16, group size 128; fresh Stack Parent |
| Visual Token | `token-75`; keep ratio 0.75, float32 scores, 4 x 4 coverage grid |
| Runtime | `mm-compile-o2`; optimization level 2, multimodal encoder compilation on, encoder CUDA Graph off, prefix caching off |
| Attention and MoE | Requested backend `auto`, observed `TRITON_MLA`; Marlin MoE, native router precision and expert topology |
| KV cache | `kv-auto`; dtype `auto` |
| Context | vLLM maximum model length 131,072 |

The model retains 64 routed experts, top-6 routing and two shared experts. The
Token selector compacts post-projector visual features while preserving the
native image preprocessing and prompt protocol.

## Stack candidates

Each candidate completed all five native LMMS tasks over 23,349 samples. The
table reports per-task retention against the same Kimi Dense denominator; the
mean is the equal-task mean of the five ratios.

| Candidate | SP / Token setting | FFN removal | ChartQA | MMStar | TextVQA | DocVQA | POPE | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `sp-0098` / `token-65` | 33.0732% | 86.9857% | 103.8932% | 85.6226% | 82.4958% | 100.9992% | 91.9993% |
| 2 | `sp-0084` / `token-70` | 28.3484% | 89.0429% | 101.8058% | 88.8328% | 87.8178% | 100.6758% | 93.6350% |
| **3 (selected)** | **`sp-0070` / `token-75`** | **23.6237%** | **91.9052%** | **101.7241%** | **91.4461%** | **92.1550%** | **100.4972%** | **95.5455%** |

All three candidates use the same fresh GPTQModel W4A16/group-128 Parent. Stack
LMMS runs use the native Transformers path; Runtime and KV settings are applied
in the final serving LMMS below.

## Final serving LMMS

| Task | Samples | Metric | Dense | Final serving | Retention |
| --- | ---: | --- | ---: | ---: | ---: |
| ChartQA | 2,500 | relaxed overall | 0.894400 | 0.824800 | 92.2182% |
| MMStar | 1,500 | category mean | 0.522915 | 0.540587 | 103.3796% |
| TextVQA | 5,000 | exact match | 0.887780 | 0.810600 | 91.3064% |
| DocVQA | 5,349 | ANLS | 0.987107 | 0.907719 | 91.9575% |
| POPE | 9,000 | F1 | 0.858343 | 0.863419 | 100.5914% |
| **Equal-task mean retention** | **23,349** |  |  |  | **95.8906%** |

The gate averages the five task-level retention ratios equally. It is not a
pooled sample score or a per-task minimum requirement.

## Matched timing

The Dense and serving backends used the same physical GPU, 128 native Search
inputs, five warmups and three measured rounds (384 requests per backend), with
batch size 1. Timing spans image opening, preprocessing, transfer, vision
encoding, generation, decoding and synchronization. Loading, warmup, cache
reset and post-run audits are excluded; the in-engine Token trace is included.

| Metric | Transformers Dense | Final serving |
| --- | ---: | ---: |
| Requests | 384 | 384 |
| Total time | 346.8998 s | 76.2998 s |
| Mean request time | 903.385 ms | 198.697 ms |
| P50 request time | 709.042 ms | 179.816 ms |
| P95 request time | 3,431.422 ms | 351.509 ms |
| Output tokens | 4,533 | 4,237 |
| Request throughput | 1.1069/s | 5.0328/s |
| Output-token throughput | 13.0672/s | 55.5309/s |

The matched request-time speedup is **4.5465x**. Natural answer lengths differ,
so output-token throughput is reported separately; its ratio is 4.2497x.

## Evidence and reproduction scope

The complete local evidence is retained under
`results/model/kimi_vl/a800/`:

- `stack/summary.json` and `stack/final/stack-freeze.json`
- `stack-serving-lmms/summary.json` and `stack-serving-lmms/final/serving-freeze.json`
- `serving-use/benchmark/summary.json`
- `preparation/dense-five-complete-tasks.json`

The public source contains the Kimi adapter, stage interfaces, JSON jobs,
runtime bridge and regression tests. A fresh run still requires the original
checkpoint, native LMMS data, Search/Dev manifests, calibration inputs and
isolated backend environments. These measurements are local evidence and are
not claimed as a source-only reproduction. See the [Kimi preparation notes](../kimi-vl-preparation.md)
and [release preparation policy](../releasing.md) for the evidence boundary.
