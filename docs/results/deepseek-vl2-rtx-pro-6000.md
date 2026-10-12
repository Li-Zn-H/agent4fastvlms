# DeepSeek-VL2 reference result

This page records the local reference measurement for the full DeepSeek-VL2
checkpoint on an RTX PRO 6000 Blackwell Server Edition. It is a report of a
completed local run; the repository does not redistribute model weights,
evaluation data, raw answers or the 249 GB result tree.

## Final result

The selected serving profile uses physical structured pruning, a freshly
quantized Parent and post-encoder visual Token selection. It retains **97.2659%**
of the model's own Dense quality across five equally weighted tasks and reaches
**4.8167x** matched batch-1 end-to-end acceleration.

| Setting | Selected value |
| --- | --- |
| Structured pruning | 28.4587% stored decoder FFN parameter removal (`sp-0114`) |
| Parent quantization | GPTQ W4A16, group size 128 |
| Visual Token | Keep ratio 0.85; 167/196 main features per native tile; layout positions protected |
| Runtime | O2; effective compilation mode 3; `FULL_AND_PIECEWISE` CUDA Graphs |
| Attention and MoE | TRITON_MLA, Marlin MoE, F32 routers; all 72 routed experts, top-6 routing and two shared experts retained |
| KV cache | Auto |

### Native serving quality

| Task | Dense | Final serving | Retention |
| --- | ---: | ---: | ---: |
| ChartQA | 0.773200 | 0.764400 | 98.8619% |
| MMStar | 0.628971 | 0.585878 | 93.1486% |
| TextVQA | 0.816160 | 0.807280 | 98.9120% |
| DocVQA | 0.925665 | 0.907579 | 98.0462% |
| POPE F1 | 0.907998 | 0.884033 | 97.3607% |
| **Equal-task mean** |  |  | **97.2659%** |

The final gate covers 23,349 native samples. The equal-task mean is the average
of the five candidate/Dense retention ratios; it is not a pooled sample score or
a per-task minimum. The final freeze is `lmms_qualified`.

### Matched timing

Both backends used the same physical GPU, native Search128 inputs and order, five
warmups and three measured rounds. The timer spans image opening, preprocessing,
transfer, vision encoding, generation, decoding and synchronization. Model
loading, warmup and post-generation audits are excluded.

| Metric | Original Transformers Dense | Final serving |
| --- | ---: | ---: |
| Requests | 384 | 384 |
| Mean request time | 0.982524 s | 0.203985 s |
| P50 request time | 0.707654 s | 0.162891 s |
| P95 request time | 2.701954 s | 0.385954 s |
| Output-token throughput | 19.4095/s | 92.9271/s |

The measured request-time speedup is **4.816650x**. Output lengths naturally
differ, so output-token throughput is reported separately. The 128-request
timing workload is performance evidence; the five-task LMMS run supplies the
quality gate.

## Stack candidates

All six candidates completed the same five native tasks. The first five were
rejected by the 95% mean-retention gate; the sixth passed.

| Candidate | FFN removal | Token keep | ChartQA | MMStar | TextVQA | DocVQA | POPE | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.1394% | 55% | 82.8246% | 85.2704% | 86.2061% | 91.5135% | 97.8176% | 88.7265% |
| 2 | 47.4041% | 60% | 84.2732% | 85.6969% | 91.2027% | 93.2883% | 97.3024% | 90.3527% |
| 3 | 42.6648% | 65% | 75.0647% | 89.9650% | 94.5599% | 94.5625% | 97.6302% | 90.3564% |
| 4 | 37.9294% | 75% | 86.6011% | 89.8742% | 97.5103% | 96.1373% | 98.8285% | 93.7903% |
| 5 | 33.1941% | 80% | 91.3088% | 90.8218% | 98.0666% | 96.5676% | 98.0653% | 94.9660% |
| **6 (selected)** | **28.4587%** | **85%** | **99.3275%** | **93.4501%** | **98.7772%** | **97.9759%** | **97.5749%** | **97.4211%** |

## Reproduction scope

The public source contains the adapter, model-specific JSONs, tests and method
documentation. A fresh run requires the original checkpoint, native LMMS data,
Search/Dev manifests, calibration inputs and isolated backend environments.
These local paths contain the complete raw evidence for the reference run:

- `results/model/deepseek_vl2/rtx_pro6000_gpu072a4781/`
- `preparation/final-integration-completion.json`
- `stack-serving-lmms/final/serving-freeze.json`
- `serving-use/benchmark/summary.json`

The public source edition reports these measurements without claiming that a
source-only checkout can reproduce them. See the [integration guide](../deepseek-vl2-preparation.md),
[full status page](../status.md#deepseek-vl2-cuda-2026-10-12) and
[release preparation notes](../releasing.md) for the evidence boundary and
licensing requirements.
