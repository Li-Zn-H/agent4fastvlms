# Reported Reference Results

The model targets are Qwen3-VL-4B-Instruct, LLaVA-1.5-7B, InternVL3.5-14B and
the full DeepSeek-VL2 checkpoint. Qwen/LLaVA measurements were checked against
their saved final freezes and benchmark summaries on 2026-10-06; InternVL
completed on 2026-10-09 and DeepSeek-VL2 on 2026-10-12.
Raw receipts and checkpoints
are not distributed with this source edition; this is not a fresh reproduction.

## Completed results

| Model | GPU | Five-task mean retention | End-to-end speedup | Dense / final mean latency |
| --- | --- | ---: | ---: | ---: |
| Qwen3-VL-4B-Instruct | RTX 4090 | 95.4011% | 2.5293x | 773.029 / 305.632 ms |
| LLaVA-1.5-7B | RTX 4090 | 95.5839% | 2.9150x | 570.785 / 195.809 ms |
| InternVL3.5-14B | RTX PRO 6000 | 94.5999% | 2.2728x | 1373.939 / 604.507 ms |

InternVL's outcome is `accelerated_quality_unmet`; its final freeze is
`lmms_evaluated`, not `lmms_qualified`. The user stopped final Runtime recovery
and selected the first complete candidate; this was not ladder exhaustion.

## Selected settings

| Setting | Qwen3-VL-4B | LLaVA-1.5-7B |
| --- | --- | --- |
| Transformers Dense dtype | BF16 | FP16 |
| Native image strategy | Dynamic resolution | Fixed 336 x 336, 576 visual tokens |
| Structured pruning | SP0544, 19.8830% decoder FFN removal | SP0411, 14.9346% decoder FFN removal |
| Quantization | Multimodal GPTQ W4A16, group 128 | Multimodal GPTQ W4A16, group 128 |
| Token keep ratio | 0.80 | 0.85, 490 retained tokens |
| Runtime | O1, observed FLASH_ATTN | O2, FLASHINFER |
| KV dtype | auto | FP8 E4M3 |
| Context limit | 32768 | 4096 |

Quantization includes declared language/vision Linear targets with high-precision
exceptions. Token selects encoded features and preserves native preprocessing.

## Final five-task quality

Scores below are native task values; retention compares each candidate with its
own Dense. The gate equally averages the five retention ratios.

| Task | Samples | Qwen Dense / final | Qwen retention | LLaVA-1.5 Dense / final | LLaVA-1.5 retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| chartqa | 2500 | 0.836800 / 0.795600 | 95.0765% | 0.179200 / 0.168400 | 93.9732% |
| mmstar | 1500 | 0.613117 / 0.571477 | 93.2084% | 0.327464 / 0.340263 | 103.9084% |
| textvqa_val | 5000 | 0.795100 / 0.726320 | 91.3495% | 0.479580 / 0.439600 | 91.6635% |
| docvqa_val | 5349 | 0.946166 / 0.935737 | 98.8977% | 0.237299 / 0.209820 | 88.4203% |
| pope | 9000 | 0.891930 / 0.878313 | 98.4733% | 0.845939 / 0.845552 | 99.9542% |

Every combination has 23349 native task samples and actual Token/Runtime/KV
mechanism evidence. LMMS was used during selection, not as an untouched test.
Search128 is a subset of Dev256. The quality target is neither a sum-score
ratio nor a minimum required separately for every task.

## Matched timing scope

Qwen/LLaVA backends used the same RTX 4090 and driver 580.105.08. Each comparison
measured 128 inputs x 3 rounds = 384 requests per backend, after five warmups.
Generation was greedy, batch 1, with at most 64 new tokens. Natural answer
lengths differ; request speed and output-token throughput are separate metrics.

| Model | Dense / final output tokens | Output-token throughput ratio |
| --- | ---: | ---: |
| Qwen3-VL-4B-Instruct | 18093 / 16257 | 2.2726x |
| LLaVA-1.5-7B | 11130 / 10770 | 2.8207x |

Timing spans image opening, preprocessing, transfer, vision encoding, generation,
answer decoding and GPU synchronization. Loading, warmup, cache reset and
post-run audits are excluded; in-model Token traces are included. Prefix cache
resets and unique media IDs preserve visual computation. See
[the benchmark protocol](serving.md#fair-benchmarking). These ratios are measured
against original Transformers Dense; they are not multiplied Skill speedups.

## Quantization backend status

GPTQModel is the validated route. LLM Compressor remains blocked by the retained
dependency pins; TorchAO and AutoRound are disabled pending validation. See
[quantization](quantization.md) for coverage and isolated environments.

## LLaVA-1.5 migration (CUDA, 2026-10-05)

The reference path completed independent Skills, seven full Stack combinations,
Runtime/KV, final serving LMMS and matched Dense timing. The seventh Stack pair
passed; final serving needed no quality fallback. Its source jobs are
[pipeline-llava15.json](../configs/pipeline-llava15.json) and
[serving-llava15.json](../configs/serving-llava15.json). Qwen uses
[pipeline.json](../configs/pipeline.json) and [serving.json](../configs/serving.json).

For a fresh target prepare the [required local assets](getting-started.md),
establish new evidence and retain it under the selected result root. Reported
reference measurements do not qualify another checkpoint, GPU or protocol.

## InternVL35-14B (CUDA, 2026-10-09)

The native `internvl_chat`/Qwen3 checkpoint completed independent Skills, ten
fresh physical Stack compositions, seven Runtime profiles, KV probing, complete
final native LMMS, image inference and matched timing. The selected settings are
physical decoder FFN removal 5% with 128-aligned widths, fresh multimodal GPTQ
INT4, Token 0.95, FlashAttention O2 and auto KV. Native 448-pixel tiling remains
unchanged, with up to 12 tiles plus thumbnail; each tile keeps 244/256 visual
tokens. Context is 8192; prefix caching and encoder compilation/graphs are off.

| Task | Samples | Dense | Final | Retention |
| --- | ---: | ---: | ---: | ---: |
| ChartQA | 2500 | 0.865600 | 0.820000 | 94.7320% |
| MMStar | 1500 | 0.678016 | 0.653055 | 96.3186% |
| TextVQA | 5000 | 0.758920 | 0.685140 | 90.2783% |
| DocVQA | 5349 | 0.919533 | 0.845482 | 91.9470% |
| POPE F1 | 9000 | 0.870828 | 0.868424 | 99.7239% |

The equal-task mean is 94.59994024%, below the 95% target by 0.4001 percentage
points. All 23,349 samples passed native prompt, sample/document-hash alignment,
actual Token, observed Runtime/KV and physical-width checks. The second final
candidate passed only its mechanism smoke before the user stopped its partial
ChartQA run; it was excluded from selection. Remaining Runtime recovery was
not run. Quality and performance outcomes remain separate.

Both final benchmark backends ran sequentially on the same RTX PRO 6000 Blackwell
Server Edition GPU, driver 590.44.01. Each used 128 native inputs, five warmups, three measured rounds,
batch 1 and greedy generation capped at 64 new tokens.

| Metric | Original Transformers Dense | Final serving |
| --- | ---: | ---: |
| Measured requests | 384 | 384 |
| Total request seconds | 527.592472 | 232.130785 |
| Mean request milliseconds | 1373.939 | 604.507 |
| P95 request milliseconds | 2173.343 | 875.243 |
| Output tokens | 14922 | 15381 |
| Output tokens/second | 28.2832 | 66.2601 |

The matched request-time speedup is 2.272824x; output-token throughput improves
2.342736x. Natural output lengths differ. The timing boundary and exclusions
match the protocol above. Independent audits verified all 768 measured requests,
round aggregates, image/question order, exact native prompt identities, same GPU
and actual mechanisms. No independent Skill speedups were multiplied.

Local evidence is under `results/model/internvl3_5_14b/rtx_pro6000/`:
`stack-serving-lmms/final/serving-freeze.json` (digest `146ba819656d50eb076a`),
`serving-use/benchmark/summary.json`, and
`preparation/final-serving-benchmark-completion.json`. See
[execution scope and checks](internvl-preparation.md),
[pipeline-internvl35.json](../configs/pipeline-internvl35.json) and
[serving-internvl35.json](../configs/serving-internvl35.json) for reuse.

## DeepSeek-VL2 (CUDA, 2026-10-12)

The full checkpoint completes the original workflow on RTX PRO 6000 Blackwell
(97,887 MiB, driver 590.44.01). The local receipt records the physical GPU identity.
The selected sixth Stack uses physical SP, fresh Parent GPTQ W4A16/group128 and
Token 0.85. All 72 routed experts, top-6 routing and two shared experts remain;
SP slices FFN intermediate channels. Native Stack passes at 97.4211% mean
retention. Final O2/Auto serving passes at **97.2659%**, with **4.816650x** matched
batch-1 end-to-end acceleration. The outcome is `goal_achieved`.

### Stack search

All six candidates complete all 23,349 native samples. FFN removal is the stored
decoder FFN-parameter fraction; Token keep applies to main spatial features,
with native layout protected. Every task retention is candidate/Dense, and the
mean weights all five tasks equally.

| Candidate | FFN removal | Token keep | ChartQA | MMStar | TextVQA | DocVQA | POPE F1 | Five-task mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.1394% | 55% | 82.8246% | 85.2704% | 86.2061% | 91.5135% | 97.8176% | 88.7265% |
| 2 | 47.4041% | 60% | 84.2732% | 85.6969% | 91.2027% | 93.2883% | 97.3024% | 90.3527% |
| 3 | 42.6648% | 65% | 75.0647% | 89.9650% | 94.5599% | 94.5625% | 97.6302% | 90.3564% |
| 4 | 37.9294% | 75% | 86.6011% | 89.8742% | 97.5103% | 96.1373% | 98.8285% | 93.7903% |
| 5 | 33.1941% | 80% | 91.3088% | 90.8218% | 98.0666% | 96.5676% | 98.0653% | 94.9660% |
| 6 (selected) | 28.4587% | 85% | 99.3275% | 93.4501% | 98.7772% | 97.9759% | 97.5749% | 97.4211% |

Five valid complete rejections trigger paired SP/Token recovery. The sixth
passes after six of ten configured attempts; the search is not exhausted.
Independent Token's fastest local timing ratio is 0.90 (1.042057x); independent
SP's recovered quality boundary is 52.1394% FFN removal. Independent GPTQ's
native Transformers path is slower at 0.619508x. These local stage measurements
are separate from final quality and matched acceleration.

### Final five-layer quality

Selected Runtime is O2, effective compilation mode 3 with
`FULL_AND_PIECEWISE` CUDA Graphs; encoder compilation is disabled. Actual
TRITON_MLA, Marlin MoE, F32 routers, physical widths and Token 167/196 main
features per tile are observed. Auto KV completes the serving path. E4M3 fails
initialization because the Triton MLA kernel needs 102,400 bytes of per-block
shared memory against SM120's 101,376-byte limit; this branch supplies no
quality or timing score. KV dtype is observed; cache sequence length is not.

| Task | Native Dense | Final serving | Retention |
| --- | ---: | ---: | ---: |
| ChartQA relaxed overall | 0.773200 | 0.764400 | 98.8619% |
| MMStar category mean | 0.628971 | 0.585878 | 93.1486% |
| TextVQA native VQA score | 0.816160 | 0.807280 | 98.9120% |
| DocVQA ANLS | 0.925665 | 0.907579 | 98.0462% |
| POPE F1 | 0.907998 | 0.884033 | 97.3607% |
| Equal-task mean retention | | | **97.2659%** |

All 23,349 original inputs and every native score/aggregate independently
validate, with actual visual selections and compacted prompts. The mean is
0.1553 percentage points below native Stack and remains above 95%. This is an
equal-task mean target, not a per-task minimum. Runtime/KV quality fallback is
unneeded; the freeze is `lmms_qualified`.

### Matched end-to-end timing

Both backends use the same physical GPU, native Search128 images/questions/order,
five warmups and three measured rounds. All 768 measured records and native
prompt identities independently validate.

| Metric | Original Transformers Dense | Final serving |
| --- | ---: | ---: |
| Mean request seconds | 0.982524 | 0.203985 |
| P50 request seconds | 0.707654 | 0.162891 |
| P95 request seconds | 2.701954 | 0.385954 |
| Requests/second | 1.017787 | 4.902323 |
| Output tokens/second | 19.409514 | 92.927116 |

The request-time speedup is **4.816650x**; output-token throughput improves
4.787709x. Timing runs from image opening to decoded answer, including native
preprocessing, transfer, vision encoding, generation and synchronization.
Loading, warmup, prefix-cache reset and post-generation prompt audit are
excluded; in-engine Token traces are included. Natural answer lengths differ.
Serving answers vary across timing rounds for 20/128 inputs, while selected
visual indices stay identical. These requests are performance evidence only;
they do not replace full native five-task quality.

Actual single-image inference also passes native prompt, Token and Runtime/KV
checks. Its first shape-compilation request is not warmed-up speed evidence.
The lazy `deepseek_vl2` pipeline ID is enabled after this measured path succeeds;
existing JSONs reuse all completed stages without launching GPU work.

Local evidence is under `results/model/deepseek_vl2/rtx_pro6000_gpu072a4781/`:
`stack-serving-lmms/final/serving-freeze.json` (digest `8c315f7c4ea69ac904f0`),
`serving-use/benchmark/summary.json`, and the independent complete quality,
inference and timing audits in `preparation/`. Detailed local histories remain
there. See [integration and checks](deepseek-vl2-preparation.md),
[pipeline-deepseek-vl2.json](../configs/pipeline-deepseek-vl2.json) and
[serving-deepseek-vl2.json](../configs/serving-deepseek-vl2.json) for reuse.
