# Reported Reference Results

The public model targets are Qwen3-VL-4B-Instruct and LLaVA-1.5-7B.
These completed local measurements were checked against their saved final
freezes and benchmark summaries on 2026-10-06. Raw receipts and checkpoints
are not distributed with this source edition; this is not a fresh reproduction.

## Completed results

| Model | Five-task mean retention | End-to-end speedup | Dense / final mean latency |
| --- | ---: | ---: | ---: |
| Qwen3-VL-4B-Instruct | 95.4011% | 2.5293x | 773.029 / 305.632 ms |
| LLaVA-1.5-7B | 95.5839% | 2.9150x | 570.785 / 195.809 ms |

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

Both backends used the same RTX 4090 and driver 580.105.08. Each comparison
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
