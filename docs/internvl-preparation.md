# InternVL3.5-14B CUDA Execution

The native InternVL3.5-14B pipeline completed local execution on RTX PRO 6000
Blackwell Server Edition. The selected configuration retains **94.5999%** of
Dense quality and achieves **2.2728x** matched batch-1 end-to-end acceleration.
Its outcome is `accelerated_quality_unmet`: the 95% quality target was not met.
See [per-task scores and timing](status.md#internvl35-14b-cuda-2026-10-09).

The user requested stopping the second final Runtime candidate and closing out
with the first complete candidate. The final freeze is `lmms_evaluated`, never
`lmms_qualified`. This explicit stop is not recovery-ladder exhaustion.
All measurements and cancelled-candidate evidence are retained locally under
`results/model/internvl3_5_14b/rtx_pro6000/`; they are excluded from this source
edition. Detailed history is archived as `preparation/internvl-execution-history.md`.

## Model and execution scope

| Area | Validated scope |
| --- | --- |
| Native checkpoint | OpenGVLab `internvl_chat`, original custom code, 40-layer Qwen3 decoder |
| Native input | 448-pixel tiles, up to 12 tiles plus thumbnail, 256 projected visual tokens per tile |
| Token | Post-encoder selection on every native tile, including thumbnail; 0.95 keeps 244/256 tokens per tile |
| Physical SP | All 40 decoder FFNs use actual sliced weights; widths are 128-aligned, with 5% aggregate removal in the selected artifact |
| Quantization | Fresh GPTQ INT4 for each SP Parent; 376 declared Linear targets, with BF16 exceptions for the language head and two projector linears |
| Runtime | FlashAttention, O2, context 8192, memory utilization 0.8; prefix caching, QK/RoPE fusion and encoder compilation/graphs off |
| KV | auto; FP8 E4M3/E5M2 startup was rejected by the frozen attention backend for this configuration |
| Hardware | RTX PRO 6000 Blackwell Server Edition, driver 590.44.01 |

Images, native tiling, prompts and task scoring were preserved. The complete
native context scan requires at most 4,877 tokens including generation, within
the configured 8,192-token serving context. Actual prompt compaction and Token
selection were checked; vLLM receipts observe KV dtype, not sequence KV length.

Use the existing isolated environments:

- `.venv-internvl`: original Dense and native LMMS coordination.
- `.venv-quant-internvl`: GPTQModel and native quantized generation.
- `.venv-runtime-internvl`: vLLM and final native serving LMMS.

Pinned versions include Torch 2.11.0+cu130, Transformers 5.14.1,
GPTQModel 7.5.0, vLLM 0.26.0 and LMMS 0.7.3. The quantizer and Runtime dependency
constraints differ; do not combine their environments. Requirements are in
[Dense/LMMS](../requirements-lmms-internvl.txt),
[quantization](../requirements-quantization-internvl-gptqmodel.txt) and
[Runtime/LMMS](../requirements-runtime-internvl.txt).

## Completed evidence

| Stage | Local evidence and result |
| --- | --- |
| Native prerequisites | Real BF16 loading, keep-all equality, all-tile compaction, physical SP, exact artifact roundtrip and isolated-interpreter generation passed |
| Data and Dense | All five offline datasets and context bound checked; original Dense Search128/Dev256, timing and all 23,349 native LMMS inputs completed |
| SP calibration | Fresh 128-input disjoint calibration; all 40 layers have complete visual-prefill, text-prefill and decode statistics |
| Independent Skills | Token, hardware width probes, SP quality ladder and full-model GPTQ export/calibration/precision/reload/quality completed |
| Stack | Ten SP50-through-SP5 compositions completed fresh quantization, reload, timing and all five native tasks; best retention 94.6534%, frozen under explicit best-effort policy |
| Runtime and KV | All seven Runtime candidates measured; FlashAttention had highest output-token throughput; auto KV completed batch 1/4/8 profiles |
| Final LMMS | First candidate completed all five tasks and 23,349 samples, with native prompt, sample/document-hash, Token, physical-width and Runtime/KV verification |
| Inference and benchmark | Native image inference passed; original Dense and final serving each completed 128 inputs x 3 rounds on the same physical GPU |
| Pipeline reuse | Read-only plan reports all eight stages complete and no next stage, while preserving `quality_target_met: false` |

Stack recovery reused the last qualified Token 0.95 source as SP widened,
according to `token_recovery_exhaustion: reuse_last_qualified`. Each changed SP
Parent received fresh quantization and full native LMMS; no downstream local
Search/Dev quality screening replaced the full tasks.

Final serving candidate `serving-a9f3c334773205d64a34` is SP5/GPTQ/Token0.95 with
FlashAttention O2 and auto KV. The second candidate, O0/auto KV, passed its
five-task mechanism smoke but was stopped at the user's request at 999/2,500
logged ChartQA generations. It has no complete full-task result and is excluded
from selection. O1 and automatic-attention O2 recovery were not run.
`preparation/serving-user-stop.json` records the instruction and confirms that
the coordinator and all its descendants exited and released the GPU.

TextVQA metadata requests encountered an unavailable network, then automatically
used the verified local dataset cache. The live jobs were not restarted for
these retries. The validated offline settings can be used for future execution
when needed; cached revision and checks remain in
`preparation/runtime-textvqa-offline-cache-check.json`.

The selected freeze is `stack-serving-lmms/final/serving-freeze.json`, digest
`146ba819656d50eb076a`. Its complete quality audit is in the candidate's
`lmms-verification.json`. Final inference and benchmark audits are
`preparation/final-serving-inference-completion.json` and
`preparation/final-serving-benchmark-completion.json`. Raw requests, outputs,
per-round timings and observed settings remain under `serving-use/`.

## Reuse and checks

These commands inspect existing evidence without starting GPU work:

```bash
.venv-internvl/bin/python -B -m search.pipeline --config configs/pipeline-internvl35.json
.venv-internvl/bin/python -B -m search.serving --config configs/serving-internvl35.json --benchmark
```

To use the selected model with another local image:

```bash
OMP_NUM_THREADS=8 .venv-internvl/bin/python -B -m search.serving \
  --config configs/serving-internvl35.json \
  --image /path/to/image.jpg --prompt "Describe this image." --execute
```

The matching benchmark is reusable with the same configuration:

```bash
OMP_NUM_THREADS=8 .venv-internvl/bin/python -B -m search.serving \
  --config configs/serving-internvl35.json --benchmark --execute
```

Single-image inference answered that the man was speaking on a phone, with
seven native tiles compacted from 1,792 to 1,708 visual tokens. Its response
reached the unchanged 64-token generation cap. One inference request is not a
warmed-up benchmark. The full matched timing protocol is in [serving](serving.md).

The 11 focused native/registration tests pass after enabling `internvl35` as a
pipeline ID. The final full CPU run reached 314 tests: 35 errors, one failure and
nine skips. The failures reference absent Qwen/LLaVA checkpoints, environments
or historical receipts; there are no new failure names versus the preceding
run. This is not a passing full suite. Exact outputs are retained as
`preparation/internvl-final-pipeline-tests.log` and
`preparation/full-cpu-final-pipeline.log`. Documentation checks also pass.

Registration admits this native topology to the existing orchestration. It
does not certify a new checkpoint, hardware platform or quality target.
Preserve all retained measurements and follow the [migration guide](model-migration.md)
for a different scope; do not reuse another model's denominators or speedup.
