# Model and Hardware Registration

The registry selects implementations; capability checks determine whether a
stage may enter environment validation. The complete pipeline targets are
`qwen3_vl`, `llava15`, `internvl35` and `deepseek_vl2`, on `cuda`.
Dense dispatches loading, input preparation, visual-token counts and
timing hooks through adapters. CUDA identity checks and Worker launches use the
hardware registry. Other Skills retain their algorithms and loops, with static
scope checks before execution.

The `internvl35` adapter and pipeline ID are enabled after complete local
Dense/Token/SP, quantization, Stack, Runtime/KV, native five-task serving LMMS,
image inference and matched Dense timing. The selected RTX PRO 6000 reference
achieves 2.2728x acceleration at 94.5999% mean retention and remains explicitly
below the 95% quality target. See its [execution scope](internvl-preparation.md)
and [reported result](status.md#internvl35-14b-cuda-2026-10-09). Registration
does not imply quality qualification or transfer measurements to another scope.

The `deepseek_vl2` adapter and pipeline ID target the full DeepSeek-VL2 checkpoint.
Complete local Dense, independent searches, physical MoE Stack, Runtime/Auto KV,
native final serving LMMS, image inference and matched original Dense timing
have model-specific evidence. Native Stack retains 97.4211%; final serving
retains 97.2659% and accelerates 4.8167x on RTX PRO 6000. Registration enables
orchestration, not quality qualification for another job. See its
[integration scope](deepseek-vl2-preparation.md) and [measurements](status.md).

## File responsibilities

| File | Responsibility |
| --- | --- |
| [adapters/base.py](../adapters/base.py) | Model/hardware interfaces, stage declarations and lazy entry resolution |
| [adapters/registry.py](../adapters/registry.py) | Registration, lookup, duplicate checks and checkpoint `model_type` selection |
| [adapters/qwen3vl.py](../adapters/qwen3vl.py) | Qwen3-VL entry points, stages, platforms and architecture scope |
| [adapters/qwen3vl_dense.py](../adapters/qwen3vl_dense.py) | Dense loading, native inputs, Token geometry and timing |
| [adapters/llava15.py](../adapters/llava15.py), [Dense implementation](../adapters/llava15_dense.py) | LLaVA-1.5-7B pipeline, fixed-resolution Token and Llama/CLIP integration, with separate local execution evidence |
| [adapters/deepseek_vl2.py](../adapters/deepseek_vl2.py), [Dense implementation](../adapters/deepseek_vl2_dense.py) | Full DeepSeek-VL2 CUDA pipeline, native dynamic tiling, physical MoE/MLA Stack and matching local execution evidence |
| [adapters/cuda.py](../adapters/cuda.py) | CUDA device selection, identity and isolated Worker launch |
| [adapters/capabilities.py](../adapters/capabilities.py) | Static Dense, Token, SP, quantization, Stack, Runtime, KV and LMMS capability matrix |
| [adapters/__init__.py](../adapters/__init__.py) | Explicit built-in registration |
| [search/context.py](../search/context.py) | JSON selection, result/baseline scope and evidence ownership |
| [search/model.py](../search/model.py) | Dense invocation and shared generation loop |
| [search/registry.py](../search/registry.py) | Read-only catalog and stage preflight CLI |

`entrypoints` use `module:function_or_class`. Catalog queries handle metadata;
`adapter.resolve(name)` imports the implementation for execution in its Worker
environment. A missing or incompatible entry produces `blocked_preflight`, without
falling back to another model.

## Queries and preflight

```bash
# Catalog only; Python standard library is sufficient.
python3.12 -B -m search.registry

# Selected model, device, result scope and all stages; requires local model metadata.
.venv-qwen3vl/bin/python -B -m search.registry \
  --model-config configs/model.json \
  --evaluation-config configs/evaluation.json --device 0

# Check Dense first when porting a model.
.venv-qwen3vl/bin/python -B -m search.registry \
  --model-config configs/model.json --stages dense --device 0
```

Queries do not import Torch/Transformers/vLLM, query GPUs or write results. A
failed check emits JSON and exits 1. The read-only pipeline also reports
`capabilities`; actual reuse still depends on stage evidence.

| Field/status | Meaning |
| --- | --- |
| `implementation=available` | Stage declared, entry points present and architecture/platform within scope |
| `ready_for_probe` | May enter stage preflight; does not certify all kernels |
| `environment=unverified` | Static checks do not validate dependencies, memory, dtype or attention backend |
| `quality=unverified` | Static checks produce no quality measurement; use matching receipts |
| `blocked_preflight` | Unknown model/platform, missing entry or unsupported device/architecture |

Quantization checks here cover model integration. GPTQModel, LLM Compressor and
other backends still need isolated requirements/`worker_python`, framework
preflight and export/reload validation. `ready_for_probe` does not certify every
quantizer, FP8 path or fusion kernel.

## JSON selection

Optional fields preserve existing responsibilities and defaults:

| JSON owner | Optional field | Default behavior |
| --- | --- | --- |
| Model descriptor | `adapter` | Resolve checkpoint `model_type`; an explicit adapter must match |
| Model descriptor | `hardware` | Infer platform from `device`, such as `cuda:0` → `cuda` |
| Evaluation | `result_root` | `results/model/qwen3vl`; other Dense models/platforms must explicitly use a separate root |
| Optimization | `worker_python` | Current interpreter for Dense/Token; pipeline Dense defaults to `coordinator_python` |
| Runtime/KV | `hardware` | `cuda`; current implementations remain CUDA-only |

Example model descriptor for a separate Qwen experiment:

```json
{
  "adapter": "qwen3_vl",
  "hardware": "cuda",
  "path": "models/Qwen3-VL-4B-Instruct",
  "device": "cuda:0",
  "dtype": "bfloat16",
  "attn_implementation": "sdpa"
}
```

Keep evaluation `data`, `generation`, `scoring` and `quality`, adding
`"result_root": "results/model/qwen3vl-new-experiment"`.
Dense writes to `<result_root>/search/<scoring.baseline>`; Token/SP discovery and
quantization/Stack/vLLM quality use that root's baseline. SP defaults to
`<result_root>/search/structured-pruning`, quantization to
`<result_root>/quantization`. Update explicit `output`, Parent, source paths,
LMMS baselines and Runtime/KV `dense_benchmark` together. The pipeline checks
scope and handoffs; `result_root` does not move files or rewrite other JSONs.

New-root Dense outputs bind model/evaluation settings and reject unowned or
mismatched baselines. Token reuse requires matching model/evaluation settings.
Legacy receipt rules preserve historical root snapshots. Changed GPU, generation
protocol or data needs a new experiment scope and performance denominator.
Read-only checks do not query the GPU and cannot certify old timings on a new
card. Do not add fields to a completed job and assume `--resume` will accept it.

Stage JSON `device` selects the physical GPU. The isolated Worker uses the
adapter's `worker_device`, currently `cuda:0`. Current CUDA stages accept physical
indices 0/1 only. Backend requirements and quantization/Runtime `worker_python`
remain independently maintained.

## Adding a model

1. Define an adapter with a unique `id`, `model_types` and actual `hardware_ids`.
2. Implement `dense_load(config) -> (model, processor)`, `processor_load(config)`,
   `prepare_inputs(processor, sample, media_root)`,
   `generation_timer(model, pruning=False)` and `image_token_count(model, inputs)`.
   Input preparation returns a batch supporting `.to(device)`. Set the adapter's
   `generation_includes_prompt` to match the actual generation output. The shared
   loop removes the input prefix when this is true; otherwise it decodes the
   entire generated tensor with `processor.batch_decode()`. Native DeepSeek-VL2
   and InternVL3.5 return answer tokens only and set it to false.
3. Declare only implemented `stages` and real `entrypoints`; implement
   `validate_checkpoint()` and required `validate_stage()` scope checks.
4. Register explicitly with `models.register(YourAdapter())` in `adapters/__init__.py`.
5. Create independent model/evaluation configs and a result root. Pass
   `--stages dense`, then establish Dense Search/Dev and performance denominators
   using the [migration guide](model-migration.md).
6. Port each remaining Skill's architecture contract. Keep `pipeline_ids=()`
   until the complete path works. Declarations do not produce validation evidence.

Qwen3-VL optimization stages currently require the implemented 4B layout: 36 layers,
2560 hidden size, original FFN width 9728; Stack uses GPTQ W4A16 / Group-128.
Other Qwen3-VL sizes may begin with Dense checks, but cannot directly enter this Stack.

LLaVA-1.5 registers `llava15` for HF `model_type=llava`: a 32-layer Llama
decoder with hidden size 4096 and FFN width 11008, plus CLIP ViT-L/14 at its
native 336-pixel resolution. Pipeline ID `llava15` is enabled after independent
Skill searches, physical Stack, Runtime/KV, final LMMS, inference and matching
Dense timing completed locally. The CLIP GPTQ path covers its native aligned
dimensions. Model-specific calibration, physical SP
recipes and all stage evidence are indexed in
[status](status.md#llava-15-migration-cuda-2026-10-05). Static readiness is not
qualification; each new checkpoint/hardware job still needs matching receipts.

## Adding hardware

Implement `HardwareAdapter`, supported stages, physical-device selection and
Worker-local naming. Supply `probe_identity(device)`,
`launch_worker(request, directory)` and required hooks, then register explicitly.
Model adapters must also declare that platform. Register execution platforms such
as CUDA; concrete card/kernel support is established by environment probes and
measurements.

Performance loops still use CUDA synchronization and memory statistics; vLLM/SP/
quantization also contain CUDA-specific internals. Other platforms require those
paths to be ported. AMD, CPU inference and multi-GPU implementations are not
registered. A new adapter cannot inherit Qwen/CUDA speed or quality claims.
