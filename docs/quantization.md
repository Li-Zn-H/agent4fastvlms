# Quantization

Independent quantization starts from Dense and accepts no SP recipe or Stack
Parent. Stack later rebuilds physical SP and freshly calibrates/quantizes it.
The independent result selects a route, not a reusable quantized SP artifact.

## Backend environments

GPTQModel W4A16, group size 128, is the validated route for the documented models.
It targets declared language and vision Linear layers. Projector/merger, output
head and other high-precision exceptions are recorded in precision manifests;
this does not mean every tensor is INT4.

| Backend | Requirements | Current scope |
| --- | --- | --- |
| GPTQModel | `requirements-quantization-gptqmodel.txt` | Validated model-specific exports |
| LLM Compressor | `requirements-quantization-llmcompressor.txt` | Blocked by retained compressed-tensors/vLLM pins |
| TorchAO | `requirements-quantization-torchao.txt` | Disabled pending validation |
| AutoRound | `requirements-quantization-autoround.txt` | Disabled pending validation |

Keep one environment per backend and set `worker_python`. Do not install a
quantizer into Dense or treat known-incompatible requirements as an installation
recipe. Preflight blocks are environment/API results, not quality rejections.

## Lifecycle and selection

Require `prepare → calibrate → export → manifest → reload`, followed by local
quality and timing checks. Verify actual INT4 coverage, high-precision exceptions,
clean Transformers reload and the configured vLLM reload. A declared dtype or
successful process exit is insufficient. Preserve candidate receipts and selected
artifacts before discarding temporary work.

```bash
# Read-only preparation check for the LLaVA-1.5 route.
.venv-qwen3vl/bin/python -B -m search.quantization --config configs/quantization-llava15.json
# Execute missing work in the configured isolated backend.
.venv-qwen3vl/bin/python -B -m search.quantization --config configs/quantization-llava15.json --execute
```

The Qwen pipeline consumes `quantization-heldout.json`; `quantization.json` is
backend exploration. Calibration separation must be established from actual
sample/image provenance. A file named "heldout" or a declared purpose alone
does not prove disjointness. Supply valid calibration for new experiments.

Native quantized Transformers can be slower than Dense; kernel/backend behavior
matters. A reload pass is not acceleration evidence. Do not confuse independent
Transformers timing, vLLM engine timing and the final matched Dense comparison.

Only complete eligible independent evidence enters [Stack](stack.md). Each new
physical Parent requires fresh quantization, mechanism/loading checks, timing and
full native LMMS. Stack does not run Search/Dev quality screening. Keep exports
and their manifests with the [receipts](../results/README.md) that reference them.
