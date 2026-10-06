# Model Migration Guide

This project currently uses Qwen3-VL as its reference model. A new model must
earn its own receipts before any Qwen3-VL result, recipe, artifact or speedup is
used. Model migration is a protocol change, not a path substitution.
Use [the Agent search guide](agent-search.md) for stage orchestration after
checking the model contracts below. The local Qwen3-VL-4B,
and LLaVA-1.5-7B CUDA paths have complete evidence; a backend accepting another
model does not establish its pipeline support.

The [registration guide](registration.md) describes the existing `adapters/`
interfaces, model/hardware registries and read-only `search.registry` catalog.
Dense execution uses the registered loader, processor, input and timing hooks;
stage preflight checks declared operations, architecture and hardware scope.
Optional model `adapter`/`hardware` and evaluation `result_root` fields select the
implementation and independent result scope. Other Skill internals still need
model-specific adaptation before a newly registered family can run the full workflow.

## Start with a capability inventory

Before running a Worker, record the following in a new model-specific config and
result root:

| Area | Questions to answer |
| --- | --- |
| Model family | What Transformers architecture and model class are loaded? Is it text-only, image-text, video, or another modality? |
| Processor | Which processor and chat-template format are required? Where are image/video placeholders created? |
| Decoder | What are the layer count, hidden size, FFN projections, attention layout, KV heads and context limit? |
| Weights | Which dtype and checkpoint format are available? Are weights dense, GPTQ, FP8 or another format? |
| Runtime | Does the pinned vLLM version support the architecture and multimodal path? Which custom class or override is needed? |
| Evaluation | Which local Search/Dev data and native LMMS tasks are valid for this model? |

Create a new `configs/model-<model>.json`, evaluation settings and result root.
Do not point a new model at `results/model/qwen3vl/`; result discovery is
model-specific and old receipts can otherwise look reusable.

This repository is not yet a fully model-agnostic library. Several Qwen3-VL
surfaces must be ported or parameterized before a new model can execute:

| Surface | Current assumption to replace |
| --- | --- |
| `adapters/qwen3vl_dense.py`, `search/model.py` | Implement the new model's registered loader, native inputs, image-token counting and phase timer; verify the shared generation contract |
| `search/context.py`, evaluation and stage JSONs | Set `evaluation.result_root`, establish a bound Dense denominator and update all explicit stage handoff paths |
| `skills/vision_token.py` | Single-image post-encoder features, DeepStack alignment and Qwen visual-token bookkeeping |
| `skills/stack_model.py`, `skills/stack_vllm.py` | Qwen3-VL text MLP, multimodal processor, mRoPE and vLLM class registration |
| `search/worker.py` | Qwen3-VL calibration probe, module inventory, vLLM quality controller and result roots |
| `search/structured_pruning.py`, `search/pruning_search.py`, `search/quantization.py`, `search/stack.py` | Topology/receipt contracts remain model-specific; source and output roots must match the configured experiment |
| `search/lmms_stack_entry.py`, `search/lmms_vllm_entry.py` | Native LMMS task adapter, Stack architecture resolver and vLLM prompt/Token bridge |

Keep these changes behind a model adapter or model-specific config branch. Do
not silently broaden a Qwen-only path and then claim that another model is
supported because its checkpoint loads.

## Establish the Dense reference first

The Dense model must produce complete Search and Dev answers, a CPU-scored
quality summary and a performance control on the target GPU. Keep the model
path, processor settings, dtype, attention implementation and generation
limits in the saved config. A model is not ready for optimization when it only
loads or produces a smoke answer.

The Dense reference supplies the denominator for quality retention and the
throughput comparison. Recreate it when the model family, processor, prompt
format, dataset, GPU protocol or generation settings change. Never compare a
new model's throughput with the Qwen3-VL Dense control.

## Check each Skill's model contract

Use the following order. A failed preflight is recorded as a block and does not
become a quality rejection.

1. **Vision Token**
   - Identify the image-token placeholder and the post-encoder visual feature
     tensor.
   - Confirm whether the model has DeepStack features, multimodal rotary
     positions or a video frame layout.
   - Port the selector only after a keep-all output-equivalence check. The
     selector must update placeholders, visual features and positions together.
   - Recompute geometry and the finite keep-ratio ladder for the new processor.
     Do not reuse Qwen3-VL retained counts or coverage assumptions.

2. **Structured Pruning (SP)**
   - Locate every decoder FFN projection and verify that physical slicing keeps
     the forward pass valid, including gated/SwiGLU variants and tied weights.
   - Derive layer count, original widths, group alignment and protected layers
     from the loaded model. Do not copy the Qwen3-VL width vector.
   - Rebuild calibration statistics and the hardware shape probe. A logical
     mask or a different model's recipe is not physical SP evidence.

3. **Quantization**
   - Start from the new model's Dense artifact. Run framework preflight and
     module inventory in an isolated requirements environment.
   - Require `prepare → calibrate → export → manifest → reload`, then the
     model's Search/Dev and runtime gates. A backend that works for Qwen3-VL
     may still lack coverage for another architecture.
   - Only the qualified model-specific route and artifact evidence may enter
     Stack; Stack freshly quantizes its SP parent. Keep backend
     failures as `blocked_preflight` when the environment or API is the cause.

4. **Stack**
   - Compose only model-specific qualified SP, quantization and Token receipts
     in the fixed order `physical SP → fresh quantization → Token runtime`.
   - Re-quantize after physical SP; do not relabel a Dense quantized artifact as
     an SP child. Implement a topology adapter before attempting vLLM.
   - Measure the composed candidate again, then run full native LMMS. No
     Search/Dev quality evaluation runs at Stack or Runtime/KV. Independent
     Skill speedups are not multiplied.

5. **Runtime and KV Cache**
   - Runtime probes require a frozen Stack artifact and a vLLM architecture
     preflight. Attention backend, compile, CUDA Graph and fusion settings are
     runtime evidence, not model weights.
     An explicit best-effort Stack handoff may proceed below its quality target
     after bounded exhaustion; complete native evidence remains required. See
     [selection policy](stack.md#selection-policy-and-tradeoffs).
   - KV Cache probes require `runtime-freeze.json`, test the configured serving
     batch sizes; quality is deferred to full serving LMMS at 95%. Report single-request
     latency separately from concurrent serving throughput.
   - A vLLM API or kernel incompatibility is `blocked_preflight`; do not weaken
     the quality gate to make the candidate pass.

6. **LMMS**
   - Add a model descriptor and adapter entry only after the model can load in
     the native LMMS task path. Verify that the adapter binds the intended
     architecture rather than silently falling back to Dense.
   - Run the full native suite for mechanism-validated model-specific Stack
     compositions. Follow the [paired recovery policy](stack.md#fast-search-and-lmms-recovery)
     if the fixed LMMS retention gate fails; `search.stack --execute` implements
     that automatic loop for supported adapters. Verify actual Token pruning in the native generation path before
     full evaluation. Native LMMS scores remain separate from local Search/Dev
     retention and vLLM serving throughput.
   - After Runtime/KV selection, port and verify the final vLLM bridge as well:
     native prompts/preprocessing/scoring, actual Token selection and observed
     runtime settings must match the intended configuration. Validate the new
     model's full input context envelope before starting the suite. Only its
     own `serving-freeze.json` with `lmms_qualified` certifies all five layers.

## Evidence and stopping rules

Every migrated model should leave these durable records under its own result
root:

```text
search/dense/                 # Dense quality and performance denominator
search/vision-token/          # Token geometry, ladder and receipts
search/structured-pruning/    # Calibration, probe and SP quality path
quantization/                 # Isolated backend receipts and selected artifact
stack/                        # Fresh composition receipts and stack-freeze.json
stack-runtime/                # Runtime candidates and runtime-freeze.json
stack-kv-cache/               # KV candidates and kv-cache-freeze.json
lmms/                         # Native final-task results
stack-serving-lmms/            # Final vLLM gate and serving-freeze.json
```

Keep `config.json`, `state.json`, `summary.json`, candidate `probe.json` and
`result.json`, selected artifacts and held-out evidence. Remove `.work/`, lock
files and failed temporary exports only after their receipts are copied. A
partial or interrupted LMMS directory is not a completed final gate.

For a new model, the Agent should report three separate outcomes for each
candidate: quality retention, single-request performance and serving-scoped
concurrency performance. A candidate is deployable only when the required
quality and reload gates pass; speed alone does not qualify it.
