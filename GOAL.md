# GLM-4V-9B Optimization Goal

Edit the inputs, then ask an Agent: "Execute GOAL.md end to end; reuse valid
evidence and resume interrupted work." Editing this file alone launches nothing.

## Inputs

```json
{
  "model": "models/GLM-4V-9B",
  "hardware": {
    "platform": "cuda",
    "device": 0
  }
}
```

The model is the local original GLM-4V-9B checkpoint. Record the actual model
files, GPU identity, driver and software versions before GPU work. The planned
experiment scope is `results/model/glm_4v_9b/a800/`; every Dense, calibration,
stage and serving handoff must remain inside that scope. Create GLM-specific
JSON jobs, requirements and adapters using [AGENTS](AGENTS.md),
[setup](docs/getting-started.md), [migration](docs/model-migration.md) and
[the workflow](docs/agent-search.md).

## Model-specific contract

The checkpoint declares `model_type: chatglm` and contains its own
`configuration_chatglm.py`, `modeling_chatglm.py`, `tokenization_chatglm.py` and
`visual.py`. Use the original local implementation through the required
`trust_remote_code` path and a Transformers environment meeting the checkpoint's
`>=4.44.0` requirement. Do not silently substitute a newer ChatGLM checkpoint,
text-only ChatGLM code or another model's processor.

The native contract is one RGB image per request, resized to 1120 x 1120 with
the checkpoint tokenizer's bicubic/normalization transform. Preserve the
`<|begin_of_image|>`/`<|endoftext|>`/`<|end_of_image|>` sequence, the native
ChatGLM chat template, image tensor dtype conversion, position IDs and stopping
behavior. The tokenizer rejects multiple images; do not create a multi-image
path by changing the protocol. The EVA2-CLIP vision tower has 63 layers,
1792-dimensional features, patch size 14 and a 2 x 2 merge, yielding 1,600
main visual features plus the model's boundary features. Validate this geometry
from the loaded checkpoint before implementing Token selection.

The checkpoint's `generation_config.json` defaults to sampling
(`do_sample=true`, temperature 0.8, top-p 0.8, max length 8192). For this
search, freeze the reproducible task profile as `do_sample=false`,
`max_new_tokens=64` and `ignore_eos=false`, with the checkpoint's three EOS IDs
(`151329`, `151336`, `151338`). Record the difference from the checkpoint
default in the Dense receipt. Dense, every candidate and final serving must use
these identical options; a sampling study would require a separate job and
result root.

The text decoder is a dense 40-layer ChatGLM stack with hidden size 4096,
FFN width 13696, 32 attention heads, two multi-query KV groups and an 8,192
token context limit. It is not an MoE model: do not carry over Kimi expert,
router, top-k or shared-FFN assumptions. Start Dense in the checkpoint's BF16
dtype and validate eager/SDPA behavior against the native implementation before
selecting an attention setting. Keep the original source weights unchanged.

Do not reuse Kimi scores, denominators, visual geometry, calibration, physical
recipes, quantized artifacts, Runtime/KV settings or speedups. GLM's model
adapter, native LMMS bridge, Token compaction, dense FFN pruning, quantization
coverage, vLLM path and result receipts must earn independent evidence. Keep
the Kimi, Qwen, LLaVA, InternVL and DeepSeek experiment roots intact.

## Objective

Find a GLM-specific combination with **at least 95% equal-task mean quality
retention and end-to-end speedup greater than 1** against this checkpoint's
original Transformers Dense implementation on the same A800. Use the same
native images, prompts, scoring, context limits and timing boundary for both
backends. Seek a qualifying combination within finite configured candidates,
not a global optimum.

## Work

1. Inspect the checkpoint, tokenizer, image transform, context envelope and
   generation output semantics. Freeze the generation profile and stopping IDs,
   then implement a lazy `chatglm` adapter and native input/timing hooks. Run
   CPU shape and prompt checks before GPU execution.
2. Add GLM-specific model, evaluation, optimization, SP, quantization, Stack,
   Runtime, KV and serving JSONs. Set every `evaluation.result_root`, explicit
   source/output/Parent path and LMMS baseline to
   `results/model/glm_4v_9b/a800/`. Keep `pipeline_ids=()` until the complete
   path has matching execution evidence.
3. Establish GLM's own Dense Search/Dev answers, full native five-task LMMS
   denominators and matched Dense performance baseline. Record actual sample
   counts and metrics from the GLM native prompt path; even if task names match
   another model, do not transfer its scores or input identity receipts.
4. Calibrate GLM's decoder FFNs on disjoint native inputs and probe physically
   sliced, group-aligned widths. Preserve attention, embeddings, the vision
   tower, multimodal projector and image boundary handling unless a separate
   model-specific contract proves otherwise. SP evidence must include real-weight
   reload and native output checks, not logical masks or a copied recipe.
5. Run independent post-encoder Token and dense-FFN SP searches, then an
   isolated GLM quantization route. Require `prepare -> calibrate -> export ->
   precision manifest -> reload` before any source enters Stack. Keep Dense,
   quantization and Runtime/serving Worker environments separate.
6. Compose `physical SP -> fresh GLM Parent quantization -> GLM Token runtime`.
   Validate prompt/image-feature compaction, physical widths, reload and timing,
   then run complete native LMMS. Stack, Runtime and KV do not use Search/Dev
   quality screening. On a valid full-LMMS quality failure, recover the next
   wider SP and next higher Token ratio together; the finite ladders belong to
   the GLM JSON jobs.
7. Probe Runtime only from the frozen GLM Stack artifact. Validate whether the
   installed vLLM release supports ChatGLM multimodal generation; implement a
   scoped GLM bridge if needed. Runtime owns attention, compilation, CUDA Graph
   and fusion evidence. Probe KV dtype/concurrency only from `runtime-freeze.json`.
8. Run final native serving LMMS with actual GLM Token, Runtime and KV checks,
   then final image inference and matched image-open-to-decoded-answer timing
   through `search.serving` against the original GLM Transformers Dense.

Use `search.pipeline` read-only first, then `--execute`, or
`--execute --resume` for an unchanged saved job. Dense LMMS and SP calibration
are preparation steps; the coordinator requires their evidence. The standard
128-input workload is timing evidence only.

## Recovery and stopping

Inspect existing GLM processes, locks and result roots before GPU work. Reuse
only complete, matching GLM receipts; never let a Kimi or another model's
receipt satisfy a GLM handoff. Resume the first missing stage without launching
duplicate coordinators. API, environment, capacity and execution errors are
blocked evidence, not quality rejections. Never lower the target, resize native
images, truncate prompts, change the one-image protocol or disable a required
mechanism to make a candidate pass.

Set `selection_policy: best_effort` explicitly in the GLM Stack and final
serving jobs if bounded quality exhaustion should retain the highest-quality
complete candidate. The 95% target remains unchanged. Partial tasks, failed
reloads and execution errors are never candidates. A below-target result is not
goal achievement. Stop at verified success, a documented external blocker or
finite search exhaustion under the configured policy.

## Deliver

Report the checkpoint identity, GLM architecture contract, A800 identity,
selected SP/quantization/Token/Runtime/KV settings, actual per-task Dense and
final scores, sample counts, equal-task mean retention, matched end-to-end
speedup, timing scope, freeze/benchmark paths, checks and reusable commands.
Keep selected artifacts and raw evidence under the GLM result root. Mark success
only with a matching `lmms_qualified` serving freeze and measured speedup >1;
otherwise report `accelerated_quality_unmet` or `no_acceleration` accurately.
Concurrent throughput and engine-only timings cannot replace matched
single-request end-to-end timing.
