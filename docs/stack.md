# Stack, Runtime and KV Cache

Stack composes independently validated Skills in a fixed order: physical FFN
pruning, fresh calibration/quantization of that Parent, then Token selection.
It has its own measured quality and speed; independent gains are never multiplied.

## Fast search and LMMS recovery

1. Read the selected sources and verify they belong to this model/experiment.
2. Build the physical Parent, freshly quantize it and enable the chosen Token ratio.
3. Check actual mechanisms and reload, then measure performance.
4. Run every configured native LMMS task against the model-specific Dense baseline.
5. On a valid complete quality result below 95%, try the next wider SP recipe and
   next higher Token keep ratio together. Repeat within the shared JSON budget.

`search.stack --execute` owns this closed loop. Stack and later stages never
run Search/Dev quality screening. The 128-input timing workload remains a
performance benchmark. Failed execution and API/preflight blocks stop the loop;
they must not trigger quality recovery or enter best-candidate selection.

`candidate_budget_exhausted` means the budget is full and an unseen paired
neighbor remains. `recovery_ladder_exhausted` means no paired neighbor exists.
Keep those outcomes distinct and preserve measurements for rejected candidates.

## Implementation roadmap

The implemented paths compose existing model-specific adapters and stage loops.
Native Stack LMMS and final serving LMMS are separate gates. `frozen_for_lmms`
alone does not mean Stack quality passed: inspect the complete linked LMMS result.
Only a matching final `lmms_qualified` serving freeze certifies all five layers.

```bash
.venv-qwen3vl/bin/python -B -m search.stack --config configs/stack-llava15.json
.venv-qwen3vl/bin/python -B -m search.stack --config configs/stack-llava15.json --execute
```

The first command is read-only. Reuse complete applicable receipts and resume
interrupted tasks. Do not copy old summaries to a new Parent or change saved
settings to make reuse appear valid.

## Runtime layers after Stack

Runtime owns attention backends, compilation, CUDA Graph and fusion options.
It starts from an eligible complete Stack and freezes observed mechanisms and
measurements in `runtime-freeze.json`. KV starts only from that freeze and owns
cache dtype and batch-size probes, preserving the Runtime and Token settings.

Their handoffs are performance-ready, with quality pending final serving LMMS.
Selection may use output-token throughput; it does not necessarily minimize
request latency when answer lengths differ. Batch-4/8 throughput is not batch-1
acceleration. A different Parent, Token ratio or context needs matching new evidence.

### Final serving LMMS gate

`search.stack_serving` chains Runtime → KV → final serving LMMS and executes by
default. Use `search.pipeline` for read-only inspection. The selected profile is
tested first with native prompts, images, scoring and actual Token/Runtime/KV.
Validate `minimum_model_len` before GPU work; do not shrink images or truncate
prompts to force inputs to fit.

A valid complete quality failure tries the same Runtime with auto KV, then
configured mechanism-validated and measured runtime controls. API/execution
failures stop the loop. Fallback exhaustion records `serving_quality_unmet` in
strict mode and does not automatically return to SP/Token recovery.

## Selection policy and tradeoffs

`strict` is the default. Both Stack and final serving may explicitly set
`selection_policy: best_effort`. The 95% quality target remains unchanged:

- Prefer a qualifying candidate and proceed to the next layer when found.
- Only after finite quality-search exhaustion, retain the highest-quality
  completely evaluated, mechanism/loading-valid candidate. Errors and partial
  evaluations remain ineligible.
- Below-target Stack uses `frozen_best_available`; below-target final serving
  uses `lmms_evaluated`. Neither is called quality-qualified.
- Retain measured quality/latency tradeoffs, including a faster lower-quality
  option where useful. Do not collapse them into a weighted score.

Final [matched benchmarking](serving.md#fair-benchmarking) reports:

| Quality and end-to-end speed | Outcome |
| --- | --- |
| Retention >=95%, speedup >1 | `goal_achieved` |
| Retention <95%, speedup >1 | `accelerated_quality_unmet` |
| Speedup <=1 | `no_acceleration` |

## Result retention

Keep Stack artifacts/freezes, native LMMS results/sample outputs, raw timings
and Runtime/KV/serving freezes with their Parent references. Final acceptance
does not overwrite search winners or base Stack settings. Use
[result retention](../results/README.md) before removing temporary files.
