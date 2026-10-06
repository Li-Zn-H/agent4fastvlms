# Configuration

The JSONs in this edition retain the two public model jobs and their dependencies.
Historical single-candidate trials and standalone artifact descriptors are omitted.
Parameter values are unchanged from the local experiments. Results, weights,
calibration and evaluation data are not bundled.

## Files and ownership

| Model | Main pipeline | Final inference and matched timing |
| --- | --- | --- |
| Qwen3-VL-4B | [pipeline.json](pipeline.json) | [serving.json](serving.json) |
| LLaVA-1.5-7B | [pipeline-llava15.json](pipeline-llava15.json) | [serving-llava15.json](serving-llava15.json) |

Unsuffixed jobs target Qwen. Each pipeline references model-specific jobs:

| Config family | Owns |
| --- | --- |
| `model` | Dense checkpoint, adapter, dtype, device, native attention |
| `evaluation` | Result scope, Search/Dev paths, generation and local scoring |
| `optimization`, `vision_token` | Dense/Token binding, selection method and ratio ladder |
| `structured_pruning` | Calibration, aligned allocation, hardware probe, quality ladder |
| `quantization` | Isolated backend, Dense calibration, export/reload and checks |
| `stack` | Independent sources, fresh Parent quantization, paired recovery, full LMMS |
| `stack_runtime` | Stack Parent, attention/compile/fusion profiles and timing |
| `stack_kv_cache` | Runtime Parent, cache dtype and batch-size probes |
| `serving_lmms` | Final five-task gate, context and measured fallback controls |
| `serving` | Final freeze and same-model Dense end-to-end comparison |

Qwen's [quantization.json](quantization.json) explores backends; the pipeline
consumes [quantization-heldout.json](quantization-heldout.json). LLaVA-1.5 uses
[quantization-llava15.json](quantization-llava15.json). See
[backend and coverage contracts](../docs/quantization.md).

## Preparing a fresh job

LLaVA-1.5 is the recommended first GPU example after [setup](../docs/getting-started.md).
All models need weights, Search/Dev manifests/images, native LMMS data and
model-specific SP/quantization calibration. Calibration filename or `purpose`
is not evidence of disjointness; check the actual sample and image identities.

The reference jobs also retain these local handoffs:

- Qwen's `stack_runtime.json.parent_benchmark` names the selected historical
  Stack candidate. For a fresh search, remove this optional override so the
  existing planner reads the new Stack freeze's `runtime_receipts.search128`.
  Its SP calibration is an imported local asset and needs compatible statistics.

These are explicit preparation requirements, not missing files to fabricate.
The existing pipeline does not automatically collect SP statistics or produce
the separate Dense LMMS denominator. See [orchestration](../docs/agent-search.md).

## Paths, resume and existing records

Resolve paths from the repository root. Keep the existing environment names or
update `coordinator_python` and all enabled `worker_python` values consistently.
For a new model/hardware/data protocol, set `evaluation.result_root` and update
every source, output, Dense denominator and Parent path together.

Do not edit snapshots to reuse incompatible evidence. Pipeline resume checks
unchanged jobs; Token/SP resume uses saved settings. Stack reuses matching
candidates from its selected current JSON. A new directory alone cannot qualify
old receipts. Follow [reuse rules](../search/README.md#reuse-before-execution).

`selection_policy` is `strict` unless explicitly set to `best_effort`. Neither
changes the 95% target. Budgets, task lists, context and generation settings stay
in the stage JSON; see [Stack policy](../docs/stack.md#selection-policy-and-tradeoffs).
