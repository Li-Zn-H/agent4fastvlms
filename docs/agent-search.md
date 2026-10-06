# Agent Search Guide

Use [GOAL](../GOAL.md) for target inputs, [setup](getting-started.md) for assets
and environments, and [configs](../configs/README.md) for model-specific jobs.
JSON defines the search space, `search/` coordinates execution and `skills/`
implements mechanisms. The Agent prepares missing adapters and prerequisites,
then uses the existing finite stage loops.

## Inspect before execution

Read the selected jobs, [status](status.md) and applicable local receipts. Verify
model, hardware, data, generation and Parent identity before reuse. A new Agent
or folder name does not justify rerunning completed work. Check processes and
locks before starting a coordinator; a saved `running` status is not liveness.

```bash
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json --execute
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json --execute --resume
```

The first command is read-only. Execution runs missing stages; resume requires
unchanged saved jobs. For another target, use [migration](model-migration.md)
and establish separate model/hardware result scopes. `ready_for_probe` is a
static capability result, not environment or quality qualification.

## Stage ownership

| Stage | Required output |
| --- | --- |
| Dense preparation | Native inference, Search/Dev, full LMMS, performance denominators and calibration |
| Token | Actual feature selection/compaction, finite ratio search, quality and timing |
| SP | Calibration, physical FFN recipes, hardware width probes and quality boundary |
| Quantization | Independent Dense export with precision manifest, reload, quality and timing |
| Stack | Physical SP → fresh quantization → Token; mechanisms/timing, full native LMMS and artifact |
| Runtime | Parent-matched attention/compile/fusion timing and `runtime-freeze.json` |
| KV Cache | Runtime-matched dtype/concurrency timing and `kv-cache-freeze.json` |
| Final serving | Complete native five-task quality with actual Runtime/KV/Token and `serving-freeze.json` |
| Usage/benchmark | Image inference and matching end-to-end timing against original Transformers Dense |

Dense native LMMS and SP calibration are separate preparation steps. The pipeline
does not manufacture those inputs. Use the model-specific calibration collector
or compatible import before SP. The 128-input timing workload after Stack is
performance-only; Stack, Runtime and KV do not use Search/Dev quality screening.

## Recovery and errors

Independent Skills keep their own quality ladders. Stack starts with qualified
independent sources and uses one paired neighbor on each complete LMMS quality
failure: wider SP plus higher Token keep ratio. Its JSON owns the shared budget.
`search.stack --execute` runs this loop; do not add parallel SP-only/Token-only
recovery branches. See [Stack](stack.md#fast-search-and-lmms-recovery).

Runtime and KV freezes are performance-ready handoffs. Final serving evaluates
the selected profile first; valid below-target quality tries same-Runtime auto
KV and then configured measured runtime controls. Final fallback exhaustion
does not automatically restart SP/Token. API, capacity and execution failures
stop the affected branch with retained error evidence.

Strict jobs stop on quality exhaustion. Explicit best-effort jobs may continue
with the highest-quality fully evaluated eligible candidate; see
[selection policy](stack.md#selection-policy-and-tradeoffs). Both retain the 95%
target and exclude partial/failed candidates. Repair locally resolvable blocks,
verify and resume without duplicate GPU jobs.

## Delivery

Require all five native tasks and the equal-task mean of per-task retention
ratios. Only `lmms_qualified` certifies the quality target for all five layers.
A matching end-to-end Dense speedup >1 is also required for goal achievement.
Report per-task scores, mean retention, settings, actual hardware, timing scope,
freeze/benchmark paths and unresolved blocks. Keep concurrent throughput separate.

Use [serving](serving.md) for the final comparison and [retention](../results/README.md)
for reusable evidence. Raw measurements remain authoritative. The source edition
contains reported summaries, not the local reference receipts.
