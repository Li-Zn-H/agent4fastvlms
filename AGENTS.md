# Agent Working Guide

Keep this local optimization framework lightweight. Use existing JSON jobs,
stage loops and receipts; do not add another orchestration or state system.
Maintain project documentation in English. Read [README](README.md), the relevant
stage guide and selected JSON before changing code or running commands.

## Preparation

- Follow [GOAL](GOAL.md) only when asked to execute a search. Reading/editing the
  template does not start a job. Use [setup](docs/getting-started.md) for a fresh
  source checkout and [the Agent guide](docs/agent-search.md) for execution.
- Inspect existing result roots and actual processes first. Reuse complete,
  applicable evidence. A matching `lmms_qualified` serving freeze and matched
  accelerated Dense benchmark complete the task; do not restart it.
- For another model/platform, read [migration](docs/model-migration.md) and
  [registration](docs/registration.md). Registration is not execution evidence.
  Keep imports lazy and catalog/planning paths read-only.
- Set `evaluation.result_root` and every explicit source/output/Parent path to
  the new experiment scope. Establish that model's own Dense denominators and
  calibration. Never transfer another model's geometry, recipes or speedups.
- Use the project-local Dense Python and each backend's isolated `worker_python`.
  Do not install quantizers in the Dense environment. Preserve source weights.

## Stage contracts

1. Dense supplies Search/Dev, native LMMS and performance baselines.
2. Token and SP have independent quality ladders. Quantization starts from Dense,
   requires an isolated requirements file and completes prepare, calibrate,
   export, precision manifest and reload checks before selection.
3. Stack composes physical SP, fresh Parent quantization, then Token. Validate
   mechanisms/reload, measure performance, then run complete native LMMS.
   Stack and Runtime/KV never run Search/Dev quality screening; a 128-input
   timing workload is performance evidence only.
4. Only valid full LMMS quality failure triggers paired recovery: the next wider
   SP and next higher Token ratio together. JSON owns ladders and finite budgets.
   Explicit `token_recovery_exhaustion: reuse_last_qualified` may reuse the last
   qualified Token source after its ladder ends while SP widens, within that budget.
   Distinguish exhausted budget with an unseen neighbor from exhausted ladder.
5. Runtime owns attention, compilation, CUDA Graph and fusion. KV starts from
   `runtime-freeze.json` and owns dtype/concurrency. Neither rewrites Stack.
   Changed Parent/Token/context requires matching evidence.
6. Final serving LMMS uses native prompts, preprocessing and scoring with actual
   Token and Runtime/KV checks. On valid quality failure, try same-Runtime auto KV,
   then measured configured runtime controls. Exhaustion does not automatically
   return to SP/Token recovery. API/execution errors stop the affected branch.

## Acceptance and reuse

The target is >=95% equal-task mean of five candidate/Dense retention ratios,
plus matched end-to-end speedup >1. This is neither a sum-score ratio nor a
per-task minimum. Do not resize images, truncate prompts, lower thresholds or
disable a Skill to bypass a block. Validate the required context before GPU work.

`strict` is the default. Explicit `best_effort` can continue only after bounded
quality exhaustion with the highest-quality complete, mechanism-valid candidate.
Stack records `frozen_best_available`; below-target final serving records
`lmms_evaluated`, never `lmms_qualified`. Partial results and errors are ineligible.
Keep separate quality and speed outcomes: `goal_achieved`,
`accelerated_quality_unmet`, or `no_acceleration`.

Use `search.pipeline` read-only first; `--execute` runs missing stages and
`--execute --resume` resumes an unchanged saved job. An exit code alone is not
a stage pass. Keep one coordinator per job. See [command defaults](search/README.md).
Final speed uses [search.serving](docs/serving.md) and the same model's original
Transformers Dense; never multiply independent Skill speedups.

## Evidence and verification

Keep configs, states, summaries, raw quality/timing, candidate receipts and
selected artifacts. Delete only unneeded temporary work after preserving evidence;
follow [result retention](results/README.md). Keep reference measurements in
[status](docs/status.md); long local histories are not part of this source edition.

Run focused tests for code changes; run the full CPU suite when changes cross
module boundaries. See [Contributing](CONTRIBUTING.md) for portable versus
asset-dependent checks. Documentation changes need `scripts/check_docs.py`,
not a GPU Worker. Public preparation does not authorize publication or license
selection; follow [release preparation](docs/releasing.md).
