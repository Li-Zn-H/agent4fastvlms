# DeepSeek-VL2 Optimization Goal

Target the full `deepseek-ai/deepseek-vl2` checkpoint, not the Small or Tiny
variants. This search is inactive until explicitly requested. Reading or editing
this file launches nothing. Reuse valid evidence and resume interrupted work
only within the authorized scope.

## Inputs

```json
{
  "model": "models/DeepSeek-VL2",
  "hardware": {
    "platform": "cuda",
    "device": 0
  }
}
```

The model is a local checkpoint directory; device is the physical GPU index.
Record actual hardware/software identity. The Agent prepares model-specific
JSONs and adapters using [AGENTS](AGENTS.md), [setup](docs/getting-started.md),
[migration](docs/model-migration.md) and [the workflow](docs/agent-search.md).

Use `results/model/deepseek_vl2/rtx_pro6000_gpu072a4781` for the current GPU.
Earlier evidence remains in `results/model/deepseek_vl2/rtx_pro6000`; the new
scope records authenticated reuse of native quality and calibration, with fresh
timing required. Choose a separate hardware scope if the device changes again.
Preserve completed experiments. Never transfer another model's scores,
calibration, recipes or speedups to the new experiment.

## Preparation and activation

This model/hardware search completed on 2026-10-12. Reuse its matching receipts;
reading or editing this template does not start a new search. A changed model,
GPU or evaluation protocol requires its own result scope and denominators.

The original full checkpoint and pinned official implementation remain intact.
The native Dense environment is `.venv-deepseek`, quantization uses
`.venv-quant-deepseek`, and Runtime/serving uses `.venv-runtime-deepseek`.
Set OMP/MKL/OpenBLAS thread counts to one for execution. See
[model-specific integration](docs/deepseek-vl2-preparation.md) and
[final measurements](docs/status.md#deepseek-vl2-cuda-2026-10-12).

Dense Search/Dev, complete native LMMS, calibration, independent Token/SP/GPTQ,
six Stack candidates, Runtime, Auto KV and final serving LMMS have matching
model-specific evidence. The selected sixth Stack uses 28.4587% decoder
FFN-parameter removal, fresh Parent GPTQ W4A16/group128 and Token keep 0.85
(167/196 main features per native tile, protected layout preserved).
Routing, all 72 routed experts, top-6 selection and two shared experts remain
native; SP removes physical intermediate channels rather than experts.
Native Stack retains 97.4211% of Dense across the five equally weighted tasks.

Final serving uses O2, compilation mode 3, `FULL_AND_PIECEWISE` CUDA Graphs,
TRITON_MLA, Marlin MoE and Auto KV. Complete native serving LMMS covers all
23,349 samples and retains **97.2659%**. Its matching freeze is
`stack-serving-lmms/final/serving-freeze.json`, digest `8c315f7c4ea69ac904f0`,
with status `lmms_qualified`. E4M3 failed kernel initialization because required
shared memory exceeded SM120's per-block limit; it supplied no quality/timing
measurement and was excluded. Auto KV completes the required path.

Single-image inference passes actual native prompt/Token/Runtime checks.
Matched Search128 timing uses the same physical GPU, five warmups and three
rounds per backend: original Transformers Dense averages 0.982524 seconds,
final serving 0.203985 seconds, giving **4.816650x** end-to-end acceleration.
`serving-use/benchmark/summary.json` records `goal_achieved`. This is batch-1
image-open-to-decoded-answer timing; loading, warmup, cache reset and post-run
prompt audit are excluded. Natural output lengths differ.

Independent completion receipts are under `preparation/`: the selected Stack,
Runtime, Auto KV, complete serving audit, `final-serving-inference-completion.json`
and `final-serving-benchmark-completion.json`. Detailed local preparation and
interruption histories are preserved there, outside the source guides.
The adapter's complete pipeline ID is enabled after the measured path succeeds;
registration alone never qualifies a new job.

## Objective

Find a combination with **at least 95% five-task mean quality retention and
end-to-end speedup greater than 1** against this model's Transformers Dense.
Use the same hardware, inputs, native protocols and timing boundaries. Seek a
qualifying combination within finite configured candidates, not a global optimum.

## Work

1. Inspect checkpoint architecture, processor, context and adapter support.
   Implement missing interfaces and validate them; declarations alone do not
   qualify support. Use separate model/hardware configs and result roots.
2. Establish model-specific Dense Search/Dev, complete native LMMS and performance
   baselines. Prepare model-specific calibration; preserve source weights.
3. Run independent Token, SP and quantization searches. Retain conservative
   recovery candidates. Keep backend environments separate from Dense.
4. Compose physical SP → freshly calibrated quantization → Token. Validate
   mechanisms/reload and performance, then full native LMMS. On valid quality
   failure, recover the next wider SP and next higher Token ratio together.
   Explicit `token_recovery_exhaustion: reuse_last_qualified` may keep the last
   qualified Token source when its ladder ends while continuing wider SP recovery.
5. Probe Runtime, then KV from its freeze. Run complete final serving LMMS with
   actual Token, compacted prompts and Runtime/KV evidence. Try configured
   measured runtime controls on valid serving quality failure.
6. Run final image inference and matching end-to-end Dense timing through
   `search.serving`. Report quality and acceleration separately.

Use `search.pipeline` read-only first, then `--execute`, or `--execute --resume`
for unchanged saved jobs. Dense LMMS and SP calibration are preparation steps;
the coordinator requires their evidence. Stack and subsequent stages have no
Search/Dev quality screening. The 128-request workload remains a timing input.

## Recovery and stopping

Reuse complete matching receipts; resume the first missing stage without duplicate
coordinators. Preserve all existing models' experiments. Resolve locally repairable
blocks and resume; API/environment/execution failures are not quality failures.
Never lower the target, silently disable mechanisms, resize native inputs or
truncate prompts to bypass a failure.

Explicit `best_effort` keeps 95% as the target. After finite quality exhaustion,
continue with the best complete eligible candidate. Partial tasks and failed
execution are never candidates. A below-target result is not goal achievement.
Stop at verified success, a documented blocker needing external input, or finite
search exhaustion under the configured policy.

## Deliver

Report model/hardware, selected settings, per-task and mean retention, end-to-end
speed and timing scope, freeze/benchmark paths, checks and reusable commands.
Keep selected artifacts and raw evidence. Mark success only with a matching
`lmms_qualified` serving freeze and measured speedup >1; otherwise report
`accelerated_quality_unmet` or `no_acceleration` accurately. Concurrent throughput
and engine-only timings cannot replace matched single-request end-to-end timing.
