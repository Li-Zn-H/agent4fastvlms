# Reusable Model Optimization Goal

Edit the inputs, then ask an Agent: "Execute GOAL.md end to end; reuse valid
evidence and resume interrupted work." Editing this file alone launches nothing.

## Inputs

```json
{
  "model": "models/Kimi-VL-A3B-Instruct",
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

Use the original Instruct checkpoint and its native non-thinking chat template,
image preprocessing, generation termination and task scoring. Do not substitute
Thinking weights. Create Kimi-specific JSON jobs and a separate local artifact
scope under `results/model/kimi_vl/a800/`, bound to the actual GPU identity.
Inspect existing processes before GPU work and avoid overlapping coordinators.
Preserve all GLM, Qwen, LLaVA, InternVL and DeepSeek experiments; their scores,
calibration, physical recipes, quantized artifacts and speedups cannot qualify
Kimi. Implement and validate missing model interfaces before running stages.

For this MoE checkpoint, preserve native routing, expert counts and top-k
selection. SP must physically prune intermediate channels with model-specific
calibration and export/reload checks; do not substitute expert dropping.

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

Set `selection_policy: best_effort` explicitly in the Kimi Stack and final
serving jobs. This keeps 95% as the target. After finite quality exhaustion,
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
