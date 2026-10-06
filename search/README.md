# Search Execution

Complete [setup](../docs/getting-started.md) before GPU work. The
[Agent guide](../docs/agent-search.md) explains cross-stage ownership; JSON jobs
own settings. Commands below are an index, not a script to run sequentially.

## Entry point

```bash
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json --execute
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json --execute --resume
```

The default inspects evidence without GPU queries or writes. `--execute` runs
missing stages; `--resume` resumes an unchanged saved orchestration job. Use
`configs/pipeline.json` for the Qwen reference. Source-only execution first needs
model metadata, data, environments, calibration and Dense native LMMS.

## Independent Token entry point

```bash
.venv-qwen3vl/bin/python -B -m search.optimize --config configs/optimization-llava15.json
.venv-qwen3vl/bin/python -B -m search.optimize --config configs/optimization-llava15.json --execute
```

Inspect the module's `--help` for explicit output and saved-snapshot resume.
Keep Token's independent quality ladder separate from Stack's paired recovery.

## Reuse before execution

Matching complete raw outputs and summaries are reusable; a folder, saved status
or successful process exit is insufficient. Verify model, hardware, sample IDs,
generation, Parent and runtime settings. Changed experiments need matching jobs
and independent result roots. Do not move evidence that retained runs reference.

Only missing stages launch isolated Workers. Dense/Token use the model environment;
quantization and Runtime use JSON-selected backend interpreters. A backend block
is not a quality failure. Retain its reason, resolve it and resume without
duplicating coordinators. Partial LMMS cannot qualify a candidate.

## Standalone commands and verification

Activate the intended project environment or use its explicit Python path.
Substitute model-specific configs for each command; unsuffixed defaults target Qwen.

| Module | Default / execution behavior |
| --- | --- |
| `search.registry` | Read-only catalog; model-specific checks require checkpoint metadata |
| `search.pipeline` | Read-only; `--execute` / `--execute --resume` |
| `search.optimize` | Read-only Token plan; `--execute` |
| `search.structured_pruning` | Read-only width probe plan; `--execute` |
| `search.pruning_search` | Read-only SP plan; `--execute`, saved-job `--resume` |
| `search.pruning_screen` | Read-only optional timing plan; `--execute` |
| `search.quantization` | Read-only preflight; `--execute` |
| `search.stack` | Read-only; `--execute` runs mechanisms/timing, LMMS and recovery |
| `search.stack_runtime` | Read-only; `--execute` needs an eligible Stack freeze |
| `search.stack_kv_cache` | Read-only; `--execute` needs a Runtime freeze |
| `search.stack_serving` | **Executes by default**; chains Runtime/KV/final LMMS |
| `search.serving_lmms` | **Executes by default**; final quality gate |
| `search.serving` | Read-only; `--execute` for inference or `--benchmark --execute` |
| `search.baseline`, `search.benchmark`, `search.run`, `search.lmms` | **Execute directly**; use only for explicitly missing evidence |

Read `--help` before standalone use. The Runtime chain accepts `--runtime-config`,
`--kv-config` and `--lmms-config`; do not silently use Qwen defaults for another
model. Native Dense LMMS and SP calibration are preparation steps, not outputs
fabricated by the coordinator. See [evaluation](../docs/evaluation.md) and
[SP calibration](../docs/structured-pruning.md#calibration-and-channel-selection).

For source verification use [Contributing](../CONTRIBUTING.md#local-verification).
For final quality/speed interpretation use [Stack policy](../docs/stack.md) and
[matched serving timing](../docs/serving.md#fair-benchmarking).
