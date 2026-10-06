# agent4vlm

A lightweight local framework for Agent-driven vision-language model optimization.
JSON jobs configure the search; isolated Workers execute it; saved evidence lets
interrupted jobs resume and completed stages be reused.

```text
Model adaptation and Dense baselines
  → independent Token / structured pruning / quantization
  → Stack composition and full LMMS with paired recovery
  → Runtime → KV Cache → final LMMS
  → matched end-to-end benchmark against the same model's Transformers Dense
```

## Start here

Python 3.12 is sufficient for these checks, without weights, GPUs or model packages:

```bash
python3.12 -B -m search.registry
python3.12 -B examples/scoring_demo.py
python3.12 -B scripts/check_docs.py
```

For real searches, follow [setup](docs/getting-started.md). Edit the model and
hardware in [GOAL.md](GOAL.md), then ask an Agent to execute it. For a prepared
LLaVA-1.5 job:

```bash
# Inspect prerequisites and reusable evidence; does not launch GPU work.
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json
# Run missing stages, or resume an unchanged saved job.
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json --execute
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline-llava15.json --execute --resume
```

Weights, Search/Dev data, calibration and measured receipts are local prerequisites
and are not distributed. The included JSONs are reference jobs, not a one-command
reproduction package. Their model-specific preparation requirements are listed in
[configs](configs/README.md). No GPU search was rerun to prepare this source edition.

## Search and acceptance

- Independent Token/SP/quantization use local quality screening. Token selects
  visual features after encoding; it preserves the model's native image resolution.
- Stack physically prunes FFNs, freshly quantizes that Parent and applies Token.
  It checks mechanisms/loading and timing, then runs full native LMMS. Quality
  failures recover the next wider SP and next higher Token ratio together.
- Runtime probes attention/compilation/fusion; KV probes cache dtype/concurrency.
  Stack and subsequent stages do not use Search/Dev quality screening.
- The target is **at least 95% equal-task mean retention and end-to-end speedup
  greater than 1** against the same model's Transformers Dense. Explicit
  `best_effort` jobs may retain the best complete candidate after finite quality
  exhaustion; below-target results remain labelled as such. Errors and partial
  evaluations cannot qualify. Independent speedups are never multiplied.

## Reported local results

| Model | Five-task mean retention | Batch-1 end-to-end speedup |
| --- | ---: | ---: |
| Qwen3-VL-4B-Instruct | 95.4011% | 2.5293x |
| LLaVA-1.5-7B | 95.5839% | 2.9150x |

Measured on RTX 4090, using 128 inputs x 3 rounds per backend. These are reported
local measurements, not results reproduced from this source-only package. See
[settings, per-task retention and scope](docs/status.md).

## Documentation and source

| Need | Entry |
| --- | --- |
| Install and prepare inputs | [Getting started](docs/getting-started.md), [small example](examples/README.md) |
| Execute and resume | [Agent guide](docs/agent-search.md), [command index](search/README.md), [configs](configs/README.md) |
| Understand the methods | [Evaluation/Token](docs/evaluation.md), [SP](docs/structured-pruning.md), [quantization](docs/quantization.md), [Stack/Runtime/KV](docs/stack.md) |
| Add a model or platform | [Migration](docs/model-migration.md), [registration](docs/registration.md) |
| Use and measure a final model | [Serving](docs/serving.md), [evidence retention](results/README.md) |
| Contribute or prepare publication | [Contributing](CONTRIBUTING.md), [security](SECURITY.md), [release notes](docs/releasing.md) |

`adapters/` owns model/hardware contracts, `skills/` implements optimization,
`search/` orchestrates execution, and `tests/` contains regression checks.
Historical trial jobs, long development logs and generated assets are omitted
from this edition. All core Python implementations and regression tests are retained.

## License

A source license has not yet been selected. This is a public-release preparation,
not a published or licensed open-source release. Model, dataset and dependency
licenses remain separate; see [remaining release work](docs/releasing.md).
