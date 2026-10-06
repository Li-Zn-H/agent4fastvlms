# Getting Started

Run commands from the repository root. The reference workflow is local,
single-GPU Qwen3-VL-4B-Instruct inference on Linux x86_64 with Python 3.12.
The pinned PyTorch build targets CUDA 13.0; use a compatible NVIDIA driver and
a GPU with enough memory for the selected model, context and backend. Memory
requirements vary by stage. Other hardware/software combinations need validation.

The repository contains code, configurations and documentation. It does not ship
model weights, evaluation images/answers, SP calibration statistics, quantized
artifacts or the reference experiment's receipts. Default JSONs describe that
experiment, not a self-contained downloadable demo.

## 1. Inspect without GPU dependencies

After obtaining the source, inspect the registry and run the small asset-independent
checks using only Python's standard library:

```bash
python3.12 -B -m search.registry
python3.12 -B scripts/check_docs.py
python3.12 -B -m unittest discover -s tests -p 'test_evaluate.py' -v
python3.12 -B -m unittest discover -s tests -p 'test_run.py' -v
```

The registry catalog does not load model packages, query GPUs or write results.
Model-specific preflight also needs a local checkpoint's metadata. The complete
CPU suite includes tests tied to local model/data/result fixtures; the two tests
above are the portable CI subset, not a claim of full backend validation.

## 2. Create the Dense environment

Create the environment if absent; reuse it if already prepared:

```bash
python3.12 -m venv --prompt qwen3vl .venv-qwen3vl
.venv-qwen3vl/bin/python -m pip install -r requirements-qwen3vl.txt
.venv-qwen3vl/bin/python -m pip check
```

If Python 3.12 has a different executable name on your system, substitute that
interpreter when creating the environment. Once created, use the project-local
interpreter explicitly or activate it and run `python`.

The reference pins include torch 2.11.0+cu130, torchvision 0.26.0+cu130,
transformers 5.14.1 and Pillow 11.3.0. Treat the
[requirements file](../requirements-qwen3vl.txt) as authoritative. Failed wheel,
CUDA or API checks are environment blocks; changing a quality threshold cannot
resolve them. Do not modify an environment supporting a completed experiment
just to test incompatible packages.

For native Dense LMMS, install the separate evaluation requirements into that
model environment when needed:

```bash
.venv-qwen3vl/bin/python -m pip install -r requirements-lmms.txt
```

Quantization and vLLM use isolated environments. For the validated route:

```bash
python3.12 -m venv .venv-quant-gptqmodel
.venv-quant-gptqmodel/bin/python -m pip install -r requirements-quantization-gptqmodel.txt
.venv-quant-gptqmodel/bin/python -m pip check
```

Set each candidate's `worker_python` to its environment. Native Stack LMMS also
needs LMMS in the Stack backend environment; follow the [evaluation guide](evaluation.md#standard-benchmarks-with-lmms)
for the backend-specific installation. Do not install all quantizer requirements
into one environment. LLM Compressor's current pins are known incompatible with
vLLM; TorchAO/AutoRound are declared but disabled. See [quantization](quantization.md).
These commands are setup instructions, not evidence that a fresh install was
validated on every platform.

## 3. Prepare weights and data

| Local asset | Reference path or configuration | Required contents |
| --- | --- | --- |
| Dense checkpoint | `models/Qwen3-VL-4B-Instruct/`, selected by `configs/model.json` | Complete weights, config, processor, tokenizer and chat template from the model publisher |
| Local Search | `data/evaluation/search-128.jsonl` | 128 valid rows under the local scoring protocol |
| Local Dev | `data/evaluation/search-256.jsonl` | 256 rows including the unchanged Search subset |
| Images | Relative to `evaluation.data.media_root` | Files referenced by each row's `media.logical_path` |
| Quantization calibration | Selected quantization JSON | Calibration inputs with the required evaluation disjointness; preserve provenance |
| SP calibration | `calibration.source` in `configs/structured_pruning.json` | Model-specific score/moment files and manifests; a width probe cannot generate them |
| Native LMMS assets | Configured task/cache paths | Official task data and access rights; the LMMS runner may download missing official files |

Obtain checkpoints and datasets from their original publishers under their own
licenses and access conditions. Keep source model weights read-only. The search
pipeline does not download local Search/Dev assets or reconstruct the reference
subset from a public identifier. Exact numerical reproduction requires the same
sample manifests and media; if those cannot be distributed, publish their
provenance and access instructions before claiming independent reproducibility.

A local JSONL row has this shape (the image and answer below are illustrative):

```json
{
  "stable_row_id": "example-0001",
  "primary_category": "example_task",
  "logical_question": "What color is the object?",
  "answers": ["red"],
  "scoring_type": "exact_or_normalized_short_answer",
  "media": {"logical_path": "media/example.jpg"}
}
```

Each row occupies one line in the actual file. IDs must be unique/nonempty;
answers are nonempty strings. Supported `scoring_type` values are
`deterministic_rule_or_structure`, `exact_or_normalized_short_answer` and
`open_response_or_explicit_review`. See [the protocol](evaluation.md#local-data).
Search/Dev overlap deliberately; Dev is a larger confirmation set, not an
independent holdout. Track data source, revision, sampling method and licenses
outside the minimal row schema.

The SP reference uses imported frozen calibration. See
[SP calibration and migration](structured-pruning.md#calibration-and-channel-selection)
for the contract and `search.structured_pruning.import_calibration` for importing
compatible statistics. A new model needs a compatible collector/importer and
its own statistics before SP execution. The pipeline reports `blocked_preflight`
when they are missing; installation alone cannot complete this stage.

## 4. Prepare an experiment scope

Read the [config catalog](../configs/README.md). For a changed model, GPU, dataset
or generation protocol, create model-specific stage JSONs and independent result
roots. Set `evaluation.result_root` and update all explicit sources, outputs,
Parent paths and Dense denominators together. Recommended layout:

```text
configs/<model-id>/<hardware-id>/
results/model/<model-id>/<hardware-id>/
```

The defaults retain historical local Qwen paths. A new directory does not make
old evidence applicable. Physical device indices and isolated Worker devices
are distinct; current CUDA stages accept physical indices 0/1 and use `cuda:0`
inside an isolated Worker. Consult [registration](registration.md).

Run static checks once the local metadata/configurations exist:

```bash
.venv-qwen3vl/bin/python -B -m search.registry \
  --model-config configs/model.json \
  --evaluation-config configs/evaluation.json --device 0
.venv-qwen3vl/bin/python -B -m search.pipeline --config configs/pipeline.json
```

Substitute your generated configs for a new experiment. The pipeline defaults to
read-only. Missing assets or unsupported adapters may block planning; inspect the
reason before execution. `ready_for_probe` is not a runtime or quality pass.

## 5. Execute and retain evidence

Use [GOAL.md](../GOAL.md) for Agent-driven preparation/execution, or follow the
[Agent guide](agent-search.md) and [stage command index](../search/README.md#standalone-commands-and-verification)
manually. Establish missing Dense local, full LMMS and performance denominators,
prepare calibration, and then execute the configured pipeline. The coordinator
reuses applicable evidence and calls existing stage loops.

Resume unchanged pipeline jobs with `--execute --resume`. Check actual processes
before starting another coordinator. Retain configs, raw outputs, summaries,
mechanism receipts and selected artifacts as described in [results](../results/README.md).
Only the final `lmms_qualified` serving freeze certifies all five optimization
layers. Use [serving](serving.md) for inference and matched end-to-end timing.

Local quality uses a ratio of task-macro scores; final LMMS uses an equal-task
mean of per-task retention ratios. Report both precisely, plus actual hardware,
software versions, sample identities, output lengths and timing boundaries.
Reference numbers in [status](status.md) are not transferable to a new target.
