# Contributing

Read [setup](docs/getting-started.md), [AGENTS](AGENTS.md), the relevant stage guide
and JSON before changing code. Keep documentation in English. Preserve stage
boundaries, lazy imports, isolated backend environments and existing receipt
contracts. Use [migration](docs/model-migration.md) for new architectures.

## Local verification

Portable checks require Python 3.12, without local assets or model packages:

```bash
python3.12 -B scripts/check_docs.py
python3.12 -B -m search.registry
python3.12 -B examples/scoring_demo.py
python3.12 -B -m unittest discover -s tests -p 'test_evaluate.py' -v
python3.12 -B -m unittest discover -s tests -p 'test_run.py' -v
```

For code changes, run the relevant focused tests in the prepared environment.
Run the full CPU suite for changes across module boundaries:

```bash
.venv-qwen3vl/bin/python -B -m unittest discover -s tests -p 'test_quantization.py' -v
.venv-qwen3vl/bin/python -B -m unittest discover -s tests -v
```

Some full-suite tests require local model metadata, data or receipts. The complete
test source is retained, but the portable subset does not establish GPU support
or reproduce quality/speed results. Report absent prerequisites explicitly.
Documentation-only work needs link/English checks, not GPU evaluation.

## Changes and evidence

Describe the problem, resulting behavior and relevant validation in each change.
Explain effects on saved jobs and resume. Never relabel partial/failed evidence
as a pass. New mechanisms and support claims need actual model/hardware checks.
Keep quality and timing scope explicit; never multiply independent speedups.

Use relative source links; mark generated result paths as local and excluded.
Keep settings in JSON, mechanisms in guides and concise measurements in
[status](docs/status.md). Retain detailed experiment histories locally.

Before sharing, inspect the file list for credentials, private prompts, weights,
data, environments and generated artifacts. `.gitignore` is not a secret scanner.
Source licensing remains deferred; see [release preparation](docs/releasing.md).
