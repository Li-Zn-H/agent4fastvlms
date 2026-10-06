# Public Source Preparation

This edition presents Qwen3-VL-4B and LLaVA-1.5-7B. It retains core adapters,
Skills, orchestration, evaluation, benchmark/recovery code, regression tests,
their main JSON jobs, setup and method guides. Shared internal model helpers stay
in place to preserve the validated implementations.

Historical candidate configs, long development histories, experiment outputs,
weights, datasets, virtual environments, local Git history and caches are omitted.
The original local experiments remain separate. No search thresholds, measured
outcomes or implementation algorithms were changed for this preparation.
Generic quantization export/reuse tests now reference the retained LLaVA-1.5 job.

Preparation validation on 2026-10-06 passed documentation/link checks, all 25
JSON config references, Python syntax checks, the synthetic example and 21
portable tests in a clean Python 3.12.12 environment without model packages.
The 14 quantization regressions also passed using the prepared local environment
and fixtures. Core implementation files and retained JSONs are byte-identical to
the local source. No fresh GPU installation or model evaluation was performed.

## What the source checks establish

[CI](../.github/workflows/ci.yml) checks English Markdown and links, the registry
catalog, the scoring example, and asset-independent scoring/routing tests on
Python 3.12. Run [the same commands](../CONTRIBUTING.md#local-verification) from
a source-only checkout. They do not load model packages or certify GPU behavior.

All reference measurements in [status](status.md) are reported local results.
Raw receipts, exact Search/Dev selection and calibration are not distributed;
this edition does not claim independent end-to-end reproduction. The two public
model configurations retain their documented preparation requirements.

## Remaining release decisions

- Select a source license with the rights holder and retain required third-party
  attribution. No source license has been assigned by this preparation.
- Confirm repository name/URL, maintainers and a private security-reporting route.
- Inspect the actual files/history to be published for credentials, private data
  and third-party redistribution requirements.
- Validate a fresh GPU environment and acquisition/preparation instructions for
  weights, native tasks, Search/Dev and model-specific calibration.
- If publishing reproducible performance claims, provide permitted input
  provenance/manifests, sanitized evidence, dependency versions, checkpoint
  identity and generation/timing scope. Report candidate selection on LMMS.

Model, dataset and dependency licenses remain separate. The synthetic example
illustrates file format and scoring; it is not a model benchmark or replacement
for evaluation data. Preparing this directory does not publish it.
