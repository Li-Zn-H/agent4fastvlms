# Evaluation and Token Selection

Independent Skill screening and native LMMS use different protocols. Preserve
their settings and denominators; a screening score is not full-suite quality.

## Local data

Search/Dev are JSONL manifests selected by the evaluation JSON. Search128 is an
unchanged subset of Dev256, which is a larger confirmation set rather than an
independent holdout. Images resolve below `data.media_root`. Each row contains:

```json
{
  "stable_row_id": "example-0001",
  "primary_category": "example_task",
  "logical_question": "What color is the image?",
  "answers": ["red"],
  "scoring_type": "exact_or_normalized_short_answer",
  "media": {"logical_path": "red-square.ppm"}
}
```

Require unique IDs, nonempty questions/answers and existing relative image paths.
Keep provenance, revisions, licenses and sampling procedures with real datasets.
The [example](../examples/README.md) demonstrates the schema and scorer only.

Supported scorers are `deterministic_rule_or_structure`,
`exact_or_normalized_short_answer` and `open_response_or_explicit_review`.
The latter two use bounded normalized answer spans and a negation-window rule;
they are not semantic judges. Exact behavior lives in [search.evaluate](../search/evaluate.py)
and its [tests](../tests/test_evaluate.py). Responses must cover exactly the
same sample IDs as Dense.

## Quality routing

Local retention divides the candidate's equal-task macro score by Dense's macro
score. The reference Search direct-pass threshold is 97.5%; below that, Dev must
retain at least 95%. Individual Skills may require explicit Dev confirmation;
consult their JSON and loop. Quantization selection requires complete export,
reload, Dev and timing evidence.

Stack and later stages do not run local Search/Dev quality screening. Their
quality decisions use complete native LMMS. A timing workload using Search
inputs does not turn that measurement into a quality gate.

## Performance measurement

Independent/native timings, runtime engine timings and final matched end-to-end
timings have distinct scopes. Keep warmups, repetitions, output lengths, sample
counts and raw measurements with each result. Do not multiply Skill speedups.
Use [the serving benchmark](serving.md#fair-benchmarking) for the final Dense ratio.

## Standard benchmarks with LMMS

Install LMMS in the relevant Dense and isolated Stack environments:

```bash
.venv-qwen3vl/bin/python -m pip install -r requirements-lmms.txt
.venv-quant-gptqmodel/bin/python -m pip install -r requirements-lmms.txt
.venv-quant-gptqmodel/bin/python -m pip check
```

Prepare a missing LLaVA-1.5 Dense denominator with:

```bash
.venv-qwen3vl/bin/python -B -m search.lmms \
  --model-config configs/model-llava15.json \
  --output results/model/llava15/cuda0/lmms/dense --resume
```

Use the matching Qwen model and output root for its baseline. The runner uses
native prompts, preprocessing, generation and scoring. Local screening's token
limit/scorer does not override native tasks. Official data may be downloaded to
the configured local cache. Acquire it under the publishers' access conditions.

| Task | Samples | Primary metric |
| --- | ---: | --- |
| ChartQA | 2500 | `relaxed_overall` |
| MMStar | 1500 | `average` |
| TextVQA validation | 5000 | Native VQA metric named `exact_match` |
| DocVQA validation | 5349 | ANLS |
| POPE | 9000 | F1 over the complete task |

### LMMS gate policy

The gate is `(1 / 5) * sum(candidate_task_score / Dense_task_score) >= 0.95`.
It is neither a sum-score ratio nor a per-task minimum. Require every task's
complete aligned samples and actual mechanisms. A smoke `--limit` or partial
run cannot qualify. Stack's gate uses native generation; final serving uses vLLM
with the native task protocol. LMMS was used for candidate selection in the
reference searches and is not an untouched held-out test.

## Vision Token search

Token selection acts on post-encoder visual features, preserving native image
preprocessing and resolution. It reduces the language-model visual sequence and
associated work/KV use; it does not remove most vision-encoder computation.
Keep-all equivalence, actual selected counts, placeholder alignment and compacted
prompts must be verified for each model/runtime bridge.

The model-specific JSON defines a finite decreasing keep-ratio ladder and failure
stopping rule. The independent runner measures eligible candidates and retains
conservative options for Stack recovery. See [Token commands](../search/README.md#independent-token-entry-point).
