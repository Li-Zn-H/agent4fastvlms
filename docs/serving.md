# Using the Final Configuration and Measuring End-to-End Speed

`search.serving` loads a qualified or explicitly evaluated best-effort `serving-freeze.json` using the existing
SP/GPTQ adapters, visual-token selector and Runtime parameter mapping. It runs
single-image inference or a Dense comparison through `search.worker`. See the
[status index](status.md) for the reference configuration and quality results.
A source-only checkout first needs its own qualified freeze and local assets;
see [getting started](getting-started.md).

## Configuration and read-only checks

[configs/serving.json](../configs/serving.json) selects the freeze, output root,
GPU, Dense model/environment, evaluation settings and benchmark warmup/repeats.
vLLM uses the final acceptance job's saved `worker_python`. SP, quantization,
Token, Runtime and KV settings come from the freeze.

```bash
.venv-qwen3vl/bin/python -B -m search.serving
.venv-qwen3vl/bin/python -B -m search.serving --benchmark
```

These default to read-only: no weight loading, GPU query or result writes. The
entry rechecks Parent digests, artifact metadata, model descriptors, complete
five-task LMMS and actual mechanism evidence. Execution requires a matching
`lmms_qualified` configuration, or an explicit `best_effort` job's complete
`lmms_evaluated` configuration. The latter remains below the quality target and
is never relabelled as qualified. See [selection policy](stack.md#selection-policy-and-tradeoffs).
The benchmark summary's `acceptance` records retention, the quality target,
matched end-to-end speedup and separate target booleans. Its `outcome` is
`goal_achieved`, `accelerated_quality_unmet`, or `no_acceleration`. Retained
alternative freezes under the final LMMS root's `final/<candidate-id>.json`
can be referenced by a serving JSON with a separate output scope for comparison.

## Single-image inference

Supply a local image path:

```bash
.venv-qwen3vl/bin/python -B -m search.serving \
  --image /path/to/image.jpg --prompt "Describe this image." --execute
```

The entry supports local images/text, the native single-image chat template,
batch 1 and greedy generation. Length and EOS behavior come from evaluation
settings; the default is at most 64 new tokens, which can limit long answers.
Changed generation settings need a new output directory. Existing LMMS retention
still belongs to the original protocol. Images/prompts are not additionally
shrunk or truncated to fit context.

`serving.json.output/inference-<request-digest>/` contains answers, actual Token
selection, compacted-prompt checks and observed attention/KV settings. A first
request can include shape compilation and is not warmed-up throughput evidence.
This entry provides local inference validation, not an HTTP server.

## Fair benchmarking

```bash
.venv-qwen3vl/bin/python -B -m search.serving --benchmark --execute
```

Transformers Dense and the final five-layer vLLM combination load in separate
environments and use the same physical GPU sequentially. The reference protocol
uses full Search128, five warmup samples and three measured repeats per backend.
Results go to `benchmark/` beneath the configured output root.

Both backends receive the native RGB decoded raster. The entry removes EXIF
orientation metadata without rotating or changing pixels, so vLLM cannot apply
an extra orientation transform absent from the Transformers Dense path.

| Item | Shared protocol |
| --- | --- |
| Inputs | Same image files, questions and order; record image digests and compare native prompt-token digests per sample |
| Generation | Matching greedy settings, maximum new tokens, Dense EOS list and repetition penalty |
| Timer start | After GPU synchronization, before opening the image file |
| Timer stop | After image loading, preprocessing, transfer, vision encoding, generation, answer decoding and GPU synchronization |
| Caches | Reset vLLM prefix cache before each request; use unique media IDs so actual visual selection runs again |
| Outside timing | Model loading, warmup, prefix-cache reset, result writes and post-generation prompt audits |
| Audit overhead inside timing | vLLM in-model Token-selection traces |
| Statistics | Per-repeat time, mean/P50/P95 request latency, request throughput, output-token throughput and raw output counts |

The primary speedup is `Dense total request time / serving total request time`.
Report output-token throughput separately because answer lengths may differ
under identical generation settings. This is a batch-1 local inference-path
comparison, not pure kernel speed, concurrent throughput or a fixed-output-length
experiment. Model loading and server queueing are outside the timing scope.

Historical Runtime/KV `dense_comparison` measurements have different timing
boundaries. Preserve them as historical evidence. New benchmarks do not rewrite
those freezes or change the passed 95% LMMS result.

## Recovery and evidence

The same command reuses matching complete Worker outputs. Changed images,
configuration, freeze or model/processor metadata cannot be mixed into that run;
use a new output directory. On failure, inspect `state.json` and `worker.log`.
Retry runs only missing work and retains prior failure records.

Keep top-level `config.json`, `state.json`, `summary.json` and backend-attempt
`request.json`, `result.json`, `runtime.json`, `outputs.jsonl`, `summary.json`
and necessary logs. Per-sample outputs include Token mechanism evidence. No
weights need copying. Existing Worker launch logic handles GPU locks and process
cleanup.
