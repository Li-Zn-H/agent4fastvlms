# Kimi-VL-A3B-Instruct Preparation

The Kimi-VL-A3B-Instruct migration completed its measured path on an NVIDIA
A800-SXM4-80GB on 2026-10-15. This page keeps the model-specific preparation and
implementation notes; the [final result report](results/kimi-vl-a3b-a800.md)
is the authoritative summary of quality, serving settings and timing.
Registration and JSON jobs describe execution but do not replace the saved
quality and benchmark evidence. The source tree does not include the local
checkpoint, data or result receipts.

## Native reference

The original local checkpoint is `models/Kimi-VL-A3B-Instruct`, with a
27-layer MLA/MoE decoder, 64 routed experts, top-6 routing and two shared
experts. The first decoder FFN is dense. Native MoonViT uses patch size 14
and 2-by-2 feature merging. Preserve the original image processor settings,
Instruct chat template, router correction tensors and generation stop IDs.
Use greedy non-thinking decoding, which the publisher permits for Instruct.
Do not use Thinking weights, modify source weights or truncate prompts.

[The model descriptor](../configs/model-kimi-vl.json) authenticates the native
source and preprocessing files. Dense uses Transformers 4.50.3 and the
unmodified checkpoint implementation in `.venv-kimi`; dependencies are pinned
in [native requirements](../requirements-kimi-vl.txt). The local environment
currently shares the installed Torch distribution through system site packages
and installs its own Transformers dependencies. It does not modify the GLM
environment. Quantization and Runtime require their own isolated environments.

Native LMMS uses an isolated `.venv-lmms-kimi` task host and the original
`.venv-kimi` GPU worker through the existing native connection. The host pins
Transformers 4.50.3 and Datasets 4.4.1 in
[LMMS requirements](../requirements-lmms-kimi-vl.txt); older Datasets 3.6 cannot
read the shared cache's native `List` feature metadata. No dataset cache or
task scoring is rewritten. [The CPU context probe](../search/kimi_context_probe.py)
checks every native input before full GPU LMMS and resumes complete task audits.

The complete CPU audit covers 23,349 native inputs across all five tasks. The
largest required context, including each task's unchanged generation allowance,
is 2,385 tokens. This is capacity evidence, not a quality score. SP calibration
has completed on 128 disjoint inputs; its loader verifies all 53 FFN units,
1,691 dense/routed/shared branches and native routing-stratum denominators.
The A800 width probe completed 780 BF16 and Marlin kernel shapes with numerical
reference checks. Width admission alone does not establish model quality or
end-to-end acceleration.

The first planned SP recipe also passes a full 27-layer native meta-model shape
check: all 53 FFN units and 1,691 branches have the declared physical widths,
with 64 routed experts, top-6 routing and two shared experts preserved. The
recipe removes 4.7247% of decoder FFN parameters. Meta parameters allocate no
checkpoint weights and execute no forwards; this checks reconstruction shapes,
not real-weight reload, quality or acceleration.

[Physical Parent export/reload](../skills/kimi_vl_artifact.py) retains original
source, processor and generation files. It replays the saved physical recipe
before strict tensor assignment and rejects missing weights or changed router
precision. CPU tiny-model checks preserve logits and cached generation exactly.
[The isolated modern-Worker bridge](../adapters/kimi_vl_compat.py) keeps native
eager MLA/MoE forwards and the original tuple cache. The same saved tiny Parent
produces identical logits and cached outputs under Transformers 4.50.3 and
5.14.1. A full-model modern `from_pretrained` smoke exposed uninitialized
nonpersistent rotary buffers despite unchanged stored parameter tensors. The
bridge now reconstructs those buffers with Kimi's original rotary constructor
after modern loading. CPU recovery checks on the complete original checkpoint
match the original Dense response prefixes for two images, with four generated
tokens per image. The complete optimized reload, serving LMMS and matched
benchmark are recorded in the final result evidence.

The independent GPTQ environment is `.venv-quant-kimi`, with
[its own requirements](../requirements-quantization-kimi-vl-gptqmodel.txt).
[Kimi calibration interfaces](../skills/kimi_vl_quantization.py) enumerate native
Linear modules, preserve BF16 routing corrections, capture MoonViT's packed
sequence offsets and align only its internal FFN channels. Kimi invokes
`kv_b_proj` as a Linear, so it belongs to sequential GPTQ calibration. Native
image grids and the multimodal projector remain unchanged.

The original processor returns FP32 pixels. Direct vision calibration now casts
them to the vision tower's dtype, matching the original outer forward before
image-feature extraction. The seven-test Hessian/export suite exercises these
FP32 inputs; it does not substitute already-cast synthetic pixels.

Native Hessian GPTQ export now writes actual packed weights, a complete precision
manifest and the calibration receipt. Independent Dense export and freshly
calibrated physical Parent export pass seven isolated CPU tests on a small native
MoE/MoonViT model. Packed CPU replay preserves logits and cached generated tokens
exactly; router values, original protocol files, complete module coverage,
packing geometry and Group-128 conventions are checked. Invalid router values,
group indices, missing calibration and changed ancestry are rejected. This is
mechanism evidence in isolation; full-checkpoint A800 reload and the vLLM path
are covered by the selected Stack and serving receipts.

The existing Worker and Stack helpers now dispatch Kimi physical export, fresh
quantization and strict loading to these interfaces. The selected recipe is
authenticated before the Stack quantizer starts. The isolated seven-test GPTQ
suite passes separately;
it also checks artifact preparation and optimized native/serving descriptors
without constructing a GPU model. Full CPU-suite limitations include absent
other-model assets and dependencies incompatible with the original Dense
environment. These failures do not qualify any GPU mechanism.

Modern Transformers automatically selected SDPA for MoonViT in the larger CPU
fixture, despite the eager decoder setting. The modern bridge now explicitly
binds both vision and decoder to native eager attention. Physical and quantized
receipts also preserve the exact precision and values of nonpersistent rotary
frequencies, and invalidate derived cosine/sine caches on replay. This avoids
changing the numerical reference when loading an exported Parent.
The updated eager bridge also passes a complete-checkpoint CPU smoke on two
native images, with four generated tokens per image matching the original GPU
Dense response prefixes. This bounded check is not optimized GPU qualification.

Native LMMS integration has completed a two-input smoke for each task, including
unchanged scoring and sample receipts. Those limited scores are ineligible as
Dense denominators. The original Transformers 4 five-task Dense run has completed
under `results/model/kimi_vl/a800/lmms/dense/`. ChartQA has completed
all 2,500 inputs with native `relaxed_overall` 0.8944. MMStar has completed all
1,500 inputs with native `average` 0.5229146964630563. TextVQA has completed all
5,000 inputs with native `exact_match` 0.88778. DocVQA has completed all 5,349
inputs with native ANLS 0.9871072081009586. A CPU audit recomputes every DocVQA
sample's native score and verifies its question and target against the unchanged
cached task; all match. POPE has completed all 9,000 inputs with native F1
0.8583425960232186. All 23,349 samples and native input receipts are verified;
`preparation/dense-five-complete-tasks.json` binds the complete denominators to
their raw results and samples. These Dense scores do not qualify an optimized
candidate. Its task host resolves
the native model through LMMS's dotted class-path manifest and uses the isolated
original Dense GPU worker.

The installed LMMS 0.7.3 evaluator hashes `request.doc`, which is `None` in this
generation path. The native sample files therefore contain the
same SHA256 of JSON `null` for every `doc_hash`. These fields cannot establish
Dense/candidate image identity. Original result/sample files remain unchanged.
[Additional native input receipts](../search/kimi_lmms_identity.py) now bind each
sample's original RGB raster and dimensions, question, native generation options,
stop strings and response to those raw files. The original Dense reference can
be reconstructed from its cached native tasks only after every recorded question
and target matches, with the dataset fingerprint saved. Its provenance is
explicitly `reconstructed_original_dense`; it does not claim historical image
observation. Future native and vLLM candidates write `observed_generation`
receipts from their actual generation path. Candidate reconstruction is rejected.
Kimi LMMS reuse and aggregation require these receipts in addition to the
existing task/sample/Token/Runtime checks; different rasters or generation options
cannot qualify even when the upstream document hashes are identical.
All five Dense tasks now have reconstructed reference receipts for all 23,349
inputs, with every recorded question/target checked and all original
result/sample files preserved. Seven input-identity tests reject changed rasters,
generation settings, native splits, raw files and reconstructed candidates;
the six serving tests and five native command/registration tests also pass.

## Independent candidate evidence

The first Token candidate, requested keep ratio 0.95, completed native GPU
Search128 screening with retention 1.0122407344440667 and passed the configured
Search direct-pass rule. Its actual visual feature count is 69,499 of 73,089
across these inputs; image preprocessing and resolution are unchanged.
The one-round independent timing is 0.8725539173465222 seconds per input versus
the reused original Dense baseline, giving 1.0350067897096553x local speedup.
This is neither complete LMMS quality nor the final combined serving benchmark.
Raw quality, Token mechanisms, timing and Worker receipts are retained under
`search/token/runner/candidates/keep-95/attempt-1/` in the Kimi result scope.
The runner stopped at the explicit one-candidate invocation limit, preserving
its unchanged JSON ladder for resume; the independent search is not exhausted.

The first SP candidate, path index 14 and actual decoder FFN removal
0.047247416156928915, also completed native GPU Search128 with retention
0.9912954777286638 and passed direct screening. Its mechanism receipt verifies
the physical shapes of all 53 FFN units, preserved router/expert counts, zero
padding/fallback and actual prefill/decode observations for all 27 layers.
Evidence is under `search/structured-pruning/runner/candidates/step-0014/`.
This independent quality run does not supply full-model speed or exported Parent
reload evidence. The unchanged SP ladder remains resumable after the first
candidate invocation limit.

The Token 0.95, 0.90 and 0.85 timing receipts have an invalid
`pruning_seconds` component: the initial post-projector event was reused on
every cached decoder call, accumulating overlapping intervals. Their raw
receipts remain unchanged. Quality and separately measured wall-clock
end-to-end/generation times remain reusable; the affected component cannot
support phase timing claims. The Kimi timer now measures this preparation
interval only before the first decoder call. Two CPU hook/timing regressions
and both original native preprocessing/generation tests pass. Subsequent
Workers use the repaired timer without restarting completed candidates.
Token 0.80 has completed Search128 with retention 0.9884393063583815 and a
full corrected 128-input timing round. Every phase interval fits within its
measured generation time; total preparation is 0.10883481615781784 seconds.
`preparation/native-token-timer-repair-validation.json` binds the CPU checks,
corrected GPU timing and unchanged affected raw files. The independent Token
loop continues through its configured finite ladder.

## Serving integration under validation

[The lazy vLLM model](../skills/kimi_vl_vllm.py) constructs physical Dense,
routed and shared FFNs with the original expert counts and top-k selection.
It computes router logits in FP32 using unchanged BF16 stored parameters and
preserves native sigmoid/noaux_tc zero masking. All native MLA projections,
including `kv_b_proj`, and MoonViT linears use the declared GPTQ policy; the
multimodal projector remains BF16. Loading checks exact router value hashes,
physical widths and the fused-MoE Marlin backend. Encoder CUDA Graph replay
remains unsupported until validated.

[The scoped MLA compatibility repair](../skills/kimi_vl_mla_compat.py) corrects
one stale dtype access in vLLM 0.26's installed chunked-prefill method. GPTQ
has no `.weight` attribute; the method must use its already-computed
`params_dtype` fallback. The repair binds only to each Kimi attention instance,
preserves the installed algorithm and refuses unfamiliar implementations.
An isolated CPU check executes the actual upstream method with a packed
projection interface, demonstrating the original error and repaired input dtype.
It does not establish CUDA kernel correctness.

[The native serving bridge](../search/kimi_serving.py) shares the authenticated
Instruct template and image processor with Dense. Token selection happens after
the multimodal projector; only the single media placeholder run is compacted.
Per-request audits require the exact native patch/merged grid, coverage-derived
retained count, complete compacted prompt and observed selection. Native LMMS
retains task generation budgets, stop strings and scoring. The final-use session
times image opening through answer decoding and synchronization, then performs
prompt audits outside timing. Dense decodes only the generated suffix.

CPU tests compare prompt preflight with original preprocessing across image
geometries and Unicode/whitespace inputs, exercise native LMMS serving dispatch,
reject changed routing/Runtime/KV evidence and check matched timing boundaries.
The actual vLLM multimodal registry and processor also pass CPU checks with
and without processor-cache reuse. Across four native image geometries and
keep ratios 1.0, 0.8 and 0.5, string and tokenized inputs preserve exact native
BF16 vision inputs, patch grids, complete compacted prompts and placeholder
ranges. Native Transformers casts pixels to that same vision dtype before
patch embedding. This check executes framework preprocessing but no GPU model.

The public independent GPTQ route now dispatches Kimi export directly and passes
the real small-model Hessian/export audit, including unchanged source weight
files and absence of SP ancestry. Independent quantization was probed through
the completed Stack path. The existing
`search.quantization --config configs/quantization-kimi-vl.json` read-only plan
finds both isolated environments and their dependencies ready for the Worker.
That plan is still read-only; execution evidence is retained under the Kimi
result root and summarized in the final report.

An additional saved synthetic fixture retains the original tokenizer, image
protocol, context, MLA head geometry and 64-expert/top-6 routing, with a smaller
two-layer decoder and one-layer vision tower. Its seeded random weights use a
smaller attention output scale to avoid routing collapse in this reduced model.
It is separate from the original checkpoint and every search artifact. Real
native calibration covers all 210 quantized modules and all 64 routed experts;
packed CPU replay preserves logits and four-token cached generation exactly.
With the configured 4-by-4 coverage grid and keep ratio 0.8, an unchanged native
224-by-224 synthetic raster yields 64 visual features, 52 retained features and
the matching physical KV length. This fixture is only a runtime mechanism probe,
never a calibration source or quality/speed denominator for the real search.

The isolated eleven-test vLLM CPU suite also checks native FP32 rotary frequencies,
BF16 cosine/sine tables and equivalent query/key permutation across positions
up to the original context limit. The actual fused-MoE router factory selects
the custom native callback and preserves all 64-expert/top-6 IDs and scaled
weights for zero, random and negative corrections. These are CPU routing checks;
the fused expert CUDA kernel remains unverified. The latest full CPU suite runs
416 tests with one failure, 38 errors and 68 skips from unavailable other-model assets and
environments; no Kimi test fails. Neither CPU suite certifies CUDA execution.

The modern Runtime environment reads all 23,349 native LMMS requests from the
existing cache. Every task's generation options and stop strings match the
original CPU context audit; first, middle and last prompt/token counts match in
each task. This checks task compatibility without scoring model outputs.
Saved serving receipt reuse now applies the same Kimi-specific router precision,
expert topology, physical FFN widths, Marlin, vision padding and Token/KV checks
as a newly constructed engine. Six serving tests reject inconsistent historical
receipts that would otherwise satisfy only the shared generic fields.

## Jobs and evidence

[The pipeline job](../configs/pipeline-kimi-vl.json) binds all outputs to
`results/model/kimi_vl/a800/` (local generated assets, excluded from source).
The recorded GPU UUID is `GPU-d4baa44d-a66f-ac51-0146-a18ac21f00ce`.
Independent Token and SP screening used Kimi's completed Search/Dev and
performance baselines. The completed Stack and final serving runs use the
model's own five-task native LMMS denominators. Shared datasets may be reused;
other models' scores, calibration, physical recipes and speedups may not.

The native CPU geometry Worker has prepared all 256 unique Search/Dev inputs.
At the first requested ratio of 0.95, rounding retains 140,659 of 147,927 visual
features across those inputs (95.0868%). This is feature-count evidence, not a
quality result. The Token coordinator can reuse it and the existing original
Dense quality/timing receipts. After the Dense LMMS coordinator releases the GPU,
the first candidate can be run with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-kimi/bin/python -B -m search.optimize \
  --config configs/optimization-kimi-vl.json \
  --output results/model/kimi_vl/a800/search/token/runner \
  --execute --max-candidates 1
```

The explicit output path is required by the Token CLI. For later invocations on
that unchanged saved job, add `--resume`; the one-candidate invocation limit
does not alter the finite ratio ladder in JSON. Do not start another coordinator
while Dense or the Token job is live.

The read-only pipeline plan remains a planning check; the measured stages were
executed through their saved stage loops. Stack and final serving explicitly
select `best_effort`, retain the 95% equal-task mean target and exclude partial
results and execution errors.
Physical SP preserves routing, all experts and top-k selection. It removes
intermediate channels using fresh Kimi calibration. Final acceleration requires
a matched comparison against original Kimi Transformers Dense.
