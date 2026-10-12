# Structured Pruning

SP physically narrows decoder FFNs using model-specific calibration and aligned
channel allocation. Its independent quality ladder supplies candidates for Stack;
it does not certify the combined model's final quality or speed.

## Calibration and channel selection

Score each FFN intermediate channel using post-SwiGLU activation moments and its
down-projection column:

```text
score[j] = sqrt((E_visual[a_j^2] + E_text[a_j^2] + E_decode[a_j^2]) / 3)
           * L2_norm(W_down[:, j])
```

Visual-prefill, text-prefill and decode strata receive equal weight. Sort channels
by increasing score, then original index; remove complete groups of 128. The
cross-layer greedy cost uses newly removed scores divided by that layer's original
score sum and removed FFN parameter count. Layers may end with different widths.
These are contribution proxies; actual quality evaluation remains necessary.

Recipes store original retained indices and per-layer widths. Physical slicing
changes gate/up rows and down columns, including applicable biases. Keep indices
ordered, reject repeated application and verify actual shapes. Report removal
as a fraction of decoder FFN parameters, not total model parameters.

Qwen reference calibration is imported from local frozen statistics using
`search.structured_pruning.import_calibration`. It is not shipped. For LLaVA-1.5,
prepare calibration inputs separately from evaluation, then collect statistics:

```bash
.venv-qwen3vl/bin/python -B scripts/calibrate_llava_sp.py \
  --model-config configs/model-llava15.json \
  --evaluation-config configs/evaluation-llava15.json \
  --config configs/structured_pruning-llava15.json
```

This command runs the model. Verify calibration provenance and disjointness from
evaluation samples/images. A hardware-width probe cannot generate calibration.
Recipe replay through the model adapter is the supported reconstruction path;
a standard Dense loader cannot infer a nonuniform physically pruned architecture.

## Hardware width probe

`search.structured_pruning` measures eligible aligned FFN shapes with isolated
Workers; `--execute` runs missing measurements. Store raw width timings and the
model/hardware context. A completed width probe is not a quality-search pass.

## Progressive quality search

`search.pruning_search` follows the JSON allocation path and coarse FFN-removal
steps. Local Search/Dev decides the path boundary; a failed candidate causes the
configured step back/confirmation. Alignment to 128 does not require quality
evaluation at every 128-channel width. Retain recipes and raw quality outputs,
without per-candidate full checkpoint copies. Resume uses frozen job settings.

The fixed step-back policy chooses the deepest admitted path point whose actual
FFN removal does not exceed the current removal plus `initial_depth_step`.
Discrete widths can make the achieved increment smaller than the configured
step. DeepSeek-VL2, for example, keeps a common width across each layer's 72
routed experts. One 128-channel routed width move removes about 0.27855 percentage
points of its original decoder FFN parameters: 17 such moves remove 4.7354 points,
whereas 18 exceed a five-point increment. Dense/shared moves have different costs;
always report the actual physical parameter fraction from that model's recipe.

The independent quality run does not time every full-model candidate.
`search.pruning_screen` provides optional representative timing; final combined
speed must still come from Stack and matched end-to-end benchmarking.
See [commands](../search/README.md#standalone-commands-and-verification) and
[evidence retention](../results/README.md).
