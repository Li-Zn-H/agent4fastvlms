# Local Results and Retention

Generated results are excluded from this source edition. Each model/hardware
experiment owns its own Dense, search and serving evidence. The reported
[reference measurements](../docs/status.md) do not supply those local assets.

## Layout

```text
results/model/<model>/<hardware>/
  search/dense/                  # local quality and performance denominators
  search/vision-token/           # independent ratio ladder and raw outputs
  search/structured-pruning/     # calibration, widths, recipes and quality
  quantization-heldout/          # independent export/reload and selected artifact
  stack/                        # composition, timing, native LMMS and artifact
  stack-runtime/                # runtime-freeze.json and measurements
  stack-kv-cache/                # kv-cache-freeze.json and measurements
  lmms/dense/                   # complete native task denominator
  stack-serving-lmms/            # final native-protocol vLLM quality and freeze
  serving-use/                  # inference and matched Dense end-to-end timing
```

Actual paths come from JSON. Keep run `config.json`, `state.json`, `summary.json`,
candidate `probe.json`/`result.json`, raw answers/timings, full native task result
and sample files, and selected artifacts/freezes. Runtime/KV reference the Parent;
they do not need another weight copy. Saved absolute references can break on moves.

## Cleanup

Preserve Dense baselines, rejected-candidate evidence, reusable calibration and
every source referenced by retained runs. A final artifact is not a cache.
Keep failure logs needed for diagnosis. Remove only completed, unreferenced
temporary `.work/`, `.selection/`, `.vllm-work/`, `retry-artifact/` and stale locks
after inspecting processes and saving required evidence.

Do not edit shared/hardlinked weight files in place or make them writable.
Write new exports to new directories. A byte-verified deduplication must preserve
paths and receipt applicability; filenames or declared precision are insufficient.

`frozen_for_lmms`, `performance_ready` and `completed` do not all mean quality
acceptance. Inspect the [stage contracts](../docs/stack.md); only matching complete
final serving evidence can certify all layers. Keep quality and timing scope
explicit and do not relabel missing results as zero or a pass.
