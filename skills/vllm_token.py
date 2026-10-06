"""Runtime configuration and selection helpers for score-based vLLM Token pruning.

The Dense environment must remain importable without vLLM.  This module therefore
contains only a small environment bridge at import time; torch and the shared
vision selector are imported lazily when the vLLM model actually processes an
image.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping


TOKEN_RUNTIME_ENV = "A4FVLMS_VLLM_TOKEN_RUNTIME"


def _validate_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    try:
        ratio = float(runtime["keep_ratio"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("vLLM Token runtime requires keep_ratio") from error
    if not 0 < ratio <= 1:
        raise ValueError("vLLM Token keep_ratio must be in (0, 1]")

    selection = runtime.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("vLLM Token runtime requires a selection mapping")
    selection = dict(selection)
    score_dtype = selection.get("score_dtype")
    if score_dtype not in ("float32", "float16"):
        raise ValueError("vLLM Token score_dtype must be float32 or float16")
    grid = selection.get("coverage_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("vLLM Token runtime requires coverage_grid")
    grid = dict(grid)
    if any(
        type(grid.get(key)) is not int or not 1 <= grid[key] <= 6
        for key in ("rows", "columns")
    ):
        raise ValueError("vLLM Token coverage grid must be 1..6")
    return {
        "keep_ratio": ratio,
        "selection": {
            "score_dtype": score_dtype,
            "coverage_grid": {
                "rows": int(grid["rows"]),
                "columns": int(grid["columns"]),
            },
        },
    }


def configure_runtime(runtime: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Expose one engine's Token settings to vLLM's spawned worker processes."""

    if runtime is None:
        os.environ.pop(TOKEN_RUNTIME_ENV, None)
        return None
    validated = _validate_runtime(runtime)
    os.environ[TOKEN_RUNTIME_ENV] = json.dumps(
        validated, sort_keys=True, separators=(",", ":")
    )
    return validated


def configured_runtime() -> dict[str, Any] | None:
    """Read the per-engine Token settings inherited by vLLM subprocesses."""

    raw = os.environ.get(TOKEN_RUNTIME_ENV)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid vLLM Token runtime JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("Invalid vLLM Token runtime payload")
    return _validate_runtime(value)


def image_keep_count(
    *, rows: int, columns: int, runtime: Mapping[str, Any]
) -> int:
    """Return the deterministic retained count used by prompt replacement."""

    from skills.vision_token import keep_count

    selection = runtime["selection"]
    return keep_count(
        int(rows),
        int(columns),
        float(runtime["keep_ratio"]),
        selection["coverage_grid"],
    )


def select_image_indices(
    features: Any,
    *,
    rows: int,
    columns: int,
    runtime: Mapping[str, Any],
) -> Any:
    """Apply the shared score/coverage selector to post-encoder features."""

    from skills.vision_token import select_indices

    return select_indices(
        features,
        int(rows),
        int(columns),
        float(runtime["keep_ratio"]),
        runtime["selection"],
    )


__all__ = [
    "TOKEN_RUNTIME_ENV",
    "configured_runtime",
    "configure_runtime",
    "image_keep_count",
    "select_image_indices",
]
