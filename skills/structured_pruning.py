"""Replay frozen channel scores and physically slice Decoder SwiGLU layers.

Imports remain lightweight until a caller applies a recipe to a loaded model.
Channel identities always refer to the original Dense model.
"""

import json
import heapq
import math
from pathlib import Path
import struct
from typing import Any


SCORE = "post_swiglu_output_contribution_rms_v1"
STRATA = ("visual_prefill", "text_prefill", "decode")


def validate_config(config, dimensions):
    if dimensions.get("model_type") == "deepseek_vl_v2":
        from skills.deepseek_vl2_pruning import validate_config as validate
        return validate(config, dimensions)
    selection = config["selection"]
    if selection["score"] != SCORE or selection["score_dtype"] != "float32":
        raise ValueError(
            "Only frozen FP32 post-SwiGLU contribution scores are supported"
        )
    group = selection["channel_group_size"]
    width = dimensions["intermediate_size"]
    if type(group) is not int or group < 1 or width % group:
        raise ValueError("channel_group_size must divide the original width")
    protected = selection["protected_layers"]
    if (
        not isinstance(protected, list)
        or any(
            type(layer) is not int or not 0 <= layer < dimensions["num_hidden_layers"]
            for layer in protected
        )
        or len(set(protected)) != len(protected)
    ):
        raise ValueError("Invalid protected_layers")
    probe = config["hardware_probe"]
    ratio = probe["minimum_width_ratio"]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not 0 < ratio <= 1
    ):
        raise ValueError("minimum_width_ratio must be in (0, 1]")
    for key in ("warmup_iterations", "measurement_iterations"):
        if type(probe[key]) is not int or probe[key] < 1:
            raise ValueError(f"{key} must be positive")


def decode_layer(payload, *, moments=False):
    """Read the old little-endian format without its registry dependencies."""
    magic = b"A4FVLMSPMOM1\0" if moments else b"A4FVLMSPSCO1\0"
    offset = len(magic) + 8
    if not payload.startswith(magic) or len(payload) < offset:
        raise ValueError("Invalid calibration header")
    layer, width = struct.unpack_from("<II", payload, len(magic))
    expected = offset + (3 * (8 + 8 * width) if moments else 4 * width)
    if not width or len(payload) != expected:
        raise ValueError("Invalid calibration length")
    if not moments:
        scores = list(struct.unpack_from(f"<{width}f", payload, offset))
        if any(not math.isfinite(value) or value < 0 for value in scores):
            raise ValueError("Invalid channel score")
        return layer, scores
    strata = []
    for name in STRATA:
        count = struct.unpack_from("<Q", payload, offset)[0]
        offset += 8
        sums = struct.unpack_from(f"<{width}f", payload, offset)
        offset += 4 * width
        squares = struct.unpack_from(f"<{width}f", payload, offset)
        offset += 4 * width
        if (
            not count
            or any(not math.isfinite(x) for x in sums)
            or any(not math.isfinite(x) or x < 0 for x in squares)
        ):
            raise ValueError("Invalid activation moments")
        strata.append(
            {"stratum": name, "count": count, "sums": sums, "sum_squares": squares}
        )
    return layer, strata


def contribution_scores(moments, down_column_norms):
    """Equal-stratum RMS times the Dense down-projection column L2 norm."""
    if [row["stratum"] for row in moments] != list(STRATA):
        raise ValueError("Expected all three ordered calibration strata")
    width = len(down_column_norms)
    if any(row["count"] <= 0 or len(row["sum_squares"]) != width for row in moments):
        raise ValueError("Invalid calibration dimensions or count")
    scores = []
    for channel, norm in enumerate(down_column_norms):
        second = math.fsum(
            row["sum_squares"][channel] / row["count"] / 3 for row in moments
        )
        if (
            not math.isfinite(second)
            or second < 0
            or not math.isfinite(norm)
            or norm < 0
        ):
            raise ValueError("Invalid contribution inputs")
        value = struct.unpack("<f", struct.pack("<f", math.sqrt(second) * norm))[0]
        if not math.isfinite(value):
            raise ValueError("Contribution overflows FP32")
        scores.append(value)
    return scores


def load_calibration(directory, dimensions):
    """Validate the complete frozen layer domain, retaining its original protocol."""
    if dimensions.get("model_type") == "deepseek_vl_v2":
        from skills.deepseek_vl2_pruning import load_calibration as load
        return load(directory, dimensions)
    directory = Path(directory)
    manifest = json.loads((directory / "score-manifest.json").read_text())
    if manifest["scorer_id"] != SCORE:
        raise ValueError("Incompatible calibration scorer")
    scores = []
    for layer in range(dimensions["num_hidden_layers"]):
        filename = f"layer-{layer:02d}.bin"
        index, row = decode_layer((directory / "scores" / filename).read_bytes())
        moment_index, moments = decode_layer(
            (directory / "moments" / filename).read_bytes(), moments=True
        )
        if (
            index != layer
            or moment_index != layer
            or len(row) != dimensions["intermediate_size"]
        ):
            raise ValueError("Calibration layer domain differs from the model")
        if len(moments[0]["sum_squares"]) != len(row):
            raise ValueError("Calibration moments and scores disagree")
        scores.append(row)
    return scores


def make_recipe(scores, widths, dimensions, selection) -> dict[str, Any]:
    """Drop the lowest scores; ties use the original channel index."""
    if dimensions.get("model_type") == "deepseek_vl_v2":
        from skills.deepseek_vl2_pruning import make_recipe as make
        return make(scores, widths, dimensions, selection)
    original = dimensions["intermediate_size"]
    count = dimensions["num_hidden_layers"]
    if len(scores) != count or len(widths) != count:
        raise ValueError("Recipe must cover every Decoder layer")
    retained = []
    for layer, (row, width) in enumerate(zip(scores, widths, strict=True)):
        if len(row) != original or any(not math.isfinite(x) or x < 0 for x in row):
            raise ValueError("Invalid score domain")
        if (
            type(width) is not int
            or not 0 < width <= original
            or width % selection["channel_group_size"]
        ):
            raise ValueError("Width must use complete channel groups")
        if layer in selection["protected_layers"] and width != original:
            raise ValueError("Cannot prune a protected layer")
        ranked = sorted(range(original), key=lambda index: (row[index], index))
        retained.append(sorted(ranked[original - width :]))
    denominator = count * original * 3 * dimensions["hidden_size"]
    removed = sum(original - width for width in widths) * 3 * dimensions["hidden_size"]
    return {
        "dimensions": dict(dimensions),
        "widths": list(widths),
        "retained_channels": retained,
        "removed_ffn_parameters": removed,
        "original_ffn_parameters": denominator,
        "depth_denominator": "all_decoder_ffn_layers",
        "actual_depth": removed / denominator,
    }


def validate_recipe(recipe, dimensions):
    if dimensions.get("model_type") == "deepseek_vl_v2":
        from skills.deepseek_vl2_pruning import validate_recipe as validate
        return validate(recipe, dimensions)
    if recipe["dimensions"] != dimensions:
        raise ValueError("Recipe dimensions differ from the Dense model")
    count = dimensions["num_hidden_layers"]
    if len(recipe["widths"]) != count or len(recipe["retained_channels"]) != count:
        raise ValueError("Incomplete recipe layer domain")
    for width, channels in zip(
        recipe["widths"], recipe["retained_channels"], strict=True
    ):
        if type(width) is not int or not 0 < width <= dimensions["intermediate_size"]:
            raise ValueError("Invalid recipe width")
        if (
            len(channels) != width
            or any(
                type(index) is not int
                or not 0 <= index < dimensions["intermediate_size"]
                for index in channels
            )
            or channels != sorted(set(channels))
        ):
            raise ValueError("Invalid original channel identities")


def slice_mlp(mlp, channels):
    """Replace gate/up rows and matching down columns, preserving dtype/device."""
    import torch

    width = mlp.gate_proj.out_features
    if (
        not channels
        or channels != sorted(set(channels))
        or any(type(index) is not int or not 0 <= index < width for index in channels)
    ):
        raise ValueError("Invalid channels for Dense MLP")
    hidden = mlp.gate_proj.in_features
    if tuple(mlp.up_proj.weight.shape) != (width, hidden) or tuple(
        mlp.down_proj.weight.shape
    ) != (hidden, width):
        raise ValueError("Inconsistent SwiGLU projection dimensions")
    if len(channels) == width:
        return
    index = torch.tensor(channels, device=mlp.gate_proj.weight.device, dtype=torch.long)
    for name, axis in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
        source = getattr(mlp, name)
        weight = source.weight.detach().index_select(axis, index).contiguous()
        # Meta construction avoids allocating and initializing a second weight copy.
        replacement = torch.nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=source.bias is not None,
            device="meta",
        )
        replacement.weight = torch.nn.Parameter(
            weight, requires_grad=source.weight.requires_grad
        )
        if source.bias is not None:
            bias = (
                source.bias.detach().index_select(0, index)
                if axis == 0
                else source.bias.detach().clone()
            )
            replacement.bias = torch.nn.Parameter(
                bias, requires_grad=source.bias.requires_grad
            )
        replacement.train(source.training)
        setattr(mlp, name, replacement)
    mlp.intermediate_size = len(channels)


def decoder_topology(model):
    """Return native text config/layers without mutating or aliasing the model."""
    if getattr(model.config, "model_type", None) == "internvl_chat":
        return model.config.llm_config, model.language_model.model.layers
    if getattr(model.config, "model_type", None) == "deepseek_vl_v2":
        return model.config.language_config, model.language.model.layers
    return model.config.text_config, model.model.language_model.layers


def apply_recipe(model, recipe):
    """Apply once to a freshly loaded Dense model; never reintroduce channels."""
    if getattr(model.config, "model_type", None) == "deepseek_vl_v2":
        from skills.deepseek_vl2_pruning import apply_recipe as apply
        return apply(model, recipe)
    config, layers = decoder_topology(model)
    dimensions = {
        key: getattr(config, key)
        for key in ("num_hidden_layers", "hidden_size", "intermediate_size")
    }
    validate_recipe(recipe, dimensions)
    if (
        hasattr(model, "structured_pruning_recipe")
        or len(layers) != dimensions["num_hidden_layers"]
    ):
        raise ValueError("Recipe requires a fresh Dense model")
    # Check the entire model before changing any layer.
    for layer in layers:
        width, hidden = dimensions["intermediate_size"], dimensions["hidden_size"]
        for name, shape in (
            ("gate_proj", (width, hidden)),
            ("up_proj", (width, hidden)),
            ("down_proj", (hidden, width)),
        ):
            if tuple(getattr(layer.mlp, name).weight.shape) != shape:
                raise ValueError("Model is not the declared Dense parent")
    for layer, channels in zip(layers, recipe["retained_channels"], strict=True):
        slice_mlp(layer.mlp, channels)
    model.structured_pruning_recipe = recipe
    return model


def audit_recipe(model, recipe):
    """Record actual compact shapes and parameter counts before evaluation."""
    if getattr(model.config, "model_type", None) == "deepseek_vl_v2":
        from skills.deepseek_vl2_pruning import audit_recipe as audit
        return audit(model, recipe)
    shapes = []
    hidden = recipe["dimensions"]["hidden_size"]
    for index, (layer, width) in enumerate(
        zip(decoder_topology(model)[1], recipe["widths"], strict=True)
    ):
        actual = {
            name: list(getattr(layer.mlp, name).weight.shape)
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        if actual != {
            "gate_proj": [width, hidden],
            "up_proj": [width, hidden],
            "down_proj": [hidden, width],
        }:
            raise ValueError("Runtime FFN shape differs from recipe")
        shapes.append({"layer": index, **actual})
    return {
        "backend": "transformers",
        "weight_shapes": shapes,
        "remaining_ffn_parameters": sum(recipe["widths"]) * 3 * hidden,
        "padding_count": 0,
        "fallback_count": 0,
    }


def validate_search(config):
    method = config["allocation"]["method"]
    if method == "moe_layer_normalized_marginal_cost_v1":
        from skills.deepseek_vl2_pruning import SCORE as moe_score
        if config["selection"]["score"] != moe_score:
            raise ValueError("MoE allocation requires routing-weighted calibration")
    elif method != "layer_normalized_marginal_cost_v1":
        raise ValueError("Unknown allocation method")
    policy = config["search"]
    budget = policy.get("maximum_quality_candidates")
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
    ):
        raise ValueError("maximum_quality_candidates must be a positive integer")
    if "failure_policy" in policy:
        if policy["failure_policy"] == "stop_on_first_dev_failure":
            # Read legacy snapshots, but keep one canonical policy name in all
            # newly written config and state records.
            policy["failure_policy"] = "step_back_on_dev_failure"
        if policy["failure_policy"] not in {
            "step_back_on_dev_failure",
        }:
            raise ValueError("Unknown failure policy")
        step = policy["initial_depth_step"]
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(step)
            or not 0 < step <= 1
        ):
            raise ValueError("Require 0 < initial step <= 1")
        recovery_step = policy.get("failure_recovery_steps", 1)
        if (
            isinstance(recovery_step, bool)
            or not isinstance(recovery_step, int)
            or recovery_step < 1
        ):
            raise ValueError("failure_recovery_steps must be a positive integer")
        return
    for key in (
        "initial_depth_step",
        "maximum_depth_step",
        "failure_step_multiplier",
        "success_step_multiplier",
    ):
        value = policy[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid {key}")
    if not 0 < policy["initial_depth_step"] <= policy["maximum_depth_step"] <= 1:
        raise ValueError("Require 0 < initial step <= maximum step <= 1")
    if (
        not 0 < policy["failure_step_multiplier"] < 1
        or policy["success_step_multiplier"] <= 1
    ):
        raise ValueError("Failure must shrink and success must expand the step")
    minimum = policy.get("minimum_depth_step", 0)
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(minimum)
        or not 0 <= minimum <= policy["initial_depth_step"]
    ):
        raise ValueError("Require 0 <= minimum step <= initial step")
    if not isinstance(policy.get("confirm_parent_on_failure", False), bool):
        raise ValueError("confirm_parent_on_failure must be boolean")
    if policy["confirmed_failures_to_stop"] != 2:
        raise ValueError("This policy requires two distinct Dev-confirmed failures")


def allocation_path(
    scores, dimensions, selection, admitted_widths, *, starting_widths=None
):
    """Greedy per-layer marginal allocation; no full combination enumeration.

    Each layer's Dense score sum stays fixed. An all-zero score vector contributes
    zero cost; ties use layer index, then target width. This is a proxy, not an
    independently measured layer-sensitivity estimate.
    """
    if dimensions.get("model_type") == "deepseek_vl_v2":
        from skills.deepseek_vl2_pruning import allocation_path as allocate
        return allocate(scores, dimensions, selection, admitted_widths, starting_widths=starting_widths)
    original = dimensions["intermediate_size"]
    layers = dimensions["num_hidden_layers"]
    widths = sorted(set(admitted_widths), reverse=True)
    group = selection["channel_group_size"]
    if (
        not widths
        or widths[0] != original
        or any(type(w) is not int or w < 1 or w % group for w in widths)
    ):
        raise ValueError(
            "Admitted widths must include Dense and complete channel groups"
        )
    if len(scores) != layers or any(
        len(row) != original or any(not math.isfinite(x) or x < 0 for x in row)
        for row in scores
    ):
        raise ValueError("Invalid full calibration score domain")
    ordered = [sorted(row) for row in scores]
    totals = [math.fsum(row) for row in scores]
    current = (
        list(starting_widths) if starting_widths is not None else [original] * layers
    )
    if len(current) != layers or any(w not in widths for w in current):
        raise ValueError("Starting widths must be admitted for every layer")
    path = [
        {
            "widths": current.copy(),
            "actual_depth": (layers * original - sum(current)) / (layers * original),
            "move": None,
        }
    ]
    heap = []

    def push(layer, position):
        if position + 1 == len(widths):
            return
        before, after = widths[position : position + 2]
        contribution = math.fsum(ordered[layer][original - before : original - after])
        parameters = (before - after) * 3 * dimensions["hidden_size"]
        cost = contribution / totals[layer] / parameters if totals[layer] else 0.0
        heapq.heappush(heap, (cost, layer, after, position))

    for layer in range(layers):
        if layer not in selection["protected_layers"]:
            push(layer, widths.index(current[layer]))
    while heap:
        cost, layer, after, position = heapq.heappop(heap)
        before = current[layer]
        current[layer] = after
        path.append(
            {
                "widths": current.copy(),
                "actual_depth": (layers * original - sum(current))
                / (layers * original),
                "move": {
                    "layer": layer,
                    "from_width": before,
                    "to_width": after,
                    "normalized_cost": cost,
                },
            }
        )
        push(layer, position + 1)
    return path


def _dev_rejections(quality):
    return sorted(
        index
        for index, result in quality.items()
        if result.get("action") == "reject"
        and result.get("stages", {}).get("dev", {}).get("action") == "reject"
    )


def coarse_candidate_ladder(path, policy, *, start_parent=0):
    """Return the deterministic quality ladder for the coarse policy."""
    if policy.get("failure_policy") not in {
        "step_back_on_dev_failure",
        "stop_on_first_dev_failure",
    }:
        return []
    parent = start_parent
    ladder = []
    while parent + 1 < len(path):
        requested = path[parent]["actual_depth"] + policy["initial_depth_step"]
        choices = [
            index
            for index in range(parent + 1, len(path))
            if path[index]["actual_depth"] <= requested + 1e-12
        ]
        index = choices[-1] if choices else parent + 1
        if index <= parent:
            break
        ladder.append(
            {
                "index": index,
                "ladder_level": len(ladder) + 1,
                "parent_candidate": parent,
                "requested_depth": requested,
                "overshoot": not choices,
            }
        )
        parent = index
    return ladder


def _recovery_evaluation(
    path, quality, policy, start_parent, failed_index, *, ladder=None
):
    """Walk one coarse candidate level toward Dense after a Dev failure.

    The normal scheduler evaluates the deepest physical path node at each
    ``initial_depth_step`` milestone.  Recovery moves between those evaluated
    candidate levels, rather than visiting every 128-channel allocation node
    inside the interval.
    """
    ladder = (
        coarse_candidate_ladder(path, policy, start_parent=start_parent)
        if ladder is None
        else ladder
    )
    by_index = {row["index"]: row for row in ladder}
    failed_position = by_index.get(failed_index)
    if failed_position is None:
        return {
            "stop_reason": "recovery_range_exhausted",
            "parent": start_parent,
            "failed_index": failed_index,
            "failed_indices": _dev_rejections(quality),
        }
    step = policy.get("failure_recovery_steps", 1)
    boundary = failed_index
    boundary_position = failed_position["ladder_level"] - 1
    while True:
        target_position = boundary_position - step
        if target_position < 0:
            parent_result = quality.get(start_parent, {})
            parent_dev = parent_result.get("stages", {}).get("dev", {})
            if parent_dev.get("action") == "pass":
                return {
                    "stop_reason": "recovered_boundary",
                    "parent": start_parent,
                    "failed_index": boundary,
                    "recovery_indices": [start_parent],
                }
            return {
                "stop_reason": "recovery_range_exhausted",
                "parent": start_parent,
                "failed_index": boundary,
                "failed_indices": _dev_rejections(quality),
            }
        target = ladder[target_position]
        candidate = target["index"]
        result = quality.get(candidate)
        job = {
            "index": candidate,
            "parent": boundary,
            "parent_candidate": target["parent_candidate"],
            "ladder_level": target["ladder_level"],
            "failed_index": boundary,
            "recovery_steps": step,
            "force_dev": True,
            "reason": "candidate_recovery",
        }
        if result is None:
            return job
        if result.get("action") == "pending" or "dev" not in result.get(
            "stages", {}
        ):
            return job
        dev = result["stages"]["dev"]
        if dev.get("action") == "pass":
            return {
                "stop_reason": "recovered_boundary",
                "parent": candidate,
                "parent_candidate": target["parent_candidate"],
                "ladder_level": target["ladder_level"],
                "failed_index": boundary,
                "recovery_indices": [
                    row["index"]
                    for row in ladder[target_position:boundary_position]
                ],
            }
        if dev.get("action") != "reject":
            raise ValueError("Only valid Dev decisions may recover the path")
        boundary = candidate
        boundary_position = target_position


def first_failure_evaluation(path, quality, policy, start_parent):
    """Use coarse milestones, then recover one candidate level per Dev failure."""
    ladder = coarse_candidate_ladder(path, policy, start_parent=start_parent)
    by_index = {row["index"]: row for row in ladder}
    failures = sorted(
        (index for index in _dev_rejections(quality) if index in by_index),
        key=lambda index: by_index[index]["ladder_level"],
    )
    if failures:
        # Only failures on the current coarse ladder can affect recovery.
        return _recovery_evaluation(
            path,
            quality,
            policy,
            start_parent,
            failures[0],
            ladder=ladder,
        )
    confirmed = start_parent
    for item in ladder:
        index = item["index"]
        job = {
            **item,
            "index": index,
            "parent": item["parent_candidate"],
            "force_dev": False,
            "reason": "main",
            "visited": [row["index"] for row in ladder if row["index"] < index],
        }
        result = quality.get(index)
        if result is None:
            return job
        if result["action"] == "pending":
            return {**job, "force_dev": True, "reason": "complete_confirmation"}
        dev = result["stages"].get("dev")
        if dev is not None and dev["action"] == "reject":
            return _recovery_evaluation(
                path,
                quality,
                policy,
                start_parent,
                index,
                ladder=ladder,
            )
        if result["action"] != "pass":
            raise ValueError("Only valid Dev rejection may start recovery")
        if dev is not None and dev["action"] == "pass":
            confirmed = index
    parent = ladder[-1]["index"] if ladder else start_parent
    return {
        "stop_reason": "declared_range_exhausted",
        "parent": parent,
        "confirmed_parent": confirmed,
        "visited": [row["index"] for row in ladder],
    }


def next_evaluation(path, quality, policy, *, start_parent=0):
    """Replay cached decisions on the frozen path and return exactly one next job.

    Dev overrides Search. Replaying after a parent's later Dev failure naturally
    resumes from a passing ancestor, without deleting any descendant evidence.
    """
    if policy.get("failure_policy") in {
        "step_back_on_dev_failure",
        "stop_on_first_dev_failure",
    }:
        return first_failure_evaluation(path, quality, policy, start_parent)
    parent, step = start_parent, policy["initial_depth_step"]
    failures = set()
    visited = []
    minimum = policy.get("minimum_depth_step", 0)
    early_confirmation = policy.get("confirm_parent_on_failure", False)
    while parent + 1 < len(path):
        requested = path[parent]["actual_depth"] + step
        upper = min((index for index in failures if index > parent), default=None)
        if (
            upper is not None
            and early_confirmation
            and parent
            and "dev" not in quality[parent]["stages"]
        ):
            return {
                "index": parent,
                "parent": parent,
                "force_dev": True,
                "reason": "confirm_parent",
                "visited": visited.copy(),
            }
        target = requested
        if upper is not None:
            target = min(
                target, (path[parent]["actual_depth"] + path[upper]["actual_depth"]) / 2
            )
        choices = [
            index
            for index in range(parent + 1, len(path))
            if path[index]["actual_depth"] <= target + 1e-12
        ]
        if minimum:
            # Evaluation spacing is independent of the 128-channel physical unit.
            spaced = [
                i
                for i in range(parent + 1, len(path))
                if path[i]["actual_depth"] + 1e-12
                >= path[parent]["actual_depth"] + minimum
                and (upper is None or i < upper)
            ]
            if not spaced and upper is not None:
                return {
                    "stop_reason": "resolution_limit_reached",
                    "parent": parent,
                    "failed_indices": sorted(failures),
                    "minimum_depth_step": minimum,
                    "visited": visited,
                }
            if not spaced:
                spaced = [len(path) - 1]  # Evaluate the finite range endpoint once.
            choices = [i for i in spaced if path[i]["actual_depth"] <= target + 1e-12]
            index = choices[-1] if choices else spaced[0]
        else:
            index = choices[-1] if choices else parent + 1
        job = {
            "index": index,
            "parent": parent,
            "requested_depth": requested,
            "target_depth": target,
            "step": step,
            "force_dev": False,
            "reason": "refinement" if upper is not None else "main",
            "overshoot": not choices,
            "visited": visited.copy(),
        }
        result = quality.get(index)
        if result is None:
            return job
        if result["action"] == "pending":
            return {**job, "force_dev": True, "reason": "complete_confirmation"}
        visited.append(index)
        if result["action"] == "pass":
            parent = index
            failures = {i for i in failures if i > parent}
            step = min(
                policy["maximum_depth_step"], step * policy["success_step_multiplier"]
            )
            continue
        if result["action"] != "reject" or "dev" not in result["stages"]:
            raise ValueError("Only valid Dev rejection may refine the path")
        failures.add(index)
        if early_confirmation and parent and "dev" not in quality[parent]["stages"]:
            return {
                **job,
                "index": parent,
                "force_dev": True,
                "reason": "confirm_parent",
            }
        if minimum:
            step = max(minimum, step * policy["failure_step_multiplier"])
            continue
        if index > parent + 1:
            step *= policy["failure_step_multiplier"]
            continue
        if parent and "dev" not in quality[parent]["stages"]:
            return {
                **job,
                "index": parent,
                "force_dev": True,
                "reason": "confirm_parent",
            }
        # Probe the next deeper legal node, reusing its independent decision.
        deeper = index + 1
        if deeper == len(path):
            return {
                "stop_reason": "declared_range_exhausted",
                "parent": parent,
                "visited": visited,
            }
        second = quality.get(deeper)
        if second is None or "dev" not in second["stages"]:
            return {
                **job,
                "index": deeper,
                "force_dev": True,
                "reason": "confirm_next_depth",
            }
        visited.append(deeper)
        if second["action"] == "reject":
            return {
                "stop_reason": "confirmed_path_boundary",
                "parent": parent,
                "failed_indices": [index, deeper],
                "visited": visited,
            }
        if second["action"] != "pass":
            raise ValueError("Unresolved deeper quality decision")
        parent = deeper
        failures = {i for i in failures if i > parent}
        step = min(
            policy["maximum_depth_step"], step * policy["success_step_multiplier"]
        )
    return {
        "stop_reason": "declared_range_exhausted",
        "parent": parent,
        "visited": visited,
    }
