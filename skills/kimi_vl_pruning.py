"""Physical Kimi MoE channel recipes; routers and expert counts stay intact.

One width is shared by all routed experts in a layer for fused-MoE compatibility.
Their retained channel identities are independent. Dense and shared branches
have their own widths. Importing this module requires no model packages.
"""

import heapq
import json
import math
from pathlib import Path

SCORE = "routed_post_swiglu_output_contribution_rms_v1"
METHOD = "moe_layer_normalized_marginal_cost_v1"
SCHEMA = "a4fvlm.kimi-vl-sp.v1"


def dimensions(config):
    def get(name, default=None):
        return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)

    result = {name: get(name, default) for name, default in (
        ("num_hidden_layers", 27), ("hidden_size", None), ("intermediate_size", None),
        ("moe_intermediate_size", None), ("n_routed_experts", None),
        ("n_shared_experts", None), ("num_experts_per_tok", None),
        ("first_k_dense_replace", 1), ("moe_layer_freq", 1))}
    for key, value in result.items():
        if type(value) is not int or value < (0 if key == "first_k_dense_replace" else 1):
            raise ValueError(f"Invalid Kimi SP dimension: {key}")
    if (result["first_k_dense_replace"] >= result["num_hidden_layers"]
            or result["num_experts_per_tok"] > result["n_routed_experts"]):
        raise ValueError("Invalid Kimi expert topology")
    return {"model_type": "kimi_vl", **result}


def units(dims):
    """Stable flat width domain while retaining actual decoder-layer identity."""
    result = []
    for layer in range(dims["num_hidden_layers"]):
        if layer < dims["first_k_dense_replace"] or layer % dims["moe_layer_freq"]:
            result.append({"layer": layer, "kind": "dense", "experts": 1,
                           "original_width": dims["intermediate_size"]})
        else:
            result.extend((
                {"layer": layer, "kind": "routed", "experts": dims["n_routed_experts"],
                 "original_width": dims["moe_intermediate_size"]},
                {"layer": layer, "kind": "shared", "experts": 1,
                 "original_width": dims["moe_intermediate_size"] * dims["n_shared_experts"]}))
    return result


def validate_config(config, dims):
    selection = config["selection"]
    if selection["score"] != SCORE or selection["score_dtype"] != "float32":
        raise ValueError("Kimi requires routing-weighted FP32 contribution scores")
    group = selection["channel_group_size"]
    if type(group) is not int or group < 1 or any(u["original_width"] % group for u in units(dims)):
        raise ValueError("Channel groups must divide every Dense/shared/routed original width")
    protected = selection["protected_layers"]
    if (not isinstance(protected, list) or len(set(protected)) != len(protected)
            or any(type(x) is not int or not 0 <= x < dims["num_hidden_layers"] for x in protected)):
        raise ValueError("Invalid protected decoder layers")
    probe = config["hardware_probe"]
    ratio = probe["minimum_width_ratio"]
    if type(ratio) not in (int, float) or not 0 < ratio <= 1:
        raise ValueError("minimum_width_ratio must be in (0, 1]")
    for key in ("warmup_iterations", "measurement_iterations"):
        if type(probe[key]) is not int or probe[key] < 1:
            raise ValueError("Probe counts must be positive integers")
    originals = {u["kind"]: u["original_width"] for u in units(dims)}
    excluded = probe.get("excluded_widths", {})
    if not isinstance(excluded, dict) or set(excluded) - set(originals):
        raise ValueError("Invalid excluded kernel-width family")
    for kind, widths in excluded.items():
        if (not isinstance(widths, list) or len(set(widths)) != len(widths)
                or any(type(w) is not int or w <= 0 or w >= originals[kind] or w % group for w in widths)):
            raise ValueError("Excluded widths must be aligned and may not remove the Dense control")
    if excluded and not probe.get("exclusion_evidence"):
        raise ValueError("Width exclusions must name their hardware failure evidence")
    if config["allocation"]["method"] != METHOD:
        raise ValueError("Kimi requires its MoE allocation method")


def _validate_scores(scores, dims):
    domain = units(dims)
    if len(scores) != len(domain):
        raise ValueError("Calibration does not cover all FFN branches")
    for unit, expert_rows in zip(domain, scores, strict=True):
        if len(expert_rows) != unit["experts"]:
            raise ValueError("Calibration does not cover all experts")
        for row in expert_rows:
            if len(row) != unit["original_width"] or any(not math.isfinite(x) or x < 0 for x in row):
                raise ValueError("Invalid Kimi channel score domain")


def load_calibration(directory, dims):
    from skills.structured_pruning import STRATA, decode_layer
    directory = Path(directory)
    manifest = json.loads((directory / "score-manifest.json").read_text())
    domain = units(dims)
    if (manifest.get("scorer_id") != SCORE or manifest.get("dimensions") != dims
            or manifest.get("units") != domain or manifest.get("strata") != list(STRATA)):
        raise ValueError("Kimi calibration identity differs from the requested topology")
    statistics = json.loads((directory / "statistics-manifest.json").read_text())
    if (statistics.get("model") != manifest.get("model") or statistics.get("dimensions") != dims
            or statistics.get("denominator") != "all_tokens_per_stratum_including_unrouted_tokens"):
        raise ValueError("Calibration moments are not bound to the model and routing protocol")
    counts = statistics["routing_counts"]
    denominators = statistics["stratum_denominators"]
    if len(counts) != len(domain) or len(denominators) != len(domain):
        raise ValueError("Missing routing-coverage domain")
    result = []
    for index, unit in enumerate(domain):
        if len(counts[index]) != unit["experts"]:
            raise ValueError("Missing expert coverage")
        rows = []
        for expert in range(unit["experts"]):
            # Routing observes all tokens. Zero hits in one stratum are an
            # actual zero contribution there, not a missing denominator. A
            # completely unobserved expert must not become a zero-cost target.
            coverage = counts[index][expert]
            if (any(type(coverage.get(s)) is not int or coverage[s] < 0 for s in STRATA)
                    or sum(coverage[s] for s in STRATA) == 0):
                raise ValueError(f"Insufficient calibration coverage for unit {index}, expert {expert}")
            name = f"unit-{index:02d}-expert-{expert:02d}.bin"
            score_id, values = decode_layer((directory / "scores" / name).read_bytes())
            moment_id, moments = decode_layer((directory / "moments" / name).read_bytes(), moments=True)
            if score_id != index or moment_id != index or len(moments[0]["sum_squares"]) != len(values):
                raise ValueError("Calibration score/moment identity mismatch")
            # Moments use all tokens in the stratum, including unrouted tokens.
            for moment in moments:
                name = moment["stratum"]
                hits = counts[index][expert][name]
                if moment["count"] != denominators[index][name] or hits > moment["count"]:
                    raise ValueError("Expert routing count exceeds the stratum denominator")
                if unit["kind"] != "routed" and hits != moment["count"]:
                    raise ValueError("Dense/shared FFNs must observe all stratum tokens")
                if hits == 0 and (any(moment["sums"]) or any(moment["sum_squares"])):
                    raise ValueError("Unrouted expert strata must have exactly zero contribution")
            rows.append(values)
        result.append(rows)
    _validate_scores(result, dims)
    return result


def make_recipe(scores, widths, dims, selection):
    _validate_scores(scores, dims)
    domain = units(dims)
    if len(widths) != len(domain):
        raise ValueError("Recipe must cover all FFN branches")
    channels = []
    for unit, rows, width in zip(domain, scores, widths, strict=True):
        original = unit["original_width"]
        if (type(width) is not int or not 0 < width <= original
                or width % selection["channel_group_size"]
                or unit["layer"] in selection["protected_layers"] and width != original):
            raise ValueError("Invalid or protected Kimi FFN width")
        channels.append([sorted(sorted(range(original), key=lambda j: (row[j], j))[original - width:])
                         for row in rows])
    factor = 3 * dims["hidden_size"]
    original_parameters = sum(u["original_width"] * u["experts"] for u in domain) * factor
    removed = sum((u["original_width"] - w) * u["experts"] for u, w in zip(domain, widths)) * factor
    return {"schema": SCHEMA, "dimensions": dict(dims), "units": domain,
            "channel_group_size": selection["channel_group_size"],
            "widths": list(widths), "retained_channels": channels,
            "original_ffn_parameters": original_parameters, "removed_ffn_parameters": removed,
            "depth_denominator": "all_decoder_dense_routed_shared_ffn_parameters",
            "actual_depth": removed / original_parameters}


def validate_recipe(recipe, dims):
    domain = units(dims)
    group = recipe.get("channel_group_size")
    if (recipe.get("schema") != SCHEMA or recipe.get("dimensions") != dims
            or recipe.get("units") != domain or type(group) is not int or group < 1
            or len(recipe["widths"]) != len(domain) or len(recipe["retained_channels"]) != len(domain)):
        raise ValueError("Kimi recipe topology differs from the Dense parent")
    for unit, width, experts in zip(domain, recipe["widths"], recipe["retained_channels"], strict=True):
        if (type(width) is not int or not 0 < width <= unit["original_width"] or width % group
                or unit["original_width"] % group
                or len(experts) != unit["experts"]):
            raise ValueError("Invalid common expert width or expert count")
        for channels in experts:
            if (len(channels) != width or channels != sorted(set(channels))
                    or any(type(x) is not int or not 0 <= x < unit["original_width"] for x in channels)):
                raise ValueError("Invalid original expert channel identities")
    original = sum(u["original_width"] * u["experts"] for u in domain) * 3 * dims["hidden_size"]
    remaining = sum(w * u["experts"] for w, u in zip(recipe["widths"], domain)) * 3 * dims["hidden_size"]
    if (recipe.get("original_ffn_parameters") != original
            or recipe.get("removed_ffn_parameters") != original - remaining
            or recipe.get("actual_depth") != (original - remaining) / original):
        raise ValueError("Recipe FFN parameter accounting differs from physical widths")


def allocation_path(scores, dims, selection, admitted_widths, *, starting_widths=None):
    """Reuse the existing finite quality ladder with a flat branch-width vector."""
    _validate_scores(scores, dims)
    domain = units(dims)
    if not isinstance(admitted_widths, dict) or set(admitted_widths) != {u["kind"] for u in domain}:
        raise ValueError("MoE hardware admission must distinguish dense, routed and shared kernels")
    admitted = [sorted(set(admitted_widths[u["kind"]]), reverse=True) for u in domain]
    for unit, choices in zip(domain, admitted):
        if (not choices or choices[0] != unit["original_width"]
                or any(type(w) is not int or w < 1 or w % selection["channel_group_size"] for w in choices)):
            raise ValueError("Admitted widths must contain the original and aligned branch widths")
    current = list(starting_widths) if starting_widths is not None else [u["original_width"] for u in domain]
    if len(current) != len(domain) or any(w not in a for w, a in zip(current, admitted)):
        raise ValueError("Starting branch widths have not passed hardware admission")
    for unit, width in zip(domain, current):
        if unit["layer"] in selection["protected_layers"] and width != unit["original_width"]:
            raise ValueError("A protected layer was already pruned")
    ordered = [[sorted(row) for row in rows] for rows in scores]
    # Normalize both shared and routed contributions to their actual decoder layer.
    totals = [0.] * dims["num_hidden_layers"]
    for unit, rows in zip(domain, scores):
        totals[unit["layer"]] += math.fsum(math.fsum(row) for row in rows)
    denominator = sum(u["original_width"] * u["experts"] for u in domain)

    def node(move):
        return {"widths": current.copy(), "actual_depth":
                (denominator - sum(w * u["experts"] for w, u in zip(current, domain))) / denominator,
                "move": move}

    heap = []
    path = [node(None)]

    def push(index):
        unit, choices = domain[index], admitted[index]
        position = choices.index(current[index])
        if position + 1 == len(choices) or unit["layer"] in selection["protected_layers"]:
            return
        before, after = choices[position:position + 2]
        contribution = math.fsum(math.fsum(row[unit["original_width"] - before:unit["original_width"] - after])
                                 for row in ordered[index])
        parameters = (before - after) * unit["experts"] * 3 * dims["hidden_size"]
        total = totals[unit["layer"]]
        heapq.heappush(heap, (contribution / total / parameters if total else 0., index, after))

    for index in range(len(domain)):
        push(index)
    while heap:
        cost, index, after = heapq.heappop(heap)
        before, current[index] = current[index], after
        path.append(node({"unit": index, "layer": domain[index]["layer"], "kind": domain[index]["kind"],
                          "from_width": before, "to_width": after, "normalized_cost": cost}))
        push(index)
    return path


def modules(model, dims):
    layers = model.language_model.model.layers
    if len(layers) != dims["num_hidden_layers"]:
        raise ValueError("Actual decoder layer count differs from the recipe")
    result = []
    for unit in units(dims):
        mlp = layers[unit["layer"]].mlp
        if unit["kind"] == "routed":
            if (len(mlp.experts) != unit["experts"] or mlp.ep_size != 1
                    or mlp.num_experts_per_tok != dims["num_experts_per_tok"]
                    or tuple(mlp.gate.weight.shape) != (unit["experts"], dims["hidden_size"])):
                raise ValueError("Kimi router/expert topology differs from the original")
            result.append(list(mlp.experts))
        else:
            result.append([mlp if unit["kind"] == "dense" else mlp.shared_experts])
    return result


def _shapes(mlp):
    return {name: list(getattr(mlp, name).weight.shape) for name in ("gate_proj", "up_proj", "down_proj")}


def _expected(width, hidden):
    return {"gate_proj": [width, hidden], "up_proj": [width, hidden], "down_proj": [hidden, width]}


def apply_recipe(model, recipe):
    from skills.structured_pruning import slice_mlp
    dims = dimensions(model.config.text_config)
    validate_recipe(recipe, dims)
    if hasattr(model, "structured_pruning_recipe"):
        raise ValueError("Kimi recipe requires a fresh Dense parent")
    branches = modules(model, dims)
    # Validate the entire domain before the first mutation.
    for unit, experts in zip(units(dims), branches):
        if any(_shapes(expert) != _expected(unit["original_width"], dims["hidden_size"]) for expert in experts):
            raise ValueError("Kimi FFN is not its original Dense shape")
    for experts, retained in zip(branches, recipe["retained_channels"], strict=True):
        for expert, channels in zip(experts, retained, strict=True):
            slice_mlp(expert, channels)
    model.structured_pruning_recipe = recipe
    return model


def audit_recipe(model, recipe):
    dims = dimensions(model.config.text_config)
    validate_recipe(recipe, dims)
    branches = modules(model, dims)
    records = []
    for unit, width, experts in zip(units(dims), recipe["widths"], branches, strict=True):
        shapes = [_shapes(expert) for expert in experts]
        if any(shape != _expected(width, dims["hidden_size"]) for shape in shapes):
            raise ValueError("Observed expert shapes differ from the physical recipe")
        records.append({**unit, "retained_width": width, "expert_shapes": shapes})
    return {"backend": "transformers", "weight_shapes": records,
            "remaining_ffn_parameters": recipe["original_ffn_parameters"] - recipe["removed_ffn_parameters"],
            "padding_count": 0, "fallback_count": 0, "router_and_expert_counts_preserved": True}
