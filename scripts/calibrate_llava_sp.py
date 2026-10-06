"""Collect native LLaVA FFN moments from held-out calibration inputs."""

from __future__ import annotations

import json
import argparse
import struct
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search.model import PROJECT_ROOT, load_model, prepare_inputs
from search.evaluate import load_samples
from search.context import require_stages, validate_result_scope
from skills.structured_pruning import contribution_scores, decode_layer, load_calibration

MAGIC_MOMENTS = b"A4FVLMSPMOM1\0"
MAGIC_SCORES = b"A4FVLMSPSCO1\0"
STRATA = ("visual_prefill", "text_prefill", "decode")


def _write_moments(path, layer, rows):
    width = len(rows[STRATA[0]]["squares"])
    payload = bytearray(MAGIC_MOMENTS)
    payload.extend(struct.pack("<II", layer, width))
    for name in STRATA:
        row = rows[name]
        payload.extend(struct.pack("<Q", row["count"]))
        payload.extend(struct.pack(f"<{width}f", *row["sums"]))
        payload.extend(struct.pack(f"<{width}f", *row["squares"]))
    Path(path).write_bytes(payload)


def _write_scores(path, layer, scores):
    payload = bytearray(MAGIC_SCORES)
    payload.extend(struct.pack("<II", layer, len(scores)))
    payload.extend(struct.pack(f"<{len(scores)}f", *scores))
    Path(path).write_bytes(payload)


def prefill_chunks(value, image_mask):
    """Partition the observed sequence using native image placeholders."""
    if value.ndim != 2 or image_mask.ndim != 1 or value.shape[0] != image_mask.numel():
        raise ValueError("Native image mask and MLP prefill length disagree")
    return (("visual_prefill", value[image_mask]), ("text_prefill", value[~image_mask]))


def validate_heldout(samples, evaluation_samples):
    sample_ids = {row["stable_row_id"] for row in samples}
    evaluation_ids = {row["stable_row_id"] for row in evaluation_samples}
    images = {row["media"]["logical_path"] for row in samples}
    evaluation_images = {row["media"]["logical_path"] for row in evaluation_samples}
    if sample_ids & evaluation_ids or images & evaluation_images:
        raise ValueError("SP calibration overlaps held-out evaluation IDs or images")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model-llava-onevision.json")
    parser.add_argument("--evaluation-config", default="configs/evaluation-llava-onevision.json")
    parser.add_argument("--config", default="configs/structured_pruning-llava-onevision.json")
    parser.add_argument("--data", help="Override the SP JSON's calibration data")
    args = parser.parse_args()
    import torch

    model_config = json.loads((PROJECT_ROOT / args.model_config).read_text())
    evaluation = json.loads((PROJECT_ROOT / args.evaluation_config).read_text())
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    data = args.data or config["calibration"].get("data", "data/evaluation/calibration-heldout-128.jsonl")
    require_stages(model_config, ("dense", "sp"))
    root = validate_result_scope(model_config, evaluation)
    calibration = (PROJECT_ROOT / config["calibration"]["source"]).resolve()
    if not calibration.is_relative_to(root) or calibration == root:
        raise ValueError("Calibration must use this model's dedicated result scope")
    if model_config.get("adapter") not in {"llava15", "llava_onevision"}:
        raise ValueError("This collector requires the implemented native LLaVA path")
    if calibration.exists():
        raise ValueError(f"Calibration already exists; inspect and reuse it: {calibration}")
    samples = load_samples(
        PROJECT_ROOT / data, media_root=PROJECT_ROOT / evaluation["data"]["media_root"],
    )
    evaluation_samples = [row for stage in ("search", "dev") for row in load_samples(
        PROJECT_ROOT / evaluation["data"][stage],
        media_root=PROJECT_ROOT / evaluation["data"]["media_root"])]
    validate_heldout(samples, evaluation_samples)
    model, processor = load_model(model_config)
    layers = model.model.language_model.layers
    dimensions = {
        "num_hidden_layers": len(layers),
        "hidden_size": int(model.config.text_config.hidden_size),
        "intermediate_size": int(model.config.text_config.intermediate_size),
    }
    states = []
    markers = []
    handles = []
    active = {"image_mask": None}
    for layer in layers:
        width = int(layer.mlp.down_proj.in_features)
        states.append({name: {"count": 0,
                             "sums": torch.zeros(width, device=model.device, dtype=torch.float32),
                             "squares": torch.zeros(width, device=model.device, dtype=torch.float32)}
                       for name in STRATA})
        call = {"prefill": True}
        markers.append(call)

        def capture(module, args, state=states[-1], call=call):
            value = args[0].detach().float()
            if value.ndim != 3 or value.shape[0] != 1:
                raise ValueError("Unexpected LLaVA MLP activation rank")
            value = value.reshape(-1, value.shape[-1])
            if call["prefill"]:
                call["prefill"] = False
                chunks = prefill_chunks(value, active["image_mask"])
            else:
                chunks = (("decode", value),)
            for name, chunk in chunks:
                if chunk.numel() == 0:
                    continue
                state[name]["count"] += int(chunk.shape[0])
                state[name]["sums"].add_(chunk.sum(dim=0))
                state[name]["squares"].add_((chunk * chunk).sum(dim=0))

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            for index, sample in enumerate(samples, 1):
                # Reset each layer's marker while preserving accumulated moments.
                for marker in markers:
                    marker["prefill"] = True
                inputs = prepare_inputs(processor, sample, evaluation["data"]["media_root"])
                inputs = inputs.to(model.device)
                active["image_mask"] = (inputs["input_ids"][0] == model.config.image_token_index)
                if not active["image_mask"].any() or active["image_mask"].all():
                    raise ValueError("Calibration requires both image and text tokens")
                generation = dict(evaluation["generation"])
                ignore_eos = generation.pop("ignore_eos", False)
                if ignore_eos:
                    generation.update(eos_token_id=None, forced_eos_token_id=None)
                model.generate(
                    **inputs,
                    **generation,
                    return_dict_in_generate=False,
                )
                if index == 1 or index % 16 == 0 or index == len(samples):
                    print(f"Calibration {index}/{len(samples)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    calibration.mkdir(parents=True, exist_ok=False)
    (calibration / "moments").mkdir()
    (calibration / "scores").mkdir()
    counts = []
    for layer_index, (layer, state) in enumerate(zip(layers, states, strict=True)):
        for name in STRATA:
            row = state[name]
            if row["count"] < 1:
                raise RuntimeError(f"Missing calibration stratum {name} for layer {layer_index}")
            row["sums"] = row["sums"].cpu().tolist()
            row["squares"] = row["squares"].cpu().tolist()
        moments_path = calibration / "moments" / f"layer-{layer_index:02d}.bin"
        _write_moments(moments_path, layer_index, state)
        norms = layer.mlp.down_proj.weight.detach().float().norm(dim=0).cpu().tolist()
        _, moments = decode_layer(moments_path.read_bytes(), moments=True)
        scores = contribution_scores(moments, norms)
        _write_scores(calibration / "scores" / f"layer-{layer_index:02d}.bin", layer_index, scores)
        counts.append({name: state[name]["count"] for name in STRATA})
    (calibration / "score-manifest.json").write_text(json.dumps({
        "scorer_id": "post_swiglu_output_contribution_rms_v1",
        "model": model_config,
        "dimensions": dimensions,
        "strata": list(STRATA),
        "source": data,
    }, indent=2) + "\n")
    (calibration / "statistics-manifest.json").write_text(json.dumps({
        "calibration_protocol": "llava_native_image_mask_mlp_down_projection_v2",
        "model": model_config,
        "dimensions": dimensions,
        "sample_count": len(samples),
        "strata": list(STRATA),
        "generation": evaluation["generation"],
        "sample_ids": [row["stable_row_id"] for row in samples],
        "evaluation_disjoint": {"sample_ids": True, "image_paths": True},
        "layer_stratum_counts": counts,
        "moments": "raw_sums_and_sum_squares_fp32",
    }, indent=2) + "\n")
    load_calibration(calibration, dimensions)
    print(json.dumps({"status": "completed", "calibration": str(calibration), "dimensions": dimensions}, indent=2))


if __name__ == "__main__":
    main()
