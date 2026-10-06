"""Reuse frozen calibration and probe finite Decoder SwiGLU widths on one GPU."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
from typing import Any

from search.model import PROJECT_ROOT
from search.context import dense_root, search_root, require_stages, validate_result_scope
from search.optimize import hardware, launch, lines, lock, read
from search.worker import save_json
from skills.structured_pruning import load_calibration, validate_config


RESULTS = PROJECT_ROOT / "results/model/qwen3vl/search/structured-pruning"
DIMENSION_KEYS = ("num_hidden_layers", "hidden_size", "intermediate_size")


def import_calibration(source, destination, dimensions):
    """Copy only compact statistics, preserving existing bytes and old manifests."""
    source, destination = Path(source), Path(destination)
    load_calibration(source, dimensions)
    files = [
        source / name for name in ("statistics-manifest.json", "score-manifest.json")
    ]
    files += [
        source / kind / f"layer-{layer:02d}.bin"
        for kind in ("moments", "scores")
        for layer in range(dimensions["num_hidden_layers"])
    ]
    for path in files:
        target = destination / path.relative_to(source)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError(f"Refusing to overwrite different calibration: {target}")
    copied = 0
    for path in files:
        target = destination / path.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
    return copied


def probe_widths(original, group, minimum_ratio):
    """The floor bounds this probe, not the quality or structural boundary."""
    minimum = math.ceil(original * minimum_ratio / group) * group
    return list(range(original, minimum - 1, -group))


def prefill_lengths(records, sample_ids):
    """Nearest-rank median/P90/max from the first complete recorded Dense round."""
    tokens = {}
    for row in records:
        if row["round"] != 1:
            continue
        sample = row["sample_id"]
        length = row["performance"]["input_tokens"]
        if sample in tokens or type(length) is not int or length < 1:
            raise ValueError("Invalid Dense input-token records")
        tokens[sample] = length
    if set(tokens) != set(sample_ids) or not tokens:
        raise ValueError("Dense metrics must cover the complete Search set")
    ordered = sorted(tokens.values())
    return sorted(
        {
            ordered[math.ceil(len(ordered) * quantile) - 1]
            for quantile in (0.5, 0.9, 1.0)
        }
    )


def snapshot(config_path, model_path, evaluation_path, device) -> dict[str, Any]:
    config, model, evaluation = (
        read(PROJECT_ROOT / path) for path in (config_path, model_path, evaluation_path)
    )
    require_stages(model, ("sp",), device=device)
    validate_result_scope(model, evaluation)
    text_config = read(PROJECT_ROOT / model["path"] / "config.json")["text_config"]
    if model.get("adapter") == "llava15":
        from adapters.llava15 import Llava15Adapter

        text_config = {**Llava15Adapter.text_defaults, **text_config}
    dimensions = {key: text_config[key] for key in DIMENSION_KEYS}
    # Qwen3-VL uses BF16 while LLaVA-OneVision's Qwen2 decoder is stored in
    # FP16. Both expose the same three-projection SwiGLU contract; keep the
    # dtype and model family in the receipt so workers cannot silently mix
    # routes.
    hidden_act = text_config.get("hidden_act", "silu")
    if model["dtype"] not in {"bfloat16", "float16"} or hidden_act != "silu":
        raise ValueError("This probe covers only the BF16/FP16 SwiGLU routes")
    validate_config(config, dimensions)
    samples = lines(PROJECT_ROOT / evaluation["data"]["search"])
    metrics_path = dense_root(evaluation, project_root=PROJECT_ROOT) / "benchmark/metrics.jsonl"
    lengths = prefill_lengths(
        lines(metrics_path), [row["stable_row_id"] for row in samples]
    )
    context = {
        "device": device,
        "backend": (
            "transformers_llama_text_mlp_eager" if model.get("adapter") == "llava15" else
            "transformers_qwen2_text_mlp_eager"
            if model.get("adapter") == "llava_onevision"
            else "transformers_qwen3_vl_text_mlp_eager"
        ),
        "dtype": model["dtype"],
        "dimensions": dimensions,
        "hidden_act": hidden_act,
        "input_distribution": "seed_0_standard_normal",
        "weights": "seed_0_torch_linear_initialization",
        "timing": "cuda_events_full_swiglu_forward",
    }
    # Preserve the established Qwen receipt context byte-for-byte so complete
    # probes remain reusable. LLaVA needs an explicit family marker because it
    # uses the Qwen2 MLP implementation and FP16 route.
    if model.get("adapter") in {"llava_onevision", "llava15"}:
        context["model_adapter"] = model["adapter"]
    shapes = [
        {"phase": phase, "input_tokens": length, "width": width}
        for phase, length in [("decode", 1), *[("prefill", size) for size in lengths]]
        for width in probe_widths(
            dimensions["intermediate_size"],
            config["selection"]["channel_group_size"],
            config["hardware_probe"]["minimum_width_ratio"],
        )
    ]
    return {
        "config": config,
        "context": context,
        "shapes": shapes,
        "metrics_source": str(metrics_path),
        "model": model,
    }


def shape_key(row):
    return row["phase"], row["input_tokens"], row["width"]


def complete_measurement(row):
    values = row.get("milliseconds", [])
    return (
        row.get("status") == "measured"
        and row.get("finite_output") is True
        and len(values) == row.get("measurement_iterations")
        and bool(values)
        and all(
            isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in values
        )
    )


def existing_measurements(destination, context):
    path = Path(destination) / "measurements.json"
    if not path.exists():
        return []
    saved = read(path)
    if saved["context"] != context:
        raise ValueError("Probe context differs; use a separate output directory")
    return saved["records"]


def pending_shapes(snap, records):
    complete = {shape_key(row) for row in records if complete_measurement(row)}
    return [shape for shape in snap["shapes"] if shape_key(shape) not in complete]


def summarize(snap, records) -> dict[str, Any]:
    requested = {shape_key(shape) for shape in snap["shapes"]}
    latest = {shape_key(row): row for row in records if shape_key(row) in requested}
    dense = {
        key[:2]: row
        for key, row in latest.items()
        if row["width"] == snap["context"]["dimensions"]["intermediate_size"]
        and complete_measurement(row)
    }
    rows = []
    for shape in snap["shapes"]:
        record = latest.get(shape_key(shape))
        if record is None:
            rows.append({**shape, "status": "pending"})
            continue
        row = {key: value for key, value in record.items() if key != "milliseconds"}
        control = dense.get(shape_key(shape)[:2])
        if complete_measurement(record):
            row["median_ms"] = statistics.median(record["milliseconds"])
            row["speedup_vs_dense_width"] = (
                statistics.median(control["milliseconds"]) / row["median_ms"]
                if control
                else None
            )
        rows.append(row)
    pending = pending_shapes(snap, records)
    return {
        "status": "completed" if not pending else "incomplete",
        "context": snap["context"],
        "remaining_shapes": len(pending),
        "shapes": rows,
        "interpretation": "Synthetic full-FFN screening only. Executable widths are not guaranteed speedups. No model quality, end-to-end performance, or structural boundary is established.",
    }


def execute(snap, output):
    """Reuse every complete shape before acquiring a GPU or launching a worker."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / ".lock"):
        records = existing_measurements(output, snap["context"])
        missing = pending_shapes(snap, records)
        if missing:
            device = snap["context"]["device"]
            with lock(f"/tmp/agent4vlm-gpu-{os.getuid()}-{device}.lock"):
                request = {
                    "operation": "structured_pruning_probe",
                    "device": device,
                    "hardware": hardware(device, snap.get("model")),
                    "model": snap.get("model", {}),
                    "probe": snap,
                    "shapes_to_measure": missing,
                    "probe_output": str(output),
                }
                attempt = 1
                while (output / f"attempt-{attempt}").exists():
                    attempt += 1
                launch(request, output / f"attempt-{attempt}")
            records = existing_measurements(output, snap["context"])
        summary = summarize(snap, records)
        save_json(output / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/structured_pruning.json")
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--evaluation-config", default="configs/evaluation.json")
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Coordinator received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    snap = snapshot(args.config, args.model_config, args.evaluation_config, args.device)
    evaluation = read(PROJECT_ROOT / args.evaluation_config)
    output = (PROJECT_ROOT / args.output).resolve() if args.output else search_root(evaluation) / "structured-pruning/hardware-probe"
    records = existing_measurements(output, snap["context"])
    result = (
        execute(snap, output)
        if args.execute
        else {
            "context": snap["context"],
            "widths": sorted({s["width"] for s in snap["shapes"]}, reverse=True),
            "prefill_lengths": sorted(
                {s["input_tokens"] for s in snap["shapes"] if s["phase"] == "prefill"}
            ),
            "total_shapes": len(snap["shapes"]),
            "pending_shapes": len(pending_shapes(snap, records)),
            "calibration_source": snap["config"]["calibration"]["source"],
        }
    )
    print(json.dumps(result, indent=2))
    if args.execute and result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
