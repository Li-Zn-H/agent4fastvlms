"""Benchmark Search samples on one CUDA device with the normal generation path."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from search.evaluate import load_samples
from search.model import PROJECT_ROOT, generate_outputs, load_model
from search.context import validate_result_scope, require_stages


def percentile(values, fraction):
    """Linearly interpolate between sorted observations, including endpoints."""
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(outputs):
    metrics = [output["performance"] for output in outputs]
    latencies = [row["end_to_end_seconds"] for row in metrics]
    result = {
        "sample_count": len(outputs),
        "total_seconds": sum(latencies),
        "mean_seconds": fmean(latencies),
        "p50_seconds": percentile(latencies, 0.50),
        "p95_seconds": percentile(latencies, 0.95),
        "generate_total_seconds": sum(row["generate_seconds"] for row in metrics),
        "output_tokens": sum(row["output_tokens"] for row in metrics),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in metrics),
    }
    for field in (
        "vision_encode_seconds",
        "pruning_seconds",
        "prefill_seconds",
        "decode_seconds",
        "decode_tokens",
    ):
        result[field] = sum(row[field] for row in metrics)
    result["decode_mean_seconds_per_token"] = (
        result["decode_seconds"] / result["decode_tokens"]
        if result["decode_tokens"]
        else None
    )
    return result


def benchmark(model, processor, config, *, output_dir, warmup=5, repeats=1):
    """Measure an already-loaded dense or optimized model; no reload per round."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    samples = load_samples(
        PROJECT_ROOT / config["data"]["search"],
        media_root=PROJECT_ROOT / config["data"]["media_root"],
    )
    if warmup > len(samples):
        raise ValueError("warmup exceeds the number of Search samples")
    destination = PROJECT_ROOT / Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    options = dict(
        media_root=config["data"]["media_root"], generation=config["generation"]
    )
    generate_outputs(model, processor, samples[:warmup], **options)
    print(f"Warmup complete: {warmup} samples", flush=True)
    rounds = []
    metrics_path = destination / "metrics.jsonl"
    for repeat in range(1, repeats + 1):
        outputs = []
        with metrics_path.open("a", encoding="utf-8") as stream:
            for index, sample in enumerate(samples, 1):
                output = generate_outputs(
                    model, processor, [sample], measure=True, **options
                )[0]
                outputs.append(output)
                record = {
                    "round": repeat,
                    "sample_id": output["sample_id"],
                    "performance": output["performance"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if index == 1 or index % 16 == 0 or index == len(samples):
                    print(
                        f"Round {repeat}/{repeats}: {index}/{len(samples)}", flush=True
                    )
        rounds.append(summarize(outputs))
        print(json.dumps(rounds[-1]), flush=True)
    totals = [row["total_seconds"] for row in rounds]
    summary = {
        "warmup_samples": warmup,
        "rounds": rounds,
        "mean_round_seconds": fmean(totals),
        "round_time_range_over_mean": (max(totals) - min(totals)) / fmean(totals),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--evaluation-config", default="configs/evaluation.json")
    parser.add_argument("--vision-token-config", default="configs/vision_token.json")
    parser.add_argument("--output", required=True, help="New result directory")
    parser.add_argument(
        "--keep-ratio",
        type=float,
        help="Apply vision token selection using configs/vision_token.json",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.keep_ratio is not None and not 0 < args.keep_ratio <= 1:
        parser.error("--keep-ratio must be in (0, 1]")
    if (PROJECT_ROOT / args.output).exists():
        parser.error("--output must be a new directory")
    config = json.loads((PROJECT_ROOT / args.evaluation_config).read_text())
    model_config = json.loads((PROJECT_ROOT / args.model_config).read_text())
    validate_result_scope(model_config, config, project_root=PROJECT_ROOT)
    require_stages(model_config, ("dense", "token") if args.keep_ratio is not None else ("dense",))
    model, processor = load_model(model_config)
    if args.keep_ratio is not None:
        from search.model import processor_adapter

        selection = json.loads(
            (PROJECT_ROOT / args.vision_token_config).read_text()
        )["selection"]
        model.vision_token_controller = processor_adapter(processor).resolve("token_controller")(
            model, selection, args.keep_ratio
        )
    benchmark(
        model,
        processor,
        config,
        output_dir=args.output,
        warmup=args.warmup,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
