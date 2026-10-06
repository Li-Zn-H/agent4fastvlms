"""Generate the dense Dev baseline and reuse its outputs for Search."""

from collections import defaultdict
import argparse
import json
from pathlib import Path
from statistics import fmean

from search.evaluate import load_samples, score_answer
from search.model import PROJECT_ROOT, generate_outputs, load_model
from search.context import dense_root, validate_result_scope, require_stages


def complete_outputs(path, samples):
    """Validate a complete stage before deciding to load Dense weights."""
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [row["sample_id"] for row in rows]
        if (len(ids) == len(set(ids)) == len(samples)
                and set(ids) == {row["stable_row_id"] for row in samples}
                and all(isinstance(row["output_text"], str) for row in rows)):
            return {row["sample_id"]: row for row in rows}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def run_baseline(model_config, config, destination, *, resume=False):
    destination = Path(destination)
    if "result_root" in config:
        validate_result_scope(model_config, config, project_root=PROJECT_ROOT)
        manifest = destination / "config.json"
        expected = {"model": model_config, "evaluation": config}
        if manifest.exists() and json.loads(manifest.read_text()) != expected:
            raise ValueError("Dense baseline config changed; use a new output")
        if not manifest.exists() and any(destination.glob("*.jsonl")):
            raise ValueError("Existing Dense outputs have no model/evaluation receipt")
    data = config["data"]
    samples = {
        split: load_samples(
            PROJECT_ROOT / data[split], media_root=PROJECT_ROOT / data["media_root"]
        )
        for split in ("dev", "search")
    }
    dev_by_id = {row["stable_row_id"]: row for row in samples["dev"]}
    if any(dev_by_id.get(row["stable_row_id"]) != row for row in samples["search"]):
        raise ValueError("Search must be an unchanged subset of Dev to reuse outputs")

    destination.mkdir(parents=True, exist_ok=True)
    if "result_root" in config and not manifest.exists():
        from search.worker import save_json
        save_json(manifest, expected)
    outputs = complete_outputs(destination / "dev.jsonl", samples["dev"]) if resume else None
    reused_dev = outputs is not None
    # Write each response immediately; an interrupted run leaves its raw outputs.
    if outputs is None:
        model, processor = load_model(model_config)
        outputs = {}
        with (destination / "dev.jsonl").open("w", encoding="utf-8") as stream:
            for index, sample in enumerate(samples["dev"], 1):
                output = generate_outputs(
                    model, processor, [sample],
                    media_root=data["media_root"], generation=config["generation"],
                )[0]
                outputs[output["sample_id"]] = output
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                stream.flush()
                if index == 1 or index % 16 == 0 or index == len(samples["dev"]):
                    print(f"Dev {index}/{len(samples['dev'])}", flush=True)

    if not reused_dev or complete_outputs(destination / "search.jsonl", samples["search"]) is None:
        with (destination / "search.jsonl").open("w", encoding="utf-8") as stream:
            for sample in samples["search"]:
                stream.write(json.dumps(outputs[sample["stable_row_id"]], ensure_ascii=False) + "\n")

    options = {key: config["scoring"][key] for key in (
        "maximum_response_tokens", "negation_window_tokens"
    )}
    summary = {}
    for split, rows in samples.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["primary_category"]].append(score_answer(
                outputs[row["stable_row_id"]]["output_text"],
                row["answers"], row["scoring_type"], **options,
            ))
        tasks = {
            task: {"sample_count": len(scores), "mean": fmean(scores)}
            for task, scores in sorted(grouped.items())
        }
        summary[split] = {
            "sample_count": len(rows),
            "macro_mean": fmean(task["mean"] for task in tasks.values()),
            "tasks": tasks,
        }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {destination}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--evaluation-config", default="configs/evaluation.json")
    parser.add_argument("--output")
    parser.add_argument("--device", help="Process-local model device (the coordinator pins the physical GPU)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.evaluation_config).read_text())
    model_config = json.loads((PROJECT_ROOT / args.model_config).read_text())
    validate_result_scope(model_config, config, project_root=PROJECT_ROOT)
    require_stages(model_config, ("dense",), project_root=PROJECT_ROOT)
    if args.device:
        model_config["device"] = args.device
    destination = (PROJECT_ROOT / args.output if args.output else
                   dense_root(config, project_root=PROJECT_ROOT) / "quality")
    run_baseline(model_config, config, destination, resume=args.resume)


if __name__ == "__main__":
    main()
