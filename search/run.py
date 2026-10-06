"""Evaluate a supplied candidate against the dense Search/Dev baseline."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Literal

from search.evaluate import compare_outputs, load_samples
from search.model import PROJECT_ROOT, generate_outputs, load_model
from search.context import dense_root, validate_result_scope, require_stages, validate_dense_reference


def quality_action(
    retention: float,
    *,
    stage: Literal["search", "dev"],
    minimum_retention: float,
    search_direct_pass_minimum_retention: float,
) -> Literal["pass", "reject", "dev"]:
    """Apply the thresholds to unrounded retention, without running a model."""
    if (
        not math.isfinite(retention)
        or retention < 0
        or not 0 < minimum_retention <= search_direct_pass_minimum_retention <= 1
    ):
        raise ValueError("Retention or quality thresholds are out of range")
    if stage not in ("search", "dev"):
        raise ValueError("stage must be search or dev")
    if stage == "dev":
        return "pass" if retention >= minimum_retention else "reject"
    return "pass" if retention >= search_direct_pass_minimum_retention else "dev"


def evaluate_candidate(
    model, processor, config, *, output_dir, reuse_outputs=None, force_dev=False
):
    """Reuse the caller's candidate model; run full Dev only when Search requests it.

    Relative paths resolve against the project root. Use a new output directory
    for each evaluation to keep previous results and incomplete runs separate.
    """
    validate_dense_reference(None, config, project_root=PROJECT_ROOT)
    destination = PROJECT_ROOT / Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    baseline_dir = dense_root(config, project_root=PROJECT_ROOT) / "quality"
    data = config["data"]
    options = {
        key: config["scoring"][key]
        for key in ("maximum_response_tokens", "negation_window_tokens")
    }
    stages = {}
    for stage in ("search", "dev"):
        samples = load_samples(
            PROJECT_ROOT / data[stage], media_root=PROJECT_ROOT / data["media_root"]
        )
        baseline = [
            json.loads(line)
            for line in (baseline_dir / f"{stage}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        # Reject an incomplete or unusable baseline before spending GPU time.
        compare_outputs(samples, baseline, baseline, **options)
        source = (reuse_outputs or {}).get(stage)
        if source:
            outputs = [
                json.loads(line)
                for line in Path(source).read_text().splitlines()
                if line.strip()
            ]
        else:
            outputs = []
            with (destination / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
                for index, sample in enumerate(samples, 1):
                    output = generate_outputs(
                        model,
                        processor,
                        [sample],
                        media_root=data["media_root"],
                        generation=config["generation"],
                    )[0]
                    outputs.append(output)
                    stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                    stream.flush()
                    if index == 1 or index % 16 == 0 or index == len(samples):
                        print(f"{stage} {index}/{len(samples)}", flush=True)
        result = compare_outputs(samples, baseline, outputs, **options)
        action = quality_action(result.retention, stage=stage, **config["quality"])
        stages[stage] = {**asdict(result), "action": action}
        print(f"{stage}: retention={result.retention:.6f}, action={action}", flush=True)
        if action != "dev" and not (stage == "search" and force_dev):
            break
    summary = {"action": action, "stages": stages}
    if reuse_outputs:
        summary["reused_outputs"] = {
            key: str(value) for key, value in reuse_outputs.items() if key in stages
        }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--evaluation-config", default="configs/evaluation.json")
    parser.add_argument("--output", required=True, help="New result directory")
    args = parser.parse_args()
    if (PROJECT_ROOT / args.output).exists():
        parser.error("--output must be a new directory")
    config = json.loads((PROJECT_ROOT / args.evaluation_config).read_text())
    model_config = json.loads((PROJECT_ROOT / args.model_config).read_text())
    validate_result_scope(model_config, config, project_root=PROJECT_ROOT)
    require_stages(model_config, ("dense",), project_root=PROJECT_ROOT)
    model, processor = load_model(model_config)
    evaluate_candidate(model, processor, config, output_dir=args.output)


if __name__ == "__main__":
    main()
