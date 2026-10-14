"""Native Kimi input receipts; preserve upstream scores and sample files."""

import hashlib
import json
from pathlib import Path


SCHEMA = "a4fvlm.kimi-lmms-inputs.v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def input_identity(task, split, doc_id, prompt, image, generation, until):
    # Hash the original task raster, before either backend's preprocessing.
    # Generation settings are those returned by the unchanged native bridge.
    raster = image.convert("RGB")
    identity = {"task": task, "split": split, "doc_id": doc_id,
                "prompt": prompt, "image_size": list(raster.size),
                "image_rgb_sha256": hashlib.sha256(raster.tobytes()).hexdigest(),
                "generation": dict(generation), "until": list(until)}
    return {**identity, "input_sha256": digest(identity)}


def completed_input(identity, response):
    return {**identity, "response_sha256": digest(response)}


def paths(root, task):
    root = Path(root)
    return root / f"{task}-native-inputs.jsonl", root / f"{task}-native-inputs.json"


def _validate_rows(rows, samples, task):
    expected = {sample["doc_id"]: sample for sample in samples}
    if len(expected) != len(samples) or len(rows) != len(samples):
        raise ValueError(f"Kimi native input count/ID mismatch for {task}")
    observed = {}
    for row in rows:
        doc_id = row.get("doc_id")
        if type(doc_id) is not int or doc_id not in expected or doc_id in observed:
            raise ValueError(f"Kimi native input count/ID mismatch for {task}")
        sample = expected[doc_id]
        identity = {key: row[key] for key in ("task", "split", "doc_id", "prompt",
                    "image_size", "image_rgb_sha256", "generation", "until")}
        if (row["task"] != task or not isinstance(row["split"], str) or not row["split"]
                or not isinstance(row["image_size"], list) or len(row["image_size"]) != 2
                or any(type(value) is not int or value <= 0 for value in row["image_size"])
                or not isinstance(row["image_rgb_sha256"], str)
                or len(row["image_rgb_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in row["image_rgb_sha256"])
                or row.get("input_sha256") != digest(identity)
                or sample.get("input") != row["prompt"]
                # Native Kimi writes the decoded response before LMMS applies
                # task-level filtering.  Dense reconstruction writes the
                # filtered response instead.  Both values are present in the
                # authoritative sample record, so accept either exact digest
                # while still rejecting any response not produced by LMMS.
                or row.get("response_sha256") not in {
                    digest(sample.get("filtered_resps")),
                    digest(sample.get("resps")),
                }):
            raise ValueError(f"Kimi native input/output binding mismatch for {task}/{doc_id}")
        observed[doc_id] = row["input_sha256"]
    return observed


def bind_receipt(root, task, *, origin="observed_generation", dataset_fingerprint=None):
    from search.lmms_gate import _sample_file, _task_result

    if origin not in {"observed_generation", "reconstructed_original_dense"}:
        raise ValueError("Unknown Kimi input receipt origin")
    root = Path(root)
    result_path, result = _task_result(root, task)
    sample_path = _sample_file(result_path, task)
    samples = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
    count = result.get("n-samples", {}).get(task, {}).get("effective")
    if type(count) is not int or count != len(samples) or count < 1:
        raise ValueError(f"Kimi native sample count mismatch for {task}")
    row_path, receipt_path = paths(root, task)
    rows = [json.loads(line) for line in row_path.read_text().splitlines() if line.strip()]
    _validate_rows(rows, samples, task)
    expected_split = result.get("configs", {}).get(task, {}).get("test_split")
    if expected_split and any(row["split"] != expected_split for row in rows):
        raise ValueError(f"Kimi native input split mismatch for {task}")
    config = json.loads((root / "config.json").read_text())
    if config.get("model", {}).get("adapter") != "kimi_vl":
        raise ValueError("Kimi input receipt requires a bound Kimi LMMS config")
    if origin == "reconstructed_original_dense" and (
            config["model"].get("stack_adapter") or config["model"].get("token_keep_ratio", 1) != 1
            or config["model"].get("execution_backend", "native_transformers") != "native_transformers"
            or not dataset_fingerprint):
        raise ValueError("Only original Dense inputs may be reconstructed")
    receipt = {"schema": SCHEMA, "status": "complete", "task": task,
               "origin": origin, "sample_count": count, "model": config["model"],
               "result_sha256": sha256(result_path), "samples_sha256": sha256(sample_path),
               "inputs_sha256": sha256(row_path), "dataset_fingerprint": dataset_fingerprint,
               "historical_image_observation": origin == "observed_generation"}
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def validated_inputs(root, task, *, candidate=False):
    from search.lmms_gate import _sample_file, _task_result

    root = Path(root)
    row_path, receipt_path = paths(root, task)
    receipt = json.loads(receipt_path.read_text())
    config = json.loads((root / "config.json").read_text())
    result_path, result = _task_result(root, task)
    sample_path = _sample_file(result_path, task)
    origin = receipt.get("origin")
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "complete"
            or receipt.get("task") != task or receipt.get("model") != config.get("model")
            or config.get("model", {}).get("adapter") != "kimi_vl"
            or receipt.get("result_sha256") != sha256(result_path)
            or receipt.get("samples_sha256") != sha256(sample_path)
            or receipt.get("inputs_sha256") != sha256(row_path)
            or origin not in {"observed_generation", "reconstructed_original_dense"}
            or candidate and origin != "observed_generation"
            or receipt.get("historical_image_observation") is not (origin == "observed_generation")
            or origin == "reconstructed_original_dense" and (
                config["model"].get("stack_adapter") or config["model"].get("token_keep_ratio", 1) != 1
                or config["model"].get("execution_backend", "native_transformers") != "native_transformers"
                or not receipt.get("dataset_fingerprint"))):
        raise ValueError(f"Kimi native input receipt binding mismatch for {task}")
    samples = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in row_path.read_text().splitlines() if line.strip()]
    if (receipt.get("sample_count") != len(samples)
            or result.get("n-samples", {}).get(task, {}).get("effective") != len(samples)):
        raise ValueError(f"Kimi native input count mismatch for {task}")
    expected_split = result.get("configs", {}).get(task, {}).get("test_split")
    if expected_split and any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"Kimi native input split mismatch for {task}")
    return _validate_rows(rows, samples, task)


def main():
    """Reconstruct original Dense inputs from unchanged native cached tasks."""
    import argparse
    import os
    from search.lmms import cache_environment
    from search.lmms_gate import _sample_file, _task_result

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    config = json.loads((root / "config.json").read_text())
    model = config["model"]
    from search.lmms_kimi_entry import validate_native_config
    validate_native_config(model)
    if model.get("stack_adapter") or config.get("limit") is not None:
        raise ValueError("Reconstruction requires complete original Dense tasks")
    os.environ.update(cache_environment(model))
    from lmms_eval.tasks import TaskManager, get_task_dict
    from search.deepseek_lmms import generation_options
    manager = TaskManager(model_name="kimi_vl")
    for name in args.tasks.split(","):
        row_path, receipt_path = paths(root, name)
        if receipt_path.exists():
            validated_inputs(root, name)
            print(f"Reusing complete native input receipt: {name}", flush=True)
            continue
        # Preserve a previous incomplete reconstruction before retrying. A
        # receipt is written only after all native prompt/target checks pass.
        for unfinished in (row_path, row_path.with_suffix(".partial.jsonl")):
            if unfinished.exists():
                number = 1
                saved = unfinished.with_suffix(f".unbound-{number}.jsonl")
                while saved.exists():
                    number += 1
                    saved = unfinished.with_suffix(f".unbound-{number}.jsonl")
                unfinished.rename(saved)
        result_path, result = _task_result(root, name)
        samples = [json.loads(line) for line in _sample_file(result_path, name).read_text().splitlines() if line.strip()]
        by_id = {row["doc_id"]: row for row in samples}
        task = get_task_dict([name], manager)[name]
        task.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False)
        if len(task.instances) != len(samples) or len(by_id) != len(samples):
            raise ValueError(f"Kimi reconstructed native task count mismatch for {name}")
        partial = row_path.with_suffix(".partial.jsonl")
        with partial.open("x") as stream:
            for index, request in enumerate(task.instances, 1):
                prompt, options, visual, doc_id, task_name, split = request.args
                sample = by_id[doc_id]
                doc = task.dataset[split][doc_id]
                if (task_name != name or prompt != sample["input"]
                        or str(task.doc_to_target(doc)) != sample["target"]):
                    raise ValueError(f"Kimi reconstructed prompt/target mismatch for {name}/{doc_id}")
                images = visual(doc)
                if len(images) != 1:
                    raise ValueError("Kimi native task requires one image")
                options, until = generation_options(options)
                identity = input_identity(name, split, doc_id, prompt, images[0], options, until)
                stream.write(json.dumps(completed_input(identity, sample["filtered_resps"])) + "\n")
                if index % 500 == 0:
                    stream.flush()
                    print(f"Native input reconstruction: {name} {index}/{len(samples)}", flush=True)
        partial.rename(row_path)
        print(json.dumps(bind_receipt(root, name, origin="reconstructed_original_dense",
                         dataset_fingerprint=task.dataset[split]._fingerprint)), flush=True)


if __name__ == "__main__":
    main()
