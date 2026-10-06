"""Run five native LMMS tasks using local Qwen3-VL weights and dataset caches."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from search.model import PROJECT_ROOT
from search.lmms_gate import _mechanism_file, _sample_file, _serving_receipt, _task_result

TASKS = ("chartqa", "mmstar", "textvqa_val", "docvqa_val", "pope")


def cache_environment(model_config=None):
    env = os.environ.copy()
    for key in ("A4FVLMSP_TOKEN_KEEP_RATIO", "A4FVLMSP_TOKEN_SELECTION", "A4FVLMSP_TOKEN_RECEIPT",
                "A4FVLM_LMMS_VLLM_MODEL", "A4FVLM_VLLM_SELECTION_TRACE", "A4FVLMS_VLLM_TOKEN_RUNTIME",
                "A4FVLM_VLLM_MODEL_RECEIPT"):
        env.pop(key, None)
    cache = PROJECT_ROOT / "data/lmms"
    env["HF_HUB_CACHE"] = str(cache / "hub")
    env["HF_DATASETS_CACHE"] = str(cache / "datasets")
    env["LMMS_EVAL_HOME"] = str(cache / "lmms")
    # Quantized backends provide Ninja/Marlin helpers in the selected
    # interpreter's bin directory.  LMMS launches a child process, so make
    # that isolated toolchain visible without modifying the Dense shell PATH.
    worker_bin = str(Path(sys.prefix) / "bin")
    env["PATH"] = worker_bin + os.pathsep + env.get("PATH", "")
    if isinstance(model_config, dict) and model_config.get("token_keep_ratio") is not None:
        env["A4FVLMSP_TOKEN_KEEP_RATIO"] = str(model_config["token_keep_ratio"])
        selection = model_config.get("token_selection")
        if isinstance(selection, dict):
            env["A4FVLMSP_TOKEN_SELECTION"] = json.dumps(
                selection, ensure_ascii=True, separators=(",", ":")
            )
    if isinstance(model_config, dict) and model_config.get("execution_backend") == "vllm":
        env["A4FVLM_LMMS_VLLM_MODEL"] = json.dumps(model_config)
    return env


def build_command(model_config, task, destination, *, limit=None):
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    adapter = model_config.get("adapter", "qwen3_vl")
    if adapter == "qwen3_vl" and model_config["dtype"] != "bfloat16":
        raise ValueError("The native Qwen3-VL adapter loads BF16 weights")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    pretrained = str((PROJECT_ROOT / model_config["path"]).resolve())
    model_arg_values = [
        f"pretrained={pretrained}",
        f"device={model_config['device']}",
        f"device_map={model_config['device']}",
        f"attn_implementation={model_config['attn_implementation']}",
    ]
    if adapter in {"llava_onevision", "llava15"}:
        model_arg_values.insert(3, f"dtype={model_config['dtype']}")
    model_args = ",".join(model_arg_values)
    # lmms-eval v2 registers the Hugging Face LLaVA wrapper as ``llava_hf``.
    # The older ``llava_hf_chat`` alias is not present in current releases,
    # even though its module still exists, so using it fails before model
    # construction and produces no task receipt.
    model_name = "llava_hf" if adapter in {"llava_onevision", "llava15"} else "qwen3_vl"
    if model_config.get("stack_adapter") is True:
        # lmms-eval's stock qwen3_vl adapter selects the dense HF class by
        # name.  The launcher patches that resolver for the physical Stack
        # topology before dispatching the normal CLI, so the final LMMS run
        # loads the retained Stack artifact rather than silently using Dense.
        command = [
            sys.executable,
            str(PROJECT_ROOT / "search" / (
                "lmms_llava_vllm_entry.py" if adapter in {"llava_onevision", "llava15"} and model_config.get("execution_backend") == "vllm"
                else "lmms_llava_stack_entry.py" if adapter in {"llava_onevision", "llava15"}
                else "lmms_vllm_entry.py" if model_config.get("execution_backend") == "vllm"
                else "lmms_stack_entry.py"
            )),
            "--model",
            model_name,
            "--model_args",
            model_args,
            "--tasks",
            task,
            "--batch_size",
            "1",
            "--output_path",
            str(destination),
            "--log_samples",
        ]
    else:
        command = [
            sys.executable, "-m", "lmms_eval", "--model", model_name,
            "--model_args", model_args, "--tasks", task, "--batch_size", "1",
            "--output_path", str(destination), "--log_samples",
        ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    return command


def _task_complete(destination, task, model_config):
    """Reuse only a complete native task with its samples and Token evidence."""
    matches = [path for path in destination.rglob("*_results.json")
               if task in json.loads(path.read_text()).get("results", {})]
    if not matches:
        return False
    path, result = _task_result(destination, task)
    expected = result.get("n-samples", {}).get(task, {}).get("effective")
    if not isinstance(expected, int) or expected < 1:
        raise ValueError(f"Missing effective sample count for {task}")
    samples = [json.loads(line) for line in _sample_file(path, task).read_text().splitlines()
               if line.strip()]
    if len(samples) != expected or len({row["doc_id"] for row in samples}) != expected:
        raise ValueError(f"Native sample count mismatch for {task}")
    ratio = model_config.get("token_keep_ratio")
    if ratio is not None and ratio < 1:
        _mechanism_file(destination, task, expected, ratio,
                        backend=model_config.get("execution_backend", "native_transformers"))
    if model_config.get("execution_backend") == "vllm":
        _serving_receipt(destination, task, model_config)
    return True


def run_lmms(model_config, tasks, *, output_dir, limit=None, resume=False):
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Select at least one task, without duplicates")
    destination = (PROJECT_ROOT / Path(output_dir)).resolve()
    commands = [build_command(model_config, task, destination, limit=limit) for task in tasks]
    effective_config = {
        "model": model_config, "tasks": tasks, "limit": limit,
    }
    if resume:
        saved = json.loads((destination / "config.json").read_text())
        if saved != effective_config:
            raise ValueError("LMMS resume config does not match the saved run")
        # Validate every reusable task before launching any new GPU work.
        completed = {task for task in tasks if _task_complete(destination, task, model_config)}
    else:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "config.json").write_text(
            json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n"
        )
        completed = set()
    for task, command in zip(tasks, commands):
        if task in completed:
            print(f"Reusing completed {task}", flush=True)
            continue
        previous = set(destination.rglob("*_results.json"))
        print(f"Running {task}" + (f" (limit={limit})" if limit else " (full split)"), flush=True)
        env = cache_environment(model_config)
        token_receipt = destination / f"{task}-token-mechanism.jsonl"
        if resume and token_receipt.exists():
            # No native result means this task must restart. Preserve its old
            # mechanism trace, so appending cannot double-count the new run.
            index = 1
            partial = destination / f"{task}-token-mechanism.partial-{index}.jsonl"
            while partial.exists():
                index += 1
                partial = destination / f"{task}-token-mechanism.partial-{index}.jsonl"
            token_receipt.rename(partial)
        if model_config.get("token_keep_ratio", 1) < 1:
            if model_config.get("stack_adapter") is not True:
                raise ValueError("Native LMMS Token pruning requires stack_adapter")
            env["A4FVLMSP_TOKEN_RECEIPT"] = str(token_receipt)
        subprocess.run(
            command,
            env=env,
            cwd=PROJECT_ROOT,
            check=True,
        )
        # Some upstream CLI failures exit successfully; require a new task result.
        results = set(destination.rglob("*_results.json")) - previous
        task_results = [json.loads(path.read_text()) for path in sorted(results)]
        task_results = [row for row in task_results if task in row["results"]]
        if not task_results:
            raise RuntimeError(f"LMMS did not produce results for {task}")
        if model_config.get("execution_backend") == "vllm":
            _task_complete(destination, task, model_config)
            # Per-sample final evidence includes the actual selected-index hashes.
            token_receipt.with_suffix(".selection.jsonl").unlink(missing_ok=True)
        if model_config.get("token_keep_ratio", 1) < 1:
            if not token_receipt.is_file():
                raise RuntimeError(f"LMMS did not produce Token mechanism evidence for {task}")
            records = [json.loads(line) for line in token_receipt.read_text().splitlines()]
            if not records or not any(row["image_tokens_after"] < row["image_tokens_before"] for row in records):
                raise RuntimeError(f"LMMS did not observe visual Token reduction for {task}")
            expected = task_results[-1].get("n-samples", {}).get(task, {}).get("effective")
            if expected is not None and len(records) != expected:
                raise RuntimeError(f"LMMS Token evidence count mismatch for {task}: {len(records)} != {expected}")
        print(f"Completed {task}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New result directory")
    parser.add_argument(
        "--model-config",
        default="configs/model.json",
        help="JSON model descriptor; may point at a validated quantized artifact",
    )
    parser.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated task IDs")
    parser.add_argument("--limit", type=int, help="Samples per task for smoke testing only")
    parser.add_argument("--resume", action="store_true", help="Reuse complete tasks in a matching result directory")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.model_config).read_text())
    from search.context import require_stages
    require_stages(config, ("lmms",))
    run_lmms(config, args.tasks.split(","), output_dir=args.output, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
