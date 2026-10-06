"""Use a qualified serving freeze, or compare it with Dense at one timing boundary.

Default invocation validates evidence on CPU. --execute uses isolated Workers;
--benchmark measures the complete Search set with both backends on the same GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter

from search.model import PROJECT_ROOT
from search.optimize import hardware, launch, lock, read
from search.worker import save_json
from skills.quantization import digest


TIMING_SCOPE = "image_open_to_decoded_answer_batch1"
PROTOCOL = "a4fvlm.serving-use.v1"


def resolve(path):
    return (PROJECT_ROOT / path).absolute()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_freeze(path):
    """Recheck the saved parent chain and full LMMS evidence before loading weights."""
    from search.lmms_gate import DEFAULT_TASKS, aggregate_lmms, complete_lmms, selection_policy
    from search.serving_lmms import plan
    from skills.stack_vllm import validate_stack_artifact

    path = resolve(path)
    freeze = read(path)
    root = path.parent.parent
    config = read(root / "config.json")
    fallback = (selection_policy(config) == "best_effort"
                and freeze.get("selection_policy") == "best_effort"
                and freeze.get("status") == "lmms_evaluated"
                and freeze.get("quality_target_met") is False)
    if freeze.get("status") != "lmms_qualified" and not fallback:
        raise ValueError("Serving requires a qualified or explicitly evaluated best-effort freeze")
    if config["tasks"] != list(DEFAULT_TASKS) or config["minimum_retention"] < .95:
        raise ValueError("Serving requires the full five-task gate at >=95%")
    planned = plan(config)
    matched = next((c for c in planned["candidates"] if c["id"] == freeze.get("id")), None)
    if (not matched or any(freeze.get(k) != v for k, v in matched.items())
            or freeze.get("stack_digest") != planned["stack_digest"]):
        raise ValueError("Serving freeze no longer matches its parent/configuration")
    model = freeze["model"]
    validate_stack_artifact(resolve(model["path"]))
    receipt = freeze["lmms"]
    if read(resolve(receipt["model_config"])) != model:
        raise ValueError("Serving LMMS descriptor differs from freeze")
    summary = aggregate_lmms(
        root / "candidates" / freeze["id"] / "lmms", resolve(config["baseline_root"]),
        tasks=config["tasks"], minimum_retention=config["minimum_retention"],
        token_keep_ratio=model["token_keep_ratio"], candidate_id=freeze["id"],
        model_config=receipt["model_config"], execution_backend="vllm",
    )
    if (summary != receipt or (summary["quality_gate"] != "passed" and not (fallback and complete_lmms(summary)))
            or (fallback and summary["quality_gate"] != "quality_rejected")):
        raise ValueError("Serving freeze has missing or changed LMMS evidence")
    return freeze, config


def generation_options(generation, eos):
    if generation.get("do_sample") is not False or type(generation.get("max_new_tokens")) is not int:
        raise ValueError("Serving use requires greedy generation with an integer token limit")
    if generation["max_new_tokens"] < 1 or type(generation.get("ignore_eos")) is not bool:
        raise ValueError("Invalid generation settings")
    stop = [] if generation["ignore_eos"] else (list(eos) if isinstance(eos, list) else [eos])
    if not generation["ignore_eos"] and any(type(x) is not int for x in stop):
        raise ValueError("Dense generation config must declare EOS token IDs")
    return {
        "max_tokens": generation["max_new_tokens"], "temperature": 0.0,
        "ignore_eos": generation["ignore_eos"], "stop_token_ids": stop,
        "repetition_penalty": 1.0,
    }


def snapshot(config_path, *, benchmark=False, image=None, prompt=None):
    from search.evaluate import load_samples

    config = read(resolve(config_path))
    freeze, saved = load_freeze(config["freeze"])
    dense = read(resolve(config["dense_model_config"]))
    evaluation = read(resolve(config["evaluation_config"]))
    eos = read(resolve(dense["path"]) / "generation_config.json")["eos_token_id"]
    options = generation_options(evaluation["generation"], eos)
    if config["device"] not in (0, 1) or type(config["device"]) is not int:
        raise ValueError("device must be 0 or 1")
    if benchmark:
        if image is not None or prompt is not None:
            raise ValueError("Benchmark uses the complete configured Search set")
        samples = load_samples(resolve(evaluation["data"]["search"]),
                               media_root=resolve(evaluation["data"]["media_root"]))
        requests = [{"sample_id": x["stable_row_id"], "prompt": x["logical_question"],
                     "image": str(resolve(evaluation["data"]["media_root"]) / x["media"]["logical_path"])}
                    for x in samples]
        warmup, repeats = config["benchmark"]["warmup"], config["benchmark"]["repeats"]
        if (type(warmup) is not int or not 0 <= warmup <= len(requests)
                or type(repeats) is not int or repeats < 1):
            raise ValueError("Invalid benchmark warmup/repeats")
    elif image is not None and prompt and prompt.strip():
        requests = [{"sample_id": "inference", "image": str(resolve(image)), "prompt": prompt}]
        warmup, repeats = 0, 1
    elif image is None and prompt is None:
        requests, warmup, repeats = [], 0, 1
    else:
        raise ValueError("Provide both --image and a nonempty --prompt")
    for row in requests:
        row["image_sha256"] = sha256(row["image"])
    # Bind processor/tokenizer/model metadata to reuse without hashing/copying weights.
    assets = {}
    for name, model in (("dense", dense), ("serving", freeze["model"])):
        root = resolve(model["path"])
        assets[name] = {p.name: sha256(p) for p in sorted(root.iterdir())
                        if p.suffix in {".json", ".jinja"} and p.is_file()}
    return {
        "protocol": PROTOCOL, "mode": "benchmark" if benchmark else "inference",
        "config": config, "freeze_digest": digest(freeze), "candidate_id": freeze["id"],
        "model": freeze["model"], "dense_model": dense, "assets": assets,
        "worker_python": str(resolve(saved["worker_python"])),
        "dense_worker_python": str(resolve(config["dense_worker_python"])),
        "generation": options, "requests": requests, "warmup": warmup, "repeats": repeats,
        "timing_scope": TIMING_SCOPE, "cache_policy": "reset_prefix_and_unique_image_per_request",
    }


def open_native_image(path):
    """Keep native decoded pixels identical through vLLM's EXIF normalization."""
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    # Transformers consumes this raster without rotating it. Prevent vLLM from
    # applying the file's orientation a second way; do not transform any pixels.
    exif = image.getexif()
    if 274 in exif:
        del exif[274]
        image.info["exif"] = exif.tobytes()
    return image


class Session:
    """One warm, single-image engine; used by both inference and the comparison."""

    def __init__(self, snap, backend, output):
        import torch
        from transformers import AutoProcessor

        self.backend, self.snap = backend, snap
        self.index = 0
        self.output = Path(output)
        if backend == "dense":
            from search.model import load_model
            self.model, self.processor = load_model(snap["dense_model"] | {"device": "cuda:0"})
            self.runtime = {"backend": "transformers", "model": snap["dense_model"]}
        else:
            from vllm import LLM
            from search.lmms_vllm_entry import runtime_receipt
            from skills.stack_vllm import prepare_vllm_artifact
            from skills.stack_runtime import build_vllm_runtime_kwargs

            model = snap["model"]
            self.processor = AutoProcessor.from_pretrained(resolve(model["path"]), local_files_only=True)
            self.trace = self.output / "selection.jsonl"
            observed_path = self.output / "observed-model.json"
            os.environ["A4FVLM_VLLM_SELECTION_TRACE"] = str(self.trace)
            os.environ["A4FVLM_VLLM_MODEL_RECEIPT"] = str(observed_path)
            adapter = prepare_vllm_artifact(resolve(model["path"]), token_runtime={
                "keep_ratio": model["token_keep_ratio"], "selection": model["token_selection"],
            })
            self.engine = LLM(
                model=str(resolve(model["path"])), dtype=model["dtype"], trust_remote_code=True,
                model_class_overrides={adapter["architecture"]: "skills.stack_vllm:" + adapter["architecture"]},
                **({"video_pruning_rate": 1e-9} if model.get("adapter", "qwen3_vl") == "qwen3_vl" else {}),
                generation_config="vllm",
                **build_vllm_runtime_kwargs(model["runtime"]),
            )
            effective = runtime_receipt(self.engine.llm_engine.vllm_config, model["runtime"])
            observed = read(observed_path)
            if (not observed["token_pruning_enabled"]
                    or observed["token_runtime"]["keep_ratio"] != model["token_keep_ratio"]
                    or observed["kv_cache_dtypes"] != [model["runtime"]["kv_cache_dtype"]]
                    or effective["max_model_len"] != model["runtime"]["vllm_max_model_len"]):
                raise ValueError("Effective serving configuration differs from freeze")
            # Requested 'auto' must resolve to the backend actually qualified in LMMS.
            qualified_runtime = read(resolve(snap["config"]["freeze"]).parent.parent /
                "candidates" / snap["candidate_id"] / "lmms/chartqa-token-mechanism.runtime.json")
            if observed["attention_backends"] != qualified_runtime["observed_model"]["attention_backends"]:
                raise ValueError("Actual attention backend differs from qualified LMMS")
            self.runtime = {"backend": "vllm", "effective": effective, "observed_model": observed,
                            "model": model, "adapter": adapter}
            self.trace.write_text("")  # Engine startup profiles are not request evidence.
            self.trace_offset = 0
        save_json(self.output / "runtime.json", self.runtime)
        torch.cuda.synchronize()

    def generate(self, request):
        """Time disk image decoding, native preparation, generation and text decoding."""
        import torch

        if self.backend != "dense" and not self.engine.reset_prefix_cache():
            raise RuntimeError("Could not clear cross-request prefix cache")
        torch.cuda.synchronize()
        started = perf_counter()
        image = open_native_image(request["image"])
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": request["prompt"]},
            ]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            options = self.snap["generation"]
            if self.backend == "dense":
                inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
                inputs.pop("token_type_ids", None)
                native_ids = inputs["input_ids"][0].tolist()
                inputs = inputs.to("cuda:0")
                eos = None if options["ignore_eos"] else options["stop_token_ids"]
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs, max_new_tokens=options["max_tokens"], do_sample=False,
                        eos_token_id=eos, forced_eos_token_id=None, repetition_penalty=1.0,
                        return_dict_in_generate=False,
                    )
                ids = output[0, len(native_ids):].tolist()
            else:
                from vllm import SamplingParams
                output = self.engine.generate([{
                    "prompt": prompt, "multi_modal_data": {"image": image},
                    "multi_modal_uuids": {"image": f"serving-use-{self.index}"},
                }], SamplingParams(**options, detokenize=False), use_tqdm=False)[0]
                ids = list(output.outputs[0].token_ids)
            text = self.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            # Validation is outside the timer. In-engine Token trace overhead remains
            # included in serving time; it cannot create an artificial speed gain.
            if self.backend != "dense":
                from search.lmms_vllm_entry import compact_prompt
                with self.trace.open() as handle:
                    handle.seek(self.trace_offset)
                    traces = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
                    self.trace_offset = handle.tell()
                if len(traces) != 1:
                    raise ValueError("Each request must execute actual visual Token selection")
                inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
                native_ids = inputs["input_ids"][0].tolist()
                image_token = self.processor.image_token_id
                expected = compact_prompt(native_ids, image_token, [
                    (t["image_tokens_before"], t["image_tokens_after"]) for t in traces])
                if list(output.prompt_token_ids) != expected:
                    raise ValueError("vLLM prompt differs from native prompt after compaction")
                mechanism = {"prompt_verified": True, "images": traces,
                             "decoder_tokens": len(expected)}
            else:
                mechanism = None
            self.index += 1
            return {"sample_id": request["sample_id"], "output_text": text,
                    "output_tokens": len(ids), "end_to_end_seconds": elapsed,
                    "native_prompt_sha256": digest(native_ids), "native_prompt_tokens": len(native_ids),
                    "token_mechanism": mechanism}
        finally:
            image.close()


def summarize(rows):
    from search.vllm_benchmark import percentile
    times = [row["end_to_end_seconds"] for row in rows]
    if not times or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("Invalid timing evidence")
    if any(type(row["output_tokens"]) is not int or row["output_tokens"] < 1 for row in rows):
        raise ValueError("Invalid output token counts")
    tokens = sum(row["output_tokens"] for row in rows)
    return {"requests": len(rows), "output_tokens": tokens, "total_seconds": sum(times),
            "mean_seconds": fmean(times), "p50_seconds": percentile(times, .5),
            "p95_seconds": percentile(times, .95), "request_throughput": len(times) / sum(times),
            "output_token_throughput": tokens / sum(times)}


def worker(request, output):
    """Invoked only through search.worker in the selected backend environment."""
    snap, backend = request["snapshot"], request["backend"]
    if snapshot(request["config_path"], benchmark=snap["mode"] == "benchmark",
                image=request.get("image"), prompt=request.get("prompt")) != snap:
        raise ValueError("Serving inputs changed before Worker startup")
    session = Session(snap, backend, output)
    for item in snap["requests"][:snap["warmup"]]:
        session.generate(item)
    rows = []
    with (output / "outputs.jsonl").open("w") as stream:
        for repeat in range(snap["repeats"]):
            for index, item in enumerate(snap["requests"], 1):
                row = session.generate(item) | {"round": repeat + 1}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if index == 1 or index % 16 == 0 or index == len(snap["requests"]):
                    print(f"{backend}: round {repeat+1}/{snap['repeats']} {index}/{len(snap['requests'])}", flush=True)
    save_json(output / "summary.json", {"status": "completed", "backend": backend,
        "snapshot_digest": digest(snap), "hardware": request["hardware"],
        "timing_scope": TIMING_SCOPE, "summary": summarize(rows),
        "rounds": [summarize([r for r in rows if r["round"] == n]) for n in range(1, snap["repeats"]+1)]})
    return {"report": str(output / "summary.json")}


def complete_output(directory, snap, backend):
    """Do not let a completed label reuse partial or mismatched measurements."""
    summary = read(directory / "summary.json")
    result = read(directory / "result.json")
    request = read(directory / "request.json")
    runtime = read(directory / "runtime.json")
    rows = [json.loads(x) for x in (directory / "outputs.jsonl").read_text().splitlines() if x.strip()]
    expected = [(n, r["sample_id"]) for n in range(1, snap["repeats"]+1) for r in snap["requests"]]
    if (result.get("status") != "completed" or summary.get("status") != "completed"
            or request.get("snapshot") != snap or summary.get("snapshot_digest") != digest(snap)
            or request.get("backend") != backend or summary.get("hardware") != request.get("hardware")
            or summary.get("backend") != backend or summary.get("timing_scope") != TIMING_SCOPE
            or [(r["round"], r["sample_id"]) for r in rows] != expected
            or summary["summary"] != summarize(rows)):
        raise ValueError("Incomplete or incompatible serving-use receipt")
    if backend == "serving":
        observed = runtime.get("observed_model", {})
        if (runtime.get("model") != snap["model"] or not observed.get("token_pruning_enabled")
                or observed.get("token_runtime", {}).get("keep_ratio") != snap["model"]["token_keep_ratio"]
                or observed.get("kv_cache_dtypes") != [snap["model"]["runtime"]["kv_cache_dtype"]]):
            raise ValueError("Missing actual runtime evidence")
        for row in rows:
            mechanism = row.get("token_mechanism") or {}
            images = mechanism.get("images", [])
            if snap["model"].get("adapter", "qwen3_vl") in {"llava_onevision", "llava15"}:
                positions_valid = all(i.get("position_type") == "1d_compacted"
                                      and i.get("positions_after") == i.get("image_tokens_after")
                                      for i in images)
            else:
                positions_valid = all(i.get("mrope_positions") == i.get("image_tokens_after") for i in images)
            if (not mechanism.get("prompt_verified") or len(images) != 1
                    or images[0].get("token_keep_ratio") != snap["model"]["token_keep_ratio"]
                    or not images[0].get("selected_indices_sha256")
                    or not positions_valid):
                raise ValueError("Missing actual Token/prompt evidence")
    return summary, rows


def compare(dense, serving):
    d, drows = dense
    s, srows = serving
    if (d["hardware"] != s["hardware"] or d["snapshot_digest"] != s["snapshot_digest"]
            or d["timing_scope"] != TIMING_SCOPE or s["timing_scope"] != TIMING_SCOPE
            or [(r["round"], r["sample_id"], r["native_prompt_sha256"]) for r in drows]
            != [(r["round"], r["sample_id"], r["native_prompt_sha256"]) for r in srows]):
        raise ValueError("Dense/serving inputs, prompt, timing scope or hardware differ")
    baseline, selected = d["summary"], s["summary"]
    return {"timing_scope": TIMING_SCOPE, "dense": baseline, "serving": selected,
            "end_to_end_speedup": baseline["total_seconds"] / selected["total_seconds"],
            "output_token_throughput_ratio": selected["output_token_throughput"] / baseline["output_token_throughput"],
            "note": "Same greedy settings; natural answer lengths may differ. Model load, warmup, cache reset and post-run audits excluded; in-engine Token trace included."}


def acceptance_outcome(retention, speedup, minimum=.95):
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (retention, speedup, minimum)):
        raise ValueError("Final acceptance needs finite measured quality and speed")
    quality_met, accelerated = retention >= minimum, speedup > 1
    return {"quality_retention": retention, "quality_target": minimum,
            "quality_target_met": quality_met, "end_to_end_speedup": speedup,
            "acceleration_target_met": accelerated,
            "goal_achieved": quality_met and accelerated,
            "outcome": ("goal_achieved" if quality_met and accelerated else
                        "accelerated_quality_unmet" if accelerated else "no_acceleration")}


def execute(config_path, *, benchmark=False, image=None, prompt=None):
    snap = snapshot(config_path, benchmark=benchmark, image=image, prompt=prompt)
    if not snap["requests"]:
        raise ValueError("Inference execution requires --image and --prompt")
    root = resolve(snap["config"]["output"]) / ("benchmark" if benchmark else "inference-" + digest(snap["requests"]))
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".lock"):
        if (root / "config.json").exists() and read(root / "config.json") != snap:
            raise ValueError("Serving-use snapshot changed; choose a new output in the JSON")
        save_json(root / "config.json", snap)
        reports = {}
        state = {"status": "running", "mode": snap["mode"], "candidate_id": snap["candidate_id"]}
        try:
            for backend in (["dense", "serving"] if benchmark else ["serving"]):
                state["active_backend"] = backend
                save_json(root / "state.json", state)
                attempts = root / backend
                attempts.mkdir(exist_ok=True)
                for directory in sorted(attempts.glob("attempt-*"), reverse=True):
                    try:
                        reports[backend] = complete_output(directory, snap, backend)
                        break
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
                if backend in reports:
                    continue
                device = snap["config"]["device"]
                with lock(Path("/tmp") / f"agent4vlm-gpu-{os.getuid()}-{device}.lock"):
                    gpu = hardware(device)
                    if reports and gpu != next(iter(reports.values()))[0]["hardware"]:
                        raise ValueError("GPU identity changed between measurements")
                    directory = attempts / f"attempt-{len(list(attempts.glob('attempt-*')))+1:03d}"
                    launch({"operation": "serving_use", "device": device, "hardware": gpu,
                            "worker_python": snap["dense_worker_python"] if backend == "dense" else snap["worker_python"],
                            "snapshot": snap, "backend": backend, "config_path": str(resolve(config_path)),
                            "image": image, "prompt": prompt}, directory)
                    reports[backend] = complete_output(directory, snap, backend)
            if snapshot(config_path, benchmark=benchmark, image=image, prompt=prompt) != snap:
                raise ValueError("Serving inputs changed during execution")
            summary = {"status": "completed", "mode": snap["mode"], "candidate_id": snap["candidate_id"],
                       "freeze": snap["config"]["freeze"], "freeze_digest": snap["freeze_digest"]}
            if benchmark:
                summary["comparison"] = compare(reports["dense"], reports["serving"])
                quality = read(resolve(snap["config"]["freeze"]))["lmms"]
                summary["acceptance"] = acceptance_outcome(
                    quality["retention"], summary["comparison"]["end_to_end_speedup"],
                    quality["minimum_retention"])
            else:
                summary["answer"] = reports["serving"][1][0]
            save_json(root / "summary.json", summary)
            state.update(status="completed", active_backend=None)
            return summary
        except BaseException as error:
            state.update(status="blocked", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save_json(root / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving.json")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--prompt")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    options = dict(benchmark=args.benchmark, image=args.image, prompt=args.prompt)
    if args.execute:
        result = execute(args.config, **options)
    else:
        snap = snapshot(args.config, **options)
        result = {"status": "ready", "mode": snap["mode"], "candidate_id": snap["candidate_id"],
                  "request_count": len(snap["requests"]), "repeats": snap["repeats"],
                  "timing_scope": TIMING_SCOPE, "model": snap["model"],
                  "output": snap["config"]["output"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
