"""Bound native single-image LMMS context from retained Dense inputs, on CPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from search.lmms_gate import DEFAULT_TASKS, _sample_file, _task_result
from search.model import PROJECT_ROOT


def probe(model_path, dense_root, model_len):
    from transformers import AutoProcessor

    model_path, dense_root = Path(model_path), Path(dense_root)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    if processor.__class__.__name__ == "LlavaProcessor":
        import transformers.models.llava.processing_llava as processing

        config = json.loads((model_path / "config.json").read_text())
        from adapters.llava15 import Llava15Adapter

        Llava15Adapter().validate_checkpoint(config)
        if (processor.patch_size != 14 or processor.vision_feature_select_strategy != "default"
                or processor.image_processor.crop_size != {"height": 336, "width": 336}):
            raise ValueError("LLaVA-1.5 processor geometry differs from native CLIP")
        visual_bound = 576
        scope = "Native zero-shot single-image LMMS; exact unpruned fixed-resolution visual count"
    elif processor.__class__.__name__ == "LlavaOnevisionProcessor":
        import transformers.models.llava_onevision.processing_llava_onevision as processing

        if processor.vision_aspect_ratio != "anyres_max_9":
            raise ValueError("This bound requires the native anyres_max_9 feature packing")
        # Account for the native downsampling deadband and newline positions.
        patch_tokens = int(processor.num_image_tokens)
        patch_side = math.isqrt(patch_tokens)
        if patch_side ** 2 != patch_tokens:
            raise ValueError("Expected a square SigLIP feature grid")
        image_size = processor.image_processor.size["height"]
        grid_height = max(h // image_size for h, _ in processor.image_processor.image_grid_pinpoints)
        visual_bound = patch_tokens + math.ceil(1.1 ** 2 * 9 * patch_tokens) + patch_side * grid_height
        scope = "Native zero-shot single-image LMMS; conservative unpruned any-resolution bound"
    else:
        raise ValueError("Expected an implemented native LLaVA processor")
    tasks = []
    for spec in DEFAULT_TASKS:
        task = spec["task"]
        result_path, result = _task_result(dense_root, task)
        samples_path = _sample_file(result_path, task)
        samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
        config = result["configs"][task]
        if config.get("num_fewshot", 0) != 0 or config.get("description") or config.get("doc_to_messages"):
            raise ValueError("Context bound requires zero-shot single-user native prompts")
        if len(samples) != result["n-samples"][task]["effective"]:
            raise ValueError("Incomplete Dense input evidence")
        max_text = 0
        for sample in samples:
            text = sample["input"]
            if not isinstance(text, str):
                raise ValueError("Missing retained native prompt text")
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text}]}],
                tokenize=False, add_generation_prompt=True)
            ids = processor.tokenizer(prompt)["input_ids"]
            if ids.count(processor.image_token_id) != 1:
                raise ValueError("Expected one native image placeholder")
            max_text = max(max_text, len(ids) - 1)
        generation = config.get("generation_kwargs", {}).get("max_new_tokens", 1024)
        required = max_text + visual_bound + generation
        tasks.append({"task": task, "samples": len(samples), "maximum_text_tokens": max_text,
                      "visual_token_upper_bound": visual_bound, "generation_tokens": generation,
                      "required_model_len_upper_bound": required,
                      "sample_file": str(samples_path),
                      "sample_file_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest()})
    required = max(row["required_model_len_upper_bound"] for row in tasks)
    return {"status": "passed" if required <= model_len else "blocked_preflight",
            "model": str(model_path), "model_len": model_len,
            "required_model_len_upper_bound": required, "tasks": tasks,
            "scope": scope,
            "model_metadata_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(model_path.iterdir())
                                      if path.suffix in {".json", ".jinja"} and path.is_file()},
            "processor_source_sha256": hashlib.sha256(Path(processing.__file__).read_bytes()).hexdigest(),
            "processor_config": processor.to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/LLaVA-OneVision-7B")
    parser.add_argument("--dense-root", default="results/model/llava_onevision/cuda0/lmms/dense")
    parser.add_argument("--model-len", type=int, default=32768)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = probe(PROJECT_ROOT / args.model, PROJECT_ROOT / args.dense_root, args.model_len)
    if args.output:
        destination = PROJECT_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "processor_config"}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
