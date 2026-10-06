"""Keep native LMMS Qwen3-VL prompts/scoring; delegate generation to vLLM.

The native chat adapter still builds messages, resizes images, tokenizes, trims
and decodes answers. Only its model loader and generate call are bridged. This
module must stay importable without torch/vLLM for CPU contract tests.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skills.stack_runtime import build_vllm_runtime_kwargs


def sampling_kwargs(generate):
    if generate.get("num_beams", 1) != 1 or generate.get("do_sample", False):
        raise ValueError("Serving LMMS currently requires greedy single-beam generation")
    eos = generate["eos_token_id"]
    return {"temperature": 0.0, "max_tokens": int(generate["max_new_tokens"]),
            "stop_token_ids": list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]}


def compact_prompt(ids, image_token_id, image_counts):
    """Expected vLLM placeholders, preserving every native text/control token."""
    result, index, image = [], 0, 0
    while index < len(ids):
        if ids[index] != image_token_id:
            result.append(ids[index])
            index += 1
            continue
        end = index
        while end < len(ids) and ids[end] == image_token_id:
            end += 1
        if image >= len(image_counts) or end - index != image_counts[image][0]:
            raise ValueError("Native image placeholders disagree with actual visual features")
        result.extend([image_token_id] * image_counts[image][1])
        image += 1
        index = end
    if image != len(image_counts):
        raise ValueError("Missing native image placeholders")
    return result


def runtime_receipt(config, runtime):
    """Read effective engine settings, fail if the requested knobs drifted."""
    compilation = config.compilation_config
    effective = {
        "optimization_level": int(config.optimization_level),
        "kv_cache_dtype": config.cache_config.cache_dtype,
        "attention_backend": str(config.attention_config.backend),
        "compile_mm_encoder": compilation.compile_mm_encoder,
        "cudagraph_mm_encoder": compilation.cudagraph_mm_encoder,
        "qk_norm_rope_fusion": compilation.pass_config.enable_qk_norm_rope_fusion,
        "compilation_mode": str(compilation.mode),
        "cudagraph_mode": str(compilation.cudagraph_mode),
        "max_model_len": config.model_config.max_model_len,
    }
    for key in ("optimization_level", "kv_cache_dtype", "compile_mm_encoder",
                "cudagraph_mm_encoder", "qk_norm_rope_fusion"):
        if key in runtime and effective[key] != runtime[key]:
            raise ValueError(f"Effective runtime mismatch: {key}")
    backend = runtime.get("attention_backend", "auto")
    if backend != "auto" and effective["attention_backend"].split(".")[-1] != backend:
        raise ValueError("Effective attention backend mismatch")
    return effective


class CapturingProcessor:
    def __init__(self, processor, model):
        self.processor, self.model = processor, model

    def __getattr__(self, key):
        return getattr(self.processor, key)

    def __call__(self, *args, **kwargs):
        if args or kwargs.get("videos") is not None:
            raise ValueError("Serving LMMS bridge supports keyword still-image inputs only")
        inputs = self.processor(**kwargs)
        texts = kwargs["text"]
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) != 1 or not kwargs.get("images") or kwargs.get("do_resize") is not False:
            raise ValueError("Serving LMMS requires native resized images and batch size 1")
        self.model.captured = (texts[0], kwargs["images"])
        return inputs


def install_bridge(descriptor):
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from lmms_eval.models.simple import qwen3_vl as simple
    from lmms_eval.models.chat import qwen3_vl as chat
    from skills.stack_vllm import prepare_vllm_artifact

    receipt_path = Path(os.environ["A4FVLMSP_TOKEN_RECEIPT"])
    trace_path = receipt_path.with_suffix(".selection.jsonl")
    os.environ["A4FVLM_VLLM_SELECTION_TRACE"] = str(trace_path)
    model_receipt_path = receipt_path.with_suffix(".model.json")
    model_receipt_path.unlink(missing_ok=True)
    os.environ["A4FVLM_VLLM_MODEL_RECEIPT"] = str(model_receipt_path)
    runtime = descriptor["runtime"]

    class VllmGenerationModel(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, pretrained, **kwargs):
            if Path(pretrained).resolve() != Path(descriptor["path"]).resolve():
                raise ValueError("LMMS artifact differs from serving descriptor")
            return cls(pretrained)

        def __init__(self, pretrained):
            super().__init__()
            self.config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
            adapter = prepare_vllm_artifact(pretrained, token_runtime={
                "keep_ratio": descriptor["token_keep_ratio"],
                "selection": descriptor["token_selection"],
            })
            if not adapter.get("custom_stack"):
                raise ValueError("Serving LMMS requires the physical Stack adapter")
            self.engine = LLM(
                model=pretrained, dtype="bfloat16", trust_remote_code=True,
                model_class_overrides={adapter["architecture"]: "skills.stack_vllm:" + adapter["architecture"]},
                video_pruning_rate=1e-9, generation_config="vllm",
                **build_vllm_runtime_kwargs(runtime),
            )
            effective = runtime_receipt(self.engine.llm_engine.vllm_config, runtime)
            observed = json.loads(model_receipt_path.read_text())
            if not observed["token_pruning_enabled"]:
                raise ValueError("Engine did not enable score-based Token pruning")
            if observed["token_runtime"]["keep_ratio"] != descriptor["token_keep_ratio"]:
                raise ValueError("Engine Token ratio mismatch")
            if observed["kv_cache_dtypes"] != [runtime.get("kv_cache_dtype", "auto")]:
                raise ValueError("Engine KV dtype mismatch")
            receipt_path.with_suffix(".runtime.json").write_text(json.dumps({
                "backend": "vllm", "model": descriptor,
                "effective": effective, "observed_model": observed, "adapter": adapter,
                "native_prompt_adapter": "lmms_eval.models.chat.qwen3_vl",
                "batch_size": 1,
            }, indent=2) + "\n")
            model_receipt_path.unlink()
            # Profiling/dummy images during engine startup are not task evidence.
            trace_path.write_text("")
            self.trace_offset = 0
            self.captured = None

        def generate(self, input_ids, **kwargs):
            if input_ids.shape[0] != 1 or self.captured is None:
                raise ValueError("Missing native preprocessing capture")
            text, images = self.captured
            self.captured = None
            sample_index = getattr(self, "sample_index", 0)
            # Unique media IDs force each audited image through the actual
            # selector, including repeated-image tasks such as POPE.
            outputs = self.engine.generate([{
                "prompt": text, "multi_modal_data": {"image": images},
                "multi_modal_uuids": {"image": [f"lmms-{sample_index}-{i}" for i in range(len(images))]},
                "mm_processor_kwargs": {"do_resize": False},
            }], SamplingParams(**sampling_kwargs(kwargs)), use_tqdm=False)
            self.sample_index = sample_index + 1
            with trace_path.open() as handle:
                handle.seek(self.trace_offset)
                traces = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
                self.trace_offset = handle.tell()
            if len(traces) != len(images):
                raise RuntimeError("vLLM did not execute the visual selector for every image")
            before = sum(row["image_tokens_before"] for row in traces)
            after = sum(row["image_tokens_after"] for row in traces)
            native_ids = input_ids[0].tolist()
            expected = compact_prompt(native_ids, self.config.image_token_id, [
                (row["image_tokens_before"], row["image_tokens_after"]) for row in traces
            ])
            output = outputs[0]
            if list(output.prompt_token_ids) != expected:
                raise RuntimeError("vLLM prompt differs from the native LMMS prompt after Token compaction")
            record = {
                "backend": "vllm", "controller": "vllm_score_based_post_encoder",
                "token_keep_ratio": descriptor["token_keep_ratio"],
                "sample_index": sample_index, "image_tokens_before": before,
                "image_tokens_after": after, "images": traces,
                "native_prompt_tokens": len(native_ids), "decoder_tokens_after": len(expected),
                "native_prompt_sha256": hashlib.sha256(json.dumps(native_ids).encode()).hexdigest(),
                "prompt_verified": True, "output_tokens": len(output.outputs[0].token_ids),
                "kv_length_observed": False,
            }
            with receipt_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            # Preserve the upstream adapter's trimming and tokenizer decoding.
            return torch.tensor([native_ids + list(output.outputs[0].token_ids)], device=input_ids.device)

    simple._resolve_model_class = lambda pretrained, is_moe: (VllmGenerationModel, "dtype")
    original_init = chat.Qwen3_VL.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.batch_size != 1 or self.world_size != 1:
            raise ValueError("Serving LMMS requires single-device batch size 1")
        self.processor = CapturingProcessor(self.processor, self.model)

    chat.Qwen3_VL.__init__ = initialize


def main():
    descriptor = json.loads(os.environ["A4FVLM_LMMS_VLLM_MODEL"])
    install_bridge(descriptor)
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()
