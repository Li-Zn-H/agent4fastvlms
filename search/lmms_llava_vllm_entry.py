"""Native LLaVA LMMS prompts/preprocessing/scoring with verified vLLM generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.lmms_vllm_entry import compact_prompt, runtime_receipt, sampling_kwargs
from skills.stack_runtime import build_vllm_runtime_kwargs


class CapturingLlavaProcessor:
    def __init__(self, processor, model):
        self.processor, self.model = processor, model

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        if args or kwargs.get("videos") is not None:
            raise ValueError("LLaVA serving bridge supports keyword still-image inputs")
        images = kwargs.get("images")
        texts = kwargs.get("text")
        texts = [texts] if isinstance(texts, str) else texts
        if not images or len(images) != 1 or not texts or len(texts) != 1:
            raise ValueError("LLaVA serving bridge requires one native image/prompt")
        if kwargs.get("do_resize") is not None:
            raise ValueError("LLaVA serving bridge must retain native image preprocessing")
        inputs = self.processor(**kwargs)
        self.model.captured = (texts[0], images)
        return inputs


def install_bridge(descriptor):
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from lmms_eval.models.simple import llava_hf as simple
    from skills.stack_vllm import prepare_vllm_artifact

    receipt_path = Path(os.environ["A4FVLMSP_TOKEN_RECEIPT"])
    trace_path = receipt_path.with_suffix(".selection.jsonl")
    model_receipt_path = receipt_path.with_suffix(".model.json")
    model_receipt_path.unlink(missing_ok=True)
    os.environ["A4FVLM_VLLM_SELECTION_TRACE"] = str(trace_path)
    os.environ["A4FVLM_VLLM_MODEL_RECEIPT"] = str(model_receipt_path)
    runtime = descriptor["runtime"]

    class VllmLlavaModel(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, pretrained, **kwargs):
            if Path(pretrained).resolve() != Path(descriptor["path"]).resolve():
                raise ValueError("LMMS artifact differs from serving descriptor")
            return cls(pretrained)

        @property
        def dtype(self):
            return getattr(torch, descriptor["dtype"])

        def __init__(self, pretrained):
            super().__init__()
            self.config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
            adapter = prepare_vllm_artifact(pretrained, token_runtime={
                "keep_ratio": descriptor["token_keep_ratio"], "selection": descriptor["token_selection"]})
            if not adapter.get("custom_stack"):
                raise ValueError("LLaVA final LMMS requires the physical Stack artifact")
            self.engine = LLM(model=pretrained, dtype=descriptor["dtype"], trust_remote_code=True,
                model_class_overrides={adapter["architecture"]: "skills.stack_vllm:" + adapter["architecture"]},
                generation_config="vllm", **build_vllm_runtime_kwargs(runtime))
            effective = runtime_receipt(self.engine.llm_engine.vllm_config, runtime)
            observed = json.loads(model_receipt_path.read_text())
            if (not observed["token_pruning_enabled"]
                    or observed["token_runtime"]["keep_ratio"] != descriptor["token_keep_ratio"]
                    or observed["kv_cache_dtypes"] != [runtime.get("kv_cache_dtype", "auto")]
                    or effective["max_model_len"] != runtime["vllm_max_model_len"]):
                raise ValueError("Observed LLaVA Runtime/KV/Token settings differ from descriptor")
            receipt_path.with_suffix(".runtime.json").write_text(json.dumps({
                "backend": "vllm", "model": descriptor, "effective": effective,
                "observed_model": observed, "adapter": adapter,
                "native_prompt_adapter": "lmms_eval.models.chat.llava_hf", "batch_size": 1,
            }, indent=2) + "\n")
            model_receipt_path.unlink()
            trace_path.write_text("")
            self.trace_offset, self.sample_index, self.captured = 0, 0, None

        def generate(self, input_ids, **kwargs):
            if input_ids.shape[0] != 1 or self.captured is None:
                raise ValueError("Missing native LLaVA preprocessing capture")
            text, images = self.captured
            self.captured = None
            outputs = self.engine.generate([{
                "prompt": text, "multi_modal_data": {"image": images},
                "multi_modal_uuids": {"image": [f"lmms-{self.sample_index}-0"]},
            }], SamplingParams(**sampling_kwargs(kwargs)), use_tqdm=False)
            with trace_path.open() as handle:
                handle.seek(self.trace_offset)
                traces = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
                self.trace_offset = handle.tell()
            if len(traces) != 1 or traces[0].get("position_type") != "1d_compacted":
                raise RuntimeError("LLaVA selector did not produce matching image evidence")
            before, after = traces[0]["image_tokens_before"], traces[0]["image_tokens_after"]
            native_ids = input_ids[0].tolist()
            expected = compact_prompt(native_ids, self.config.image_token_index, [(before, after)])
            output = outputs[0]
            if list(output.prompt_token_ids) != expected:
                raise RuntimeError("vLLM LLaVA prompt differs from native LMMS after compaction")
            generated = list(output.outputs[0].token_ids)
            with receipt_path.open("a") as handle:
                handle.write(json.dumps({
                    "backend": "vllm", "controller": "vllm_score_based_post_encoder",
                    "token_keep_ratio": descriptor["token_keep_ratio"], "sample_index": self.sample_index,
                    "image_tokens_before": before, "image_tokens_after": after, "images": traces,
                    "native_prompt_tokens": len(native_ids), "decoder_tokens_after": len(expected),
                    "native_prompt_sha256": hashlib.sha256(json.dumps(native_ids).encode()).hexdigest(),
                    "prompt_verified": True, "output_tokens": len(generated), "kv_length_observed": False,
                }) + "\n")
            self.sample_index += 1
            return torch.tensor([native_ids + generated], device=input_ids.device)

    adapter = descriptor.get("adapter")
    if adapter not in {"llava_onevision", "llava15"}:
        raise ValueError("Unsupported LLaVA serving adapter")
    simple.model_map["llava" if adapter == "llava15" else "llava_onevision"] = VllmLlavaModel
    original_init = simple.LlavaHf.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.batch_size != 1 or self.world_size != 1:
            raise ValueError("LLaVA serving LMMS requires single-device batch size 1")
        self._image_processor = CapturingLlavaProcessor(self._image_processor, self.model)

    simple.LlavaHf.__init__ = initialize


def main():
    install_bridge(json.loads(os.environ["A4FVLM_LMMS_VLLM_MODEL"]))
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()
