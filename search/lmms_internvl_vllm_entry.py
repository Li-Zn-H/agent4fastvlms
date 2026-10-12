"""Native InternVL LMMS prompts/pixels/scoring with verified vLLM generation."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def unexpanded_prompt(prompt, image_tokens):
    image = "<img>" + "<IMG_CONTEXT>" * image_tokens + "</img>"
    if prompt.count(image) != 1:
        raise ValueError("Native InternVL prompt must contain one complete image run")
    return prompt.replace(image, "<image>", 1)


def generation_receipt(native_ids, image_token_id, output, traces, ratio, index):
    from search.lmms_vllm_entry import compact_prompt

    if len(traces) != 1 or traces[0].get("position_type") != "1d_compacted":
        raise RuntimeError("InternVL vLLM must observe selection for one native image")
    image = traces[0]
    before, after = image["image_tokens_before"], image["image_tokens_after"]
    if (image.get("token_keep_ratio") != ratio or image.get("tiles_selected", 0) * 256 != before
            or image.get("positions_after") != after or not image.get("selected_indices_sha256")):
        raise RuntimeError("InternVL vLLM did not select every native tile")
    expected = compact_prompt(native_ids, image_token_id, [(before, after)])
    if list(output.prompt_token_ids) != expected:
        raise RuntimeError("InternVL vLLM prompt differs from native LMMS after compaction")
    generated = list(output.outputs[0].token_ids)
    return generated, {
        "backend": "vllm", "controller": "vllm_score_based_post_encoder",
        "token_keep_ratio": ratio, "sample_index": index,
        "image_tokens_before": before, "image_tokens_after": after, "images": traces,
        "native_prompt_tokens": len(native_ids), "decoder_tokens_after": len(expected),
        "native_prompt_sha256": hashlib.sha256(json.dumps(native_ids).encode()).hexdigest(),
        "prompt_verified": True, "output_tokens": len(generated), "kv_length_observed": False,
    }


class InternVLVllmBackend:
    model_name = "InternVL"
    native_prompt_adapter = "lmms_eval.models.simple.internvl3"

    def __init__(self, descriptor, receipt_path):
        from skills.stack_vllm import prepare_vllm_artifact, validate_stack_artifact
        from skills.stack_runtime import build_vllm_runtime_kwargs
        from search.lmms_vllm_entry import runtime_receipt

        artifact = (PROJECT_ROOT / descriptor["path"]).resolve()
        validate_stack_artifact(artifact)
        from vllm import LLM

        self.descriptor = descriptor
        self.receipt = Path(receipt_path)
        self.trace = self.receipt.with_suffix(".selection.jsonl")
        model_receipt = self.receipt.with_suffix(".model.json")
        model_receipt.unlink(missing_ok=True)
        os.environ["A4FVLM_VLLM_SELECTION_TRACE"] = str(self.trace)
        os.environ["A4FVLM_VLLM_MODEL_RECEIPT"] = str(model_receipt)
        runtime = descriptor["runtime"]
        adapter = prepare_vllm_artifact(artifact, token_runtime={
            "keep_ratio": descriptor["token_keep_ratio"], "selection": descriptor["token_selection"]})
        if not adapter.get("custom_stack"):
            raise ValueError(f"Final {self.model_name} serving requires its physical Stack artifact")
        self.engine = LLM(model=str(artifact),
                          dtype=descriptor["dtype"], trust_remote_code=True,
                          model_class_overrides=adapter["model_class_overrides"],
                          generation_config="vllm", **build_vllm_runtime_kwargs(runtime))
        effective = runtime_receipt(self.engine.llm_engine.vllm_config, runtime)
        observed = json.loads(model_receipt.read_text())
        self.validate_observed(observed, adapter, effective)
        self.runtime = {"backend": "vllm", "model": descriptor, "effective": effective,
                        "observed_model": observed, "adapter": adapter,
                        "native_prompt_adapter": self.native_prompt_adapter, "batch_size": 1}
        self.receipt.with_suffix(".runtime.json").write_text(json.dumps(self.runtime, indent=2) + "\n")
        model_receipt.unlink()
        self.trace.write_text("")
        self.trace_offset, self.index = 0, 0

    def validate_observed(self, observed, adapter, effective):
        descriptor = self.descriptor
        runtime = descriptor["runtime"]
        if (observed.get("token_pruning_enabled") is not True
                or observed.get("token_runtime", {}).get("keep_ratio") != descriptor["token_keep_ratio"]
                or observed.get("physical_widths") != adapter["widths"]
                or observed.get("kv_cache_dtypes") != [runtime.get("kv_cache_dtype", "auto")]
                or effective["max_model_len"] != runtime["vllm_max_model_len"]):
            raise ValueError(f"Actual {self.model_name} Runtime/KV/Token differs from its serving descriptor")

    def infer(self, image, prompt, sampling):
        from vllm import SamplingParams

        # The five native tasks supply decoded RGB rasters without EXIF
        # rotation. Preserve those pixels before vLLM normalizes PIL images.
        native_image = image.convert("RGB")
        exif = native_image.getexif()
        if 274 in exif:
            del exif[274]
            native_image.info["exif"] = exif.tobytes()
        try:
            return self.engine.generate([{
                "prompt": prompt, "multi_modal_data": {"image": native_image},
                "multi_modal_uuids": {"image": f"{self.model_name}-native-{self.index}"},
            }], SamplingParams(**sampling), use_tqdm=False)[0]
        finally:
            native_image.close()

    def audit(self, output, native_ids, image_token_id):
        with self.trace.open() as stream:
            stream.seek(self.trace_offset)
            traces = [json.loads(line) for line in stream.read().splitlines() if line.strip()]
            self.trace_offset = stream.tell()
        generated, receipt = generation_receipt(native_ids, image_token_id, output, traces,
                                               self.descriptor["token_keep_ratio"], self.index)
        with self.receipt.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt) + "\n")
        self.index += 1
        return generated, receipt

    def generate(self, image, native_prompt, native_ids, image_token_id, sampling):
        prompt = unexpanded_prompt(native_prompt, native_ids.count(image_token_id))
        output = self.infer(image, prompt, sampling)
        return self.audit(output, native_ids, image_token_id)


def install_bridge(descriptor):
    import torch
    from lmms_eval.models.simple import internvl3
    from adapters.internvl35_dense import load_processor, native_model_class
    from search.lmms_vllm_entry import sampling_kwargs

    captured_images = []
    original_load_image = internvl3.load_image
    def load_image(image, *args, **kwargs):
        result = original_load_image(image, *args, **kwargs)
        captured_images.append(image)
        return result
    internvl3.load_image = load_image

    class CapturingTokenizer:
        def __init__(self, tokenizer, model):
            self.tokenizer, self.model = tokenizer, model

        def __getattr__(self, name):
            return getattr(self.tokenizer, name)

        def __call__(self, text, *args, **kwargs):
            self.model.native_prompt = text
            return self.tokenizer(text, *args, **kwargs)

    class Model(torch.nn.Module):
        def __init__(self, pretrained, device):
            super().__init__()
            processor = load_processor(descriptor, artifact=True)
            self.config = processor.config
            self.template = processor.config.template
            self.system_message = processor.conversation_factory(self.template).system_message
            self.num_image_token = processor.tokens_per_tile
            self.device = torch.device(device)
            self.native_chat = native_model_class(pretrained).chat
            self.backend = InternVLVllmBackend(descriptor, os.environ["A4FVLMSP_TOKEN_RECEIPT"])
            self.native_prompt = None

        def eval(self):
            return self

        def to(self, device):
            if torch.device(device) != self.device:
                raise ValueError("InternVL LMMS cannot move the serving engine between devices")
            return self

        def chat(self, tokenizer, *args, **kwargs):
            return self.native_chat(self, CapturingTokenizer(tokenizer, self), *args, **kwargs)

        def generate(self, *, input_ids, pixel_values, attention_mask, **options):
            if input_ids.shape[0] != 1 or len(captured_images) != 1 or self.native_prompt is None:
                raise ValueError("Missing native InternVL image/prompt capture")
            image = captured_images.pop()
            generated, _ = self.backend.generate(image, self.native_prompt, input_ids[0].tolist(),
                                                 self.img_context_token_id, sampling_kwargs(options))
            self.native_prompt = None
            return torch.tensor([generated], device=input_ids.device, dtype=torch.long)

    def load(pretrained, **kwargs):
        if Path(pretrained).resolve() != (PROJECT_ROOT / descriptor["path"]).resolve():
            raise ValueError("InternVL LMMS artifact differs from its serving descriptor")
        device = kwargs.get("device_map")
        if not isinstance(device, str) or not device.startswith("cuda:"):
            raise ValueError("InternVL serving LMMS requires one explicit CUDA device")
        return Model(pretrained, device)
    internvl3.AutoModel = SimpleNamespace(from_pretrained=load)


def main():
    install_bridge(json.loads(os.environ["A4FVLM_LMMS_VLLM_MODEL"]))
    runpy.run_module("lmms_eval", run_name="__main__")


if __name__ == "__main__":
    main()
