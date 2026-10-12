"""Native DeepSeek prompts, generation semantics and matched serving timing."""

import hashlib
import json
from math import ceil
from pathlib import Path
from time import perf_counter

from search.lmms_internvl_vllm_entry import InternVLVllmBackend


def sampling_options(options, eos, until):
    allowed = {"max_new_tokens", "do_sample", "num_beams", "use_cache", "return_dict_in_generate",
               "temperature", "top_p", "top_k", "repetition_penalty", "pad_token_id", "eos_token_id"}
    if (set(options) - allowed or options.get("do_sample", False)
            or options.get("num_beams", 1) != 1 or not options.get("use_cache", True)
            or options.get("return_dict_in_generate", False)):
        raise ValueError("DeepSeek serving requires supported greedy cached single-beam options")
    if (type(options.get("max_new_tokens")) is not int or options["max_new_tokens"] < 1
            or not isinstance(until, list) or any(not isinstance(s, str) for s in until)):
        raise ValueError("DeepSeek serving requires a positive output budget and native stop strings")
    stop = options.get("eos_token_id", eos)
    stop_ids = [] if stop is None else list(stop) if isinstance(stop, (list, tuple)) else [int(stop)]
    # EOS is disabled only when explicitly disabled by the native request.
    return {"temperature": 0., "max_tokens": options["max_new_tokens"],
            "stop_token_ids": stop_ids, "ignore_eos": stop is None,
            "stop": [s for s in until if s], "include_stop_str_in_output": True,
            "repetition_penalty": options.get("repetition_penalty", 1.),
            "skip_special_tokens": True, "spaces_between_special_tokens": False}


def generation_receipt(native_ids, image_token_id, output, traces, ratio, selection, side, crop, index):
    from search.lmms_vllm_entry import compact_prompt
    from skills.deepseek_vl2_token import image_layout
    from skills.deepseek_vl2_vllm import image_keep_count
    if len(traces) != 1:
        raise RuntimeError("DeepSeek must observe selection for exactly one native image")
    nx, ny = crop
    tiles, protected, before = image_layout(side, nx, ny)
    after = image_keep_count(side, nx, ny, {"keep_ratio": ratio, "selection": selection})
    image = traces[0]
    expected_values = {"position_type": "1d_compacted", "token_keep_ratio": ratio,
                       "image_tokens_before": before, "image_tokens_after": after,
                       "positions_after": after, "tiles_selected": len(tiles),
                       "protected_layout_tokens": len(protected)}
    if any(image.get(k) != v for k, v in expected_values.items()) or not image.get("selected_indices_sha256"):
        raise RuntimeError("DeepSeek visual selection differs from its native global/local layout")
    expected = compact_prompt(native_ids, image_token_id, [(before, after)])
    if list(output.prompt_token_ids) != expected or len(output.outputs) != 1:
        raise RuntimeError("DeepSeek vLLM prompt differs from native LMMS after compaction")
    generated = list(output.outputs[0].token_ids)
    return generated, {"backend": "vllm", "controller": "vllm_score_based_post_encoder",
        "token_keep_ratio": ratio, "sample_index": index,
        "image_tokens_before": before, "image_tokens_after": after, "images": traces,
        "native_prompt_tokens": len(native_ids), "decoder_tokens_after": len(expected),
        "native_prompt_sha256": hashlib.sha256(json.dumps(native_ids).encode()).hexdigest(),
        "prompt_verified": True, "output_tokens": len(generated), "kv_length_observed": False}


def validate_runtime_observation(header, observed, token_runtime, kv_dtype):
    from skills.deepseek_vl2_vllm import layer_widths
    widths = [{"layer": i, **row} for i, row in enumerate(layer_widths(header))]
    grid = header["vision_config"]["image_size"] // header["vision_config"]["patch_size"]
    enabled = bool(token_runtime and token_runtime["keep_ratio"] < 1)
    if (observed.get("moe_layer_widths") != widths or observed.get("vision_patch_grid") != [grid, grid]
            or observed.get("router_logits_dtype") != "float32" or observed.get("moe_backends") != ["MARLIN"]
            or observed.get("physical_widths") != header.get("a4fvlm_stack_widths")
            or observed.get("token_runtime") != token_runtime
            or observed.get("token_pruning_enabled") is not enabled
            or observed.get("kv_cache_dtypes") != [kv_dtype] or not observed.get("attention_backends")):
        raise ValueError("DeepSeek serving changed native vision, routing, Token/KV or physical expert widths")


class DeepSeekVllmBackend(InternVLVllmBackend):
    model_name = "DeepSeek-VL2"
    native_prompt_adapter = "search.deepseek_lmms.DeepSeekVL2LMMS"

    def validate_observed(self, observed, adapter, effective):
        super().validate_observed(observed, adapter, effective)
        from adapters.deepseek_vl2_dense import PROJECT_ROOT
        header = json.loads((PROJECT_ROOT / self.descriptor["path"] / "config.json").read_text())
        validate_runtime_observation(header, observed,
            {"keep_ratio": self.descriptor["token_keep_ratio"], "selection": self.descriptor["token_selection"]},
            self.descriptor["runtime"].get("kv_cache_dtype", "auto"))

    def audit(self, output, inputs, processor):
        with self.trace.open() as stream:
            stream.seek(self.trace_offset)
            traces = [json.loads(line) for line in stream.read().splitlines() if line.strip()]
            self.trace_offset = stream.tell()
        native = processor.native
        side = ceil((native.image_size // native.patch_size) / native.downsample_ratio)
        generated, receipt = generation_receipt(inputs["input_ids"][0].tolist(), native.image_token_id,
            output, traces, self.descriptor["token_keep_ratio"], self.descriptor["token_selection"], side,
            [int(v) for v in inputs["images_spatial_crop"][0, 0].tolist()], self.index)
        with self.receipt.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt) + "\n")
        self.index += 1
        return generated, receipt

    def generate(self, image, question, inputs, processor, sampling):
        output = self.infer(image, processor.unexpanded_prompt(question), sampling)
        return self.audit(output, inputs, processor)


class DeepSeekSession:
    def __init__(self, snap, backend, output):
        import torch
        from adapters.deepseek_vl2_dense import load_model, load_processor
        from search.serving import read, resolve, save_json
        self.snap, self.backend = snap, backend
        descriptor = snap["dense_model"] if backend == "dense" else snap["model"]
        self.max_length = read(resolve(descriptor["path"]) / "config.json")["language_config"]["max_position_embeddings"]
        if backend == "dense":
            self.model, self.processor = load_model({**snap["dense_model"], "device": "cuda:0"})
            self.runtime = {"backend": "transformers", "model": snap["dense_model"]}
        else:
            self.processor = load_processor(snap["model"])
            self.native = DeepSeekVllmBackend(snap["model"], Path(output) / "token-mechanism.jsonl")
            self.runtime = self.native.runtime
            qualified = read(resolve(snap["config"]["freeze"]).parent.parent /
                             "candidates" / snap["candidate_id"] / "lmms/chartqa-token-mechanism.runtime.json")
            if self.runtime["observed_model"]["attention_backends"] != qualified["observed_model"]["attention_backends"]:
                raise ValueError("Actual DeepSeek attention backend differs from qualified LMMS")
        if (not snap["generation"]["ignore_eos"]
                and snap["generation"]["stop_token_ids"] != [self.processor.tokenizer.eos_token_id]):
            raise ValueError("DeepSeek serving must preserve its native tokenizer EOS")
        save_json(Path(output) / "runtime.json", self.runtime)
        torch.cuda.synchronize()

    def generate(self, request):
        import torch
        from adapters.deepseek_vl2_dense import prepare_image
        from search.serving import open_native_image, digest
        if self.backend != "dense" and not self.native.engine.reset_prefix_cache():
            raise RuntimeError("Could not clear DeepSeek cross-request prefix cache")
        torch.cuda.synchronize()
        started = perf_counter()
        image = open_native_image(request["image"])
        try:
            options = self.snap["generation"]
            if self.backend == "dense":
                inputs = prepare_image(self.processor, image, request["prompt"])
                native_ids = inputs["input_ids"][0].tolist()
                with torch.inference_mode():
                    result = self.model.generate(**inputs.to("cuda:0"),
                        max_new_tokens=options["max_tokens"], do_sample=False, use_cache=True,
                        eos_token_id=None if options["ignore_eos"] else options["stop_token_ids"],
                        forced_eos_token_id=None, repetition_penalty=1., return_dict_in_generate=False)
                ids = result[0].tolist()
            else:
                prompt = self.processor.unexpanded_prompt(request["prompt"])
                expected_ids = self.processor.prompt_token_ids(request["prompt"], image.size)
                if len(expected_ids) + options["max_tokens"] > self.max_length:
                    raise ValueError("Original DeepSeek context exceeds capacity; no truncation")
                result = self.native.infer(image, prompt, {**options, "detokenize": False})
                ids = list(result.outputs[0].token_ids)
            text = self.processor.batch_decode([ids], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            if self.backend == "dense":
                mechanism = None
            else:
                inputs = prepare_image(self.processor, image, request["prompt"])
                native_ids = inputs["input_ids"][0].tolist()
                if native_ids != expected_ids:
                    raise ValueError("DeepSeek serving context preflight differs from native preprocessing")
                _, receipt = self.native.audit(result, inputs, self.processor)
                mechanism = {"prompt_verified": receipt["prompt_verified"], "images": receipt["images"],
                             "decoder_tokens": receipt["decoder_tokens_after"]}
            return {"sample_id": request["sample_id"], "output_text": text, "output_tokens": len(ids),
                    "end_to_end_seconds": elapsed, "native_prompt_sha256": digest(native_ids),
                    "native_prompt_tokens": len(native_ids), "token_mechanism": mechanism}
        finally:
            image.close()
