"""Native Kimi serving prompts, physical mechanism receipts and matched timing."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

from search.lmms_internvl_vllm_entry import InternVLVllmBackend


def sampling_options(options, eos, until):
    # Both native bridges support the same greedy cached generation contract.
    from search.deepseek_serving import sampling_options as convert
    try:
        return convert(options, eos, until)
    except ValueError as error:
        raise ValueError(str(error).replace("DeepSeek", "Kimi")) from error


def generation_receipt(native_ids, image_token_id, output, traces, runtime, grid, merge, index):
    from search.lmms_vllm_entry import compact_prompt
    from skills.kimi_vl_vllm import retained_count
    ph, pw = grid
    mh, mw = merge
    if ph < 1 or pw < 1 or ph % mh or pw % mw or len(traces) != 1:
        raise RuntimeError("Kimi must observe selection for exactly one merge-aligned native image")
    rows, columns = ph // mh, pw // mw
    before, after = rows * columns, retained_count(rows, columns, runtime)
    image = traces[0]
    expected_values = {"method": "vllm_score_based_post_encoder", "position_type": "1d_compacted",
                       "token_keep_ratio": runtime["keep_ratio"], "patch_grid": [ph, pw],
                       "merged_grid": [rows, columns], "image_tokens_before": before,
                       "image_tokens_after": after, "positions_after": after}
    if any(image.get(k) != v for k, v in expected_values.items()) or not image.get("selected_indices_sha256"):
        raise RuntimeError("Kimi visual selection differs from its native post-projector grid")
    expected = compact_prompt(native_ids, image_token_id, [(before, after)])
    if list(output.prompt_token_ids) != expected or len(output.outputs) != 1:
        raise RuntimeError("Kimi vLLM prompt differs from native LMMS after compaction")
    generated = list(output.outputs[0].token_ids)
    return generated, {"backend": "vllm", "controller": "vllm_score_based_post_encoder",
        "token_keep_ratio": runtime["keep_ratio"], "sample_index": index,
        "image_tokens_before": before, "image_tokens_after": after, "images": traces,
        "native_prompt_tokens": len(native_ids), "decoder_tokens_after": len(expected),
        "native_prompt_sha256": hashlib.sha256(json.dumps(native_ids).encode()).hexdigest(),
        "prompt_verified": True, "output_tokens": len(generated), "kv_length_observed": False}


def validate_runtime_observation(header, observed, token_runtime, kv_dtype):
    from skills.kimi_vl_vllm import layer_widths
    widths = [{"layer": i, **row} for i, row in enumerate(layer_widths(header))]
    padding = header["a4fvlm_kimi_gptq"]["vision_padding"]
    if (observed.get("moe_layer_widths") != widths
            or observed.get("physical_widths") != header.get("a4fvlm_stack_widths")
            or observed.get("router_logits_dtype") != "float32"
            or observed.get("router_parameter_dtype") != "bfloat16"
            or observed.get("native_routing") != "noaux_tc_sigmoid_zero_mask"
            or observed.get("verified_router_tensors") != 2 * sum("routed" in row for row in widths)
            or observed.get("expert_topology") != {"routed": header["text_config"]["n_routed_experts"],
                                                   "shared": header["text_config"]["n_shared_experts"],
                                                   "top_k": header["text_config"]["num_experts_per_tok"]}
            or observed.get("moe_backends") != ["MARLIN"]
            or observed.get("vision_padding") != {"original_width": padding["original_width"],
                                                   "padded_width": padding["padded_width"]}
            or observed.get("token_runtime") != token_runtime
            or observed.get("token_pruning_enabled") is not bool(token_runtime and token_runtime["keep_ratio"] < 1)
            or observed.get("kv_cache_dtypes") != [kv_dtype] or not observed.get("attention_backends")):
        raise ValueError("Kimi serving changed native routing, physical FFNs, vision padding or Token/KV")


class KimiVllmBackend(InternVLVllmBackend):
    model_name = "Kimi-VL"
    native_prompt_adapter = "search.kimi_lmms.KimiVLLMMS"

    def validate_observed(self, observed, adapter, effective):
        super().validate_observed(observed, adapter, effective)
        from adapters.kimi_vl_dense import PROJECT_ROOT
        header = json.loads((PROJECT_ROOT / self.descriptor["path"] / "config.json").read_text())
        validate_runtime_observation(header, observed,
            {"keep_ratio": self.descriptor["token_keep_ratio"], "selection": self.descriptor["token_selection"]},
            self.descriptor["runtime"].get("kv_cache_dtype", "auto"))

    def audit(self, output, inputs, processor):
        with self.trace.open() as stream:
            stream.seek(self.trace_offset)
            traces = [json.loads(line) for line in stream.read().splitlines() if line.strip()]
            self.trace_offset = stream.tell()
        generated, receipt = generation_receipt(inputs["input_ids"][0].tolist(),
            processor.tokenizer.convert_tokens_to_ids(processor.image_token), output, traces,
            {"keep_ratio": self.descriptor["token_keep_ratio"], "selection": self.descriptor["token_selection"]},
            inputs["image_grid_hws"][0].tolist(), processor.image_processor.merge_kernel_size, self.index)
        with self.receipt.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt) + "\n")
        self.index += 1
        return generated, receipt

    def generate(self, image, question, inputs, processor, sampling):
        from adapters.kimi_vl_dense import unexpanded_prompt
        output = self.infer(image, unexpanded_prompt(processor, question), sampling)
        return self.audit(output, inputs, processor)


class KimiSession:
    def __init__(self, snap, backend, output):
        import torch
        from adapters.kimi_vl_dense import load_model, load_processor
        from search.serving import read, resolve, save_json
        self.snap, self.backend = snap, backend
        descriptor = snap["dense_model"] if backend == "dense" else snap["model"]
        self.max_length = read(resolve(descriptor["path"]) / "config.json")["text_config"]["max_position_embeddings"]
        if backend == "dense":
            self.model, self.processor = load_model({**snap["dense_model"], "device": "cuda:0"})
            self.runtime = {"backend": "transformers", "model": snap["dense_model"]}
        else:
            self.processor = load_processor(snap["model"])
            self.native = KimiVllmBackend(snap["model"], Path(output) / "token-mechanism.jsonl")
            self.runtime = self.native.runtime
            qualified = read(resolve(snap["config"]["freeze"]).parent.parent /
                             "candidates" / snap["candidate_id"] / "lmms/chartqa-token-mechanism.runtime.json")
            if self.runtime["observed_model"]["attention_backends"] != qualified["observed_model"]["attention_backends"]:
                raise ValueError("Actual Kimi attention backend differs from accepted LMMS")
        if (not snap["generation"]["ignore_eos"]
                and snap["generation"]["stop_token_ids"] != [self.processor.tokenizer.eos_token_id]):
            raise ValueError("Kimi serving must preserve its native tokenizer EOS")
        save_json(Path(output) / "runtime.json", self.runtime)
        torch.cuda.synchronize()

    def generate(self, request):
        import torch
        from adapters.kimi_vl_dense import prepare_image, prompt_token_ids, unexpanded_prompt
        from search.serving import open_native_image, digest
        if self.backend != "dense" and not self.native.engine.reset_prefix_cache():
            raise RuntimeError("Could not clear Kimi cross-request prefix cache")
        torch.cuda.synchronize()
        started = perf_counter()
        image = open_native_image(request["image"])
        try:
            options = self.snap["generation"]
            if self.backend == "dense":
                inputs = prepare_image(self.processor, image, request["prompt"])
                native_ids = inputs["input_ids"][0].tolist()
                if len(native_ids) + options["max_tokens"] > self.max_length:
                    raise ValueError("Original Kimi context exceeds capacity; no truncation")
                with torch.inference_mode():
                    result = self.model.generate(**inputs.to("cuda:0"),
                        max_new_tokens=options["max_tokens"], do_sample=False, use_cache=True,
                        eos_token_id=None if options["ignore_eos"] else options["stop_token_ids"],
                        forced_eos_token_id=None, repetition_penalty=options["repetition_penalty"],
                        return_dict_in_generate=False)
                # Original native Transformers returns the complete prompt plus suffix.
                ids = result[0, len(native_ids):].tolist()
            else:
                expected_ids = prompt_token_ids(self.processor, request["prompt"], image.size)
                if len(expected_ids) + options["max_tokens"] > self.max_length:
                    raise ValueError("Original Kimi context exceeds capacity; no truncation")
                result = self.native.infer(image, unexpanded_prompt(self.processor, request["prompt"]),
                                           {**options, "detokenize": False})
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
                    raise ValueError("Kimi context preflight differs from native preprocessing")
                _, receipt = self.native.audit(result, inputs, self.processor)
                mechanism = {"prompt_verified": receipt["prompt_verified"], "images": receipt["images"],
                             "decoder_tokens": receipt["decoder_tokens_after"]}
            return {"sample_id": request["sample_id"], "output_text": text, "output_tokens": len(ids),
                    "end_to_end_seconds": elapsed, "native_prompt_sha256": digest(native_ids),
                    "native_prompt_tokens": len(native_ids), "token_mechanism": mechanism}
        finally:
            image.close()
