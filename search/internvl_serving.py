"""Native InternVL implementation of the shared serving timing boundary."""
from pathlib import Path
from time import perf_counter


class InternVLSession:
    def __init__(self, snap, backend, output):
        import torch
        from adapters.internvl35_dense import load_model, load_processor
        from search.serving import read, resolve, save_json

        self.snap, self.backend = snap, backend
        if backend == "dense":
            self.model, self.processor = load_model({**snap["dense_model"], "device": "cuda:0"})
            self.runtime = {"backend": "transformers", "model": snap["dense_model"]}
        else:
            from search.lmms_internvl_vllm_entry import InternVLVllmBackend
            self.processor = load_processor(snap["model"], artifact=True)
            self.native = InternVLVllmBackend(snap["model"], Path(output) / "token-mechanism.jsonl")
            self.runtime = self.native.runtime
            qualified = read(resolve(snap["config"]["freeze"]).parent.parent /
                             "candidates" / snap["candidate_id"] / "lmms/chartqa-token-mechanism.runtime.json")
            if self.runtime["observed_model"]["attention_backends"] != qualified["observed_model"]["attention_backends"]:
                raise ValueError("Actual InternVL attention backend differs from qualified LMMS")
        self.separator = self.processor.conversation_factory(self.processor.config.template).sep.strip()
        if not snap["generation"]["ignore_eos"]:
            native_eos = self.processor.tokenizer.convert_tokens_to_ids(self.separator)
            if snap["generation"]["stop_token_ids"] != [native_eos]:
                raise ValueError("InternVL serving must use the native conversation separator EOS")
        save_json(Path(output) / "runtime.json", self.runtime)
        torch.cuda.synchronize()

    def generate(self, request):
        import torch
        from adapters.internvl35_dense import prepare_image
        from search.serving import open_native_image, digest

        if self.backend != "dense" and not self.native.engine.reset_prefix_cache():
            raise RuntimeError("Could not clear InternVL cross-request prefix cache")
        torch.cuda.synchronize()
        started = perf_counter()
        image = open_native_image(request["image"])
        try:
            options = self.snap["generation"]
            if self.backend == "dense":
                inputs = prepare_image(self.processor, image, request["prompt"])
                native_ids = inputs["input_ids"][0].tolist()
                inputs = inputs.to("cuda:0")
                with torch.inference_mode():
                    result = self.model.generate(
                        **inputs, max_new_tokens=options["max_tokens"], do_sample=False,
                        eos_token_id=None if options["ignore_eos"] else options["stop_token_ids"],
                        forced_eos_token_id=None, repetition_penalty=1.0,
                        return_dict_in_generate=False,
                    )
                # Native InternVL returns only the generated suffix.
                ids = result[0].tolist()
            else:
                prompt = self.processor.unexpanded_prompt(request["prompt"])
                result = self.native.infer(image, prompt, {**options, "detokenize": False})
                ids = list(result.outputs[0].token_ids)
            text = self.processor.tokenizer.batch_decode([ids], skip_special_tokens=True)[0]
            text = text.split(self.separator)[0].strip()
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            # As in the shared Session, native prompt audits and result writes
            # follow the timer. The in-engine selector trace remains timed.
            if self.backend == "dense":
                mechanism = None
            else:
                inputs = prepare_image(self.processor, image, request["prompt"])
                native_ids = inputs["input_ids"][0].tolist()
                _, receipt = self.native.audit(result, native_ids, self.processor.image_token_id)
                mechanism = {"prompt_verified": receipt["prompt_verified"], "images": receipt["images"],
                             "decoder_tokens": receipt["decoder_tokens_after"]}
            return {"sample_id": request["sample_id"], "output_text": text, "output_tokens": len(ids),
                    "end_to_end_seconds": elapsed, "native_prompt_sha256": digest(native_ids),
                    "native_prompt_tokens": len(native_ids), "token_mechanism": mechanism}
        finally:
            image.close()
