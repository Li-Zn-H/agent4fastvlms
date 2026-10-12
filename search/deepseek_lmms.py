"""Native DeepSeek processor/generation with LMMS-owned tasks and scoring."""

import json
import os
from pathlib import Path

import torch
from lmms_eval.api.model import lmms
from tqdm import tqdm

from adapters.deepseek_vl2_dense import load_model, load_processor, prepare_image
from search.lmms_deepseek_entry import PROJECT_ROOT, validate_native_config


def generation_options(kwargs):
    """Keep native task generation limits; reject unsupported settings."""
    options = dict(kwargs)
    until = options.pop("until", [])
    if isinstance(until, str):
        until = [until]
    if not isinstance(until, list) or any(not isinstance(x, str) for x in until):
        raise ValueError("Native stopping sequences must be strings")
    options.setdefault("max_new_tokens", 1024)
    temperature = options.pop("temperature", 0)
    options.setdefault("do_sample", temperature > 0)
    if options["do_sample"]:
        options["temperature"] = temperature if temperature > 0 else 1.
    elif options.get("top_p") is None:
        options.pop("top_p", None)
    if type(options["max_new_tokens"]) is not int or options["max_new_tokens"] < 1:
        raise ValueError("Native generation requires a positive output budget")
    if options.get("use_cache", True) is not True or options.get("num_beams", 1) != 1:
        raise ValueError("DeepSeek native evaluation currently requires cached single-beam generation")
    options.update(use_cache=True, return_dict_in_generate=False)
    return options, until


class DeepSeekVL2LMMS(lmms):
    def __init__(self, pretrained, device="cuda:0", device_map=None,
                 attn_implementation="eager", batch_size=1, **kwargs):
        super().__init__()
        if kwargs or int(batch_size) != 1:
            raise ValueError(f"DeepSeek LMMS requires batch size one and known arguments: {kwargs}")
        config = json.loads(os.environ["A4FVLM_LMMS_DEEPSEEK_MODEL"])
        validate_native_config(config)
        if Path(pretrained).resolve() != (PROJECT_ROOT / config["path"]).resolve():
            raise ValueError("DeepSeek LMMS pretrained path differs from its descriptor")
        if attn_implementation != config["attn_implementation"]:
            raise ValueError("DeepSeek LMMS attention implementation differs from its descriptor")
        self._device = torch.device(device_map or device)
        self.serving = config.get("execution_backend") == "vllm"
        self.isolated = bool(config.get("stack_adapter"))
        descriptor = {**config, "device": str(self._device)}
        if self.serving:
            from types import SimpleNamespace
            from search.deepseek_serving import DeepSeekVllmBackend
            self.processor = load_processor(descriptor)
            # The serving wrapper only needs artifact metadata. Importing the
            # native model here would require xFormers/attrdict in vLLM's
            # isolated environment even though it never executes that model.
            artifact = (PROJECT_ROOT / descriptor["path"]).resolve()
            header = json.loads((artifact / "config.json").read_text())
            self._model = SimpleNamespace(config=SimpleNamespace(**{
                **header, "language_config": SimpleNamespace(**header["language_config"]),
                "_name_or_path": str(artifact),
            }))
            self.backend = DeepSeekVllmBackend(descriptor, os.environ["A4FVLMSP_TOKEN_RECEIPT"])
        elif self.isolated:
            from search.deepseek_native_worker import create_native_proxy
            self.processor = load_processor(descriptor)
            self._model = create_native_proxy(descriptor)
        else:
            self._model, self.processor = load_model(descriptor)
        self.pretrained = pretrained
        self.batch_size_per_gpu = 1

    @property
    def model(self):
        return self._model

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.config.language_config.max_position_embeddings

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests):
        raise NotImplementedError("The configured five tasks use native generation")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("The configured five tasks are single-turn image tasks")

    def generate_until(self, requests):
        responses = []
        with torch.inference_mode():
            for request in tqdm(requests, desc="Native DeepSeek-VL2"):
                context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
                document = self.task_dict[task][split][doc_id]
                images = doc_to_visual(document)
                if not isinstance(images, list) or len(images) != 1:
                    raise ValueError("DeepSeek native five-task evaluation requires exactly one image")
                inputs = prepare_image(self.processor, images[0], context)
                if not self.isolated:
                    inputs = inputs.to(self.device)
                options, until = generation_options(gen_kwargs)
                if inputs["input_ids"].shape[1] + options["max_new_tokens"] > self.max_length:
                    raise ValueError(f"Native context exceeds capacity for {task}/{doc_id}; no truncation")
                if self.serving:
                    from search.deepseek_serving import sampling_options
                    sampling = sampling_options(options, self.eot_token_id, until)
                    ids, _ = self.backend.generate(images[0], context, inputs, self.processor, sampling)
                    generated = torch.tensor([ids], dtype=torch.long)
                else:
                    if any(until):
                        options = {**options, "stop_strings": [x for x in until if x]}
                        if not self.isolated:
                            options["tokenizer"] = self.tokenizer
                    generated = self.model.generate(**inputs, **options)
                response = self.processor.batch_decode(generated, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)[0]
                for stop in until:
                    if stop:
                        response = response.split(stop, 1)[0]
                responses.append(response)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
        return responses
