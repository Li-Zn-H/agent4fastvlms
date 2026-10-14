"""Native Kimi prompts and pixels with LMMS-owned tasks and scoring."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from lmms_eval.api.model import lmms
from tqdm import tqdm

from adapters.kimi_vl_dense import load_processor, prepare_image
from search.deepseek_lmms import generation_options
from search.kimi_native_worker import NativeBackend
from search.lmms_kimi_entry import PROJECT_ROOT, validate_native_config


class KimiVLLMMS(lmms):
    def __init__(self, pretrained, device='cuda:0', device_map=None,
                 attn_implementation='eager', batch_size=1, **kwargs):
        super().__init__()
        if kwargs or int(batch_size) != 1:
            raise ValueError(f'Kimi LMMS requires batch size one and known arguments: {kwargs}')
        config = json.loads(os.environ['A4FVLM_LMMS_KIMI_MODEL'])
        validate_native_config(config)
        if Path(pretrained).resolve() != (PROJECT_ROOT / config['path']).resolve():
            raise ValueError('Kimi LMMS checkpoint differs from descriptor')
        if attn_implementation != config['attn_implementation']:
            raise ValueError('Kimi LMMS attention implementation differs from descriptor')
        self._device = torch.device(device_map or device)
        self.processor = load_processor(config)
        self.serving = config.get('execution_backend') == 'vllm'
        if self.serving:
            from search.kimi_serving import KimiVllmBackend
            self.backend = KimiVllmBackend({**config, 'device': str(self._device)},
                                           os.environ['A4FVLMSP_TOKEN_RECEIPT'])
        else:
            self.backend = NativeBackend({**config, 'device': str(self._device),
                                          'native_worker_python': config.get('native_worker_python', '.venv-kimi/bin/python'),
                                          'token_keep_ratio': config.get('token_keep_ratio', 1.0)})
        header = json.loads((PROJECT_ROOT / config['path'] / 'config.json').read_text())
        self._model = SimpleNamespace(config=SimpleNamespace(**{
            **header, 'text_config': SimpleNamespace(**header['text_config']),
            '_name_or_path': str((PROJECT_ROOT / config['path']).resolve())}))
        self.pretrained = pretrained
        self.input_receipt = Path(os.environ['A4FVLM_LMMS_KIMI_INPUT_RECEIPT'])

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
        return 163585

    @property
    def max_length(self):
        return self.config.text_config.max_position_embeddings

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests):
        raise NotImplementedError('The configured five tasks use native generation')

    def generate_until_multi_round(self, requests):
        raise NotImplementedError('The configured five tasks are single-turn image tasks')

    def generate_until(self, requests):
        responses = []
        with torch.inference_mode():
            for request in tqdm(requests, desc='Native Kimi-VL Instruct'):
                context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
                images = doc_to_visual(self.task_dict[task][split][doc_id])
                if not isinstance(images, list) or len(images) != 1:
                    raise ValueError('Kimi native five-task evaluation requires exactly one image')
                inputs = prepare_image(self.processor, images[0], context)
                options, until = generation_options(gen_kwargs)
                from search.kimi_lmms_identity import input_identity, completed_input
                identity = input_identity(task, split, doc_id, context, images[0], options, until)
                if inputs['input_ids'].shape[1] + options['max_new_tokens'] > self.max_length:
                    raise ValueError(f'Kimi native context exceeded for {task}/{doc_id}; no truncation')
                if self.serving:
                    from search.kimi_serving import sampling_options
                    ids, _ = self.backend.generate(images[0], context, inputs, self.processor,
                                                   sampling_options(options, self.eot_token_id, until))
                    suffix = torch.tensor([ids], dtype=torch.long)
                else:
                    if any(until):
                        options['stop_strings'] = [x for x in until if x]
                    suffix = self.backend.generate(**inputs, **options)
                response = self.processor.batch_decode(suffix, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)[0]
                for stop in until:
                    if stop:
                        response = response.split(stop, 1)[0]
                responses.append(response)
                with self.input_receipt.open('a') as stream:
                    stream.write(json.dumps(completed_input(identity, response)) + '\n')
                self.cache_hook.add_partial('generate_until', (context, gen_kwargs), response)
        return responses
