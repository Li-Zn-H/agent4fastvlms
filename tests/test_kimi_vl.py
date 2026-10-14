"""Kimi registration, native preprocessing and cached-generation contracts."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from adapters import AdapterError
from adapters.kimi_vl import KimiVLAdapter
from adapters.kimi_vl_dense import native_path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Kimi-VL-A3B-Instruct"


@unittest.skipUnless((MODEL / "config.json").is_file(), "local Kimi metadata required")
class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        self.header = json.loads((MODEL / "config.json").read_text())

    def test_catalog_remains_lazy_and_does_not_claim_complete_pipeline(self):
        script = """import sys
from search.registry import catalog
value = catalog(model='kimi_vl')['models'][0]
assert value['implemented_stages'] == ['dense', 'token', 'sp', 'quantization', 'stack', 'runtime', 'kv_cache', 'lmms']
assert value['pipeline_ids'] == []
assert not any(name in sys.modules for name in ('torch', 'transformers', 'vllm'))
"""
        subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, check=True)

    def test_rejects_changed_routing_and_expert_counts(self):
        KimiVLAdapter().validate_checkpoint(self.header)
        for key, value in [('n_routed_experts', 32), ('num_experts_per_tok', 4),
                           ('scoring_func', 'softmax'), ('topk_method', 'greedy')]:
            changed = copy.deepcopy(self.header)
            changed['text_config'][key] = value
            with self.subTest(key=key), self.assertRaises(AdapterError):
                KimiVLAdapter().validate_checkpoint(changed)

    def test_independent_quantization_snapshot_is_lazy_and_full_pipeline_remains_guarded(self):
        script = """import sys
from search.quantization import snapshot
value = snapshot('configs/quantization-kimi-vl.json')
assert value['model']['adapter'] == 'kimi_vl'
assert value['output'].endswith('/results/model/kimi_vl/a800/quantization')
assert not any(name in sys.modules for name in ('torch', 'transformers', 'vllm', 'gptqmodel', 'PIL'))
"""
        subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, check=True)
        adapter = KimiVLAdapter()
        adapter.validate_stage("quantization", self.header)
        for stage in ("stack", "runtime", "kv_cache", "lmms"):
            adapter.validate_stage(stage, self.header)
        with self.assertRaisesRegex(AdapterError, "under validation"):
            adapter.validate_checkpoint(self.header, pipeline=True)

    def test_all_stage_preflight_and_stack_plan_are_lazy_without_qualified_sources(self):
        script = """import json, sys
from search.registry import preflight
from search.stack import snapshot
model = json.load(open('configs/model-kimi-vl.json'))
report = preflight(model, device=0)
assert report['status'] == 'ready_for_probe'
assert all(row['environment'] == 'unverified' and row['quality'] == 'unverified'
           for row in report['stages'].values())
value = snapshot('configs/stack-kimi-vl.json')
assert value['status'] == 'blocked' and value['candidate_count'] == 0
assert 'no_passing_structured_pruning_recipe' in value['blockers']
assert 'no_qualified_quantization_route:gptqmodel_w4a16' in value['blockers']
assert not any(name in sys.modules for name in ('torch','transformers','vllm','gptqmodel','PIL'))
"""
        subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, check=True)

    def test_rejects_thinking_protocol_and_modified_native_source(self):
        with self.assertRaisesRegex(ValueError, 'non-thinking'):
            native_path({**self.config, 'protocol': 'thinking'})
        with tempfile.TemporaryDirectory() as directory:
            for name in self.config['native_source_sha256']:
                Path(directory, name).write_bytes((MODEL / name).read_bytes())
            Path(directory, 'chat_template.jinja').write_text('Changed template')
            with self.assertRaisesRegex(ValueError, 'chat_template'):
                native_path({**self.config, 'path': directory})


try:
    import torch
    import transformers
    NATIVE_ENV = transformers.__version__ == "4.50.3"
except ImportError:
    NATIVE_ENV = False


@unittest.skipUnless(NATIVE_ENV and (MODEL / "config.json").is_file(),
                     "isolated Kimi native environment required")
class NativeTests(unittest.TestCase):
    def test_preprocessing_matches_unmodified_processor_and_keeps_native_grid(self):
        from PIL import Image
        from adapters.kimi_vl_dense import load_processor, prepare_image
        config = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        processor = load_processor(config)
        image = Image.new('RGB', (281, 421), (25, 120, 210))
        inputs = prepare_image(processor, image, 'What color is the image?')
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                     {'type': 'text', 'text': 'What color is the image?'}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        original = processor(images=[image], text=[prompt], padding=True, truncation=False, return_tensors='pt')
        for key in inputs:
            self.assertTrue(torch.equal(inputs[key], original[key]), key)
        self.assertEqual(inputs['image_grid_hws'].tolist(), [[32, 22]])
        self.assertEqual(int((inputs['input_ids'] == 163605).sum()), 176)
        self.assertNotIn('think', prompt.lower())

    def test_unmodified_tiny_model_returns_prompt_and_uses_cached_native_moe(self):
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        config = AutoConfig.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
        cls = get_class_from_dynamic_module(config.auto_map['AutoModelForCausalLM'], str(MODEL), local_files_only=True)
        text = config.text_config
        for key, value in {'vocab_size': 256, 'hidden_size': 64, 'intermediate_size': 128,
                           'moe_intermediate_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4,
                           'num_key_value_heads': 4, 'n_routed_experts': 4, 'n_shared_experts': 1,
                           'num_experts_per_tok': 2, 'kv_lora_rank': 16, 'qk_rope_head_dim': 8,
                           'qk_nope_head_dim': 16, 'v_head_dim': 16, 'bos_token_id': 1,
                           'eos_token_id': 2, 'pad_token_id': 0}.items():
            setattr(text, key, value)
        vision = config.vision_config
        for key, value in {'hidden_size': 32, 'intermediate_size': 64, 'num_attention_heads': 4,
                           'num_hidden_layers': 1, 'init_pos_emb_height': 2, 'init_pos_emb_width': 2}.items():
            setattr(vision, key, value)
        config.media_placeholder_token_id = 100
        config.vocab_size = 256
        config._attn_implementation = 'eager'
        model = cls(config).eval()
        model.language_model.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        # The tiny numerical fixture uses FP32 linears, but readiness still
        # requires the original checkpoint's BF16 correction storage.
        gate = model.language_model.model.layers[1].mlp.gate
        gate.e_score_correction_bias.data = gate.e_score_correction_bias.data.to(torch.bfloat16)
        ids = torch.tensor([[1, 100, 25]])
        inputs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                  'pixel_values': torch.randn(4, 3, 14, 14), 'image_grid_hws': torch.tensor([[2, 2]])}
        calls = []
        def observe(owner, args, output):
            cache = output.past_key_values
            calls.append(cache.get_seq_length() if hasattr(cache, 'get_seq_length') else cache[0][0].shape[-2])
        handle = model.language_model.model.register_forward_hook(
            observe)
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=3,
                                           eos_token_id=None, pad_token_id=0, use_cache=True)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(generated[:, :3], ids))
        self.assertEqual(generated.shape[1], 6)
        self.assertEqual(calls, [3, 4, 5])
        self.assertEqual(len(model.language_model.model.layers[1].mlp.experts), 4)
        from skills.kimi_vl_token import KimiTokenController
        selection = {'score_dtype': 'float32', 'coverage_grid': {'rows': 1, 'columns': 1}}
        options = {'max_new_tokens': 3, 'do_sample': False, 'eos_token_id': None,
                   'pad_token_id': 0, 'use_cache': True}
        with torch.inference_mode():
            keepall = KimiTokenController(model, selection, 1).generate(dict(inputs), options)
        self.assertTrue(torch.equal(keepall, generated))
        four_ids = torch.tensor([[1, 100, 100, 100, 100, 25]])
        four = {'input_ids': four_ids, 'attention_mask': torch.ones_like(four_ids),
                'pixel_values': torch.randn(16, 3, 14, 14), 'image_grid_hws': torch.tensor([[4, 4]])}
        original_extract = model._extract_image_features
        hooks = len(model.language_model.model._forward_hooks)
        controller = KimiTokenController(model, selection, .5)
        with torch.inference_mode():
            output = controller.generate(four, options)
        self.assertEqual(output.shape[1], 7)
        self.assertEqual(controller.last_metrics['image_tokens_before'], 4)
        self.assertEqual(controller.last_metrics['image_tokens_after'], 2)
        self.assertEqual(controller.last_metrics['kv_tokens_after_last_forward'], 6)
        self.assertTrue(torch.equal(four['input_ids'], torch.tensor([[1, 100, 100, 25]])))
        self.assertEqual(model._extract_image_features, original_extract)
        self.assertEqual(len(model.language_model.model._forward_hooks), hooks)
        # Exercise the real existing IPC framing with the unmodified tiny
        # model, including the suffix-only boundary needed by the task host.
        import socket
        import threading
        from multiprocessing.connection import Connection
        from search.kimi_native_worker import serve
        from search.internvl_native_worker import send_message, receive_message
        parent_socket, child_socket = socket.socketpair()
        parent, child = Connection(parent_socket.detach()), Connection(child_socket.detach())
        model._agent4vlm_router_dtypes = {}
        thread = threading.Thread(target=serve, args=(child, {'token_keep_ratio': 1}),
                                  kwargs={'loader': lambda config: (model, None)})
        thread.start()
        try:
            self.assertTrue(parent.poll(10))
            self.assertEqual(receive_message(parent)['status'], 'ready')
            send_message(parent, {'index': 0, 'kwargs': {**inputs, **options}})
            self.assertTrue(parent.poll(10))
            response = receive_message(parent)
            self.assertEqual(response['status'], 'completed')
            self.assertTrue(torch.equal(response['output_ids'], generated[:, 3:]))
            send_message(parent, None)
            thread.join(10)
            self.assertFalse(thread.is_alive())
        finally:
            parent.close()
            thread.join(10)
        from skills.kimi_vl_pruning import dimensions, units, make_recipe, apply_recipe, audit_recipe
        from skills.kimi_vl_calibration import RoutingMoments
        collector = RoutingMoments(model)
        collector.start_sample(inputs['input_ids'][0] == 100)
        try:
            with torch.inference_mode():
                model.generate(**inputs, **options)
            for unit, experts, denominators in zip(collector.domain, collector.states, collector.counts):
                for stratum in ('visual_prefill', 'text_prefill', 'decode'):
                    hits = sum(state[stratum]['hits'] for state in experts)
                    self.assertEqual(hits, denominators[stratum] * (2 if unit['kind'] == 'routed' else 1))
        finally:
            collector.close()
        dims = dimensions(model.config.text_config)
        domain = units(dims)
        scores = [[[float(i + 1) for i in range(unit['original_width'])]
                   for expert in range(unit['experts'])] for unit in domain]
        recipe = make_recipe(scores, [unit['original_width'] // 2 for unit in domain], dims,
                             {'channel_group_size': 16, 'protected_layers': []})
        routing = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                   if '.mlp.gate.' in name}
        apply_recipe(model, recipe)
        audit = audit_recipe(model, recipe)
        self.assertTrue(audit['router_and_expert_counts_preserved'])
        self.assertEqual(recipe['actual_depth'], .5)
        self.assertEqual(len(model.language_model.model.layers[1].mlp.experts), 4)
        for name, parameter in model.named_parameters():
            if name in routing:
                self.assertTrue(torch.equal(parameter, routing[name]), name)
        with torch.inference_mode():
            pruned = model.generate(**inputs, **options)
        self.assertEqual(pruned.shape[1], 6)
        with self.assertRaisesRegex(ValueError, 'fresh Dense'):
            apply_recipe(model, recipe)
