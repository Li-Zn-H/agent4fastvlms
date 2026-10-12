"""Precision contracts and real GPTQ calibration on a small native MoE model."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from skills.deepseek_vl2_quantization import expected_inventory, high_precision_exceptions, vision_padding
from skills.quantization import classify_module, digest

ROOT = Path(__file__).resolve().parents[1]


def header():
    return dict(model_type='deepseek_vl_v2', tile_tag='2D', global_view_pos='head',
        vision_config={'image_size': 56, 'patch_size': 14, 'width': 128, 'layers': 1, 'heads': 2, 'mlp_ratio': 1.25},
        projector_config={'input_dim': 128, 'n_embed': 128},
        language_config={'vocab_size': 128, 'hidden_size': 128, 'intermediate_size': 256,
            'moe_intermediate_size': 256, 'num_hidden_layers': 2,
            'num_attention_heads': 2, 'num_key_value_heads': 2,
            'kv_lora_rank': 128, 'q_lora_rank': None, 'qk_rope_head_dim': 64,
            'qk_nope_head_dim': 64, 'v_head_dim': 64,
            'n_routed_experts': 2, 'n_shared_experts': 2, 'num_experts_per_tok': 2,
            'first_k_dense_replace': 1, 'n_group': 1, 'topk_group': 1,
            'topk_method': 'noaux_tc', 'scoring_func': 'sigmoid',
            'max_position_embeddings': 128, 'bos_token_id': 0, 'eos_token_id': 1})


class PolicyTests(unittest.TestCase):
    def test_native_mla_exceptions_and_full_expert_inventory(self):
        h = header()
        h['language_config'].update(num_hidden_layers=30, n_routed_experts=72, num_experts_per_tok=6)
        h['vision_config'].update(layers=27, width=1152, mlp_ratio=3.7362)
        policy = expected_inventory(h)
        self.assertEqual(len(policy), 6590)
        self.assertEqual(list(policy.values()).count('int4_w4a16'), 6552)
        self.assertEqual(list(policy.values()).count('bf16'), 38)
        self.assertEqual(policy['language.model.layers.29.mlp.experts.71.down_proj'], 'int4_w4a16')
        self.assertEqual(policy['language.model.layers.29.self_attn.kv_b_proj'], 'bf16')
        self.assertEqual(policy['vision.attn_pool.mlp.fc1'], 'bf16')
        self.assertEqual(classify_module('language.model.layers.1.mlp.gate', model_adapter='deepseek_vl2'), 'bf16')
        exceptions = high_precision_exceptions(h)
        self.assertEqual(exceptions['language.model.layers.29.mlp.gate.e_score_correction_bias']['dtype'], 'float32')
        self.assertEqual(vision_padding(h)['padded_width'], 4352)
        with self.assertRaisesRegex(ValueError, 'q_proj'):
            expected_inventory({**h, 'language_config': {**h['language_config'], 'q_lora_rank': 128}})

    def test_lazy_import(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import skills.deepseek_vl2_quantization; import sys; '
            'assert not any(x in sys.modules for x in ("torch", "transformers", "gptqmodel", "vllm"))'],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_refuses_live_cache_and_cleans_up_on_failure(self):
        import torch
        from skills.deepseek_vl2_quantization import capture_inputs
        class Block(torch.nn.Module):
            def forward(self, hidden, **kwargs):
                raise AssertionError('capture must stop the first block')
        block = Block()
        x = torch.randn(1, 4, 128)
        rows = capture_inputs(block, [x], lambda value: block(value, past_key_value=None, use_cache=False))
        torch.testing.assert_close(rows[0][0], x, rtol=0, atol=0)
        self.assertEqual(rows[0][1], {'past_key_value': None, 'use_cache': False})
        for kwargs in ({'past_key_value': object(), 'use_cache': False}, {'use_cache': True}):
            with self.assertRaisesRegex(ValueError, 'cache'):
                capture_inputs(block, [x], lambda value: block(value, **kwargs))
            self.assertFalse(block._forward_pre_hooks)


@unittest.skipUnless(importlib.util.find_spec('gptqmodel') and
                     (ROOT / '.cache/deepseek-vl2/upstream/deepseek_vl2').is_dir(),
                     'isolated GPTQModel and native DeepSeek source required')
class NativeGPTQTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.prior = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.prior)
        torch.manual_seed(482)
        from adapters.deepseek_vl2_dense import native_model_class
        self.descriptor = json.loads((ROOT / 'configs/model-deepseek-vl2.json').read_text())
        cls = native_model_class(self.descriptor)
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
        self.native = cls(DeepseekVLV2Config(**header())).to(torch.bfloat16).eval()
        # Native noaux_tc stores the correction separately in F32.
        self.native.language.model.layers[1].mlp.gate.e_score_correction_bias.data = torch.zeros(2)
        # xFormers has no CPU kernel. This test-only implementation preserves
        # its attention equation; production always uses the native GPU op.
        def cpu_attention(q, k, v, p=0., **kwargs):
            self.assertEqual(p, 0.)
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        context = patch('xformers.ops.memory_efficient_attention', cpu_attention)
        context.start()
        self.addCleanup(context.stop)

    def inputs(self):
        import torch
        from transformers import BatchFeature
        from skills.deepseek_vl2_token import image_layout
        rows = []
        for nx in (1, 2):
            count = image_layout(2, nx, 1)[2]
            ids = torch.tensor([[3] + [98] * count + list(range(10, 42))])
            rows.append(BatchFeature({'input_ids': ids, 'attention_mask': torch.ones_like(ids, dtype=torch.bool),
                'images': torch.randn(1, nx + 1, 3, 56, 56, dtype=torch.bfloat16),
                'images_seq_mask': ids == 98, 'images_spatial_crop': torch.tensor([[[nx, 1]]])}))
        return rows

    def test_padding_preserves_function_and_does_not_visit_pooling_head(self):
        import torch
        from skills.deepseek_vl2_quantization import pad_vision
        pixels = self.inputs()[0]['images'][0]
        handle = self.native.vision.attn_pool.register_forward_pre_hook(
            lambda *args: self.fail('native ignore_head must bypass pooling'))
        self.addCleanup(handle.remove)
        with torch.inference_mode():
            before = self.native.vision(pixels)
            padded = pad_vision(self.native)
            after = self.native.vision(pixels)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertEqual(padded['added_channels'], 96)
        self.assertEqual(self.native.vision.attn_pool.mlp.fc1.out_features, 160)

    def test_fresh_physical_export_calibrates_every_expert_and_binds_evidence(self):
        self.check_export(pruned=True)

    def test_independent_dense_export_has_no_sp_ancestry(self):
        self.check_export(pruned=False)

    def test_export_rejects_parent_drift_before_calibration(self):
        import torch
        from skills.deepseek_vl2_artifact import QUANT_METHOD, write_parent
        from skills.deepseek_vl2_pruning import dimensions, units, make_recipe, apply_recipe
        from skills.deepseek_vl2_quantization import export_multimodal_gptq
        dims = dimensions(self.native.config.language_config)
        scores = [[[float(i) for i in range(unit['original_width'])]
                   for _ in range(unit['experts'])] for unit in units(dims)]
        recipe = make_recipe(scores, [128, 128, 256], dims,
                             {'channel_group_size': 128, 'protected_layers': []})
        apply_recipe(self.native, recipe)
        class Processor:
            def save_pretrained(self, path):
                (Path(path) / 'processor-fixture.json').write_text('{}')
        processor = Processor()
        rows = [{'stable_row_id': str(index)} for index in range(2)]
        calibration = {'data': 'fixture', 'media_root': '.', 'limit': len(rows)}
        candidate = {'route': 'gptqmodel_w4a16', 'bits': 4, 'group_size': 128,
                     'vision_quantization': QUANT_METHOD}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'parent'
            write_parent(self.native, processor, recipe, parent)
            descriptor = {**self.descriptor, 'path': str(parent), 'device': 'cpu'}
            names = [name for name, module in self.native.named_modules()
                     if isinstance(module, torch.nn.Linear)]
            changed_width = copy.deepcopy(self.native)
            changed_width.language.model.layers[1].mlp.shared_experts.down_proj = torch.nn.Linear(
                128, 128, bias=False, dtype=torch.bfloat16)
            changed_recipe = copy.deepcopy(self.native)
            changed_recipe.structured_pruning_recipe['retained_channels'][0][0] = list(range(128))
            changed_dtype = copy.deepcopy(self.native)
            changed_dtype.language.model.layers[1].self_attn.q_proj.float()
            for label, loaded in (('width', changed_width), ('recipe', changed_recipe),
                                  ('dtype', changed_dtype)):
                with self.subTest(drift=label), patch('search.evaluate.load_samples', return_value=rows), \
                        patch('skills.deepseek_vl2_quantization.quantize_native_modules',
                              side_effect=AssertionError('Calibration started for an invalid Parent')) as quantize:
                    destination = root / label
                    with self.assertRaisesRegex(ValueError, 'Parent'):
                        export_multimodal_gptq(candidate=candidate, model=descriptor, processor=processor,
                            calibration=calibration, output_dir=destination, module_names=names,
                            loaded_model=loaded)
                    quantize.assert_not_called()
                    self.assertFalse(destination.exists())

    def check_export(self, *, pruned):
        import torch
        from skills.deepseek_vl2_artifact import (
            QUANT_METHOD, write_parent, load_physical_model, validate_quantized_bindings, audit_quantized_artifact,
            config_dict,
        )
        from skills.deepseek_vl2_pruning import dimensions, units, make_recipe, apply_recipe
        from skills.deepseek_vl2_quantization import export_multimodal_gptq, unpack_gptq
        dims = dimensions(self.native.config.language_config)
        scores = [[[float(i) for i in range(u['original_width'])] for _ in range(u['experts'])] for u in units(dims)]
        recipe = make_recipe(scores, [128, 128, 256], dims, {'channel_group_size': 128, 'protected_layers': []})
        if pruned:
            apply_recipe(self.native, recipe)
        class Processor:
            def save_pretrained(self, path):
                (Path(path) / 'processor-fixture.json').write_text('{}')
        processor = Processor()
        batches = self.inputs()
        rows = [{'stable_row_id': str(i), 'batch': batch} for i, batch in enumerate(batches)]
        calibration = {'data': 'fixture', 'media_root': '.', 'limit': len(rows)}
        candidate = {'route': 'gptqmodel_w4a16', 'bits': 4, 'group_size': 128,
                     'vision_quantization': QUANT_METHOD, 'calibration': calibration}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, output = root / 'parent', root / 'output'
            descriptor = {**self.descriptor, 'path': str(parent), 'device': 'cpu'}
            if pruned:
                write_parent(self.native, processor, recipe, parent)
                native = load_physical_model(descriptor)
            else:
                from skills.internvl35_artifact import save_tensors
                parent.mkdir()
                (parent / 'config.json').write_text(json.dumps(config_dict(self.native.config)))
                save_tensors(self.native.state_dict(), parent)
                native = self.native
            preserved = {key: value.clone() for key, value in native.state_dict().items()
                         if any(x in key for x in ('kv_b_proj.', '.mlp.gate.', 'projector.', 'lm_head.', 'attn_pool.'))}
            names = [name for name, value in native.named_modules() if isinstance(value, torch.nn.Linear)]
            with patch('search.evaluate.load_samples', return_value=rows), patch(
                    'search.model.prepare_inputs', side_effect=lambda processor, row, media_root: row['batch']):
                if pruned:
                    from skills.stack_adapters import StackAdapterError, export_pruned_gptqmodel
                    selected = {'route': candidate['route'], 'candidate_config': candidate}
                    with self.assertRaisesRegex(StackAdapterError, 'differs from selected recipe'):
                        export_pruned_gptqmodel(candidate=selected, parent_dir=parent, processor=processor,
                            calibration=calibration, output_dir=output, model=descriptor,
                            module_names=names, recipe_digest='another-parent')
                    self.assertFalse(output.exists())
                    # The native model above was physically reloaded from this
                    # Parent. Reuse it only to avoid loading a second CPU copy;
                    # calibration, solving, packing and export are real.
                    with patch('adapters.deepseek_vl2_dense.load_model', return_value=(native, processor)):
                        result = export_pruned_gptqmodel(candidate=selected, parent_dir=parent,
                            processor=processor, calibration=calibration, output_dir=output,
                            model=descriptor, module_names=names, recipe_digest=digest(recipe))
                else:
                    result = export_multimodal_gptq(candidate=candidate, model=descriptor, processor=processor,
                        calibration=calibration, output_dir=output, module_names=names, loaded_model=native)
            manifest = json.loads(Path(result['manifest']).read_text())
            config = json.loads((output / 'config.json').read_text())
            validate_quantized_bindings(output, config, manifest)
            if pruned:
                from search.lmms import build_command, cache_environment
                lmms_descriptor = {**descriptor, 'path': str(output), 'stack_adapter': True,
                                   'token_keep_ratio': .5,
                                   'token_selection': {'score_dtype': 'float32', 'coverage_grid': {'rows': 1, 'columns': 1}},
                                   'native_worker_python': sys.executable}
                self.assertIn('deepseek_vl2', build_command(lmms_descriptor, 'mmstar', Path('unused')))
                self.assertEqual(json.loads(cache_environment(lmms_descriptor)['A4FVLM_LMMS_DEEPSEEK_MODEL']),
                                 lmms_descriptor)
                serving_descriptor = {**lmms_descriptor, 'execution_backend': 'vllm',
                                      'runtime': {'vllm_max_model_len': 4096, 'optimization_level': 0,
                                                  'compile_mm_encoder': False, 'cudagraph_mm_encoder': False}}
                self.assertIn('deepseek_vl2', build_command(serving_descriptor, 'mmstar', Path('unused')))
                self.assertEqual(json.loads(cache_environment(serving_descriptor)['A4FVLM_LMMS_VLLM_MODEL']),
                                 serving_descriptor)
            self.assertEqual('stack' in manifest, pruned)
            self.assertEqual((output / 'sp-recipe.json').exists(), pruned)
            from skills.stack_vllm import validate_stack_artifact, prepare_vllm_artifact
            if pruned:
                self.assertEqual(validate_stack_artifact(output)['widths'], [128, 128, 256])
                from skills.stack_model import validate_resumable_export
                request = {'model': descriptor, 'evaluation': {'generation': {'max_new_tokens': 8}},
                           'candidate': {'structured_pruning': {'recipe_digest': digest(recipe)},
                                         'quantization': selected, 'token_keep_ratio': .5},
                           'calibration': calibration, 'quantization_runtime': {'quantize': True},
                           'selection': {'score_dtype': 'float32',
                                         'coverage_grid': {'rows': 1, 'columns': 1}}}
                reused = validate_resumable_export(output, request, request)
                self.assertEqual(reused['artifact_digest'], manifest['artifact_digest'])
                self.assertTrue(reused['reused_export_only'])
                self.assertFalse(reused['quality_reused'])
                for key in ('calibration', 'candidate', 'selection'):
                    changed = copy.deepcopy(request)
                    changed[key] = {}
                    with self.assertRaisesRegex(ValueError, 'request mismatch'):
                        validate_resumable_export(output, request, changed)
                wrong_parent = copy.deepcopy(request)
                wrong_parent['candidate']['structured_pruning']['recipe_digest'] = 'another-parent'
                with self.assertRaisesRegex(ValueError, 'physical parent/calibration'):
                    validate_resumable_export(output, wrong_parent, wrong_parent)
            with patch.dict('os.environ', {}, clear=True):
                prepared = prepare_vllm_artifact(output)
                self.assertEqual(prepared['model_type'], 'deepseek_vl2')
                self.assertEqual(prepared['custom_stack'], pruned)
                self.assertEqual(list(prepared['model_class_overrides'].values()),
                                 ['skills.deepseek_vl2_vllm:A4FVLMDeepseekVLV2ForCausalLM'])
            audit = audit_quantized_artifact(output, manifest)
            self.assertEqual(audit['status'], 'passed')
            self.assertEqual(audit['verified_int4_modules'], 22)
            from safetensors.torch import load_file
            saved = {}
            for filename in output.glob('*.safetensors'):
                saved.update(load_file(filename))
            for name, precision in manifest['module_assignment'].items():
                if precision == 'int4_w4a16':
                    packed = {suffix: saved[name + '.' + suffix] for suffix in ('qweight', 'qzeros', 'scales', 'g_idx')}
                    torch.testing.assert_close(native.get_submodule(name).weight,
                                               unpack_gptq(packed).to(torch.bfloat16), rtol=0, atol=0)
            for name, value in preserved.items():
                torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
            self.assertEqual(saved['language.model.layers.1.mlp.experts.1.down_proj.g_idx'].numel(), 128 if pruned else 256)
            self.assertEqual(saved['language.model.layers.1.mlp.shared_experts.down_proj.g_idx'].numel(), 256 if pruned else 512)
            receipt_path = output / 'vision-quantization.json'
            receipt = json.loads(receipt_path.read_text())
            self.assertTrue(all(row['module'].startswith('vision.blocks.') for row in receipt['modules'][:4]))
            self.assertTrue(all(row['observed_tokens'] > 0 for row in receipt['modules']))
            receipt['modules'].pop()
            receipt_path.write_text(json.dumps(receipt))
            changed = copy.deepcopy(manifest)
            changed['visual_quantization']['calibration_receipt_digest'] = digest(receipt)
            changed['artifact_digest'] = digest({k: v for k, v in changed.items() if k != 'artifact_digest'})
            with self.assertRaisesRegex(ValueError, 'complete native routed calibration'):
                validate_quantized_bindings(output, config, changed)
            from safetensors.torch import save_file
            files = list(output.glob('*.safetensors'))
            self.assertEqual(len(files), 1)
            key = 'language.model.layers.1.mlp.gate.e_score_correction_bias'
            saved[key] = saved[key].to(torch.bfloat16)
            save_file(saved, str(files[0]))
            corrupt = audit_quantized_artifact(output, manifest)
            self.assertEqual(corrupt['status'], 'blocked_preflight')
            self.assertTrue(any(row.get('tensor') == key for row in corrupt['errors']))

    def test_unobserved_expert_is_an_error_and_hooks_are_removed(self):
        import torch
        from skills.deepseek_vl2_quantization import quantize_blocks
        layer = self.native.language.model.layers[1]
        layer.mlp.gate.top_k = 1
        layer.mlp.gate.e_score_correction_bias.data[:] = torch.tensor([100., -100.])
        hidden = torch.randn(1, 12, 128, dtype=torch.bfloat16)
        kwargs = {'position_ids': torch.arange(12)[None], 'use_cache': False,
                  'attention_mask': torch.full((1, 1, 12, 12), float('-inf'), dtype=torch.bfloat16).triu(1)}
        groups = [('mlp.experts.0.gate_proj', 'mlp.experts.1.gate_proj')]
        with self.assertRaisesRegex(ValueError, 'Unobserved DeepSeek GPTQ modules'):
            quantize_blocks([layer], 'fixture', [(hidden, kwargs)], groups, {}, [])
        self.assertFalse(layer.mlp.experts[0].gate_proj._forward_hooks)
        self.assertFalse(layer.mlp.experts[1].gate_proj._forward_hooks)


if __name__ == '__main__':
    unittest.main()
