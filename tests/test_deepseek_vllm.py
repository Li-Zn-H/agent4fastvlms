"""CPU geometry, width and precision contracts for the DeepSeek vLLM port."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from skills.deepseek_vl2_vllm import layer_widths, image_keep_count, compact_processed_image

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {'keep_ratio': .5, 'selection': {'score_dtype': 'float32',
                                         'coverage_grid': {'rows': 4, 'columns': 4}}}


def header():
    return {'model_type': 'deepseek_vl_v2', 'language_config': {
        'num_hidden_layers': 2, 'hidden_size': 128, 'intermediate_size': 512,
        'moe_intermediate_size': 256, 'n_routed_experts': 4, 'n_shared_experts': 2,
        'num_experts_per_tok': 2, 'first_k_dense_replace': 1, 'q_lora_rank': None},
        'vision_config': {'layers': 1}}


class ContractTests(unittest.TestCase):
    def test_independent_routed_and_shared_widths_preserve_expert_counts(self):
        config = header()
        self.assertEqual(layer_widths(config), [{'dense': 512}, {'routed': 256, 'shared': 512}])
        config['a4fvlm_stack_widths'] = [384, 128, 384]
        before = copy.deepcopy(config)
        self.assertEqual(layer_widths(config), [{'dense': 384}, {'routed': 128, 'shared': 384}])
        self.assertEqual(config, before)
        for widths in ([128, 128], [128, 512, 128], [128, 127, 128], [128, True, 128]):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                layer_widths({**config, 'a4fvlm_stack_widths': widths})

    def test_placeholder_compaction_preserves_text_and_protected_token_budget(self):
        import torch
        from skills.deepseek_vl2_token import image_layout, select_image_indices
        nx, ny = 3, 2
        _, protected, count = image_layout(14, nx, ny)
        ids = torch.tensor([[1, 5] + [98] * count + [7, 8]])
        batch = {'input_ids': ids, 'images_seq_mask': ids[0] == 98,
                 'images_spatial_crop': torch.tensor([[nx, ny]]),
                 'num_image_tokens': torch.tensor([count]), 'pixel_values': torch.zeros(7, 3, 384, 384)}
        unchanged = compact_processed_image(copy.deepcopy(batch), 98, 14, None)
        torch.testing.assert_close(unchanged['input_ids'], ids, rtol=0, atol=0)
        compacted = compact_processed_image(copy.deepcopy(batch), 98, 14, RUNTIME)
        keep = image_keep_count(14, nx, ny, RUNTIME)
        self.assertEqual(keep, 7 * 98 + len(protected))
        self.assertEqual(compacted['input_ids'].tolist(), [[1, 5] + [98] * keep + [7, 8]])
        self.assertEqual(compacted['num_image_tokens'].tolist(), [keep])
        self.assertEqual(int(compacted['images_seq_mask'].sum()), keep)
        torch.testing.assert_close(compacted['pixel_values'], batch['pixel_values'], rtol=0, atol=0)
        features = torch.randn(count, 16)
        selected = select_image_indices(features, 14, nx, ny, .5, RUNTIME['selection'])
        self.assertEqual(len(selected), keep)
        self.assertTrue(set(protected).issubset(selected.tolist()))
        invalid = copy.deepcopy(batch)
        invalid['input_ids'][0, 8] = 7
        with self.assertRaisesRegex(ValueError, 'placeholder run'):
            compact_processed_image(invalid, 98, 14, RUNTIME)

    def test_lazy_import(self):
        script = ('import skills.deepseek_vl2_vllm; import sys; '
                  'assert not any(n in sys.modules for n in ("torch", "transformers", "vllm", "timm"))')
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('timm'), 'vision runtime required')
    def test_original_384_patch14_grid_is_not_pixel_padded(self):
        import torch
        from skills.deepseek_vl2_vllm import build_vision
        cfg = SimpleNamespace(image_size=384, patch_size=14, width=32, layers=1, heads=2,
                              mlp_ratio=4, ignore_head=True, class_token=False, global_pool='map')
        model = build_vision(cfg).eval()
        with torch.inference_mode():
            patches = model.patch_embed(torch.zeros(1, 3, 384, 384))
        self.assertEqual(tuple(patches.shape), (1, 729, 32))
        self.assertEqual(model.patch_embed.grid_size, (27, 27))
        self.assertFalse(model.patch_embed.dynamic_img_pad)
        self.assertFalse(model.dynamic_img_size)


@unittest.skipUnless(importlib.util.find_spec('vllm'), 'isolated vLLM interpreter required')
class RuntimePolicyTests(unittest.TestCase):
    def test_mapped_gptq_policy_covers_fused_linears_and_preserves_mla_weight_access(self):
        from skills.deepseek_vl2_quantization import expected_inventory
        from vllm.model_executor.layers.quantization.utils.gptq_utils import is_layer_gptq_quantized
        from vllm.model_executor.models.deepseek_vl2 import DeepseekVLV2ForCausalLM
        policy = expected_inventory(header())
        names = [name for name, precision in policy.items() if precision == 'int4_w4a16']
        names = DeepseekVLV2ForCausalLM.hf_to_vllm_mapper.apply_list(names)
        mapping = {'gate_up_proj': ['gate_proj', 'up_proj']}
        for prefix in ('language_model.model.layers.0.mlp.gate_up_proj',
                       'language_model.model.layers.1.mlp.shared_experts.gate_up_proj',
                       'language_model.model.layers.1.self_attn.q_proj',
                       'language_model.model.layers.1.self_attn.kv_a_proj_with_mqa',
                       'vision.blocks.0.attn.qkv', 'vision.blocks.0.mlp.fc2'):
            self.assertTrue(is_layer_gptq_quantized(prefix, names, mapping), prefix)
        for prefix in ('language_model.model.layers.1.self_attn.kv_b_proj',
                       'projector.layers.0', 'vision.attn_pool.q'):
            self.assertFalse(is_layer_gptq_quantized(prefix, names, mapping), prefix)
        partial = [name for name in names if not name.endswith('shared_experts.up_proj')]
        with self.assertRaisesRegex(ValueError, 'not all shards'):
            is_layer_gptq_quantized('language_model.model.layers.1.mlp.shared_experts.gate_up_proj', partial, mapping)

    def test_registry_class_materializes_without_cuda_allocation(self):
        import torch
        from skills.deepseek_vl2_vllm import create_model_class
        cls = create_model_class()
        self.assertEqual(cls.__name__, 'A4FVLMDeepseekVLV2ForCausalLM')
        self.assertFalse(cls.supports_encoder_cudagraph)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
