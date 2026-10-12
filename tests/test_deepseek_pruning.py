"""MoE recipe accounting and native masked-versus-physical equivalence."""

import copy
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from skills import deepseek_vl2_pruning as sp

ROOT = Path(__file__).resolve().parents[1]


def tiny_dimensions():
    return sp.dimensions(dict(num_hidden_layers=2, hidden_size=16, intermediate_size=32,
                              moe_intermediate_size=16, n_routed_experts=4, n_shared_experts=2,
                              num_experts_per_tok=2, first_k_dense_replace=1))


def scores_for(dims):
    return [[[float((j + expert) % u['original_width'] + 1) for j in range(u['original_width'])]
             for expert in range(u['experts'])] for u in sp.units(dims)]


SELECTION = {"channel_group_size": 8, "protected_layers": []}


class RecipeTests(unittest.TestCase):
    def test_stack_architecture_dispatch_agrees_for_saved_json_and_live_config(self):
        from skills.stack_model import stack_architecture_for_config
        from skills.deepseek_vl2_artifact import STACK_ARCHITECTURE
        for config in ({"model_type": "deepseek_vl_v2"}, SimpleNamespace(model_type="deepseek_vl_v2")):
            self.assertEqual(stack_architecture_for_config(config), STACK_ARCHITECTURE)

    def test_global_denominators_preserve_zero_routing_without_unobserved_experts(self):
        from scripts.calibrate_llava_sp import _write_moments, _write_scores
        from skills.structured_pruning import STRATA
        dims = tiny_dimensions()
        domain = sp.units(dims)
        denominators = [{name: 10 for name in STRATA} for _ in domain]
        coverage = [[{name: (0 if u['kind'] == 'routed' and name == 'decode' else 10)
                      for name in STRATA} for _ in range(u['experts'])] for u in domain]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'moments').mkdir()
            (root / 'scores').mkdir()
            manifest = {'scorer_id': sp.SCORE, 'dimensions': dims, 'units': domain, 'strata': list(STRATA), 'model': {'test': True}}
            statistics = {'model': manifest['model'], 'dimensions': dims, 'routing_counts': coverage,
                          'stratum_denominators': denominators,
                          'denominator': 'all_tokens_per_stratum_including_unrouted_tokens'}
            (root / 'score-manifest.json').write_text(json.dumps(manifest))
            for index, unit in enumerate(domain):
                for expert in range(unit['experts']):
                    width = unit['original_width']
                    rows = {name: {'count': 10, 'sums': [0.] * width,
                                   'squares': [float(coverage[index][expert][name])] * width} for name in STRATA}
                    filename = f'unit-{index:02d}-expert-{expert:02d}.bin'
                    _write_moments(root / 'moments' / filename, index, rows)
                    _write_scores(root / 'scores' / filename, index, [1.] * width)
            (root / 'statistics-manifest.json').write_text(json.dumps(statistics))
            self.assertEqual(len(sp.load_calibration(root, dims)), 3)
            statistics['routing_counts'][1][0] = {name: 0 for name in STRATA}
            (root / 'statistics-manifest.json').write_text(json.dumps(statistics))
            with self.assertRaisesRegex(ValueError, 'Insufficient calibration'):
                sp.load_calibration(root, dims)

    def test_shape_admission_separates_expert_kernel_families(self):
        from search.structured_pruning import shape_key, complete_measurement
        row = {'phase': 'prefill', 'input_tokens': 10, 'width': 128, 'kind': 'routed',
               'kernel': 'marlin_fused_moe_w4a16', 'status': 'measured', 'finite_output': True,
               'measurement_iterations': 1, 'milliseconds': [.5]}
        self.assertNotEqual(shape_key(row), shape_key({**row, 'kind': 'shared'}))
        self.assertNotEqual(shape_key(row), shape_key({**row, 'kernel': 'torch_bf16'}))
        self.assertFalse(complete_measurement(row))
        self.assertTrue(complete_measurement({**row, 'reference_checked': True}))

    def test_lazy_import(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "from skills import deepseek_vl2_pruning; import sys; "
            "assert not any(x in sys.modules for x in ('torch','transformers','vllm'))"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_widths_keep_all_experts_with_independent_channel_identities(self):
        dims = tiny_dimensions()
        recipe = sp.make_recipe(scores_for(dims), [24, 8, 16], dims, SELECTION)
        from skills.structured_pruning import make_recipe, validate_recipe
        self.assertEqual(make_recipe(scores_for(dims), [24, 8, 16], dims, SELECTION), recipe)
        validate_recipe(recipe, dims)
        sp.validate_recipe(recipe, dims)
        self.assertEqual([u['kind'] for u in recipe['units']], ['dense', 'routed', 'shared'])
        self.assertEqual(recipe['original_ffn_parameters'], (32 + 4 * 16 + 32) * 16 * 3)
        self.assertEqual(recipe['removed_ffn_parameters'], (8 + 4 * 8 + 16) * 16 * 3)
        self.assertEqual(len(recipe['retained_channels'][1]), 4)
        self.assertNotEqual(recipe['retained_channels'][1][0], recipe['retained_channels'][1][1])
        for edit in ('expert_count', 'width', 'duplicate', 'accounting'):
            invalid = copy.deepcopy(recipe)
            if edit == 'expert_count':
                invalid['retained_channels'][1].pop()
            elif edit == 'width':
                invalid['widths'][1] = 7
            elif edit == 'duplicate':
                invalid['retained_channels'][1][0][1] = invalid['retained_channels'][1][0][0]
            else:
                invalid['removed_ffn_parameters'] += 1
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                sp.validate_recipe(invalid, dims)

    def test_allocation_is_nested_and_requires_each_kernel_family(self):
        dims = tiny_dimensions()
        scores = scores_for(dims)
        admitted = {'dense': [32, 24, 16], 'routed': [16, 8], 'shared': [32, 24, 16]}
        path = sp.allocation_path(scores, dims, SELECTION, admitted)
        self.assertEqual(path[0]['actual_depth'], 0)
        self.assertEqual(path[-1]['widths'], [16, 8, 16])
        previous = None
        for node in path:
            recipe = sp.make_recipe(scores, node['widths'], dims, SELECTION)
            self.assertEqual(recipe['actual_depth'], node['actual_depth'])
            if previous is not None:
                for old, new in zip(previous['retained_channels'], recipe['retained_channels']):
                    self.assertTrue(all(set(n).issubset(o) for o, n in zip(old, new)))
            previous = recipe
        with self.assertRaises(ValueError):
            sp.allocation_path(scores, dims, SELECTION, [32, 24, 16, 8])
        protected = sp.allocation_path(scores, dims, {**SELECTION, 'protected_layers': [1]}, admitted)
        self.assertEqual(protected[-1]['widths'], [16, 16, 32])


def native_available():
    try:
        return (importlib.metadata.version('transformers') == '4.46.3'
                and (ROOT / '.cache/deepseek-vl2/upstream/deepseek_vl2').is_dir())
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(native_available(), 'isolated native DeepSeek environment required')
class PhysicalTests(unittest.TestCase):
    def test_physical_parent_roundtrip_and_recipe_tamper_rejection(self):
        import torch
        from adapters.deepseek_vl2_dense import native_model_class
        from skills.deepseek_vl2_artifact import write_parent, load_physical_model
        descriptor = json.loads((ROOT / 'configs/model-deepseek-vl2.json').read_text())
        model_class = native_model_class(descriptor)
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
        config = self.make_model()[0].config.language_config
        model = model_class(DeepseekVLV2Config(tile_tag='2D', global_view_pos='head',
            language_config=config, vision_config={'image_size':28,'patch_size':14,'width':16,'layers':1,'heads':2,'mlp_ratio':2},
            projector_config={'input_dim':16,'n_embed':16})).float().eval()
        model.language.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        dims = sp.dimensions(model.config.language_config)
        recipe = sp.make_recipe(scores_for(dims), [24, 8, 16], dims, SELECTION)
        sp.apply_recipe(model, recipe)
        inputs = torch.tensor([[1, 5, 7, 8]])
        class Processor:
            def save_pretrained(self, path):
                (Path(path) / 'processor-fixture.json').write_text('{}')
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            destination = Path(directory) / 'parent'
            write_parent(model, Processor(), recipe, destination)
            loaded = load_physical_model({**descriptor, 'path':str(destination), 'device':'cpu', 'dtype':'float32'})
            torch.testing.assert_close(model.language(inputs, use_cache=False).logits,
                                       loaded.language(inputs, use_cache=False).logits, rtol=0, atol=0)
            self.assertEqual(sp.audit_recipe(loaded, recipe), sp.audit_recipe(model, recipe))
            path = destination / 'sp-recipe.json'
            changed = json.loads(path.read_text())
            changed['retained_channels'][1][0] = list(range(8))
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'bound to its recipe'):
                load_physical_model({**descriptor, 'path':str(destination), 'device':'cpu', 'dtype':'float32'})

    def make_model(self):
        import torch
        from adapters.deepseek_vl2_dense import native_classes
        native_classes(json.loads((ROOT / 'configs/model-deepseek-vl2.json').read_text()))
        from deepseek_vl2.models.modeling_deepseek import DeepseekV2Config, DeepseekV2ForCausalLM
        dims = tiny_dimensions()
        config = DeepseekV2Config(**{k: v for k, v in dims.items() if k != 'model_type'},
            vocab_size=32, num_attention_heads=2, num_key_value_heads=2,
            kv_lora_rank=8, q_lora_rank=None, qk_rope_head_dim=4, qk_nope_head_dim=4,
            v_head_dim=4, n_group=1, topk_group=1, topk_method='noaux_tc', scoring_func='sigmoid',
            max_position_embeddings=64)
        torch.manual_seed(123)
        language = DeepseekV2ForCausalLM(config).float().eval()
        language.model.layers[1].mlp.gate.e_score_correction_bias.data.zero_()
        model = SimpleNamespace(config=SimpleNamespace(language_config=config), language=language)
        return model, dims

    def test_native_logits_match_masking_and_router_weights_are_unchanged(self):
        import torch
        model, dims = self.make_model()
        reference = copy.deepcopy(model)
        gate = model.language.model.layers[1].mlp.gate
        router = {k: v.clone() for k, v in gate.state_dict().items()}
        recipe = sp.make_recipe(scores_for(dims), [24, 8, 16], dims, SELECTION)
        with torch.no_grad():
            for experts, retained in zip(sp.modules(reference, dims), recipe['retained_channels']):
                for expert, keep in zip(experts, retained):
                    removed = sorted(set(range(expert.down_proj.in_features)) - set(keep))
                    expert.down_proj.weight[:, removed] = 0
            sp.apply_recipe(model, recipe)
            inputs = torch.tensor([[1, 5, 7, 8, 9]])
            expected = reference.language(inputs, use_cache=False).logits
            actual = model.language(inputs, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(sp.audit_recipe(model, recipe)['remaining_ffn_parameters'],
                         recipe['original_ffn_parameters'] - recipe['removed_ffn_parameters'])
        self.assertTrue(all(torch.equal(v, gate.state_dict()[k]) for k, v in router.items()))
        with self.assertRaisesRegex(ValueError, 'fresh Dense'):
            sp.apply_recipe(model, recipe)

    def test_routing_moments_match_all_token_weighted_reference(self):
        import torch
        from skills.deepseek_vl2_calibration import RoutingMoments
        from skills.structured_pruning import STRATA
        model, dims = self.make_model()
        layer = model.language.model.layers[1].mlp
        x = torch.randn(1, 5, dims['hidden_size'])
        image_mask = torch.tensor([False, True, True, False, True])
        with torch.inference_mode():
            expected = layer(x)
            ids, weights, _ = layer.gate(x)
            collector = RoutingMoments(model)
            try:
                collector.start_sample(image_mask)
                observed = layer(x)
                self.assertTrue(torch.equal(expected, observed))
            finally:
                collector.close()
            for expert_index, expert in enumerate(layer.experts):
                weight = torch.where(ids == expert_index, weights, 0).sum(dim=1)
                values = expert.act_fn(expert.gate_proj(x[0])) * expert.up_proj(x[0])
                values = values.float() * weight[:, None].float()
                for mask, name in ((image_mask, STRATA[0]), (~image_mask, STRATA[1])):
                    state = collector.states[1][expert_index][name]
                    torch.testing.assert_close(state['squares'], values[mask].square().sum(dim=0))
                    torch.testing.assert_close(state['sums'], values[mask].sum(dim=0))
                    self.assertEqual(state['hits'], int(((ids == expert_index).any(dim=1) & mask).sum()))
                    self.assertEqual(collector.counts[1][name], int(mask.sum()))
        self.assertFalse(collector.handles)


if __name__ == '__main__':
    unittest.main()
