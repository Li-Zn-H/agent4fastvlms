"""Native Kimi task command and environment binding without GPU work."""

import json
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from search.lmms import build_command, cache_environment
from search.lmms_kimi_entry import validate_native_config

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('lmms_eval'), 'isolated LMMS host required')
class RegistryTests(unittest.TestCase):
    def test_real_manifest_resolves_native_class_without_loading_gpu_weights(self):
        from lmms_eval.models import MODEL_REGISTRY_V2
        from search.lmms_kimi_entry import register_native_model
        from search.kimi_lmms import KimiVLLMMS
        register_native_model()
        self.assertIs(MODEL_REGISTRY_V2.get_model_class('kimi_vl'), KimiVLLMMS)


@unittest.skipUnless((ROOT / 'models/Kimi-VL-A3B-Instruct/config.json').is_file(),
                     'local Kimi metadata required')
class CommandTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/model-kimi-vl.json').read_text())

    def test_native_tasks_remain_full_and_have_dedicated_entry(self):
        for task in ('chartqa', 'mmstar', 'textvqa_val', 'docvqa_val', 'pope'):
            command = build_command(self.config, task, Path('/tmp/kimi-contract'))
            self.assertTrue(command[1].endswith('search/lmms_kimi_entry.py'))
            self.assertEqual(command[command.index('--model') + 1], 'kimi_vl')
            self.assertNotIn('--limit', command)
            self.assertNotIn('--gen_kwargs', command)
            self.assertIn('--log_samples', command)

    def test_environment_removes_stale_other_model_and_token_settings(self):
        with patch.dict(os.environ, {'A4FVLM_LMMS_DEEPSEEK_MODEL': 'stale',
                                     'A4FVLM_LMMS_KIMI_MODEL': 'stale',
                                     'A4FVLMSP_TOKEN_KEEP_RATIO': '.5'}):
            environment = cache_environment(self.config)
        self.assertEqual(json.loads(environment['A4FVLM_LMMS_KIMI_MODEL']), self.config)
        self.assertNotIn('A4FVLM_LMMS_DEEPSEEK_MODEL', environment)
        self.assertNotIn('A4FVLMSP_TOKEN_KEEP_RATIO', environment)

    def test_dense_rejects_unvalidated_stack_or_thinking(self):
        for changed in ({'stack_adapter': True}, {'protocol': 'thinking'},
                        {'execution_backend': 'vllm'}, {'token_keep_ratio': .5}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_native_config({**self.config, **changed})

    def test_optimized_native_descriptor_requires_token_isolation_and_physical_gptq(self):
        import sys
        config = {**self.config, 'stack_adapter': True, 'token_keep_ratio': .5,
                  'token_selection': {'score_dtype': 'float32', 'coverage_grid': {'rows': 1, 'columns': 1}},
                  'native_worker_python': sys.executable}
        for change, message in (({}, 'freshly quantized physical Parent'),
                                ({'native_worker_python': 'missing/python'}, 'native_worker_python'),
                                ({'token_selection': {}}, 'selection policy'),
                                ({'token_keep_ratio': 1}, 'keep ratio'),
                                ({'execution_backend': 'vllm'}, 'native context 131072'),
                                ({'execution_backend': 'vllm', 'runtime': {'vllm_max_model_len': 131072}},
                                 'freshly quantized physical Parent')):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                build_command({**config, **change}, 'mmstar', Path('unused'))
