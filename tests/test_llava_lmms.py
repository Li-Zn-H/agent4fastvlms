"""Native answer trimming and measured decoder cache evidence for LLaVA."""

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from adapters.llava_onevision_dense import LlavaVisionTokenController
from search.lmms_llava_stack_entry import restore_native_prefix


class LlavaLmmsTests(unittest.TestCase):
    def test_vllm_bridge_preserves_native_processor_call(self):
        from search.lmms_llava_vllm_entry import CapturingLlavaProcessor
        from unittest.mock import Mock

        processor, model = Mock(return_value={"native": True}), SimpleNamespace()
        wrapper = CapturingLlavaProcessor(processor, model)
        image = object()
        kwargs = {"images": [image], "videos": None, "text": "native prompt", "return_tensors": "pt"}
        self.assertEqual(wrapper(**kwargs), {"native": True})
        processor.assert_called_once_with(**kwargs)
        self.assertEqual(model.captured, ("native prompt", [image]))
        with self.assertRaisesRegex(ValueError, "native image"):
            wrapper(**kwargs, do_resize=False)
        with self.assertRaisesRegex(ValueError, "one native"):
            wrapper(images=[image, image], text="prompt")

    def test_gate_requires_consistent_compacted_1d_positions(self):
        from search.lmms_gate import _mechanism_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = {"image_tokens_before": 10, "image_tokens_after": 5,
                     "position_type": "1d_compacted", "positions_after": 5,
                     "selected_indices_sha256": "hash", "token_keep_ratio": .5}
            row = {"backend": "vllm", "controller": "vllm_score_based_post_encoder",
                   "token_keep_ratio": .5, "sample_index": 0, "image_tokens_before": 10,
                   "image_tokens_after": 5, "images": [image], "native_prompt_tokens": 12,
                   "decoder_tokens_after": 7, "native_prompt_sha256": "hash", "prompt_verified": True}
            path = root / "pope-token-mechanism.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(_mechanism_file(root, "pope", 1, .5, backend="vllm"), path)
            image["positions_after"] = 10
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Invalid vLLM"):
                _mechanism_file(root, "pope", 1, .5, backend="vllm")

    def test_compaction_preserves_generated_answer_for_native_trimming(self):
        import torch

        class Decoder(torch.nn.Module):
            def forward(self, length):
                return SimpleNamespace(past_key_values=SimpleNamespace(get_seq_length=lambda: length))

        class Model:
            def __init__(self):
                self.config = SimpleNamespace(image_token_index=9)
                self.model = SimpleNamespace(language_model=Decoder(), get_image_features=self.features)

            def features(self):
                return SimpleNamespace(pooler_output=[torch.arange(12, dtype=torch.float32).reshape(4, 3)])

            def generate(self, input_ids, attention_mask, **kwargs):
                if kwargs.get("use_cache") is not True:
                    raise ValueError("quantizer calibration cache setting was not overridden")
                self_mask = attention_mask.tolist()
                if attention_mask.shape != input_ids.shape:
                    raise ValueError("attention mask still uses the unpruned prompt")
                if self_mask != [[0, 1, 1, 1]]:
                    raise ValueError("text padding changed during Token selection")
                self.model.get_image_features()
                self.model.language_model(input_ids.shape[1])
                self.model.language_model(input_ids.shape[1] + 1)
                return torch.cat((input_ids, torch.tensor([[101, 102]])), dim=1)

        model = Model()
        selection = {"score_dtype": "float32", "coverage_grid": {"rows": 1, "columns": 1}}
        controller = LlavaVisionTokenController(model, selection, .5)
        original_ids = torch.tensor([[1, 9, 9, 9, 9, 2]])
        inputs = {"input_ids": original_ids, "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1]])}
        output = controller.generate(inputs, {})
        metrics = controller.last_metrics
        self.assertEqual(metrics["decoder_tokens_after"], 4)
        self.assertEqual(metrics["decode_steps"], 1)
        self.assertEqual(metrics["output_tokens"], 2)
        self.assertEqual(metrics["kv_tokens_after_last_forward"], 5)
        restored = restore_native_prefix(output, original_ids, metrics["decoder_tokens_after"])
        self.assertEqual(restored[:, original_ids.shape[1]:].tolist(), [[101, 102]])
        self.assertEqual(restored[:, :original_ids.shape[1]].tolist(), original_ids.tolist())
        self.assertEqual(len(model.model.language_model._forward_hooks), 0)
        self.assertEqual(len(model.model.language_model._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
