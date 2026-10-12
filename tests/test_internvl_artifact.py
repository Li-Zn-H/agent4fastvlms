"""Native image and physical artifact contracts; CPU and small random weights."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODEL = Path("models/InternVL3_5-14B")


class ImageProtocolTests(unittest.TestCase):
    @unittest.skipUnless((MODEL / "config.json").exists(), "native InternVL code/tokenizer fixture required")
    def test_artifact_processor_retains_physical_architecture_and_native_prompt(self):
        from adapters.internvl35_dense import load_processor
        from skills.internvl35_artifact import STACK_ARCHITECTURE

        dense = load_processor({"path": str(MODEL), "dtype": "bfloat16"})
        header = json.loads((MODEL / "config.json").read_text())
        header.update(architectures=[STACK_ARCHITECTURE],
                      a4fvlm_stack_widths=[17280, 17152] * 20,
                      a4fvlm_stack_recipe_digest="processor-test-only",
                      a4fvlm_stack_physical_compaction=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dense.save_pretrained(path)
            (path / "config.json").write_text(json.dumps(header))
            artifact = load_processor({"path": str(path), "dtype": "bfloat16"}, artifact=True)
            self.assertEqual(artifact.config.architectures, [STACK_ARCHITECTURE])
            for tiles in (1, 7, 13):
                prompt = dense.prompt("Describe the image.", tiles)
                self.assertEqual(artifact.prompt("Describe the image.", tiles), prompt)
                self.assertEqual(artifact.tokenizer(prompt)["input_ids"], dense.tokenizer(prompt)["input_ids"])
            # A Dense architecture cannot carry physical widths. Validation
            # must inspect the saved architecture before any normalization.
            header["architectures"] = ["InternVLChatModel"]
            (path / "config.json").write_text(json.dumps(header))
            with self.assertRaisesRegex(ValueError, "physical width metadata"):
                load_processor({"path": str(path), "dtype": "bfloat16"}, artifact=True)

    def test_isolated_transform_matches_pinned_native_pixels_including_ties(self):
        import torch
        from PIL import Image
        from lmms_eval.models.simple.internvl3 import load_image as native
        from adapters.internvl35_processing import load_image

        prior = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for size in ((448, 448), (896, 448), (448, 896), (1344, 896),
                         (2257, 1764), (100, 100), (2000, 100), (100, 2000)):
                with self.subTest(size=size):
                    # Nonuniform pixels expose ordering, crop and resampling changes.
                    image = Image.effect_noise(size, 40).convert("RGB")
                    torch.testing.assert_close(load_image(image), native(image), rtol=0, atol=0)
        finally:
            torch.set_num_threads(prior)


@unittest.skipUnless((MODEL / "config.json").exists(), "native InternVL code fixture required")
class ArtifactTests(unittest.TestCase):
    model_path = MODEL
    adapter_id = "internvl35"

    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoConfig, AutoModel
        cls.torch = torch
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(131)
        config = AutoConfig.from_pretrained(cls.model_path, trust_remote_code=True, local_files_only=True)
        config.llm_config.hidden_size = 128
        config.llm_config.intermediate_size = 256
        config.llm_config.num_hidden_layers = 2
        config.llm_config.layer_types = ["full_attention"] * 2
        config.llm_config.num_attention_heads = 4
        config.llm_config.num_key_value_heads = 2
        config.llm_config.head_dim = 32
        config.llm_config.vocab_size = 128
        config.llm_config.eos_token_id = 2
        config.llm_config.bos_token_id = 1
        config.llm_config.pad_token_id = 0
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 2
        config.vision_config.num_attention_heads = 4
        config.vision_config.image_size = 112
        config.force_image_size = 112
        cls.template = AutoModel.from_config(config, trust_remote_code=True, use_flash_attn=False).float().eval()
        cls.template.language_model.set_attn_implementation("sdpa")
        cls.template.img_context_token_id = 98

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.prior_threads)

    def parent(self, destination):
        from skills.structured_pruning import apply_recipe, make_recipe
        from skills.stack_model import write_pruned_parent
        model = copy.deepcopy(self.template)
        dimensions = {"num_hidden_layers": 2, "hidden_size": 128, "intermediate_size": 256}
        recipe = make_recipe([list(range(256))] * 2, [128, 256], dimensions,
                             {"channel_group_size": 128, "protected_layers": []})
        apply_recipe(model, recipe)
        def save_processor(path):
            for source in self.model_path.glob("*.py"):
                shutil.copyfile(source, Path(path) / source.name)
        processor = SimpleNamespace(save_pretrained=save_processor, image_token_id=98,
                                    config=model.config,
                                    conversation_factory=lambda name: SimpleNamespace(sep="eos"),
                                    tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda value: 2))
        write_pruned_parent(model, processor, recipe, destination)
        return model, processor

    def load(self, destination, processor):
        from skills.stack_model import load_stack_model
        with patch("adapters.internvl35_dense.load_processor", return_value=processor):
            return load_stack_model({"adapter": self.adapter_id, "path": str(destination),
                                     "dtype": "float32", "device": "cpu", "attn_implementation": "sdpa"})[0]

    def test_physical_parent_roundtrip_preserves_exact_weights_and_native_generation(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent"
            model, processor = self.parent(path)
            reloaded = self.load(path, processor)
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[name], weight, rtol=0, atol=0)
        self.assertEqual([x.mlp.gate_proj.out_features for x in reloaded.language_model.model.layers], [128, 256])
        ids = torch.tensor([[10] + [98] * 16 + [11, 12]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "pixel_values": torch.randn(1, 3, 112, 112)}
        options = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0, "eos_token_id": None}
        with torch.inference_mode():
            torch.testing.assert_close(model.generate(**inputs, **options),
                                       reloaded.generate(**inputs, **options), rtol=0, atol=0)

    def test_changed_parent_widths_and_missing_tensor_are_rejected(self):
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent"
            _, processor = self.parent(path)
            manifest_path = path / "stack-parent.json"
            original = manifest_path.read_text()
            manifest = json.loads(original)
            manifest["widths"] = [256, 128]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "parent manifest"):
                self.load(path, processor)
            manifest_path.write_text(original)
            tensors = load_file(str(path / "model.safetensors"))
            tensors.pop("language_model.model.layers.0.mlp.down_proj.weight")
            save_file(tensors, str(path / "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                self.load(path, processor)


if __name__ == "__main__":
    unittest.main()
