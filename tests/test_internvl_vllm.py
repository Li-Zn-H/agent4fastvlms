"""CPU contracts for native vLLM preprocessing and all-tile Token selection."""
import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from skills.internvl35_vllm import select_tiles, tile_keep_count

MODEL = Path("models/InternVL3_5-14B")
RUNTIME = {"keep_ratio": .5, "selection": {"score_dtype": "float32",
                                         "coverage_grid": {"rows": 6, "columns": 6}}}


class TileTests(unittest.TestCase):
    @unittest.skipUnless((MODEL / 'config.json').exists(), 'native InternVL config required')
    def test_runtime_preparation_defaults_to_spawn_and_preserves_explicit_override(self):
        import os
        from unittest.mock import patch
        from skills.stack_vllm import prepare_vllm_artifact
        with patch.dict(os.environ, {}, clear=True):
            prepared = prepare_vllm_artifact(MODEL)
            self.assertEqual(prepared['worker_start_method'], 'spawn')
            self.assertEqual(os.environ['VLLM_WORKER_MULTIPROC_METHOD'], 'spawn')
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'
            self.assertEqual(prepare_vllm_artifact(MODEL)['worker_start_method'], 'fork')

    def test_serving_receipt_checks_actual_native_prompt_and_all_tiles(self):
        from search.lmms_internvl_vllm_entry import generation_receipt, unexpanded_prompt
        from search.lmms_gate import _mechanism_file
        import json
        import tempfile
        native = [1] + [98] * 768 + [2, 3]
        compacted = [1] + [98] * 384 + [2, 3]
        output = SimpleNamespace(prompt_token_ids=compacted, outputs=[SimpleNamespace(token_ids=[4, 5])])
        traces = [{"position_type": "1d_compacted", "positions_after": 384, "image_tokens_before": 768,
                   "image_tokens_after": 384, "tiles_selected": 3, "token_keep_ratio": .5,
                   "selected_indices_sha256": "selected"}]
        generated, receipt = generation_receipt(native, 98, output, traces, .5, 0)
        self.assertEqual(generated, [4, 5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture-token-mechanism.jsonl"
            path.write_text(json.dumps(receipt) + "\n")
            self.assertEqual(_mechanism_file(Path(directory), "fixture", 1, .5, backend="vllm"), path)
        for changed in ("text", "missing_tile", "missing_selection"):
            result, images = copy.deepcopy(output), copy.deepcopy(traces)
            if changed == "text":
                result.prompt_token_ids[-1] = 100
            elif changed == "missing_tile":
                images[0]["tiles_selected"] = 2
            else:
                images.clear()
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                generation_receipt(native, 98, result, images, .5, 0)
        prompt = "prefix <img>" + "<IMG_CONTEXT>" * 768 + "</img> suffix"
        self.assertEqual(unexpanded_prompt(prompt, 768), "prefix <image> suffix")
        with self.assertRaises(ValueError):
            unexpanded_prompt(prompt, 384)

    def test_different_tile_scores_do_not_erase_low_score_tiles_or_thumbnail(self):
        import torch
        features = torch.cat([torch.full((256, 8), score) for score in (100., 1., 10.)])
        indices = select_tiles(features, RUNTIME)
        self.assertEqual(indices.numel(), 384)
        self.assertTrue(bool((indices[1:] > indices[:-1]).all()))
        self.assertEqual(torch.bincount(indices // 256).tolist(), [128, 128, 128])
        self.assertEqual(tile_keep_count({**RUNTIME, "keep_ratio": .01}), 36)
        with self.assertRaises(ValueError):
            select_tiles(features[:-1], RUNTIME)


@unittest.skipUnless(importlib.util.find_spec("vllm") and (MODEL / "config.json").exists(),
                     "isolated Runtime environment and native config required")
class RuntimeProcessorTests(unittest.TestCase):
    def test_native_lmms_pixels_survive_vllm_exif_normalization(self):
        from PIL import Image
        from vllm.multimodal.image import normalize_image
        from search.lmms_internvl_vllm_entry import InternVLVllmBackend

        image = Image.new("RGB", (73, 41), "blue")
        image.putpixel((2, 3), (255, 0, 0))
        image.getexif()[274] = 6
        image.info["exif"] = image.getexif().tobytes()
        expected = image.tobytes()
        self.assertEqual(normalize_image(image).size, (41, 73))
        def generate(requests, sampling, use_tqdm):
            request = requests[0]
            normalized = normalize_image(request["multi_modal_data"]["image"])
            self.assertEqual(normalized.size, image.size)
            self.assertEqual(normalized.tobytes(), expected)
            self.assertEqual(request["prompt"], "native prompt")
            self.assertEqual(request["multi_modal_uuids"]["image"], "internvl-native-0")
            return ["generated"]
        backend = object.__new__(InternVLVllmBackend)
        backend.engine = SimpleNamespace(generate=generate)
        backend.index = 0
        self.assertEqual(backend.infer(image, "native prompt", {"temperature": 0, "max_tokens": 2}),
                         "generated")
        self.assertEqual(image.getexif()[274], 6)
        self.assertEqual(image.tobytes(), expected)

    def test_native_lmms_transparent_pixels_are_not_composited_by_vllm(self):
        from PIL import Image
        from vllm.multimodal.image import convert_image_mode, normalize_image
        from search.lmms_internvl_vllm_entry import InternVLVllmBackend

        image = Image.new("RGBA", (4, 3), (10, 20, 30, 0))
        expected = image.convert("RGB").tobytes()
        self.assertNotEqual(convert_image_mode(image, "RGB").tobytes(), expected)
        def generate(requests, sampling, use_tqdm):
            observed = convert_image_mode(normalize_image(requests[0]["multi_modal_data"]["image"]), "RGB")
            self.assertEqual(observed.tobytes(), expected)
            return ["generated"]
        backend = object.__new__(InternVLVllmBackend)
        backend.engine, backend.index = SimpleNamespace(generate=generate), 0
        self.assertEqual(backend.infer(image, "native prompt", {"temperature": 0, "max_tokens": 2}),
                         "generated")
        self.assertEqual(image.mode, "RGBA")

    def test_actual_vllm_processor_retains_native_pixels_ties_and_compacted_prompt(self):
        self.check_native_processor(MODEL)

    def check_native_processor(self, model_path):
        import torch
        from PIL import Image
        from lmms_eval.models.simple.internvl3 import load_image
        from adapters.internvl35_dense import load_processor
        from skills.internvl35_vllm import processor_classes, create_model_class
        from skills.vllm_token import configure_runtime
        from search.lmms_vllm_entry import compact_prompt
        prior = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, prior)
        configure_runtime(RUNTIME)
        self.addCleanup(configure_runtime, None)
        NativeImage, NativeProcessor, _ = processor_classes()
        native = load_processor({"path": str(model_path), "dtype": "bfloat16"})
        processor = NativeProcessor(image_processor=NativeImage(448, 1, 12, True, True),
                                    tokenizer=native.tokenizer, image_seq_length=256)
        for size in ((448, 448), (896, 448), (1344, 896), (2257, 1764), (2000, 100), (100, 2000)):
            with self.subTest(size=size):
                image = Image.effect_noise(size, 20).convert("RGB")
                pixels = load_image(image)
                conversation = native.conversation_factory(native.config.template)
                conversation.append_message(conversation.roles[0], "<image>\nDescribe.")
                conversation.append_message(conversation.roles[1], None)
                observed = processor(text=conversation.get_prompt(), images=[image], return_tensors="pt")
                torch.testing.assert_close(observed["pixel_values_flat"], pixels, rtol=0, atol=0)
                self.assertEqual(processor.get_num_image_tokens(image_width=size[0], image_height=size[1]), len(pixels) * 128)
                ids = native.tokenizer(native.prompt("Describe.", len(pixels)))["input_ids"]
                expected = compact_prompt(ids, native.image_token_id, [(len(pixels) * 256, len(pixels) * 128)])
                self.assertEqual(observed["input_ids"][0].tolist(), expected)
        cls = create_model_class()
        self.assertEqual(cls.__name__, "A4FVLMInternVLChatModel")


if __name__ == "__main__":
    unittest.main()
