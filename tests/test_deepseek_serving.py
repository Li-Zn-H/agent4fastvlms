"""Native DeepSeek serving prompts, protected-layout receipts and timing scope."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search.deepseek_serving import sampling_options, generation_receipt, DeepSeekVllmBackend

SELECTION = {"score_dtype": "float32", "coverage_grid": {"rows": 4, "columns": 4}}


def trace(nx=2, ny=1):
    from skills.deepseek_vl2_token import image_layout
    from skills.deepseek_vl2_vllm import image_keep_count
    tiles, protected, count = image_layout(14, nx, ny)
    keep = image_keep_count(14, nx, ny, {"keep_ratio": .5, "selection": SELECTION})
    return {"token_keep_ratio": .5, "image_tokens_before": count, "image_tokens_after": keep,
            "positions_after": keep, "position_type": "1d_compacted", "tiles_selected": len(tiles),
            "protected_layout_tokens": len(protected), "selected_indices_sha256": "fixture-selection"}


class ServingContractTests(unittest.TestCase):
    def test_sampling_preserves_native_budget_eos_stops_and_rejects_unsupported_generation(self):
        options = dict(max_new_tokens=1024, do_sample=False, use_cache=True, return_dict_in_generate=False)
        before = copy.deepcopy(options)
        actual = sampling_options(options, 1, ["", "STOP"])
        self.assertEqual(options, before)
        self.assertEqual(actual["max_tokens"], 1024)
        self.assertEqual(actual["stop_token_ids"], [1])
        self.assertEqual(actual["stop"], ["STOP"])
        self.assertTrue(actual["include_stop_str_in_output"])
        self.assertFalse(actual["ignore_eos"])
        self.assertTrue(sampling_options({**options, "eos_token_id": None}, 1, [])["ignore_eos"])
        for change in ({"do_sample": True}, {"num_beams": 2}, {"use_cache": False},
                       {"max_new_tokens": 0}, {"unrecognized_option": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                sampling_options({**options, **change}, 1, [])

    def test_native_global_local_separators_and_full_prompt_are_audited(self):
        from search.lmms_gate import _mechanism_file
        for nx, ny in ((1, 1), (2, 1), (1, 9), (3, 2)):
            image = trace(nx, ny)
            native = [1, 7] + [98] * image["image_tokens_before"] + [8, 9]
            output = SimpleNamespace(prompt_token_ids=[1, 7] + [98] * image["image_tokens_after"] + [8, 9],
                                     outputs=[SimpleNamespace(token_ids=[4, 1])])
            args = (native, 98, output, [image], .5, SELECTION, 14, [nx, ny], 0)
            generated, receipt = generation_receipt(*args)
            self.assertEqual(generated, [4, 1])
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = root / "fixture-token-mechanism.jsonl"
                path.write_text(json.dumps(receipt) + "\n")
                self.assertEqual(_mechanism_file(root, "fixture", 1, .5, backend="vllm"), path)
            for key, value in (("protected_layout_tokens", 0), ("tiles_selected", 0),
                               ("image_tokens_after", image["image_tokens_after"] - 1),
                               ("selected_indices_sha256", "")):
                invalid = [dict(image, **{key: value})]
                with self.assertRaises(RuntimeError):
                    generation_receipt(native, 98, output, invalid, .5, SELECTION, 14, [nx, ny], 0)
            output.prompt_token_ids[-1] = 77
            with self.assertRaisesRegex(RuntimeError, "prompt differs"):
                generation_receipt(*args)

    def test_runtime_receipt_rejects_changed_vision_grid_expert_width_and_moe_kernel(self):
        from skills.deepseek_vl2_vllm import layer_widths
        header = json.loads(Path("models/DeepSeek-VL2/config.json").read_text())
        # Export materializes native defaults; the original source omits 384.
        header["vision_config"]["image_size"] = 384
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        rows = layer_widths(header)
        widths = [v for row in rows for v in row.values()]
        header["a4fvlm_stack_widths"] = widths
        (Path(folder.name) / "config.json").write_text(json.dumps(header))
        backend = DeepSeekVllmBackend.__new__(DeepSeekVllmBackend)
        backend.descriptor = {"path": folder.name, "token_keep_ratio": .5,
                              "token_selection": SELECTION, "runtime": {"vllm_max_model_len": 4096}}
        observed = dict(token_pruning_enabled=True, token_runtime={"keep_ratio": .5, "selection": SELECTION},
            physical_widths=widths, kv_cache_dtypes=["auto"], attention_backends=["FLASH_ATTN_MLA"],
            moe_layer_widths=[{"layer": i, **row} for i, row in enumerate(rows)],
            vision_patch_grid=[27, 27], router_logits_dtype="float32", moe_backends=["MARLIN"])
        backend.validate_observed(observed, {"widths": widths}, {"max_model_len": 4096})
        for key, value in (("vision_patch_grid", [28, 28]), ("moe_layer_widths", []),
                           ("router_logits_dtype", "bfloat16"), ("moe_backends", ["TRITON"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                backend.validate_observed({**observed, key: value}, {"widths": widths}, {"max_model_len": 4096})


def native_environment():
    from test_deepseek_lmms_bridge import native_environment as available
    return available()


@unittest.skipUnless(all(importlib.util.find_spec(name) is not None
                        for name in ("torch", "lmms_eval")), "LMMS/runtime packages required")
class ServingInitializationTests(unittest.TestCase):
    def test_serving_initialization_reads_artifact_context_without_native_model_import(self):
        import torch
        from search.deepseek_lmms import DeepSeekVL2LMMS
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder)
            header = {"model_type": "deepseek_vl_v2", "language_config": {
                "max_position_embeddings": 4096, "hidden_size": 2560},
                "vision_config": {"patch_size": 14}, "projector_config": {"n_embed": 2560}}
            (artifact / "config.json").write_text(json.dumps(header))
            descriptor = {"path": str(artifact), "adapter": "deepseek_vl2",
                          "dtype": "bfloat16", "attn_implementation": "eager",
                          "stack_adapter": True, "execution_backend": "vllm"}
            processor, backend = object(), object()
            cuda_initialized = torch.cuda.is_initialized()
            with patch.dict(os.environ, {"A4FVLM_LMMS_DEEPSEEK_MODEL": json.dumps(descriptor),
                    "A4FVLMSP_TOKEN_RECEIPT": str(artifact / "token.jsonl")}), \
                 patch("search.deepseek_lmms.validate_native_config"), \
                 patch("search.deepseek_lmms.load_processor", return_value=processor), \
                 patch("search.deepseek_serving.DeepSeekVllmBackend", return_value=backend) as build, \
                 patch("adapters.deepseek_vl2_dense.native_classes",
                       side_effect=AssertionError("Serving must not import the native CUDA model")):
                model = DeepSeekVL2LMMS(pretrained=str(artifact))
            self.assertTrue(model.serving)
            self.assertEqual(model.max_length, header["language_config"]["max_position_embeddings"])
            self.assertEqual(model.config.language_config.hidden_size, 2560)
            self.assertEqual(model.config.model_type, header["model_type"])
            self.assertIs(model.processor, processor)
            self.assertIs(model.backend, backend)
            build.assert_called_once_with({**descriptor, "device": "cuda:0"}, str(artifact / "token.jsonl"))
            self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)


@unittest.skipUnless(native_environment(), "original native DeepSeek environment required")
class NativeServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from adapters.deepseek_vl2_dense import load_processor
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.descriptor = json.loads(Path("configs/model-deepseek-vl2.json").read_text())
        cls.processor = load_processor(cls.descriptor)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_prompt_preflight_matches_native_processor_across_geometries_and_whitespace(self):
        from PIL import Image
        from adapters.deepseek_vl2_dense import prepare_image
        for size in ((384, 384), (768, 384), (17, 400)):
            for question in ("Describe.", "  中文描述 \n", "<image>\nWhat is written?\n"):
                with self.subTest(size=size, question=question):
                    inputs = prepare_image(self.processor, Image.new("RGB", size), question)
                    self.assertEqual(self.processor.prompt_token_ids(question, size), inputs["input_ids"][0].tolist())
        with self.assertRaises(ValueError):
            self.processor.unexpanded_prompt("<image> <image>")

    def test_snapshot_uses_native_language_eos_without_generation_file(self):
        from search.serving import snapshot
        with patch("search.serving.load_freeze", return_value=({"model": self.descriptor, "id": "test"},
                {"worker_python": ".venv-runtime-deepseek/bin/python"})):
            result = snapshot("configs/serving-deepseek-vl2.json")
        self.assertEqual(result["generation"]["stop_token_ids"], [self.processor.tokenizer.eos_token_id])

    def test_dense_and_serving_keep_suffix_prompt_timing_and_original_context(self):
        import torch
        from PIL import Image
        from transformers import BatchFeature
        from adapters.deepseek_vl2_dense import prepare_image
        from search.deepseek_serving import DeepSeekSession
        from search.lmms_vllm_entry import compact_prompt
        from skills.quantization import digest
        eos = self.processor.tokenizer.eos_token_id
        ids = self.processor.tokenizer.encode("A red object.", add_special_tokens=False) + [eos]
        options = dict(max_tokens=64, temperature=0., ignore_eos=False, stop_token_ids=[eos], repetition_penalty=1.)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_path = root / "image.png"
            Image.new("RGB", (768, 384), (220, 20, 10)).save(image_path)
            with Image.open(image_path) as image:
                inputs = prepare_image(self.processor, image, "Describe.")
            native_ids = inputs["input_ids"][0].tolist()
            item = trace()
            compacted = compact_prompt(native_ids, self.processor.image_token_id,
                                       [(item["image_tokens_before"], item["image_tokens_after"])])
            output = SimpleNamespace(prompt_token_ids=compacted, outputs=[SimpleNamespace(token_ids=ids)])
            request = dict(sample_id="fixture", image=str(image_path), prompt="Describe.")
            captures, rows = [], []
            for backend in ("dense", "serving"):
                session = DeepSeekSession.__new__(DeepSeekSession)
                session.snap, session.backend = {"generation": options}, backend
                session.processor, session.max_length = self.processor, 4096
                def generate(**kwargs):
                    captures.append(kwargs)
                    return torch.tensor([ids])
                session.model = SimpleNamespace(generate=generate)
                actual = DeepSeekVllmBackend.__new__(DeepSeekVllmBackend)
                actual.descriptor = dict(token_keep_ratio=.5, token_selection=SELECTION)
                actual.index, actual.trace_offset = 0, 0
                actual.receipt, actual.trace = root / "mechanism.jsonl", root / "selection.jsonl"
                actual.trace.write_text(json.dumps(item) + "\n")
                def infer(image, prompt, sampling):
                    self.assertEqual(prompt, self.processor.unexpanded_prompt("Describe."))
                    self.assertEqual(sampling, {**options, "detokenize": False})
                    return output
                actual.infer = infer
                actual.engine = SimpleNamespace(reset_prefix_cache=lambda: True)
                session.native = actual
                original_audit = actual.audit
                with patch("torch.cuda.synchronize"), \
                     patch.object(BatchFeature, "to", lambda self, *args, **kwargs: self), \
                     patch("search.deepseek_serving.perf_counter", side_effect=[10., 12.]) as timer, \
                     patch.object(actual, "audit") as audit:
                    def after_timer(*args):
                        self.assertEqual(timer.call_count, 2)
                        return original_audit(*args)
                    audit.side_effect = after_timer
                    rows.append(session.generate(request))
                    if backend == "serving":
                        audit.assert_called_once()
                if backend == "serving":
                    session.max_length = len(native_ids)
                    with patch("torch.cuda.synchronize"), patch.object(actual, "infer") as inference:
                        with self.assertRaisesRegex(ValueError, "no truncation"):
                            session.generate(request)
                        inference.assert_not_called()
            for row in rows:
                self.assertEqual(row["output_text"], "A red object.")
                self.assertEqual(row["output_tokens"], len(ids))
                self.assertEqual(row["native_prompt_sha256"], digest(native_ids))
                self.assertEqual(row["end_to_end_seconds"], 2.)
            self.assertEqual(captures[0]["input_ids"][0].tolist(), native_ids)
            self.assertEqual(captures[0]["eos_token_id"], [eos])
            self.assertIsNone(rows[0]["token_mechanism"])
            self.assertTrue(rows[1]["token_mechanism"]["prompt_verified"])


if __name__ == "__main__":
    unittest.main()
