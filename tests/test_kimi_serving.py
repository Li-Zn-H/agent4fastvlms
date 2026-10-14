"""Kimi native prompt, routed topology, suffix and end-to-end serving contracts."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search.kimi_serving import generation_receipt, sampling_options, validate_runtime_observation

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {"score_dtype": "float32", "coverage_grid": {"rows": 2, "columns": 2}}
RUNTIME = {"keep_ratio": .5, "selection": SELECTION}


def trace(grid):
    from skills.kimi_vl_vllm import retained_count
    ph, pw = grid
    after = retained_count(ph // 2, pw // 2, RUNTIME)
    return {"method": "vllm_score_based_post_encoder", "token_keep_ratio": .5,
            "patch_grid": list(grid), "merged_grid": [ph // 2, pw // 2],
            "image_tokens_before": ph * pw // 4, "image_tokens_after": after,
            "positions_after": after, "position_type": "1d_compacted",
            "selected_indices_sha256": "fixture-selection"}


class ServingContractTests(unittest.TestCase):
    def test_native_grid_compaction_audits_entire_prompt_and_receipt_counts(self):
        from search.lmms_gate import _mechanism_file
        for grid in ((8, 8), (12, 4), (2, 2)):
            item = trace(grid)
            native = [1, 7] + [98] * item["image_tokens_before"] + [8, 9]
            output = SimpleNamespace(prompt_token_ids=[1, 7] + [98] * item["image_tokens_after"] + [8, 9],
                                     outputs=[SimpleNamespace(token_ids=[4, 1])])
            args = (native, 98, output, [item], RUNTIME, grid, [2, 2], 0)
            generated, receipt = generation_receipt(*args)
            self.assertEqual(generated, [4, 1])
            if item["image_tokens_after"] < item["image_tokens_before"]:
                with tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    path = root / "fixture-token-mechanism.jsonl"
                    path.write_text(json.dumps(receipt) + "\n")
                    self.assertEqual(_mechanism_file(root, "fixture", 1, .5, backend="vllm"), path)
            for key, value in (("patch_grid", [2, 2]), ("merged_grid", [1, 1]),
                               ("positions_after", 0), ("selected_indices_sha256", "")):
                if item[key] == value:
                    continue
                with self.subTest(grid=grid, key=key), self.assertRaises(RuntimeError):
                    generation_receipt(native, 98, output, [{**item, key: value}], RUNTIME, grid, [2, 2], 0)
            output.prompt_token_ids[-1] = 77
            with self.assertRaisesRegex(RuntimeError, "prompt differs"):
                generation_receipt(*args)

    def test_runtime_requires_exact_router_precision_topology_and_physical_widths(self):
        from test_kimi_vllm import header
        from skills.kimi_vl_vllm import layer_widths
        config = header()
        config["a4fvlm_stack_widths"] = [384, 128, 384]
        config["a4fvlm_kimi_gptq"] = {"vision_padding": {"original_width": 4304, "padded_width": 4352}}
        observed = {"moe_layer_widths": [{"layer": i, **row} for i, row in enumerate(layer_widths(config))],
                    "physical_widths": config["a4fvlm_stack_widths"], "router_logits_dtype": "float32",
                    "router_parameter_dtype": "bfloat16", "native_routing": "noaux_tc_sigmoid_zero_mask",
                    "verified_router_tensors": 2, "expert_topology": {"routed": 4, "shared": 2, "top_k": 2},
                    "moe_backends": ["MARLIN"], "vision_padding": config["a4fvlm_kimi_gptq"]["vision_padding"],
                    "token_runtime": RUNTIME, "token_pruning_enabled": True,
                    "kv_cache_dtypes": ["auto"], "attention_backends": ["FLASH_ATTN_MLA"]}
        validate_runtime_observation(config, observed, RUNTIME, "auto")
        from search.lmms_gate import _serving_receipt
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps(config))
            descriptor = {"adapter": "kimi_vl", "path": str(root), "token_keep_ratio": .5,
                          "token_selection": SELECTION, "runtime": {"kv_cache_dtype": "auto"}}
            receipt = {"backend": "vllm", "model": descriptor, "observed_model": observed}
            saved = root / "fixture-token-mechanism.runtime.json"
            saved.write_text(json.dumps(receipt))
            self.assertEqual(_serving_receipt(root, "fixture", descriptor), receipt)
            # These records satisfy the shared generic fields but violate
            # Kimi's mechanism. Reuse must reject them before another GPU run.
            for key, value in (("router_parameter_dtype", "float32"),
                               ("expert_topology", {"routed": 4, "shared": 2, "top_k": 1}),
                               ("moe_layer_widths", []), ("moe_backends", ["TRITON"]),
                               ("vision_padding", {}),
                               ("token_runtime", {**RUNTIME, "selection": {}})):
                with self.subTest(saved_receipt=key):
                    saved.write_text(json.dumps({**receipt, "observed_model": {**observed, key: value}}))
                    with self.assertRaisesRegex(ValueError, "Kimi serving changed"):
                        _serving_receipt(root, "fixture", descriptor)
        for key, value in (("router_parameter_dtype", "float32"), ("router_logits_dtype", "bfloat16"),
                           ("native_routing", "noaux_tc_inf_mask"), ("verified_router_tensors", 0),
                           ("expert_topology", {"routed": 4, "shared": 2, "top_k": 1}),
                           ("moe_layer_widths", []), ("moe_backends", ["TRITON"]),
                           ("vision_padding", {}), ("token_runtime", None), ("kv_cache_dtypes", ["fp8"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_runtime_observation(config, {**observed, key: value}, RUNTIME, "auto")

    def test_sampling_preserves_native_eos_task_budget_and_stop_strings(self):
        options = {"max_new_tokens": 1024, "do_sample": False, "use_cache": True,
                   "return_dict_in_generate": False}
        before = copy.deepcopy(options)
        sampling = sampling_options(options, 163585, ["", "STOP"])
        self.assertEqual(options, before)
        self.assertEqual(sampling["max_tokens"], 1024)
        self.assertEqual(sampling["stop_token_ids"], [163585])
        self.assertEqual(sampling["stop"], ["STOP"])
        self.assertTrue(sampling["include_stop_str_in_output"])
        self.assertTrue(sampling_options({**options, "eos_token_id": None}, 163585, [])["ignore_eos"])
        for change in ({"do_sample": True}, {"use_cache": False}, {"max_new_tokens": 0}, {"unexpected": True}):
            with self.assertRaisesRegex(ValueError, "Kimi"):
                sampling_options({**options, **change}, 163585, [])


@unittest.skipUnless((ROOT / "models/Kimi-VL-A3B-Instruct/config.json").is_file()
                     and importlib.util.find_spec("transformers"), "native Kimi metadata/environment required")
class NativeServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adapters.kimi_vl_dense import load_processor
        cls.descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        cls.processor = load_processor(cls.descriptor)

    def test_prompt_preflight_matches_original_pixels_path_across_geometries_and_unicode(self):
        from PIL import Image
        from adapters.kimi_vl_dense import prepare_image, prompt_token_ids, unexpanded_prompt
        for size in ((281, 421), (32, 60), (14, 14), (1500, 2049)):
            for question in ("Describe.", "  中文描述 \n", "<image>\nWhat is written?\n"):
                with self.subTest(size=size, question=question), Image.new("RGB", size) as image:
                    native = prepare_image(self.processor, image, question)
                    self.assertEqual(prompt_token_ids(self.processor, question, size), native["input_ids"][0].tolist())
                    self.assertEqual(unexpanded_prompt(self.processor, question).count(self.processor.image_token), 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            unexpanded_prompt(self.processor, "What is <|media_pad|>?")

    def test_matched_timing_decodes_dense_suffix_and_audits_serving_after_timer(self):
        import torch
        from PIL import Image
        from transformers import BatchFeature
        from adapters.kimi_vl_dense import prepare_image, unexpanded_prompt
        from search.kimi_serving import KimiSession, KimiVllmBackend
        from search.lmms_vllm_entry import compact_prompt
        from skills.quantization import digest
        eos = self.processor.tokenizer.eos_token_id
        ids = self.processor.tokenizer.encode("A red object.", add_special_tokens=False) + [eos]
        options = {"max_tokens": 64, "temperature": 0., "ignore_eos": False,
                   "stop_token_ids": [eos], "repetition_penalty": 1.}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_path = root / "image.png"
            Image.new("RGB", (112, 112), (220, 20, 10)).save(image_path)
            with Image.open(image_path) as image:
                inputs = prepare_image(self.processor, image, "Describe.")
            native_ids = inputs["input_ids"][0].tolist()
            item = trace(inputs["image_grid_hws"][0].tolist())
            compacted = compact_prompt(native_ids, 163605, [(item["image_tokens_before"], item["image_tokens_after"])])
            output = SimpleNamespace(prompt_token_ids=compacted, outputs=[SimpleNamespace(token_ids=ids)])
            captures, rows = [], []
            request = {"sample_id": "fixture", "image": str(image_path), "prompt": "Describe."}
            for backend in ("dense", "serving"):
                session = KimiSession.__new__(KimiSession)
                session.snap, session.backend = {"generation": options}, backend
                session.processor, session.max_length = self.processor, 131072
                def generate(**kwargs):
                    captures.append(kwargs)
                    return torch.tensor([native_ids + ids])
                session.model = SimpleNamespace(generate=generate)
                actual = KimiVllmBackend.__new__(KimiVllmBackend)
                actual.descriptor = {"token_keep_ratio": .5, "token_selection": SELECTION}
                actual.index, actual.trace_offset = 0, 0
                actual.receipt, actual.trace = root / "mechanism.jsonl", root / "selection.jsonl"
                actual.trace.write_text(json.dumps(item) + "\n")
                def infer(image, prompt, sampling):
                    self.assertEqual(prompt, unexpanded_prompt(self.processor, "Describe."))
                    self.assertEqual(sampling, {**options, "detokenize": False})
                    return output
                actual.infer, actual.engine = infer, SimpleNamespace(reset_prefix_cache=lambda: True)
                session.native = actual
                original_audit = actual.audit
                with patch("torch.cuda.synchronize"), patch.object(BatchFeature, "to", lambda self, *a, **k: self), \
                     patch("search.kimi_serving.perf_counter", side_effect=[10., 12.]) as timer, \
                     patch.object(actual, "audit") as audit:
                    def after_timer(*args):
                        self.assertEqual(timer.call_count, 2)
                        return original_audit(*args)
                    audit.side_effect = after_timer
                    rows.append(session.generate(request))
                    if backend == "serving":
                        audit.assert_called_once()
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
            self.assertIsNone(rows[0]["token_mechanism"])
            self.assertTrue(rows[1]["token_mechanism"]["prompt_verified"])


@unittest.skipUnless(importlib.util.find_spec("lmms_eval"), "isolated LMMS host required")
class ServingLMMSContractTests(unittest.TestCase):
    def test_vllm_native_task_dispatch_preserves_prompt_budget_and_stop_strings(self):
        import torch
        from PIL import Image
        from adapters.kimi_vl_dense import load_processor
        from search.kimi_lmms import KimiVLLMMS
        descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        processor = load_processor(descriptor)
        config = {**descriptor, "execution_backend": "vllm", "stack_adapter": True}
        ids = processor.tokenizer.encode("red", add_special_tokens=False)
        native = SimpleNamespace(generate=lambda *args: (ids, {}))
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"A4FVLM_LMMS_KIMI_MODEL": json.dumps(config),
                                     "A4FVLMSP_TOKEN_RECEIPT": "unused.jsonl",
                                     "A4FVLM_LMMS_KIMI_INPUT_RECEIPT": str(Path(directory) / "inputs.jsonl")}), \
             patch("search.kimi_lmms.validate_native_config"), \
             patch("search.kimi_serving.KimiVllmBackend", return_value=native) as build, \
             patch("search.kimi_lmms.NativeBackend", side_effect=AssertionError("vLLM must not launch native GPU")):
            model = KimiVLLMMS(pretrained=str(ROOT / descriptor["path"]))
        self.assertTrue(model.serving)
        build.assert_called_once_with({**config, "device": "cuda:0"}, "unused.jsonl")
        model.task_dict = {"fixture": {"test": [{}]}}
        model.cache_hook = SimpleNamespace(add_partial=lambda *args: None)
        with tempfile.TemporaryDirectory() as directory, Image.new("RGB", (112, 112)) as image, \
             patch.object(native, "generate", return_value=(ids, {})) as generate:
            model.input_receipt = Path(directory) / "inputs.jsonl"
            request = SimpleNamespace(args=("What color?", {"max_new_tokens": 17, "until": ["STOP"]},
                                            lambda _: [image], 0, "fixture", "test"))
            self.assertEqual(model.generate_until([request]), ["red"])
            args = generate.call_args.args
            self.assertIs(args[0], image)
            self.assertEqual(args[1], "What color?")
            self.assertEqual(args[-1]["max_tokens"], 17)
            self.assertEqual(args[-1]["stop"], ["STOP"])
            self.assertEqual(args[-1]["stop_token_ids"], [163585])
            from search.kimi_lmms_identity import digest
            observed = json.loads(model.input_receipt.read_text())
            self.assertEqual(observed["prompt"], "What color?")
            self.assertEqual(observed["response_sha256"], digest("red"))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
