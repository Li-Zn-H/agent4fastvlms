"""Native Kimi routing/pixel contracts and lazy vLLM construction checks."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from skills.kimi_vl_vllm import layer_widths, compact_processed_image, retained_count, native_grid

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Kimi-VL-A3B-Instruct"
RUNTIME = {"keep_ratio": .5, "selection": {"score_dtype": "float32", "coverage_grid": {"rows": 2, "columns": 2}}}


def header():
    return {"model_type": "kimi_vl", "text_config": {
        "num_hidden_layers": 2, "hidden_size": 128, "intermediate_size": 512,
        "moe_intermediate_size": 256, "n_routed_experts": 4, "n_shared_experts": 2,
        "num_experts_per_tok": 2, "first_k_dense_replace": 1, "q_lora_rank": None},
        "vision_config": {"num_hidden_layers": 1}}


class ContractTests(unittest.TestCase):
    def test_nonuniform_widths_preserve_all_routed_and_shared_branches(self):
        config = header()
        self.assertEqual(layer_widths(config), [{"dense": 512}, {"routed": 256, "shared": 512}])
        config["a4fvlm_stack_widths"] = [384, 128, 384]
        before = copy.deepcopy(config)
        self.assertEqual(layer_widths(config), [{"dense": 384}, {"routed": 128, "shared": 384}])
        self.assertEqual(config, before)
        for invalid in ([128, 128], [128, 512, 128], [128, 127, 128], [128, True, 128]):
            with self.subTest(widths=invalid), self.assertRaises(ValueError):
                layer_widths({**config, "a4fvlm_stack_widths": invalid})

    def test_compaction_keeps_all_native_pixels_grid_and_text_tokens(self):
        import torch
        ids = torch.tensor([[1, 5] + [98] * 16 + [7, 8]])
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "image_grid_hws": torch.tensor([[8, 8]]), "pixel_values": torch.randn(64, 3, 14, 14)}
        keepall = compact_processed_image(copy.deepcopy(batch), 98, [2, 2], None)
        self.assertTrue(torch.equal(keepall["input_ids"], ids))
        actual = compact_processed_image(copy.deepcopy(batch), 98, [2, 2], RUNTIME)
        keep = retained_count(4, 4, RUNTIME)
        self.assertEqual(actual["input_ids"].tolist(), [[1, 5] + [98] * keep + [7, 8]])
        self.assertEqual(actual["attention_mask"].shape, actual["input_ids"].shape)
        for key in ("pixel_values", "image_grid_hws"):
            self.assertTrue(torch.equal(actual[key], batch[key]))
        broken = copy.deepcopy(batch)
        broken["input_ids"][0, 8] = 7
        with self.assertRaisesRegex(ValueError, "placeholder run"):
            compact_processed_image(broken, 98, [2, 2], RUNTIME)

    def test_catalog_import_does_not_initialize_model_packages(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            'import skills.kimi_vl_vllm; import sys; '
            'assert not any(n in sys.modules for n in ("torch","transformers","vllm","gptqmodel"))'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless((MODEL / "config.json").is_file() and importlib.util.find_spec("transformers"),
                     "authenticated Kimi source and model environment required")
class NativeProtocolTests(unittest.TestCase):
    def test_routing_matches_original_gate_including_negative_corrections(self):
        import torch
        from adapters.kimi_vl_dense import native_model_class
        from skills.kimi_vl_vllm import native_router_logits, native_routing
        descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        cls = native_model_class(descriptor)
        config = cls.config_class(**json.loads((MODEL / "config.json").read_text())).text_config
        config.hidden_size, config.n_routed_experts, config.num_experts_per_tok = 128, 8, 2
        config.n_group, config.topk_group, config.routed_scaling_factor = 2, 1, 1.25
        base = cls.__mro__[1] if cls.__module__ == "adapters.kimi_vl_compat" else cls
        Gate = sys.modules[base.__module__].MoEGate
        gate = Gate(config).to(torch.bfloat16).eval()
        generator = torch.Generator().manual_seed(812)
        gate.weight.data.copy_(torch.randn(8, 128, generator=generator, dtype=torch.bfloat16))
        hidden = torch.randn(1, 64, 128, generator=generator, dtype=torch.bfloat16)
        for correction in (torch.zeros(8, dtype=torch.bfloat16),
                           torch.randn(8, generator=generator, dtype=torch.bfloat16),
                           torch.full((8,), -2., dtype=torch.bfloat16)):
            gate.e_score_correction_bias.data.copy_(correction)
            with torch.inference_mode():
                original_ids, original_weights, _ = gate(hidden)
                logits = native_router_logits(hidden.flatten(0, 1), gate.weight)
                weights, ids = native_routing(logits, correction, config)
            self.assertTrue(torch.equal(ids.long(), original_ids))
            self.assertTrue(torch.equal(weights, original_weights))
        with self.assertRaisesRegex(ValueError, "BF16"):
            native_router_logits(hidden.flatten(0, 1), gate.weight.float())

    def test_native_grid_and_wrapped_processor_preserve_original_pixels(self):
        import torch
        from PIL import Image
        from adapters.kimi_vl_dense import load_processor, prepare_image
        from skills.kimi_vl_vllm import processor_class
        from skills.vllm_token import configure_runtime
        descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        native = load_processor(descriptor)
        Wrapper = processor_class(descriptor)
        wrapper = Wrapper(image_processor=native.image_processor, tokenizer=native.tokenizer,
                          chat_template=native.chat_template)
        self.addCleanup(configure_runtime, None)
        for size in ((281, 421), (32, 60), (14, 14), (1500, 2049)):
            with self.subTest(size=size):
                image = Image.new("RGB", size, (23, 124, 210))
                configure_runtime(None)
                reference = prepare_image(native, image, "What color is the image?")
                observed = prepare_image(wrapper, image, "What color is the image?")
                for key in reference:
                    self.assertTrue(torch.equal(reference[key], observed[key]), key)
                ph, pw = reference["image_grid_hws"][0].tolist()
                mh, mw = native.image_processor.merge_kernel_size
                self.assertEqual(native_grid(*size, native.image_processor), (ph // mh, pw // mw))
                configure_runtime(RUNTIME)
                compacted = prepare_image(wrapper, image, "What color is the image?")
                self.assertEqual(int((compacted["input_ids"] == 163605).sum()), retained_count(ph // mh, pw // mw, RUNTIME))
                for key in ("pixel_values", "image_grid_hws"):
                    self.assertTrue(torch.equal(reference[key], compacted[key]), key)


@unittest.skipUnless(importlib.util.find_spec("vllm"), "isolated vLLM interpreter required")
class RuntimePolicyTests(unittest.TestCase):
    @unittest.skipUnless((MODEL / "config.json").is_file(), "authenticated Kimi source required")
    def test_actual_fused_router_factory_preserves_native_64_expert_top6_weights(self):
        import torch
        from adapters.kimi_vl_dense import native_model_class
        from skills.kimi_vl_vllm import native_router_logits, native_routing
        from vllm.model_executor.layers.fused_moe.router.custom_routing_router import CustomRoutingRouter
        from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
        descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        cls = native_model_class(descriptor)
        config = cls.config_class(**json.loads((MODEL / "config.json").read_text())).text_config
        config.hidden_size = 128
        base = cls.__mro__[1] if cls.__module__ == "adapters.kimi_vl_compat" else cls
        gate = sys.modules[base.__module__].MoEGate(config).to(torch.bfloat16).eval()
        generator = torch.Generator().manual_seed(1049)
        gate.weight.data.copy_(torch.randn(64, 128, generator=generator, dtype=torch.bfloat16))
        hidden = torch.randn(1, 83, 128, generator=generator, dtype=torch.bfloat16)
        def routing(hidden_states, gating_output, topk, renormalize):
            self.assertEqual(topk, 6)
            self.assertEqual(renormalize, config.norm_topk_prob)
            return native_routing(gating_output, gate.e_score_correction_bias, config)
        router = create_fused_moe_router(
            top_k=6, global_num_experts=64, renormalize=config.norm_topk_prob,
            custom_routing_function=routing, e_score_correction_bias=gate.e_score_correction_bias,
            scoring_func=config.scoring_func, routed_scaling_factor=1.0)
        self.assertIsInstance(router, CustomRoutingRouter)
        # The callback owns Kimi's scale. Factory dispatch must not substitute
        # biased/grouped routing or normalize/multiply its weights a second time.
        for correction in (torch.zeros(64, dtype=torch.bfloat16),
                           torch.randn(64, generator=generator, dtype=torch.bfloat16),
                           torch.full((64,), -2., dtype=torch.bfloat16)):
            gate.e_score_correction_bias.data.copy_(correction)
            with torch.inference_mode():
                expected_ids, expected_weights, _ = gate(hidden)
                logits = native_router_logits(hidden.flatten(0, 1), gate.weight)
                for index_dtype in (None, torch.int32, torch.int64):
                    weights, ids = router.select_experts(hidden.flatten(0, 1), logits, index_dtype)
                    self.assertTrue(torch.equal(ids.long(), expected_ids))
                    self.assertTrue(torch.equal(weights, expected_weights))
                    self.assertEqual(ids.dtype, index_dtype or torch.int32)
        self.assertIsNone(router.eplb_state)
        self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless((MODEL / "config.json").is_file(), "authenticated Kimi source required")
    def test_vllm_rotary_preserves_native_positions_and_query_key_permutation(self):
        import torch
        from adapters.kimi_vl_dense import native_model_class
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
        descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        cls = native_model_class(descriptor)
        base = cls.__mro__[1] if cls.__module__ == "adapters.kimi_vl_compat" else cls
        native = sys.modules[base.__module__]
        config = json.loads((MODEL / "config.json").read_text())["text_config"]
        dim, maximum, theta = config["qk_rope_head_dim"], config["max_position_embeddings"], config["rope_theta"]
        original = native.DeepseekV3RotaryEmbedding(dim, maximum, theta)
        with set_current_vllm_config(VllmConfig()):
            runtime = RotaryEmbedding(dim, dim, maximum, theta, is_neox_style=False, dtype=torch.bfloat16)
        # Native loading retains FP32 frequencies even with BF16 parameters.
        self.assertEqual(original.inv_freq.dtype, torch.float32)
        positions = torch.tensor([0, 1, 7, 127, 255, 256, 511, 2048, maximum - 1])
        generator = torch.Generator().manual_seed(407)
        q = torch.randn(1, 16, len(positions), dim, generator=generator, dtype=torch.bfloat16)
        k = torch.randn(1, 1, len(positions), dim, generator=generator, dtype=torch.bfloat16)
        cos, sin = original(q, seq_len=maximum)
        runtime_cos, runtime_sin = runtime.get_cos_sin(maximum)
        self.assertTrue(torch.equal(runtime_cos, cos[:, :dim // 2]))
        self.assertTrue(torch.equal(runtime_sin, sin[:, :dim // 2]))
        expected_q, expected_k = native.apply_rotary_pos_emb(q, k, cos, sin, positions.unsqueeze(0))
        observed_q, observed_k = runtime.forward_native(
            positions, q[0].transpose(0, 1).contiguous().flatten(1),
            k[0].transpose(0, 1).contiguous().flatten(1))
        def native_order(tensor, heads):
            tensor = tensor.reshape(len(positions), heads, dim).transpose(0, 1).unsqueeze(0)
            return tensor.reshape(1, heads, len(positions), dim // 2, 2).transpose(-1, -2).flatten(-2)
        # Native returns halves; vLLM returns interleaved pairs. Both queries
        # and cached keys use the same permutation, preserving their dot product.
        self.assertTrue(torch.equal(native_order(observed_q, 16), expected_q))
        self.assertTrue(torch.equal(native_order(observed_k, 1), expected_k))
        self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless((MODEL / "config.json").is_file(), "authenticated Kimi source required")
    def test_actual_vllm_processor_and_cache_preserve_native_pixels_grid_and_compacted_prompt(self):
        import shutil
        import tempfile
        import torch
        from PIL import Image
        from unittest.mock import patch
        from adapters.kimi_vl_dense import load_processor, prepare_image, unexpanded_prompt
        from search.lmms_vllm_entry import compact_prompt
        from skills.kimi_vl_vllm import create_model_class
        from skills.vllm_token import configure_runtime
        from vllm.config import ModelConfig
        from vllm.model_executor.models import ModelRegistry
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.multimodal.cache import MultiModalProcessorOnlyCache
        from vllm.multimodal.processing import ProcessorInputs
        from vllm.multimodal.processing.context import TimingContext
        descriptor = json.loads((ROOT / "configs/model-kimi-vl.json").read_text())
        cls = create_model_class()
        self.addCleanup(configure_runtime, None)
        # Metadata-only fixture: no model construction or weight copy. Use the
        # original full checkpoint's real tokenizer/template/image protocol.
        with tempfile.TemporaryDirectory() as folder, patch.dict(ModelRegistry.models):
            path = Path(folder)
            for name in (*descriptor["native_source_sha256"], "tiktoken.model", "config.json"):
                shutil.copyfile(MODEL / name, path / name)
            (path / "manifest.json").write_text(json.dumps({"model": descriptor}))
            ModelRegistry.register_model("KimiVLForConditionalGeneration", cls)
            config = ModelConfig(model=str(path), tokenizer=str(path), trust_remote_code=True,
                                 dtype="bfloat16", max_model_len=131072, enforce_eager=True)
            native = load_processor({**descriptor, "path": str(path)})
            for ratio in (1., .8, .5):
                runtime = {**RUNTIME, "keep_ratio": ratio}
                configure_runtime(runtime)
                for cached in (False, True):
                    cache = MultiModalProcessorOnlyCache(config) if cached else None
                    processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=native.tokenizer, cache=cache)
                    for size in ((281, 421), (32, 60), (14, 14), (1500, 2049)):
                        with Image.new("RGB", size, (23, 124, 210)) as image:
                            for index, question in enumerate(("What color?", "  中文描述 \n")):
                                prompt = unexpanded_prompt(native, question)
                                if index:
                                    prompt = native.tokenizer.encode(prompt, add_special_tokens=False)
                                inputs = ProcessorInputs(prompt=prompt,
                                    mm_data_items=processor.info.parse_mm_data({"image": image}))
                                output = processor.apply(inputs, TimingContext(enabled=False))
                                reference = prepare_image(native, image, question)
                                ph, pw = reference["image_grid_hws"][0].tolist()
                                keep = retained_count(ph // 2, pw // 2, runtime)
                                native_ids = reference["input_ids"][0].tolist()
                                expected = compact_prompt(native_ids, 163605, [(ph * pw // 4, keep)])
                                with self.subTest(size=size, ratio=ratio, cache=cached, tokenized=bool(index)):
                                    self.assertEqual(output["prompt_token_ids"], expected)
                                    data = output["mm_kwargs"].get_data()
                                    self.assertTrue(torch.equal(data["image_grid_hws"], reference["image_grid_hws"]))
                                    # Native forward casts pixels to the vision dtype
                                    # before patch embedding, as vLLM's context does.
                                    self.assertTrue(torch.equal(data["pixel_values"], reference["pixel_values"].to(torch.bfloat16)))
                                    placeholders = output["mm_placeholders"]["image"]
                                    self.assertEqual(len(placeholders), 1)
                                    self.assertEqual(placeholders[0].length, keep)
                                    self.assertEqual(expected[placeholders[0].offset:placeholders[0].offset + keep],
                                                     [163605] * keep)
                    if cache is not None:
                        self.assertGreater(cache.make_stats().hits, 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_upstream_chunked_prefill_accepts_packed_projection_without_weight(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from skills.kimi_vl_mla_compat import bind_packed_prefill
        from vllm.model_executor.layers.attention import mla_attention as upstream
        from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
        captured = []
        class PackedProjection:
            params_dtype = torch.bfloat16
            quant_method = AutoGPTQLinearMethod.__new__(AutoGPTQLinearMethod)
            def __call__(self, inputs):
                captured.append(inputs)
                return torch.cat([inputs, inputs], dim=-1), None
        class Probe:
            _compute_prefill_context = upstream.MLACommonBaseImpl._compute_prefill_context
            _concat_k_nope_k_pe = upstream.MLACommonBaseImpl._concat_k_nope_k_pe
        probe = Probe()
        probe.kv_b_proj, probe.kv_cache_dtype = PackedProjection(), "auto"
        probe.kv_lora_rank, probe.num_heads = 2, 1
        probe.qk_nope_head_dim, probe.v_head_dim = 2, 2
        probe._use_flashinfer_concat_mla_k = False
        workspace = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        chunk = SimpleNamespace(seq_tot=[2], workspace=workspace, block_table=None,
                                cu_seq_lens=[None], token_to_seq=[None], chunk_total_token=[2],
                                starts=[None], has_empty_context=[False])
        def attend(**kwargs):
            self.assertEqual(kwargs["k"].shape, (2, 1, 3))
            return kwargs["v"], torch.zeros(2)
        metadata = SimpleNamespace(num_prefills=1, prefill=SimpleNamespace(
            q_data_type=torch.bfloat16, chunked_context=chunk, block_table=None,
            prefill_backend=SimpleNamespace(run_prefill_context_chunk=attend)))
        with patch.object(upstream.current_platform, "fp8_dtype", return_value=torch.float8_e4m3fn), \
             patch.object(upstream.ops, "gather_and_maybe_dequant_cache"):
            with self.assertRaisesRegex(AttributeError, "weight"):
                probe._compute_prefill_context(torch.zeros(2, 1, 3), torch.empty(0), metadata, torch.ones(1))
            bind_packed_prefill(probe)
            output, _ = probe._compute_prefill_context(torch.zeros(2, 1, 3), torch.empty(0), metadata, torch.ones(1))
        self.assertEqual(captured[0].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(captured[0], workspace[:, :2].to(torch.bfloat16)))
        self.assertTrue(torch.equal(output[:, 0], captured[0]))
        self.assertFalse(hasattr(probe.kv_b_proj, "weight"))
        self.assertFalse(torch.cuda.is_initialized())

    def test_policy_quantizes_native_mla_and_vision_but_preserves_mapper(self):
        from skills.kimi_vl_quantization import expected_inventory
        from vllm.model_executor.layers.quantization.utils.gptq_utils import is_layer_gptq_quantized
        policy = expected_inventory(header())
        selected = [name for name, precision in policy.items() if precision == "int4_w4a16"]
        mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        for name in ("language_model.model.layers.0.mlp.gate_up_proj",
                     "language_model.model.layers.1.mlp.shared_experts.gate_up_proj",
                     "language_model.model.layers.1.self_attn.kv_b_proj",
                     "vision_tower.encoder.blocks.0.wqkv", "vision_tower.encoder.blocks.0.mlp.fc1"):
            self.assertTrue(is_layer_gptq_quantized(name, selected, mapping), name)
        for name in ("multi_modal_projector.linear_1", "language_model.lm_head"):
            self.assertFalse(is_layer_gptq_quantized(name, selected, mapping), name)
        missing = [name for name in selected if not name.endswith("shared_experts.up_proj")]
        with self.assertRaisesRegex(ValueError, "not all shards"):
            is_layer_gptq_quantized("language_model.model.layers.1.mlp.shared_experts.gate_up_proj", missing, mapping)

    def test_model_registry_class_constructs_without_gpu_or_checkpoint_allocation(self):
        import torch
        from skills.kimi_vl_vllm import create_model_class
        cls = create_model_class()
        self.assertEqual(cls.__name__, "A4FVLMKimiVLForConditionalGeneration")
        self.assertFalse(cls.supports_encoder_cudagraph)
        self.assertEqual(cls.packed_modules_mapping, {"gate_up_proj": ["gate_proj", "up_proj"]})
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
