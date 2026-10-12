"""Native InternVL precision policy, without quantizer imports or weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from skills.quantization import assignment, classify_module, validate_module_inventory


class InternVLPrecisionTests(unittest.TestCase):
    adapter_id = "internvl35"
    decoder_layers = 40

    def names(self):
        language = [f"language_model.model.layers.{i}.{name}"
                    for i in range(self.decoder_layers)
                    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
        vision = [f"vision_model.encoder.layers.{i}.{name}"
                  for i in range(24) for name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")]
        return language, vision, ["mlp1.1", "mlp1.3", "language_model.lm_head"]

    def test_both_towers_are_quantized_but_native_mapper_and_head_are_preserved(self):
        language, vision, preserved = self.names()
        policy = assignment(language + vision + preserved, model_adapter=self.adapter_id)
        self.assertTrue(all(policy[name] == "int4_w4a16" for name in language + vision))
        self.assertTrue(all(policy[name] == "bf16" for name in preserved))
        inventory = validate_module_inventory(policy, model_adapter=self.adapter_id)
        self.assertTrue(inventory["valid"])
        self.assertEqual(inventory["language_linear_count"], self.decoder_layers * 7)
        self.assertEqual(inventory["vision_linear_count"], 96)
        self.assertEqual(inventory["mapper_count"], 2)
        self.assertEqual(inventory["int4_count"], self.decoder_layers * 7 + 96)
        self.assertEqual(inventory["bf16_count"], 3)

    def test_decoder_only_inventory_cannot_claim_multimodal_coverage(self):
        language, _, preserved = self.names()
        inventory = validate_module_inventory(language + preserved, model_adapter=self.adapter_id)
        self.assertFalse(inventory["valid"])
        self.assertEqual(inventory["missing_families"], ["vision_linear"])

    def test_other_families_cannot_satisfy_internvl_inventory(self):
        inventory = validate_module_inventory([
            "model.language_model.layers.0.self_attn.q_proj",
            "model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
        ], model_adapter=self.adapter_id)
        self.assertFalse(inventory["valid"])
        self.assertEqual(set(inventory["missing_families"]),
                         {"language_linear", "vision_linear", "vision_mapper"})
        for name in ("language_model.model.layers.0.self_attn.q_norm",
                     "language_model.model.embed_tokens", "vision_model.embeddings.patch_embedding",
                     "vision_model.encoder.layers.0.norm1", "mlp1.0"):
            self.assertEqual(classify_module(name, model_adapter=self.adapter_id), "bf16")


class ArtifactBindingTests(unittest.TestCase):
    adapter_id = "internvl35"
    decoder_type = "qwen3"

    def test_config_calibration_and_parent_cannot_be_swapped_on_reload(self):
        from skills.quantization import make_manifest, digest
        from skills.internvl35_artifact import QUANT_METHOD, STACK_ARCHITECTURE, validate_quantized_bindings
        config = {"model_type": "internvl_chat", "architectures": [STACK_ARCHITECTURE],
                  "llm_config": {"model_type": self.decoder_type, "num_hidden_layers": 1, "intermediate_size": 256},
                  "a4fvlm_stack_widths": [128], "a4fvlm_stack_physical_compaction": True,
                  "a4fvlm_stack_recipe_digest": "recipe",
                  "a4fvlm_internvl_gptq": {"method": QUANT_METHOD, "format": "gptq_v1"},
                  "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128,
                                          "desc_act": False, "sym": True}}
        parent = {"schema": "a4fvlm.stack-parent.v1", "architecture": STACK_ARCHITECTURE,
                  "widths": [128], "physical_compaction": True, "recipe_digest": "recipe",
                  "model_config_digest": "source"}
        calibration = {"data": "heldout.jsonl", "limit": 2}
        module = "language_model.model.layers.0.mlp.up_proj"
        receipt = {"method": QUANT_METHOD, "calibration": calibration,
                   "sample_ids": ["a", "b"], "sample_count": 2,
                   "sequential_projection_groups": True, "vision_before_decoder": True,
                   "source_config_digest": "source",
                   "modules": [{"module": module, "method": "gptq_hessian", "fallback": None,
                                "samples": 20, "bits": 4, "group_size": 128}]}
        manifest = make_manifest(model={"adapter": self.adapter_id, "path": "parent"},
                                 config={"route": "gptqmodel_w4a16", "bits": 4,
                                         "group_size": 128, "calibration": calibration},
                                 calibration=calibration, module_names=[module], status="exported")
        manifest["visual_quantization"] = {"method": QUANT_METHOD, "artifact_config_digest": digest(config),
                                           "calibration_receipt_digest": digest(receipt)}
        manifest["stack"] = {"parent_recipe_digest": "recipe", "parent_manifest_digest": digest(parent),
                             "fresh_calibration": True}
        def rebind(value):
            value["artifact_digest"] = digest({k: v for k, v in value.items() if k != "artifact_digest"})
        rebind(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for change in (None, "config", "receipt", "parent", "incomplete", "stale_source", "decoder_family"):
                with self.subTest(change=change):
                    c, r, p, m = map(copy.deepcopy, (config, receipt, parent, manifest))
                    if change == "config":
                        c["template"] = "changed"
                    elif change == "receipt":
                        r["sample_ids"] = ["other", "b"]
                    elif change == "parent":
                        p["model_config_digest"] = "other-parent"
                    elif change == "decoder_family":
                        m["model"]["adapter"] = "llava15"
                        m["model_digest"] = digest(m["model"])
                        rebind(m)
                    elif change in {"incomplete", "stale_source"}:
                        if change == "incomplete":
                            r["modules"] = []
                        else:
                            r["source_config_digest"] = "other-parent"
                        m["visual_quantization"]["calibration_receipt_digest"] = digest(r)
                        rebind(m)
                    (root / "stack-parent.json").write_text(json.dumps(p))
                    (root / "vision-quantization.json").write_text(json.dumps(r))
                    if change is None:
                        validate_quantized_bindings(root, c, m)
                    else:
                        with self.assertRaises(ValueError):
                            validate_quantized_bindings(root, c, m)


if __name__ == "__main__":
    unittest.main()
