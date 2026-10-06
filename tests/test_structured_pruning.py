"""Frozen statistics, physical channel compaction, and reuse-first width probes."""

import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

from search.structured_pruning import (
    execute,
    import_calibration,
    pending_shapes,
    prefill_lengths,
    probe_widths,
    summarize,
)
from search.worker import run_pruning_probe, save_json
from skills.structured_pruning import (
    SCORE,
    STRATA,
    apply_recipe,
    contribution_scores,
    decode_layer,
    load_calibration,
    make_recipe,
    slice_mlp,
    validate_config,
)


DIMENSIONS: dict[str, Any] = {
    "num_hidden_layers": 2,
    "hidden_size": 4,
    "intermediate_size": 8,
}
SELECTION = {
    "score": SCORE,
    "score_dtype": "float32",
    "channel_group_size": 2,
    "protected_layers": [],
}
CONFIG = {
    "selection": SELECTION,
    "hardware_probe": {
        "minimum_width_ratio": 0.5,
        "warmup_iterations": 5,
        "measurement_iterations": 20,
    },
}


def tiny_model():
    layers = [
        SimpleNamespace(mlp=Qwen3VLTextMLP(Qwen3VLTextConfig(**DIMENSIONS)))
        for _ in range(2)
    ]
    return SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(**DIMENSIONS)),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
    )


class CalibrationTests(unittest.TestCase):
    def test_binary_validation_and_equal_strata(self):
        payload = b"A4FVLMSPSCO1\0" + struct.pack("<II8f", 0, 8, *range(8))
        self.assertEqual(decode_layer(payload), (0, list(range(8))))
        for bad in (
            payload[:-1],
            payload + b"x",
            b"bad",
            payload[:-4] + struct.pack("<f", float("nan")),
        ):
            with self.assertRaises(ValueError):
                decode_layer(bad)
        moments = [
            {"stratum": name, "count": count, "sum_squares": [count * value]}
            for name, count, value in zip(STRATA, (1, 10, 100), (1, 4, 9))
        ]
        expected = ((1 + 4 + 9) / 3) ** 0.5 * 2
        self.assertAlmostEqual(contribution_scores(moments, [2])[0], expected, places=5)

    def test_import_is_complete_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "target"
            for folder in ("scores", "moments"):
                (source / folder).mkdir(parents=True)
            for layer in range(2):
                header = struct.pack("<II", layer, 8)
                (source / "scores" / f"layer-{layer:02d}.bin").write_bytes(
                    b"A4FVLMSPSCO1\0" + header + struct.pack("<8f", *range(8))
                )
                payload = (
                    b"A4FVLMSPMOM1\0"
                    + header
                    + (struct.pack("<Q16f", 3, *([1.0] * 16)) * 3)
                )
                (source / "moments" / f"layer-{layer:02d}.bin").write_bytes(payload)
            save_json(source / "score-manifest.json", {"scorer_id": SCORE})
            save_json(
                source / "statistics-manifest.json", {"calibration_protocol": "legacy"}
            )
            self.assertEqual(import_calibration(source, target, DIMENSIONS), 6)
            self.assertEqual(import_calibration(source, target, DIMENSIONS), 0)
            self.assertEqual(len(load_calibration(target, DIMENSIONS)), 2)
            (target / "scores/layer-00.bin").write_bytes(b"different")
            with self.assertRaises(ValueError):
                import_calibration(source, target, DIMENSIONS)
            (source / "moments/layer-01.bin").unlink()
            with self.assertRaises(FileNotFoundError):
                import_calibration(source, target, DIMENSIONS)


class SlicingTests(unittest.TestCase):
    def test_recipe_ranking_nested_and_protection(self):
        scores = [[1.0] * 8, list(range(8))]
        recipe = make_recipe(scores, [6, 4], DIMENSIONS, SELECTION)
        self.assertEqual(
            recipe["retained_channels"], [list(range(2, 8)), list(range(4, 8))]
        )
        self.assertEqual(recipe["removed_ffn_parameters"], 6 * 3 * 4)
        self.assertEqual(recipe["actual_depth"], 6 / 16)
        deeper = make_recipe(scores, [4, 2], DIMENSIONS, SELECTION)
        for before, after in zip(
            recipe["retained_channels"], deeper["retained_channels"]
        ):
            self.assertTrue(set(after) < set(before))
        with self.assertRaises(ValueError):
            make_recipe(
                scores, [6, 4], DIMENSIONS, {**SELECTION, "protected_layers": [0]}
            )
        with self.assertRaises(ValueError):
            make_recipe(scores, [5, 4], DIMENSIONS, SELECTION)

    def test_physical_forward_and_fresh_restore(self):
        torch.manual_seed(0)
        model = tiny_model()
        dense = copy.deepcopy(model)
        recipe = make_recipe([list(range(8))] * 2, [4, 6], DIMENSIONS, SELECTION)
        apply_recipe(model, recipe)
        x = torch.randn(2, 3, 4)
        for layer, reference, keep in zip(
            model.model.language_model.layers,
            dense.model.language_model.layers,
            recipe["retained_channels"],
        ):
            mlp, original = layer.mlp, reference.mlp
            self.assertEqual(list(mlp.gate_proj.weight.shape), [len(keep), 4])
            self.assertEqual(list(mlp.down_proj.weight.shape), [4, len(keep)])
            self.assertEqual(
                sum(p.numel() for p in mlp.parameters()), len(keep) * 4 * 3
            )
            activations = original.act_fn(original.gate_proj(x)) * original.up_proj(x)
            activations[..., [index for index in range(8) if index not in keep]] = 0
            torch.testing.assert_close(mlp(x), original.down_proj(activations))
        restored = apply_recipe(copy.deepcopy(dense), json.loads(json.dumps(recipe)))
        for a, b in zip(
            model.model.language_model.layers, restored.model.language_model.layers
        ):
            torch.testing.assert_close(a.mlp(x), b.mlp(x), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            apply_recipe(model, recipe)

    def test_invalid_recipe_does_not_partially_modify_model(self):
        model = tiny_model()
        recipe = make_recipe([list(range(8))] * 2, [4, 4], DIMENSIONS, SELECTION)
        recipe["retained_channels"][1] = [4, 4, 5, 6]
        with self.assertRaises(ValueError):
            apply_recipe(model, recipe)
        self.assertEqual(
            model.model.language_model.layers[0].mlp.gate_proj.out_features, 8
        )

    def test_bias_dtype_and_keep_all(self):
        mlp = Qwen3VLTextMLP(Qwen3VLTextConfig(**DIMENSIONS)).double()
        original = mlp.gate_proj
        slice_mlp(mlp, list(range(8)))
        self.assertIs(mlp.gate_proj, original)
        for name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(mlp, name)
            module.bias = torch.nn.Parameter(
                torch.randn(module.out_features, dtype=torch.float64)
            )
        reference = copy.deepcopy(mlp)
        slice_mlp(mlp, [1, 3, 5, 7])
        x = torch.randn(2, 4, dtype=torch.float64)
        activation = reference.act_fn(reference.gate_proj(x)) * reference.up_proj(x)
        activation[..., [0, 2, 4, 6]] = 0
        torch.testing.assert_close(mlp(x), reference.down_proj(activation))
        self.assertEqual(mlp.gate_proj.weight.dtype, torch.float64)


class ProbeTests(unittest.TestCase):
    def test_finite_widths_and_representative_lengths(self):
        validate_config(CONFIG, DIMENSIONS)
        self.assertEqual(probe_widths(9728, 128, 0.5), list(range(9728, 4863, -128)))
        self.assertEqual(len(probe_widths(9728, 128, 0.5)), 39)
        rows = [
            {"round": 1, "sample_id": str(i), "performance": {"input_tokens": i + 1}}
            for i in range(10)
        ]
        self.assertEqual(prefill_lengths(rows, [str(i) for i in range(10)]), [5, 9, 10])
        with self.assertRaises(ValueError):
            prefill_lengths(rows[:-1], [str(i) for i in range(10)])

    def test_complete_reuse_never_queries_or_launches_gpu(self):
        snap: dict[str, Any] = {
            "context": {"device": 0, "dimensions": DIMENSIONS},
            "shapes": [{"phase": "decode", "input_tokens": 1, "width": 8}],
        }
        row = {
            **snap["shapes"][0],
            "status": "measured",
            "finite_output": True,
            "milliseconds": [2.0, 3.0],
            "measurement_iterations": 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            save_json(
                Path(temporary) / "measurements.json",
                {"context": snap["context"], "records": [row]},
            )
            with (
                patch(
                    "search.structured_pruning.hardware",
                    side_effect=AssertionError("GPU queried"),
                ),
                patch(
                    "search.structured_pruning.launch",
                    side_effect=AssertionError("Worker launched"),
                ),
            ):
                self.assertEqual(execute(snap, temporary)["status"], "completed")

    def test_failed_width_does_not_remove_narrower_shapes(self):
        shapes = [
            {"phase": "prefill", "input_tokens": 10, "width": width}
            for width in (8, 6, 4)
        ]
        snap = {"context": {"dimensions": DIMENSIONS}, "shapes": shapes}
        records = [
            {
                **shapes[0],
                "status": "measured",
                "finite_output": True,
                "milliseconds": [2.0],
                "measurement_iterations": 1,
            },
            {**shapes[1], "status": "oom"},
        ]
        self.assertEqual(pending_shapes(snap, records), shapes[1:])
        summary = summarize(snap, records)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["shapes"][0]["speedup_vs_dense_width"], 1.0)

    def test_plan_imports_no_inference_framework(self):
        code = (
            "import sys; import search.structured_pruning; "
            "assert not {'torch', 'transformers'} & sys.modules.keys()"
        )
        subprocess.run([sys.executable, "-B", "-c", code], check=True)

    def test_worker_checkpoints_and_continues_after_failed_width(self):
        cuda = Mock()
        cuda.Event.return_value.elapsed_time.return_value = 1.0
        cuda.max_memory_allocated.return_value = 1024
        cpu_torch = SimpleNamespace(
            **{
                **vars(torch),
                "cuda": cuda,
                "manual_seed": lambda seed: None,
                "device": lambda name: torch.device("cpu"),
            }
        )

        def build(config):
            if config.intermediate_size == 6:
                raise RuntimeError("Synthetic unsupported width")
            return Qwen3VLTextMLP(config)

        shapes = [
            {"phase": "decode", "input_tokens": 1, "width": width}
            for width in (8, 6, 4)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            request = {
                "probe_output": temporary,
                "hardware": "CPU test fixture",
                "shapes_to_measure": shapes,
                "probe": {
                    "context": {"dimensions": DIMENSIONS},
                    "config": {
                        "hardware_probe": {
                            "warmup_iterations": 1,
                            "measurement_iterations": 2,
                        }
                    },
                },
            }
            with (
                patch.dict(sys.modules, {"torch": cpu_torch}),
                patch(
                    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextMLP",
                    side_effect=build,
                ),
            ):
                result = run_pruning_probe(request)
            rows = json.loads((Path(temporary) / "measurements.json").read_text())[
                "records"
            ]
            self.assertEqual(
                [row["status"] for row in rows],
                ["measured", "execution_error", "measured"],
            )
            self.assertEqual(result["failed_shapes"], 1)
            self.assertEqual(rows[2]["weight_shapes"]["down_proj"], [4, 4])


if __name__ == "__main__":
    unittest.main()
