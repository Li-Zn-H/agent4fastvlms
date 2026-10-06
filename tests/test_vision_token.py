"""Selection semantics, finite search scheduling, and decoder compaction."""

import unittest
from types import SimpleNamespace

import torch

from search.vision_token import candidate_ratios
from skills.vision_token import VisionTokenController, keep_count, select_indices

SELECTION = {"score_dtype": "float32", "coverage_grid": {"rows": 2, "columns": 2}}
SEARCH = {
    "start_keep_ratio": 0.95,
    "minimum_keep_ratio": 0.1,
    "keep_ratio_step": 0.05,
    "consecutive_dev_failures": 2,
}


class SelectionTests(unittest.TestCase):
    def test_coverage_and_order(self):
        features = torch.arange(16.0).reshape(16, 1)
        self.assertEqual(
            select_indices(features, 4, 4, 0.25, SELECTION).tolist(), [5, 7, 13, 15]
        )
        self.assertEqual(
            select_indices(features, 4, 4, 0.375, SELECTION).tolist(),
            [5, 7, 12, 13, 14, 15],
        )

    def test_ties_and_small_grid(self):
        features = torch.ones(16, 2)
        self.assertEqual(
            select_indices(features, 4, 4, 0.375, SELECTION).tolist(),
            [0, 1, 2, 3, 8, 10],
        )
        self.assertEqual(keep_count(1, 2, 0.1, SELECTION["coverage_grid"]), 2)
        self.assertEqual(
            select_indices(features[:2], 1, 2, 0.1, SELECTION).tolist(), [0, 1]
        )
        with self.assertRaises(ValueError):
            keep_count(4, 4, 0, SELECTION["coverage_grid"])

    def test_reference_equivalence(self):
        torch.manual_seed(7)
        features = torch.randn(77, 12)
        scores = features.norm(dim=-1).tolist()
        grid = {"rows": 6, "columns": 6}
        cells = {}
        for i in range(77):
            cell = (i // 11 * 6 // 7, i % 11 * 6 // 11)
            cells.setdefault(cell, []).append(i)
        representatives = {
            min(v, key=lambda i: (-scores[i], i)) for v in cells.values()
        }
        for ratio in candidate_ratios(SEARCH):
            count = keep_count(7, 11, ratio, grid)
            remaining = sorted(
                set(range(77)) - representatives, key=lambda i: (-scores[i], i)
            )
            expected = sorted(
                representatives | set(remaining[: count - len(representatives)])
            )
            self.assertEqual(
                select_indices(
                    features, 7, 11, ratio, {**SELECTION, "coverage_grid": grid}
                ).tolist(),
                expected,
            )


class SearchTests(unittest.TestCase):
    def test_ratios(self):
        self.assertEqual(candidate_ratios(SEARCH), [i / 100 for i in range(95, 9, -5)])
        self.assertEqual(
            candidate_ratios({**SEARCH, "minimum_keep_ratio": 0.12})[-2:], [0.15, 0.12]
        )
        for key, value in [
            ("keep_ratio_step", 0),
            ("minimum_keep_ratio", 0),
            ("start_keep_ratio", float("nan")),
        ]:
            with self.assertRaises(ValueError):
                candidate_ratios({**SEARCH, key: value})


class AdapterTests(unittest.TestCase):
    def test_failed_generation_removes_hooks_and_request_state(self):
        language = torch.nn.Identity()

        def fail(**kwargs):
            raise RuntimeError("inference failed")

        model = SimpleNamespace(
            config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
            model=SimpleNamespace(language_model=language, rope_deltas=torch.ones(1)),
            generate=fail,
        )
        controller = VisionTokenController(model, SELECTION, 0.5)
        inputs = {
            "input_ids": torch.ones(1, 20, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 8, 8]]),
        }
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            controller.generate(inputs, {"do_sample": False})
        self.assertFalse(language._forward_pre_hooks)
        self.assertFalse(language._forward_hooks)
        self.assertIsNone(controller.kept)
        self.assertIsNone(model.model.rope_deltas)

    def test_prefill_and_decode_keep_rotary_positions_and_shorten_cache(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(deepstack_visual_indexes=[0, 1, 2])
            )
        )
        controller = VisionTokenController(model, SELECTION, 0.25)
        controller.rows, controller.columns, controller.original_length = 4, 4, 20
        visual = torch.tensor([[False, False] + [True] * 16 + [False, False]])
        embeds = torch.arange(40.0).reshape(1, 20, 2)
        deep = [embeds[0, 2:18] + i for i in range(3)]
        positions = torch.arange(80).reshape(4, 1, 20)
        _, args = controller._before(
            None,
            (),
            {
                "inputs_embeds": embeds,
                "attention_mask": torch.ones(1, 20),
                "position_ids": positions,
                "visual_pos_masks": visual,
                "deepstack_visual_embeds": deep,
            },
        )
        kept = controller.kept
        self.assertEqual(args["inputs_embeds"].shape[1], 8)
        self.assertEqual(kept[:2].tolist(), [0, 1])
        self.assertEqual(kept[-2:].tolist(), [18, 19])
        self.assertTrue(torch.equal(args["position_ids"][1:], positions[1:, :, kept]))
        self.assertEqual(args["position_ids"][0, 0].tolist(), list(range(8)))
        selected = kept[2:-2] - 2
        for index in range(3):
            self.assertTrue(
                torch.equal(
                    args["deepstack_visual_embeds"][index], deep[index][selected]
                )
            )
        for step in range(3):
            past = 8 + step
            cache = SimpleNamespace(get_seq_length=lambda: past)
            decode_positions = torch.tensor(
                [[[20 + step]], [[40 + step]], [[60 + step]], [[80 + step]]]
            )
            _, args = controller._before(
                None,
                (),
                {
                    "inputs_embeds": torch.zeros(1, 1, 2),
                    "attention_mask": torch.ones(1, 21 + step),
                    "position_ids": decode_positions,
                    "past_key_values": cache,
                },
            )
            self.assertEqual(args["attention_mask"].shape[1], past + 1)
            self.assertTrue(torch.equal(args["position_ids"][1:], decode_positions[1:]))
            self.assertEqual(args["position_ids"][0].item(), past)
            controller._after(
                None,
                (),
                {},
                SimpleNamespace(
                    past_key_values=SimpleNamespace(get_seq_length=lambda: past + 1)
                ),
            )
        controller.reset()
        self.assertIsNone(controller.kept)
        self.assertIsNone(controller.last_metrics)


if __name__ == "__main__":
    unittest.main()
