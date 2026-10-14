"""Kimi timing regression: post-projector selection happens only at prefill."""

from types import SimpleNamespace
import unittest

from adapters.kimi_vl_dense import GenerationTimer

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "CPU Torch required")
class KimiTimingTests(unittest.TestCase):
    def make_model(self):
        clock = [0.0]

        class Event:
            def __init__(self):
                self.at = clock[0]

            def elapsed_time(self, end):
                return end.at - self.at

        class Module(torch.nn.Module):
            def __init__(self, durations):
                super().__init__()
                self.durations = iter(durations)

            def forward(self):
                clock[0] += next(self.durations)

        vision, projector, decoder = Module([10]), Module([5]), Module([20, 3, 4])
        model = SimpleNamespace(vision_tower=vision, multi_modal_projector=projector,
                                language_model=SimpleNamespace(model=decoder))
        return model, clock, Event

    def test_selection_interval_does_not_include_later_cached_decoding(self):
        model, clock, event = self.make_model()
        timer = GenerationTimer(model, pruning=True)
        timer.event = event
        with timer:
            model.vision_tower()
            model.multi_modal_projector()
            clock[0] += 2  # Feature selection and compacted embedding preparation.
            model.language_model.model()
            model.language_model.model()
            model.language_model.model()
        metrics = timer.metrics()
        self.assertEqual(metrics["vision_encode_seconds"], .010)
        self.assertEqual(metrics["pruning_seconds"], .002)
        self.assertEqual(metrics["prefill_seconds"], .020)
        self.assertEqual(metrics["decode_seconds"], .007)
        self.assertEqual(metrics["decode_tokens"], 2)
        for module in (model.vision_tower, model.multi_modal_projector,
                       model.language_model.model):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_dense_has_no_selection_interval_and_removes_hooks_on_error(self):
        model, _, event = self.make_model()
        timer = GenerationTimer(model, pruning=False)
        timer.event = event
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with timer:
                model.vision_tower()
                model.multi_modal_projector()
                model.language_model.model()
                model.language_model.model()
                raise RuntimeError("generation failed")
        metrics = timer.metrics()
        self.assertEqual(metrics["pruning_seconds"], 0)
        self.assertEqual(metrics["prefill_seconds"], .020)
        self.assertEqual(metrics["decode_seconds"], .003)
        for module in (model.vision_tower, model.multi_modal_projector,
                       model.language_model.model):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
