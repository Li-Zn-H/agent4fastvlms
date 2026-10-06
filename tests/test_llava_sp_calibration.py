"""Native LLaVA calibration partitions and binary moment contract."""
import tempfile
from pathlib import Path
import unittest

from scripts.calibrate_llava_sp import STRATA, _write_moments, prefill_chunks, validate_heldout
from skills.structured_pruning import contribution_scores, decode_layer


class LlavaCalibrationTests(unittest.TestCase):
    def test_prefill_follows_noncontiguous_image_mask(self):
        import torch

        values = torch.tensor([[1., 2.], [10., 20.], [3., 4.], [30., 40.]])
        chunks = dict(prefill_chunks(values, torch.tensor([False, True, False, True])))
        torch.testing.assert_close(chunks["visual_prefill"], values[[1, 3]])
        torch.testing.assert_close(chunks["text_prefill"], values[[0, 2]])
        with self.assertRaisesRegex(ValueError, "length disagree"):
            prefill_chunks(values, torch.tensor([True, False]))

    def test_moments_store_sums_without_averaging_twice(self):
        rows = {name: {"count": 2, "sums": [4., 6.], "squares": [10., 20.]}
                for name in STRATA}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer.bin"
            _write_moments(path, 0, rows)
            layer, moments = decode_layer(path.read_bytes(), moments=True)
        self.assertEqual(layer, 0)
        self.assertEqual(moments[0]["sums"], (4., 6.))
        self.assertEqual(moments[0]["sum_squares"], (10., 20.))
        scores = contribution_scores(moments, [2., 3.])
        self.assertAlmostEqual(scores[0], 2 * 5**0.5, places=5)
        self.assertAlmostEqual(scores[1], 3 * 10**0.5, places=5)

    def test_calibration_rejects_shared_evaluation_ids_or_images(self):
        def row(identity, image):
            return {"stable_row_id": identity, "media": {"logical_path": image}}

        validate_heldout([row("cal", "a")], [row("dev", "b")])
        for candidate in (row("dev", "a"), row("cal", "b")):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                validate_heldout([candidate], [row("dev", "b")])


if __name__ == "__main__":
    unittest.main()
