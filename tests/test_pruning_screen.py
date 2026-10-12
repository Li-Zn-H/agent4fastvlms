"""Representative freezing, Dev promotion and reuse without GPU execution."""

from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from search import pruning_screen as ps


def quality(confirmed=False):
    stages = {"search": {"action": "pass"}}
    if confirmed:
        stages["dev"] = {"action": "pass"}
    return {"action": "pass", "stages": stages}


class ScreenTests(unittest.TestCase):
    def test_representatives_and_metric_tradeoffs(self):
        self.assertEqual(ps.representatives({}), [])
        self.assertEqual(
            ps.representatives({i: quality() for i in range(1, 12)}), [1, 6, 11]
        )
        self.assertEqual(
            ps.representatives({1: quality(), 2: {"action": "reject"}}), [1]
        )
        dense = {"mean_round_seconds": 10, "rounds": [{"peak_allocated_bytes": 100}]}
        slower_smaller = {
            "mean_round_seconds": 11,
            "rounds": [{"peak_allocated_bytes": 90}],
        }
        self.assertTrue(ps.comparison(dense, slower_smaller)["improved"])
        self.assertFalse(ps.comparison(dense, dense)["improved"])

    def test_inherited_recipe_is_referenced_without_copying(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "old"
            new = Path(folder) / "new"
            recipe = old / "candidates" / "step-0001" / "recipe.json"
            recipe.parent.mkdir(parents=True)
            recipe.write_text("{}")
            new.mkdir()
            (new / "config.json").write_text(
                json.dumps({"extension": {"source": str(old)}})
            )
            self.assertEqual(ps.recipe_source(new, 1), recipe.resolve())
            self.assertFalse((new / "candidates").exists())
            revised = Path(folder) / "revised"
            revised.mkdir()
            (revised / "config.json").write_text(
                json.dumps({"revision": {"source": str(new)}})
            )
            self.assertEqual(ps.recipe_source(revised, 1), recipe.resolve())

    def test_timing_removes_diagnostic_hooks_and_skips_quality(self):
        from search.worker import execute

        handle = Mock()
        mlp = Mock()
        mlp.register_forward_hook.return_value = handle
        model = SimpleNamespace(
            config=SimpleNamespace(text_config=None),
            model=SimpleNamespace(
                language_model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])
            )
        )
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            destination = Path(folder)
            recipe = destination / "recipe.json"
            recipe.write_text(json.dumps({"widths": [8]}))
            stack.enter_context(
                patch("search.model.load_model", return_value=(model, None))
            )
            stack.enter_context(
                patch(
                    "skills.structured_pruning.audit_recipe",
                    return_value={"valid": True},
                )
            )
            evaluate = stack.enter_context(patch("search.run.evaluate_candidate"))

            def timed(*args, **kwargs):
                handle.remove.assert_called_once()
                return {"measured": True}

            stack.enter_context(patch("search.benchmark.benchmark", side_effect=timed))
            result = execute(
                {
                    "operation": "structured_pruning_candidate",
                    "recipe": str(recipe),
                    "model": {},
                    "evaluation": {},
                    "evaluate": False,
                    "measure": True,
                },
                destination,
            )
            evaluate.assert_not_called()
            self.assertEqual(result, {"benchmark": {"measured": True}})

    def test_resume_reuses_benchmarks_and_confirmations_without_gpu(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            runner = root / "runner"
            runner.mkdir()
            snap = {
                "model": {},
                "evaluation": {"scoring": {"baseline": "dense"}},
                "config": {"calibration": {"source": "unused"}},
                "dimensions": {},
                "device": 0,
            }
            (runner / "config.json").write_text(json.dumps(snap))
            (runner / "state.json").write_text(json.dumps({"status": "completed"}))
            q = {i: quality() for i in [1, 2, 3]}
            sources = {i: {"search": "saved"} for i in q}
            for i in q:
                candidate = runner / "candidates" / f"step-{i:04d}"
                candidate.mkdir(parents=True)
                (candidate / "recipe.json").write_text(
                    json.dumps({"actual_depth": i / 10})
                )
            stack.enter_context(patch.object(ps, "RESULTS", root))
            stack.enter_context(patch.object(ps, "load_calibration", return_value=[]))
            stack.enter_context(
                patch.object(ps, "discover", side_effect=lambda *args: (sources, q))
            )
            stack.enter_context(
                patch.object(ps, "lock", side_effect=lambda *args: nullcontext())
            )
            gpu = stack.enter_context(patch.object(ps, "hardware", return_value="gpu"))

            def inspect(snap, ratio, paths):
                if not paths:
                    return None
                dense = paths[0].parent.name == "dense"
                return {
                    "source": str(paths[0]),
                    "summary": {
                        "mean_round_seconds": 10 if dense else 8,
                        "rounds": [{"peak_allocated_bytes": 100 if dense else 80}],
                    },
                }

            stack.enter_context(
                patch.object(ps, "inspect_benchmark", side_effect=inspect)
            )

            def launch(request, directory):
                directory.mkdir()
                (directory / "request.json").write_text(json.dumps(request))
                (directory / "result.json").write_text(
                    json.dumps({"worker_seconds": 1, "gpu_reserved_seconds": 1})
                )
                if request["force_dev"]:
                    i = int(directory.parent.name.split("-")[1])
                    q[i] = quality(True)

            worker = stack.enter_context(patch.object(ps, "launch", side_effect=launch))
            first = ps.screen(runner, execute=True)
            self.assertEqual(worker.call_count, 6)
            self.assertEqual(first["status"], "completed")
            self.assertTrue(
                all(r["confirmed_improvement"] for r in first["candidates"].values())
            )
            gpu.reset_mock()
            worker.reset_mock()
            second = ps.screen(runner, execute=True)
            gpu.assert_not_called()
            worker.assert_not_called()
            self.assertEqual(second["worker_seconds"], 6)
            self.assertEqual(second["selected"], first["selected"])


if __name__ == "__main__":
    unittest.main()
