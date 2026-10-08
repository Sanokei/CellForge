"""Additive control-harness regressions. Tiny fixtures are not study outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("nca_feasibility_preflight", Path(__file__).parents[1] / "scripts/nca_feasibility_preflight.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture_task():
    return {"task_id": "11111111", "train": [
        {"input": [[1, 2]], "output": [[1, 2]]},
        {"input": [[3]], "output": [[4]]}]}


class NCAFeasibilityPreflightTests(unittest.TestCase):
    def test_four_arms_same_t8_structure_and_1488_distinct_positions(self):
        graphs = [M.control_graph(arm) for arm in M.ARMS]
        structure = M.graph_structure(graphs[0])
        for graph in graphs:
            self.assertEqual(M.graph_structure(graph), structure)
            self.assertEqual(graph.meta["nca_recipe"]["iterations"], 8)
            self.assertEqual(sum(a.size for a in graph.params.values()), 1488)
            self.assertEqual(len(graph.params), 96)
            active = [n["use"] for n in graph.defs["nca_step"]["nodes"] if n["use"] in graph.defs and graph.defs[n["use"]]["kind"] == "cell"]
            self.assertEqual(len(set(active)), 48)
        self.assertEqual(sum(n["use"] == "nca_step" for n in graphs[0].defs["nca_gene"]["nodes"]), 8)

    def test_random_is_fully_seeded_seven_value_draw(self):
        first, second = M.control_graph("paired_random_1"), M.control_graph("paired_random_1")
        self.assertEqual(first.hash(), second.hash())
        self.assertTrue(all(np.isin(a, M.VOCABULARY).all() for a in first.params.values()))
        self.assertEqual(sum(a.size for a in first.params.values()), 1488)
        import random
        rng = random.Random(2026100801)
        expected = [rng.choice(M.VOCABULARY) for _ in range(1488)]
        actual = [float(x) for name in sorted(first.params) for x in first.params[name].ravel()]
        self.assertEqual(actual, expected)

    def test_second_draw_and_author_arms_are_prohibited(self):
        for name in ("paired_random_2", "founder", "mutation", "random"):
            with self.assertRaises(ValueError):
                M.control_graph(name)

    def test_existing_smoothing_is_explicit_reference_exception(self):
        smoothing = M.control_graph("neighbor_majority")
        self.assertTrue(any(not np.isin(a, M.VOCABULARY).all() for a in smoothing.params.values()))
        self.assertIn("exception", M.future_protocol()["fixed_references"])
        self.assertTrue(all(not a.any() for a in M.control_graph("zero").params.values()))

    def test_24_bias_redundant_directions_preserved(self):
        graph = M.control_graph("identity")
        self.assertEqual(sum(graph.params[f"nca.h{row}.c{chunk}.b"].size for row in range(12) for chunk in range(3)), 36)
        # Rebalancing chunk biases by (+1,-1,0) leaves row sum and fixed graph unchanged.
        altered = graph.copy()
        for row in range(12):
            altered.params[f"nca.h{row}.c0.b"] += 1
            altered.params[f"nca.h{row}.c1.b"] -= 1
        _, _, _, _, _, Predictor, _ = M.runtime_imports()
        task = fixture_task()
        self.assertEqual(Predictor(graph)(task["train"], [[1, 2]]), Predictor(altered)(task["train"], [[1, 2]]))

    def test_leave_one_out_excludes_scored_pair_in_original_order(self):
        task = fixture_task()
        self.assertEqual(M.leave_one_out(task, 0), [task["train"][1]])
        self.assertEqual(M.leave_one_out(task, 1), [task["train"][0]])
        with self.assertRaises(ValueError):
            M.leave_one_out({"train": task["train"][:1]}, 0)

    def test_real_fixed_graph_prediction_only_on_tiny_fixture(self):
        _, _, _, _, _, Predictor, _ = M.runtime_imports()
        row = M.evaluate_demo(Predictor(M.control_graph("identity")), fixture_task(), 0)
        self.assertEqual(row["prediction"], [[1, 2]])
        self.assertTrue(row["exact"])
        self.assertEqual(row["pixel_errors"], 0)

    def test_runtime_errors_are_null_not_wrong_predictions(self):
        def fail(*args):
            raise RuntimeError("fixture failure")
        row = M.evaluate_demo(fail, fixture_task(), 0)
        self.assertIsNone(row["exact"])
        self.assertIsNone(row["normalized_pixel_error"])
        summary = M.summarize([row, M.error_row("11111111", 1, {"type": "error"})], [fixture_task()])
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["runtime_errors"], 2)
        self.assertIsNone(summary["exact_demonstrations"])
        self.assertIsNone(summary["exact_tasks"])

    def test_metric_is_unweighted_mean_per_demo(self):
        rows = [{"task_id": "11111111", "demo_index": i, "status": "valid", "exact": False,
                 "normalized_pixel_error": value} for i, value in enumerate((0.5, 1.0))]
        summary = M.summarize(rows, [fixture_task()])
        self.assertEqual(summary["mean_normalized_pixel_error"], 0.75)
        self.assertEqual(summary["exact_tasks"], 0)
        self.assertEqual(summary["demonstration_denominator"], 2)

    def test_coverage_constants_and_future_stop_protocol(self):
        self.assertEqual(len(M.ALL_IDS), 11)
        self.assertEqual(len(M.COMPATIBLE_IDS), 6)
        self.assertNotIn("e7639916", M.ALL_IDS)
        p = M.future_protocol()
        self.assertEqual(p["maximum_author_calls"], 2)
        self.assertEqual(p["charged_positions_per_consumed_author_attempt"], 1488)
        self.assertEqual(p["maximum_positions_each_arm"], 2976)
        self.assertEqual(p["paired_random_seeds_in_attempt_order"], [2026100801, 2026100802])
        self.assertIn("stops", p["invalid_or_refused"])
        self.assertIn("preserve", p["positive_feasibility"])
        self.assertIn("No random", p["founder_packet"])

    def test_duplicate_json_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    M.read_json(path)

    def test_query_data_rejected(self):
        *_, diagnostic = M.runtime_imports()
        bad = dict(fixture_task(), test=[{"input": [[1]], "output": [[2]]}])
        with self.assertRaises(ValueError):
            diagnostic.validate_task(bad)

    def test_hash_and_stop_fail_closed_before_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "MANIFEST.json").write_text("{}")
            (run / "MANIFEST.sha256").write_text("wrong")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                M.verify_freeze(run, "wrong")
            (run / "STOP").touch()
            with self.assertRaisesRegex(ValueError, "STOP"):
                M.verify_freeze(run, "wrong")

    def test_dangling_stop_symlink_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "STOP").symlink_to(run / "missing-stop-target")
            with self.assertRaisesRegex(ValueError, "STOP"):
                M.verify_freeze(run, "irrelevant")

    def test_no_overwrite_of_artifacts_or_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            path = run / "fixed.json"
            M.write_json(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                M.write_json(path, {"a": 2})
            with self.assertRaisesRegex(ValueError, "fresh run"):
                M.freeze(run, run)

    def test_replay_excludes_only_timing_fields(self):
        first = {"exact": True, "elapsed_seconds": 1, "exec_stats": {"macs": 10}}
        second = dict(first, elapsed_seconds=2)
        self.assertEqual(M.semantic_result(first), M.semantic_result(second))
        second["exact"] = False
        self.assertNotEqual(M.semantic_result(first), M.semantic_result(second))

    def test_perfect_or_invalid_control_blocks_author(self):
        manifest = {"arms": [{"arm": name, "tensor_sha256": name} for name in M.ARMS], "panel": {}, "accounting": {},
                    "future_author_protocol": M.future_protocol()}
        results = []
        for name in M.ARMS:
            rows = [{"prediction": [[1]]}] * 17
            results.append({"arm": name, "demonstrations": rows, "summary": {"valid": True, "exact_demonstrations": 0,
                            "exact_tasks": 0, "mean_normalized_pixel_error": 1, "solved_demo_keys": []}})
        base = M.summarize_baseline(results, manifest)
        self.assertEqual(base["selected_baseline"], "identity")
        self.assertFalse(base["author_dispatch_authorized"])
        results[3]["summary"]["exact_demonstrations"] = 17
        self.assertIn("baseline_perfect", M.summarize_baseline(results, manifest)["stop_reasons"])
        results[3]["summary"]["valid"] = False
        self.assertIn("invalid_control", M.summarize_baseline(results, manifest)["stop_reasons"])

    def test_tensor_tamper_or_shape_change_rejected(self):
        graph = M.control_graph("identity")
        _, _, _, _, array_hash, _, _ = M.runtime_imports()
        payload = {"schema": "nca-literal-tensors/1", "arm": "identity", "tensors": {key: value.tolist() for key, value in graph.params.items()}}
        manifest = {"arms": [{"arm": "identity", "tensor_file": "weights.json", "graph_sha256": graph.hash(),
                              "tensor_sha256": M.digest(payload["tensors"]), "per_tensor_sha256": {k: array_hash(a) for k, a in graph.params.items()}}],
                    "graph": {"structure_sha256": M.digest(M.graph_structure(graph))}}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            path = run / "weights.json"
            M.write_json(path, payload)
            self.assertEqual(M.load_graph(run, manifest, "identity").hash(), graph.hash())
            payload["tensors"]["nca.h0.c0.b"] = [1]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "per-tensor"):
                M.load_graph(run, manifest, "identity")
            payload["tensors"]["nca.h0.c0.b"] = [0, 0]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "shape"):
                M.load_graph(run, manifest, "identity")


if __name__ == "__main__":
    unittest.main()
