"""Checker regressions only. These fixtures are not a capability benchmark."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "nca_capacity_diagnostic", Path(__file__).parents[1] / "scripts/nca_capacity_diagnostic.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def task(task_id, source, target):
    return {"task_id": task_id, "train": [{"input": source, "output": target}]}


class NCACapacityDiagnosticTests(unittest.TestCase):
    def test_size_change_is_ruled_out(self):
        result = MODULE.analyze([task("11111111", [[1]], [[1, 1]])])
        self.assertEqual(result["whole_panel_necessary_conditions"], "ruled_out")
        self.assertEqual(len(result["shape_mismatch_pairs"]), 1)

    def test_black_is_distinct_from_outside_mask(self):
        self.assertNotEqual(MODULE.neighborhood([[0]], 0, 0),
                            MODULE.neighborhood([[0, 0]], 0, 0))

    def test_conflicting_identical_input_across_tasks(self):
        result = MODULE.analyze([task("11111111", [[0]], [[1]]),
                                 task("22222222", [[0]], [[2]])])
        self.assertEqual(len(result["demonstration_invariance_conflicts"]), 1)
        self.assertEqual(result["cross_task_conflict_edges"], [["11111111", "22222222"]])
        self.assertEqual(result["same_size_unavoidable_pixel_errors_lower_bound"], 1)
        self.assertTrue(all(row["individual_necessary_conditions"] == "not_ruled_out"
                            for row in result["task_status"]))

    def test_within_task_conflict_at_distance_beyond_radius(self):
        source = [[0] * 30 for _ in range(30)]
        target = [row[:] for row in source]
        target[10][10] = 1
        result = MODULE.analyze([task("11111111", source, target)])
        self.assertGreater(result["task_status"][0]["within_task_conflict_classes"], 0)
        self.assertEqual(result["task_status"][0]["individual_necessary_conditions"], "ruled_out")

    def test_radius_boundary_is_included_and_beyond_is_not(self):
        source = [[0] * 30 for _ in range(30)]
        at_boundary = [row[:] for row in source]
        at_boundary[15][23] = 1
        beyond = [row[:] for row in source]
        beyond[15][24] = 1
        original = MODULE.neighborhood(source, 15, 15)
        self.assertNotEqual(original, MODULE.neighborhood(at_boundary, 15, 15))
        self.assertEqual(original, MODULE.neighborhood(beyond, 15, 15))

    def test_identical_input_conflicting_demos_within_task(self):
        payload = task("11111111", [[0]], [[1]])
        payload["train"].append({"input": [[0]], "output": [[2]]})
        result = MODULE.analyze([payload])
        self.assertEqual(result["task_status"][0]["within_task_conflict_classes"], 1)

    def test_query_fields_rejected_and_bool_is_not_color(self):
        payload = task("11111111", [[0]], [[1]])
        payload["test"] = [{"input": [[0]]}]
        with self.assertRaises(ValueError):
            MODULE.analyze([payload])
        with self.assertRaises(ValueError):
            MODULE.analyze([task("11111111", [[False]], [[1]])])

    def test_pass_is_only_not_ruled_out_and_order_stable(self):
        first = task("11111111", [[1]], [[1]])
        second = task("22222222", [[2]], [[2]])
        result = MODULE.analyze([first, second])
        self.assertEqual(result["whole_panel_necessary_conditions"], "not_ruled_out")
        self.assertEqual(result, MODULE.analyze([second, first]))
        with self.assertRaises(ValueError):
            MODULE.analyze([first, first])

    def test_manifest_loader_rejects_changed_data_or_missing_panel_member(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            payload = repo / "demo.json"
            payload.write_text(json.dumps(task("11111111", [[1]], [[1]])))
            inventory = repo / "inventory.json"
            inventory.write_text(json.dumps({"base_commit": "pinned", "entries": [
                {"task_id": "11111111", "path": "AUTHOR.json", "git_blob_sha1": "blob"}]}))
            row = {"task_id": "11111111", "source_packet_path": "AUTHOR.json",
                   "source_git_blob_sha1": "blob", "derived_path": "demo.json",
                   "derived_sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            receipt = {"base_commit": "pinned", "input_inventory_sha256":
                       hashlib.sha256(inventory.read_bytes()).hexdigest(), "rows": [row]}
            materialization = repo / "materialization.json"
            materialization.write_text(json.dumps(receipt))
            tasks, _ = MODULE.load_frozen_panel(repo, inventory, materialization)
            self.assertEqual(len(tasks), 1)
            receipt["rows"] = []
            materialization.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                MODULE.load_frozen_panel(repo, inventory, materialization)
            receipt["rows"] = [row]
            materialization.write_text(json.dumps(receipt))
            payload.write_text(json.dumps(task("11111111", [[1]], [[2]])))
            with self.assertRaises(ValueError):
                MODULE.load_frozen_panel(repo, inventory, materialization)

    def test_manifest_loader_rejects_inventory_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            inventory = repo / "inventory.json"
            inventory.write_text('{"base_commit":"pinned","entries":[]}')
            materialization = repo / "materialization.json"
            materialization.write_text(json.dumps({"input_inventory_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "inventory hash mismatch"):
                MODULE.load_frozen_panel(repo, inventory, materialization)


if __name__ == "__main__":
    unittest.main()
