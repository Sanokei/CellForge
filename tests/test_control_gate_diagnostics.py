"""Expected-defect diagnostics, NOT scientific acceptance or historical replay.

These assertions expose the current owner-controlled gate behavior. Once an
owner-authorized repair lands, update these additive diagnostics accordingly.
No protected gate or real experiment artifact is changed by this module.
"""
import types
import unittest
from unittest.mock import patch

from tests import arc_goal_checks as A
from tests import standard_checks as S


class ProtectedControlGateDiagnostics(unittest.TestCase):
    def setUp(self):
        self.task = {"train": [], "test": [{"input": [[1]], "output": [[1]]}]}
        self.weights = {"W": [[1.0, 2.0]]}

    def test_current_dictionary_zeroing_loses_tensor_structure(self):
        self.assertEqual(S._zeros_like(self.weights), 0.0)
        self.assertEqual(S._zeros_like(self.weights["W"]), [[0.0, 0.0]])
        self.assertEqual(self.weights, {"W": [[1.0, 2.0]]})

    def test_current_solved_conflates_execution_error_and_wrong_answer(self):
        def invalid(weights, task):
            self.assertNotIn("output", task["test"][0])
            raise TypeError("fixture rejects malformed control")
        failing = types.SimpleNamespace(predict=invalid)
        wrong = types.SimpleNamespace(predict=lambda w, t: [[[[0]]]])
        self.assertFalse(A.solved(failing, 0.0, self.task))
        self.assertFalse(A.solved(wrong, self.weights, self.task))
        self.assertIn("output", self.task["test"][0])

    def test_current_ablation_can_pass_with_only_invalid_control_execution(self):
        seen = []
        def predict(weights, task):
            seen.append(weights)
            self.assertNotIn("output", task["test"][0])
            if not isinstance(weights, dict):
                raise TypeError("fixture requires named tensors")
            # Deliberately ignores numeric values: shape-correct zeros solve too.
            return [[task["test"][0]["input"]]]
        ev = types.SimpleNamespace(predict=predict)
        score = {"split": "training", "task_ids": ["fixture"],
                 "weights_file": "fixture.json", "evaluator": "fixture.py",
                 "claimed": {"per_task": {"fixture": True}}}
        self.assertTrue(A.solved(ev, {"W": [[0.0, 0.0]]}, self.task))
        seen.clear()
        with patch.object(A, "load_score", return_value=score), \
             patch.object(A.os.path, "isdir", return_value=True), \
             patch.object(S, "weights_of", return_value=self.weights), \
             patch.object(A, "load_eval", return_value=ev), \
             patch.object(S, "load", return_value=self.task):
            self.assertEqual(A.check_weights_matter("fixture-only", root="fixture-root"), [])
        self.assertEqual(seen, [0.0])


if __name__ == "__main__":
    unittest.main()
