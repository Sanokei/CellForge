"""Shared-policy measurement must keep one genome and fail closed on controls."""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import arc3_shared_pilot as P


def result(levels, error=None):
    return dict(levels=levels, used=1500, error=error, level_steps=[50] * levels)


class SharedMeasurement(unittest.TestCase):
    def measure(self, error=None):
        entry = dict(source_lines=["def act(params, mem, frame, avail, info):",
                                   "    return (avail[0], 0, 0)"], params={"W": [1.]})
        controls = dict(zero=result(0, error), blind=result(0),
                        random=[result(0) for _ in range(3)])
        with patch.object(P.E, "play", return_value=result(1)) as play, \
             patch.object(P.E, "controls", return_value=controls) as ctl:
            measured = P.evaluate(entry)
            for call in play.call_args_list + ctl.call_args_list:
                self.assertIs(call.args[1], entry["params"])
                self.assertEqual(call.args[3], 1500)
                self.assertEqual(call.kwargs["seed"], 0)
            self.assertEqual([c.args[2] for c in play.call_args_list], P.GAMES)
        return measured

    def test_same_genome_and_budget_on_entire_panel(self):
        self.assertEqual(self.measure()["credited_total"], 3)

    def test_crashed_control_cannot_establish_weight_dependence(self):
        measured = self.measure("worker crash")
        self.assertEqual(measured["raw_total"], 3)
        self.assertEqual(measured["credited_total"], 0)

    def test_replay_checks_action_cost_and_errors_as_well_as_levels(self):
        first = self.measure()
        import copy
        altered = copy.deepcopy(first)
        altered["games"][0]["result"]["used"] += 1
        self.assertNotEqual(P.metrics(first), P.metrics(altered))
