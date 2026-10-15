"""Independent reproductions of the three second-audit blockers; no ARC plays."""
import ast
import copy
import pathlib
import unittest
from unittest.mock import patch
import common
import analyze_results
import test_review as F
import supervisor as V
from common import *
from analyze_results import decision

class StableSemanticSchema(unittest.TestCase):
    def test_empty_python312_type_params_does_not_change_identity(self):
        source="def act(params, mem, frame, avail, info):\n    return avail[0], 0, 0\n"
        expected=normalized_source(source); tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef):
                node.type_params=[]
                node._fields=tuple(node._fields)+("type_params",)
        with patch("common.ast.parse",return_value=tree):
            self.assertEqual(normalized_source(source),expected)
    def test_nonempty_new_semantics_are_rejected(self):
        source="def f():\n    return 1\n"; tree=ast.parse(source)
        tree.body[0].type_params=["new semantics"]; tree.body[0]._fields+=("type_params",)
        with patch("common.ast.parse",return_value=tree),self.assertRaises(Blocked):
            normalized_source(source)
    def test_repr_is_json_with_frozen_schema(self):
        import json
        parsed=json.loads(normalized_source("x = 1"))
        self.assertEqual(parsed["schema"],"cellforge-semantic-ast/1")

class ActiveCancellation(unittest.TestCase):
    setUp=F.ControllerCrashPoints.setUp
    tearDown=F.ControllerCrashPoints.tearDown
    enable=F.ControllerCrashPoints.enable
    def test_active_cancel_stops_tree_and_accounts_once(self):
        self.enable(); self.controller.tick(); self.assertEqual(self.backend.starts,1)
        before=read(self.root/"REGISTRY.json"); self.assertEqual(before["active"],self.fp)
        (self.root/"cancel").mkdir(); (self.root/"cancel"/self.fp).write_text("isolated cancellation")
        self.controller.tick()
        after=read(self.root/"REGISTRY.json")
        self.assertEqual(self.backend.stops,1)
        self.assertIsNone(after["active"]); self.assertEqual(after["conditions"][self.fp]["status"],"cancelled")
        self.assertEqual(after["studies"][self.ph]["attempted"],1); self.assertEqual(after["consecutive_failures"],0)
        run=self.controller.context(after,self.fp)[3]; receipt=read(run/"TERMINAL.json")
        self.assertTrue(receipt["evidence"]["job_tree_empty"])
        # Pausing prevents another distinct pending fixture from launching.
        (self.root/"DISPATCH_ENABLED").unlink(); self.controller.tick()
        self.assertEqual(read(self.root/"REGISTRY.json")["sequence"],1); self.assertEqual(self.backend.stops,1)
        self.assertEqual(self.backend.starts,1)

class ZeroComparator(unittest.TestCase):
    def results(self,zero):
        data={c["fingerprint"]:F.ComparisonGate().rows(c,full_levels=1) for c in F.PLAN["conditions"]}
        for rows in data.values():
            for row in rows:
                if row["arm"]=="zero" and row["result"]["game"]=="ar25": row["result"]["levels"]=zero
        return data
    def test_founder_one_zero_two_fails(self):
        result=decision(F.PLAN,self.results(2))
        self.assertTrue(all(f["comparisons_complete"] for f in result["founders"]))
        self.assertFalse(any(f["engineering_gate_passed"] for f in result["founders"]))
    def test_founder_equal_zero_fails(self):
        self.assertFalse(any(f["engineering_gate_passed"] for f in decision(F.PLAN,self.results(1))["founders"]))
    def test_founder_beats_zero_and_every_other_control_can_pass(self):
        self.assertTrue(all(f["engineering_gate_passed"] for f in decision(F.PLAN,self.results(0))["founders"]))
        self.assertIn("zero",analyze_results.COMPARATORS)

if __name__=="__main__": unittest.main(verbosity=2)

