import unittest,copy,pathlib
from cellforge.construction.arc_weight_moe import *
class TestARCWeightMoE(unittest.TestCase):
 def test_query_answers_stripped_before_model_view(self):
  task={"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]],"output":[[9]],"other":"secret"}]}
  got=stripped_task(task);self.assertEqual(got["test"],[{"input":[[3]]}]);self.assertEqual(got["train"][0]["output"],[[2]])
 def test_zero_score_rejected_without_holding_cell_service(self):
  evidence={"partition":"development","outcomes":{"a":False},"solved":0,"total":1}
  decision=admission(evidence,copy.deepcopy(evidence),[],.01)
  self.assertFalse(decision["accepted"]);self.assertIn("owner_ARC_floor",decision["reasons"])
 def test_full_task_regression_cannot_hide_in_aggregate_gain(self):
  parent={"partition":"development","outcomes":{"a":True,"b":False,"c":False},"solved":1,"total":3}
  child={"partition":"development","outcomes":{"a":False,"b":True,"c":True},"solved":2,"total":3}
  self.assertFalse(admission(parent,child,[],0)["accepted"])
 def test_reporting_selection_rejected(self):
  p={"partition":"reporting","outcomes":{"a":True},"solved":1,"total":1}
  with self.assertRaises(ValueError):admission(p,p,[],0)
