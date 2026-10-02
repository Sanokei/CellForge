import unittest,pathlib,tempfile,shutil,json,hashlib
from cellforge.construction.action_verification import verify
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestActionVerification(unittest.TestCase):
 def test_saved_trial_recomputes_and_never_grants_scientific_acceptance(self):
  r=verify(ROOT/"runs/weights-action-arguments-v1")
  self.assertTrue(r["mechanism_replay"]["passed"],r)
  self.assertFalse(r["scientific_headline_allowed"])
 def test_missing_evidence_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   self.assertFalse(verify(d)["mechanism_replay"]["passed"])
 def test_tampered_scores_fail_even_with_updated_self_declared_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)
   for f in (ROOT/"runs/weights-action-arguments-v1").iterdir():
    if f.is_file():shutil.copyfile(f,p/f.name)
   r=json.loads((p/"RESULT.json").read_text());r["results"][0]["scores"]["reporting"]["state_correct"]=12
   (p/"RESULT.json").write_text(json.dumps(r))
   h=json.loads((p/"INTEGRITY.json").read_text());h["RESULT.json"]=hashlib.sha256((p/"RESULT.json").read_bytes()).hexdigest()
   (p/"INTEGRITY.json").write_text(json.dumps(h))
   r=verify(p)
   self.assertFalse(r["mechanism_replay"]["passed"])
   self.assertIn("score_replay_differs",r["mechanism_replay"]["errors"])
 def test_stale_weight_evidence_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)
   for f in (ROOT/"runs/weights-action-arguments-v1").iterdir():
    if f.is_file():shutil.copyfile(f,p/f.name)
   r=json.loads((p/"FROZEN.json").read_text());r[1]["W"][0]=[0]*8
   (p/"FROZEN.json").write_text(json.dumps(r))
   self.assertFalse(verify(p)["mechanism_replay"]["passed"])

class TestPopulationVerification(unittest.TestCase):
 def trial(self):return ROOT/"runs/weights-action-population-corrected-g3"
 def rebind(self,p,name):
  h=json.loads((p/"FINAL_INTEGRITY.json").read_text());h[name]=hashlib.sha256((p/name).read_bytes()).hexdigest()
  (p/"FINAL_INTEGRITY.json").write_text(json.dumps(h))
 def test_population_replay_reconstructs_selection_and_budgets(self):
  r=verify(self.trial())
  self.assertTrue(r["population_replay"]["passed"],r)
  self.assertFalse(r["evidence_quality"]["passed"])
  self.assertFalse(r["runtime_acceptance"])
  self.assertIn("fixture_filename_not_copyable:reporting:newbare:JOURNAL",r["evidence_quality"]["errors"])
  self.assertFalse(r["scientific_headline_allowed"])
  self.assertEqual(len(r["population_replay"]["counts"]),3)
 def test_fabricated_live_membership_fails_even_after_rehash(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);shutil.copytree(self.trial(),p,dirs_exist_ok=True)
   r=json.loads((p/"RESULT.json").read_text())
   r["contexts"][0]["population"]["live"]["action-arguments-lexical"]=[]
   (p/"RESULT.json").write_text(json.dumps(r));self.rebind(p,"RESULT.json")
   r=verify(p)
   self.assertFalse(r["mechanism_replay"]["passed"])
   self.assertTrue(any("membership" in e for e in r["mechanism_replay"]["errors"]))
 def test_unmatched_budget_and_unexecuted_gate_fail_after_rehash(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);shutil.copytree(self.trial(),p,dirs_exist_ok=True)
   r=json.loads((p/"MANIFEST.json").read_text());r["charged_positions"]["router"]["random"]=261
   (p/"MANIFEST.json").write_text(json.dumps(r));self.rebind(p,"MANIFEST.json")
   gate=json.loads((p/"RUNTIME_GATE_RESULT.json").read_text());gate["executed"]=False
   (p/"RUNTIME_GATE_RESULT.json").write_text(json.dumps(gate));self.rebind(p,"RUNTIME_GATE_RESULT.json")
   r=verify(p);self.assertFalse(r["mechanism_replay"]["passed"])
   self.assertIn("router_charged_capacity_not_equal",r["mechanism_replay"]["errors"])
   self.assertIn("runtime_gate_not_executed_or_failed",r["mechanism_replay"]["errors"])
