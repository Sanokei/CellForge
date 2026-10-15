import unittest,copy
from cellforge.construction.action_phase_protocol import *
from cellforge.construction.joint_weight_population import digest

class TestActionPhaseProtocol(unittest.TestCase):
 def params(self,n=0):
  return {"argument.W":[[n]*12 for _ in range(3)],"argument.b":[0,0,0]}
 def scores(self,ok):
  return {k:{"partition":"development","source":"isolated-action-fixture-safety" if k=="anchor" else "isolated-action-fixture-development",
   "manifest":k,"outcomes":{"case":ok},"false_positive_given_consistency":0,"traces":[{"id":"case","correct":ok,"files_correct":True}]} for k in ("fresh","archive","anchor")}
 def pop(self,n=0):
  p=ActionCellPopulation();p.seed(FAMILY,self.params(n),{"author":"fixture"},self.scores(False));return p
 def author(self,p):
  cid=p.live[FAMILY][0];recipient=p.cells[cid];feedback={"traces":[]}
  prompt={"parent_id":cid,"parent_params":recipient["params"],"development_evidence":feedback,"partition":"development",
   "feature_order":list(FEATURES),"role_order":["arg1","arg2","file"],
   "edit_contract":{"positions":39,"range":[-9,9],"operator":"literal_full_matrix","interface":"tool-action-joint/2"}}
  response={"model":"gpt-6.1-sol","reasoning":"low","parent_id":cid,"parent_params":recipient["params"],"tensors":self.params(1),
   "feature_order":list(FEATURES),"role_order":["arg1","arg2","file"],"prompt_sha256":digest(prompt),
   "feedback_sha256":digest(feedback),"charged_positions":39,"interface":"tool-action-joint/2"}
  return response,prompt,feedback
 def test_raw_parent_prompt_tensor_shape_and_budget_binding(self):
  p=self.pop();r,q,f=self.author(p);recipient=p.cells[p.live[FAMILY][0]]
  self.assertEqual(bind_author(r,q,recipient,f)[1]["positions"],39)
  for mode in ("parent","prompt","feature","width","budget","io","numeric"):
   rr=copy.deepcopy(r);qq=copy.deepcopy(q)
   if mode=="parent":rr["parent_id"]="forged"
   elif mode=="prompt":qq["development_evidence"]={"reporting":True}
   elif mode=="feature":rr["feature_order"]=list(reversed(FEATURES))
   elif mode=="width":rr["tensors"]["argument.W"][0].append(1)
   elif mode=="budget":rr["charged_positions"]=38
   elif mode=="io":rr["interface"]="unreviewed/99"
   else:rr["tensors"]["argument.W"][0][0]=10
   with self.subTest(mode=mode),self.assertRaises(ValueError):bind_author(rr,qq,recipient,f)
 def test_random_parent_is_independent_survivor_and_charges_precede_execution(self):
  llm=self.pop(0);rand=self.pop(-1);r,q,f=self.author(llm);journal=[]
  def evaluate(params):
   self.assertTrue(journal);return self.scores(params==self.params(1))
  actual=apply_phase({"llm":llm.snapshot(),"random":rand.snapshot()},101,1,r,q,f,evaluate,journal.append)
  self.assertNotEqual(actual["llm"]["parent"],actual["random"]["parent"])
  self.assertTrue(actual["llm"]["decision"]["accepted"]);self.assertFalse(actual["random"]["decision"]["accepted"])
  self.assertEqual([x["positions"] for x in journal],[39,39])
  self.assertEqual(actual["random"]["population"]["live"],rand.snapshot()["live"])
 def test_report_selection_rejected_without_admission(self):
  p=self.pop();r,q,f=self.author(p);journal=[]
  def evaluate(params):
   s=self.scores(True);s["fresh"]["partition"]="reporting";return s
  with self.assertRaisesRegex(ValueError,"reporting"):
   apply_phase({"llm":p.snapshot(),"random":p.snapshot()},101,1,r,q,f,evaluate,journal.append)
  self.assertEqual(len(journal),1);self.assertEqual(len(p.charges),0)
 def test_caps_and_fixture_rates_are_computed_not_boolean_labels(self):
  with self.assertRaises(ValueError):argument_caps(self.params(10))
  s=self.scores(True);s["anchor"]["false_positive_given_consistency"]=None
  with self.assertRaises(ValueError):validate_scores(s)
 def test_invalid_authoring_is_charged_before_binding_rejection(self):
  p=self.pop();r,q,f=self.author(p);r["tensors"]["argument.W"][0].append(0);charges=[]
  with self.assertRaises(ValueError):
   apply_phase({"llm":p.snapshot(),"random":p.snapshot()},101,1,r,q,f,lambda v:self.scores(True),charges.append)
  self.assertEqual(len(charges),1);self.assertEqual(charges[0]["positions"],39)
 def test_fabricated_outcomes_or_fixture_rate_rejected(self):
  s=self.scores(True);s["fresh"]["outcomes"]["case"]=False
  with self.assertRaises(ValueError):validate_scores(s)
  s=self.scores(True);s["anchor"]["traces"][0]["files_correct"]=False
  with self.assertRaises(ValueError):validate_scores(s)
 def test_dropped_rejected_charge_and_fabricated_transition_rejected(self):
  p=self.pop();r,q,f=self.author(p);charges=[]
  arms=apply_phase({"llm":p.snapshot(),"random":p.snapshot()},101,1,r,q,f,lambda v:self.scores(v==self.params(1)),charges.append)
  expected={"phase":1,"contexts":[{"seed":101,"arms":arms}]}
  self.assertTrue(validate_phase_receipt(expected,copy.deepcopy(expected),charges))
  with self.assertRaisesRegex(ValueError,"charged"):validate_phase_receipt(expected,expected,charges[:-1])
  forged=copy.deepcopy(expected);forged["contexts"][0]["arms"]["random"]["decision"]["accepted"]=True
  with self.assertRaisesRegex(ValueError,"selection"):validate_phase_receipt(expected,forged,charges)
