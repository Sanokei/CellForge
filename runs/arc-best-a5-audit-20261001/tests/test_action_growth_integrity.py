import unittest,tempfile,pathlib,json,copy,hashlib
from cellforge.construction.anchored_action_protocol import verify_input_anchor,validate_snapshot,resolve_live_router,REQUIRED
from cellforge.construction.joint_weight_population import digest
from cellforge.construction.action_growth_verification import verify_growth

ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestGrowthIntegrity(unittest.TestCase):
 def snapshot(self):
  params={"W":[[1]],"b":[0]};cid=digest({"structure":"action-router","params":params})
  cell={"id":cid,"structure":"action-router","params":params,"parents":[],"state":"live","scores":{}}
  return {"cells":{cid:cell},"live":{"action-router":[cid]},"histories":{"action-router":[{"graph_hash":cid,"scores":{}}]}},cid
 def fixture(self,p):
  row=lambda i:{"id":i,"request":"write 'x' to LOG","initial":{},"expected":{"LOG":"x"},"action":"write_file"}
  values={"MANIFEST.json":{k:[row(k)] for k in ("fresh","archive","safety","reporting")},
   "parent-result.json":{"contexts":[{"population":self.snapshot()[0]}]},"baseline.json":{},
   "AUTHORING_CONTRACT.json":{},"INTERFACES.json":{},"RUNTIME_SOURCES.json":["runtime.py"]}
  for n,v in values.items():(p/n).write_text(json.dumps(v))
  (p/"run.py").write_text("# not executed");(p/"runtime.py").write_text("# not executed")
  dependencies={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in [*REQUIRED,"runtime.py"]}
  (p/"DECLARATION.json").write_text(json.dumps({"schema":"anchored-action-inputs/1","dependencies":dependencies},sort_keys=True))
  return hashlib.sha256((p/"DECLARATION.json").read_bytes()).hexdigest()
 def rehash(self,p,n):
  d=json.loads((p/"DECLARATION.json").read_text());d["dependencies"][n]=hashlib.sha256((p/n).read_bytes()).hexdigest()
  (p/"DECLARATION.json").write_text(json.dumps(d,sort_keys=True))
 def test_input_preflight_never_grants_runtime_acceptance(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);anchor=self.fixture(p);receipt=verify_input_anchor(p,anchor)
   self.assertTrue(receipt["input_anchor_verified"]);self.assertFalse(receipt["runtime_acceptance"])
 def test_rehashed_forged_reporting_and_budget_cannot_replace_external_anchor(self):
  for changed in ("request","expected","partition","budget"):
   with self.subTest(changed=changed),tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d);anchor=self.fixture(p);m=json.loads((p/"MANIFEST.json").read_text())
    if changed=="request":m["reporting"][0]["request"]="write 'y' to LOG"
    elif changed=="expected":m["reporting"][0]["expected"]["LOG"]="y"
    elif changed=="partition":m["fresh"]=m.pop("reporting")
    else:m["budget"]=999
    (p/"MANIFEST.json").write_text(json.dumps(m));self.rehash(p,"MANIFEST.json")
    with self.assertRaisesRegex(ValueError,"anchor mismatch"):verify_input_anchor(p,anchor)
 def test_missing_and_partial_coverage_rejected_even_with_new_anchor(self):
  for mode in ("empty","partial","missing"):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d);self.fixture(p);decl=json.loads((p/"DECLARATION.json").read_text())
    if mode=="empty":decl["dependencies"]={}
    elif mode=="partial":decl["dependencies"].pop("parent-result.json")
    else:(p/"baseline.json").unlink()
    (p/"DECLARATION.json").write_text(json.dumps(decl))
    anchor=hashlib.sha256((p/"DECLARATION.json").read_bytes()).hexdigest()
    with self.assertRaises(ValueError):verify_input_anchor(p,anchor)
 def test_unbound_runtime_and_changed_parent_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);anchor=self.fixture(p);(p/"runtime.py").write_text("forged")
   with self.assertRaisesRegex(ValueError,"dependency changed"):verify_input_anchor(p,anchor)
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);anchor=self.fixture(p);(p/"parent-result.json").write_text("{}")
   with self.assertRaisesRegex(ValueError,"dependency changed"):verify_input_anchor(p,anchor)
 def test_fabricated_parent_tensor_edge_and_history_rejected(self):
  for mode in ("id","tensor","edge","history"):
   s,cid=self.snapshot()
   if mode=="id":s["cells"][cid]["id"]="fake"
   elif mode=="tensor":s["cells"][cid]["params"]["b"]=[1]
   elif mode=="edge":s["cells"][cid]["parents"]=[cid]
   else:s["histories"]["action-router"][0]["graph_hash"]="fake"
   with self.subTest(mode=mode),self.assertRaises(ValueError):validate_snapshot(s)
 def test_selected_router_uses_live_tensors_and_rejects_retired(self):
  s,cid=self.snapshot();self.assertEqual(resolve_live_router(s,cid),s["cells"][cid]["params"])
  s["cells"][cid]["state"]="retired"
  with self.assertRaises(ValueError):resolve_live_router(s,cid)
 def test_historical_growth_is_explicitly_untrusted(self):
  result=verify_growth(ROOT/"runs/weights-action-growth-g2")
  self.assertFalse(result["mechanism_replay"]["passed"])
  self.assertIn("historical_run_has_no_independently_anchored_preregistration",result["mechanism_replay"]["errors"])
  self.assertFalse(result["runtime_acceptance"])
