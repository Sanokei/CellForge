import unittest,copy,json,pathlib
from cellforge.construction.arc_weight_moe import *
from cellforge.construction.joint_weight_population import digest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestARCMoEService(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.core=make_core(ROOT/"runs/weights-real-learning-v2/pop1_core.py")
  cls.t=json.loads((ROOT/"runs/weights-real-learning-v2/parent.json").read_text())["tensors"]
 def rows(self):
  return [{"id":"fixture","task":{"train":[{"input":[[1]],"output":[[1]]}],
                                "test":[{"input":[[1]],"output":[[1]]}]}}]
 def test_real_numeric_backend_has_query_blind_predict_and_earned_pool(self):
  rows=self.rows();pool=measured_pool(self.t,rows,self.core);evidence=score(self.t,rows,self.core)
  self.assertEqual(evidence["total"],1);self.assertTrue(any(c["state"]=="live" for c in pool["cells"].values()))
  changed=copy.deepcopy(rows[0]["task"]);changed["test"][0]["output"]=[[9]]
  self.assertEqual(predict(self.t,rows[0]["task"],self.core),predict(self.t,changed,self.core))
 def test_unearned_cell_or_forged_pool_cannot_assemble(self):
  pool=measured_pool(self.t,self.rows(),self.core);ids=list(pool["cells"]);q={"partition":"development"}
  raw={"parent_hash":digest(self.t),"pool_hash":digest(pool),"prompt_hash":digest(q),"model":"gpt-6.1-sol","reasoning":"low",
       "rationale":"test binding only","cell_ids":ids[:2],"router":{"route.W":[[1,0],[0,1]],"route.b":[0,0]},"charged_positions":6}
  pool["cells"][ids[0]]["state"]="failed-founder";raw["pool_hash"]=digest(pool)
  with self.assertRaisesRegex(ValueError,"unearned"):assemble(raw,q,pool,digest(self.t),self.core)
 def test_assembly_inherits_actual_numeric_parent_cells_and_router(self):
  pool=measured_pool(self.t,self.rows(),self.core);ids=[cid for cid,c in pool["cells"].items() if c["state"]=="live"][:2]
  self.assertEqual(len(ids),2);q={"partition":"development"}
  raw={"parent_hash":digest(self.t),"pool_hash":digest(pool),"prompt_hash":digest(q),"model":"gpt-6.1-sol","reasoning":"low",
   "rationale":"fixture earned assembly","cell_ids":ids,"router":{"route.W":[[1,0],[0,1]],"route.b":[0,0]},"charged_positions":6}
  joint=assemble(raw,q,pool,digest(self.t),self.core)
  for i,cid in enumerate(ids):
   for name,v in pool["cells"][cid]["params"].items():self.assertEqual(joint[str(i)+":"+name],v)
  self.assertEqual(joint["route.W"],raw["router"]["route.W"])
