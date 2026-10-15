import unittest,pathlib,json,copy,itertools,numpy as np
from cellforge.construction.adhesion import AdhesionCandidate,validate_adhesion
ROOT=pathlib.Path(__file__).resolve().parents[1]
CAT=json.loads((ROOT/"runs/baseline-logic-math-cells/cells.json").read_text())["cells"]
SPEC=next(p["spec"] for p in json.loads((ROOT/"tests/fixtures/logic_adhesion_author.json").read_text())["proposals"] if p["name"]=="lt-selector")
def forward(cell,bits):
 v=list(bits)
 for l in cell["layers"]:v=[int(sum(w*x for w,x in zip(row,v))+b>0) for row,b in zip(l["W"],l["b"])]
 return v
def score(spec,cat):
 c=AdhesionCandidate(spec,cat,forward);hits=0
 for x in itertools.product((0,1),repeat=4):
  expected=max(x[0]+2*x[1],x[2]+2*x[3])
  prediction=c.execute({"bits":list(x)})
  hits+=prediction["packet"]["bits"]==[expected&1,expected>>1]
 return hits
class TestAdhesionBehavior(unittest.TestCase):
 def test_labels_callbacks_and_host_max_cannot_execute(self):
  c=AdhesionCandidate(SPEC,CAT,forward)
  with self.assertRaises(ValueError):c.execute({"bits":[0,0,1,1],"expected":[1,1]})
  for kind in ("host-max","callback","lookup"):
   bad=copy.deepcopy(SPEC);bad["nodes"][1]["kind"]=kind
   with self.assertRaises(ValueError):validate_adhesion(bad,CAT)
 def test_cycles_and_missing_inputs_cannot_execute(self):
  bad=copy.deepcopy(SPEC);bad["nodes"][0]["inputs"]["a0"]={"node":"low","port":"y"}
  with self.assertRaises(ValueError):validate_adhesion(bad,CAT)
  bad=copy.deepcopy(SPEC);del bad["nodes"][2]["inputs"]["s"]
  with self.assertRaises(ValueError):validate_adhesion(bad,CAT)
 def test_actual_numeric_contribution_not_cached_scores(self):
  self.assertEqual(score(SPEC,CAT),16)
  for name,expected in (("cmp2",10),("mux",1)):
   cat=copy.deepcopy(CAT)
   for layer in cat[name]["layers"]:
    for key in ("W","b"):layer[key]=np.zeros_like(layer[key]).tolist()
   self.assertEqual(score(SPEC,cat),expected)
  spec=copy.deepcopy(SPEC);spec["nodes"][1]["params"]={"W":[[0,0,0]],"b":[0]}
  self.assertEqual(score(spec,CAT),10)
 def test_sharing_and_numeric_provenance(self):
  r=validate_adhesion(SPEC,CAT)
  self.assertEqual((r["inherited_unique_positions"],r["expanded_inherited_positions"],r["new_numeric_positions"]),(46,57,4))
  self.assertIsNone(r["population_child_id"])
 def test_unused_donor_cannot_inflate_inheritance(self):
  bad=copy.deepcopy(SPEC);bad["nodes"].append({"id":"fake","kind":"cell","cell":"not","inputs":{"a":{"external":"a0"}}});bad["parents"].append("not")
  with self.assertRaises(ValueError):validate_adhesion(bad,CAT)
