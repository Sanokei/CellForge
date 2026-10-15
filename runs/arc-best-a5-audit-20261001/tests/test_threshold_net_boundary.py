import copy,importlib.util,itertools,json,unittest
from pathlib import Path
from cellforge.construction.typed_population import ThresholdNetArtifact,validate_threshold_net,project_bit_features
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/baseline-logic-math-cells'
spec=importlib.util.spec_from_file_location('threshold_baseline_fixture',BASE/'test_cells.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class ThresholdBoundaryTests(unittest.TestCase):
 def test_actual18founders_exhaustive_typed_parity(self):
  self.assertEqual(len(base.CELLS),18)
  for name,cell in base.CELLS.items():
   artifact=ThresholdNetArtifact(cell,base.forward)
   for x in itertools.product((0,1),repeat=len(cell['inputs'])):
    packet=artifact.predict({'bits':list(x)})
    self.assertEqual(packet['bits'],base.REFS[name](*x))
    self.assertEqual(packet['output_ports'],cell['outputs'])
 def test_actual_authored_numeric_carry_glue_chains16cases(self):
  ha=ThresholdNetArtifact(base.CELLS['half_adder'],base.forward)
  fa=ThresholdNetArtifact(base.CELLS['full_adder'],base.forward)
  flat=ThresholdNetArtifact(base.CELLS['add2'],base.forward)
  glue={'W':[[0,1]],'b':[0]}
  for a0,a1,b0,b1 in itertools.product((0,1),repeat=4):
   low=ha.predict({'bits':[a0,b0]})
   carry=project_bit_features(glue,low,output_ports=['cin'])
   high=fa.predict({'bits':[a1,b1,carry['bits'][0]]})
   self.assertEqual([low['bits'][0]]+high['bits'],flat.predict({'bits':[a0,a1,b0,b1]})['bits'])
 def test_bad_layers_labels_and_wrong_input_width_fail(self):
  cell=base.CELLS['and'];artifact=ThresholdNetArtifact(cell,base.forward)
  for req in ({'bits':[1,1],'target':[1]},{'bits':[1]},{'bits':['1',1]},{'bits':[1.,1]},{'bits':[2,0]}):
   with self.assertRaises(ValueError):artifact.predict(req)
  for kind in ('ragged','nonfinite','extra','output','duplicate'):
   b=copy.deepcopy(cell)
   if kind=='ragged':b['layers'][0]['W']=[[1]]
   if kind=='nonfinite':b['layers'][0]['b']=[float('nan')]
   if kind=='extra':b['layers'][0]['fit']=True
   if kind=='output':b['outputs']=['one','two']
   if kind=='duplicate':b['inputs']=['a','a']
   with self.assertRaises(ValueError):validate_threshold_net(b)
  with self.assertRaises(ValueError):ThresholdNetArtifact(cell,lambda c,x:[.5]).predict({'bits':[1,1]})
 def test_invalid_glue_and_no_implicit_repair(self):
  packet={'bits':[0,1],'output_ports':['sum','carry']}
  for params in ({'W':[[1]],'b':[0]},{'W':[[0,2]],'b':[0]},{'W':[[0,1]],'b':[float('nan')]}):
   with self.assertRaises(ValueError):project_bit_features(params,packet,output_ports=['cin'])
  with self.assertRaises(ValueError):validate_threshold_net(json.loads((BASE/'tool_cells.json').read_text()))

if __name__=='__main__':unittest.main(verbosity=2)
