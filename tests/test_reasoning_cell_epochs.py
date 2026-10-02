import unittest,json,pathlib,tempfile,copy
from cellforge.construction.reasoning_cell_epochs import grade,validate,identity,init,prompts
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestReasoningCellEpochs(unittest.TestCase):
 def cell(self):return {'niche':'all_present','activation':'step','params':{'W1':[[1,-1]],'b1':[.5]},'typed_interface':'count-required->boolean/1'}
 def test_actual_numeric_weights_change_deduction_behavior(self):
  c=self.cell();rows=[{'id':'all','partition':'development','input':[2,2],'expected':1},{'id':'missing','partition':'development','input':[1,2],'expected':0}]
  self.assertEqual(grade(c,rows)['solved'],2);c['params']['b1']=[2];self.assertEqual(grade(c,rows)['solved'],1)
 def test_structural_variant_executes_and_is_distinct_genotype(self):
  a=self.cell();b=copy.deepcopy(a);b['params']={'W1':[[1,-1],[-1,1]],'b1':[.5,-.5],'W2':[[1,0]],'b2':[0]}
  self.assertNotEqual(identity(a),identity(b));self.assertEqual(validate(b),9)
  self.assertEqual(grade(b,[{'id':'all','partition':'development','input':[4,4],'expected':1}])['solved'],1)
 def test_foreign_io_and_nonboolean_output_rejected(self):
  c=self.cell();c['params']['W1']=[[1]]
  with self.assertRaises(ValueError):validate(c)
  c=self.cell();c['params'].update(W2=[[2]],b2=[0]);self.assertTrue(grade(c,[{'id':'x','partition':'development','input':[1,1],'expected':1}])['errors'])
 def test_frozen_reporting_cannot_drive_selection(self):
  with self.assertRaisesRegex(ValueError,'reporting'):grade(self.cell(),[{'id':'x','partition':'frozen_reporting','input':[1,1],'expected':1}])
