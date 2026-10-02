import unittest,copy,json,tempfile
from pathlib import Path
from cellforge.construction import selected_cell_registry as r
from cellforge.construction import arithmetic_subtraction_adapter as a
class RegistryReplay(unittest.TestCase):
 def test_literal_weights_change_executed_digit(self):
  c={'family':'decimal-subtraction','params':copy.deepcopy(a.BASELINE_PARAMS)}
  request={'input':{'a':7,'b':2,'borrow':0}}
  self.assertEqual(r.predict(c,request),{'digit':9,'borrow':0})
  c['params']['sum.W'][0][1]=-1
  self.assertEqual(r.predict(c,request),{'digit':5,'borrow':0})
 def test_spatial_numeric_execution(self):
  c={'family':'spatial-cardinal','params':{'W1':[[0]*90],'b1':[-1]}}
  self.assertEqual(r.predict(c,{'input':[0]*90}),0)
  c['params']['b1']=[1]
  self.assertEqual(r.predict(c,{'input':[0]*90}),1)
 def test_false_native_scores_fail_replay_before_publication(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'arithmetic';root.mkdir();cid='native';cell={'params':a.BASELINE_PARAMS,'evidence':{'traces':[{'id':'x','input':{'a':7,'b':2,'borrow':0},'expected':{'digit':5,'borrow':0}}],'outcomes':{'x':True}}}
   (root/'STATE-0.json').write_text(json.dumps({'cells':{cid:cell},'contexts':[{'arms':{k:{'live':cid} for k in ('llm','random')}}]}));(root/'DECLARATION.json').write_text('{}')
   out=Path(t)/'out'
   with self.assertRaisesRegex(ValueError,'fails independent numeric replay'):r.refresh(root,Path(t)/'unused',out)
   self.assertFalse((out/'LATEST.json').exists())
if __name__=='__main__':unittest.main()
