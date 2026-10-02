import unittest,json,pathlib,copy
from cellforge.construction.arc_weight_moe import primitive_pool,make_core
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestPrimitiveAdmission(unittest.TestCase):
 def fixture(self):
  tensors=json.loads((ROOT/'runs/weights-real-learning-v2/parent.json').read_text())['tensors']
  core=make_core(ROOT/'runs/weights-real-learning-v2/pop1_core.py')
  rows=[{'id':str(i),'partition':'development','input':[[6,1],[2,6]],'expected':[[0,1],[2,0]]} for i in range(8)]
  return tensors,core,{'schema':'grid-primitive-admission/1','development':{'colormap':rows},'frozen_reporting':{}}
 def test_measured_primitive_can_earn_assembly_without_whole_ARC_success(self):
  t,c,m=self.fixture();p=primitive_pool(t,m,c);live=[v for v in p['cells'].values() if v['state']=='live']
  self.assertEqual([v['structure'] for v in live],['colormap']);self.assertEqual(sum(live[0]['scores']['zero_control'].values()),0)
 def test_frozen_answers_cannot_change_eligibility(self):
  t,c,m=self.fixture();a=primitive_pool(t,m,c);m['frozen_reporting']={'colormap':[{'expected':[[9]]}]}
  b=primitive_pool(t,m,c);self.assertEqual(a['live'],b['live'])
 def test_reporting_rows_cannot_enter_selection(self):
  t,c,m=self.fixture();m['development']['colormap'][0]['partition']='frozen_reporting'
  with self.assertRaisesRegex(ValueError,'reporting'):primitive_pool(t,m,c)
 def test_zero_weight_or_trivial_identity_cannot_earn_admission(self):
  t,c,m=self.fixture()
  for row in m['development']['colormap']:row['expected']=row['input']
  self.assertFalse(primitive_pool(t,m,c)['live'])
