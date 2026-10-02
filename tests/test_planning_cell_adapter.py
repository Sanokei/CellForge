import copy
import unittest
from unittest.mock import patch
import planning_cell_adapter as a
class PlanningTests(unittest.TestCase):
 def test_zero_control_real_headroom(self):
  evidence=a.grade(a.BASELINE_CELL)
  self.assertEqual((evidence['solved'],evidence['total'],evidence['completed']),(0,8,0))
  self.assertFalse(evidence['errors']);self.assertFalse(evidence['learning_claim'])
  self.assertEqual(a.validate(a.BASELINE_CELL),28)
 def test_separate_bfs_reference_valid_minimal_plans(self):
  for case in a.DEVELOPMENT_CASES:
   q=case['input'];answer=a.reference(q);self.assertTrue(answer['reachable'])
   state=copy.deepcopy(q['initial'])
   for action in answer['plan']:
    state,error=a.transition(state,action,q);self.assertIsNone(error)
   self.assertEqual(state['delivered'],q['goal'])
  self.assertEqual(a.reference(a.DEVELOPMENT_CASES[0]['input'])['shortest_steps'],3)
  self.assertEqual(a.reference(a.DEVELOPMENT_CASES[1]['input'])['shortest_steps'],4)
  self.assertEqual(a.reference(a.DEVELOPMENT_CASES[2]['input'])['shortest_steps'],1)
 def test_candidate_never_calls_bfs_or_masks_illegal(self):
  with patch.object(a,'reference',side_effect=AssertionError('reference must not enter candidate execution')):
   answer=a.predict(a.BASELINE_PARAMS,a.DEVELOPMENT_CASES[0]['input'])
  self.assertEqual(answer['plan'],['acquire','acquire'])
  self.assertEqual(answer['status'],'invalid_action')
  self.assertEqual(answer['trace'][-1]['error'],'cargo_capacity_exceeded')
  self.assertGreater(answer['execution']['node_execs'],0)
 def test_actual_weights_choose_actions(self):
  p=copy.deepcopy(a.BASELINE_PARAMS);p['policy.b1'][2]=1
  out=a.predict(p,a.DEVELOPMENT_CASES[2]['input'])
  self.assertEqual(out['plan'],['deliver']);self.assertEqual(out['status'],'complete')
  out=a.predict(p,a.DEVELOPMENT_CASES[0]['input'])
  self.assertEqual(out['status'],'invalid_action');self.assertEqual(out['trace'][0]['error'],'missing_required_inventory')
 def test_physical_resource_rules(self):
  q=a.DEVELOPMENT_CASES[0]['input'];s=dict(q['initial'])
  self.assertEqual(a.transition(s,'recharge',q)[1],'battery_already_full')
  s['energy']=0
  self.assertEqual(a.transition(s,'acquire',q)[1],'insufficient_energy')
  self.assertEqual(a.transition(s,'recharge',q)[0]['energy'],q['max_energy'])
 def test_step_deadline_and_projected_negative_features(self):
  q=copy.deepcopy(a.DEVELOPMENT_CASES[0]['input']);q['capacity']=3;q['max_steps']=1
  self.assertEqual(a.predict(a.BASELINE_PARAMS,q)['status'],'deadline')
  f=a.features(q['initial'],q,0)
  self.assertEqual(f.shape,(4,6));self.assertLess(f[2,1],0)
 def test_bfs_unreachable_has_no_fabricated_plan(self):
  q=copy.deepcopy(a.DEVELOPMENT_CASES[0]['input']);q['max_steps']=2
  answer=a.reference(q);self.assertFalse(answer['reachable']);self.assertIsNone(answer['plan'])
 def test_fail_closed_bounds_and_literal_types(self):
  p=copy.deepcopy(a.BASELINE_PARAMS);p['policy.W1'][0][0]=True
  with self.assertRaises(ValueError):a.validate(p)
  q=copy.deepcopy(a.DEVELOPMENT_CASES[0]['input']);q['initial']['goods']=2
  with self.assertRaises(ValueError):a.predict(a.BASELINE_PARAMS,q)
 def test_manifest_and_feedback_frozen_excluded(self):
  evidence=a.grade(a.BASELINE_CELL)
  self.assertFalse(a.MANIFEST['frozen_evaluated'])
  self.assertEqual(set(evidence['outcomes']),set(a.MANIFEST['development_ids']))
  self.assertFalse(set(evidence['outcomes'])&set(a.MANIFEST['frozen_ids']))
  for t in evidence['traces']:
   self.assertIn('input',t);self.assertIn('expected',t);self.assertIn('trace',t['predicted'])
 def test_exact_crossover(self):
  left=copy.deepcopy(a.BASELINE_CELL);right=copy.deepcopy(left);right['params']['policy.b1']=[1,2,0,0]
  child=a.crossover(left,right,[24])
  self.assertEqual(child['params']['policy.b1'],[0.,2.,0.,0.])
  self.assertEqual(len(child['inheritance']),28);self.assertEqual(child['authored_positions'],0)
  with self.assertRaises(ValueError):a.crossover(left,left,[0])
if __name__=='__main__':unittest.main()
