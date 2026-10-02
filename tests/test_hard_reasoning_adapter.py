import copy
import unittest
import hard_reasoning_adapter as a
class HardReasoningTests(unittest.TestCase):
 def oracle_params(self):
  # Mathematical fixture only; not a service founder or a learned proposal.
  return {'rule.W1':[[1,-1,1,-1,1,-1]],'rule.b1':[0.5],'any.W1':[[1]],'any.b1':[0]}
 def test_baseline_headroom(self):
  result=a.grade(a.BASELINE_CELL)
  self.assertEqual((result['solved'],result['total']),(2,8))
  self.assertFalse(result['learning_claim']);self.assertFalse(result['errors'])
  self.assertEqual(a.validate(a.BASELINE_CELL),9)
 def test_multistep_full_development_execution(self):
  self.assertEqual(a.grade(self.oracle_params())['solved'],8)
  c=a.DEVELOPMENT_CASES[0];out=a.predict(self.oracle_params(),c['input'])
  self.assertEqual(out['answer'],{'status':'entailed','positive_round':3,'negative_round':None})
  self.assertGreater(out['execution']['node_execs'],0)
  self.assertTrue(any(t['round']==3 and t['rule_index']==2 and t['fired'] for t in out['trace']))
 def test_missing_negative_is_unknown(self):
  q=a.DEVELOPMENT_CASES[1]['input'];out=a.predict(self.oracle_params(),q)
  self.assertEqual(out['answer']['status'],'unknown')
  baseline=a.predict(a.BASELINE_PARAMS,q);self.assertEqual(baseline['answer']['status'],'entailed')
 def test_conflict_and_exact_timing(self):
  q=a.DEVELOPMENT_CASES[4]['input'];out=a.predict(self.oracle_params(),q)
  self.assertEqual(out['answer'],{'status':'both','positive_round':3,'negative_round':4})
  self.assertEqual(out['answer'],a.reference(q)['answer'])
 def test_no_early_fixedpoint_stop(self):
  q=a.DEVELOPMENT_CASES[6]['input'];out=a.predict(self.oracle_params(),q)
  self.assertEqual(out['answer']['positive_round'],5)
  self.assertEqual(out['rounds'],6)
 def test_weights_control_firings_and_actual_multistep_answers(self):
  q=a.DEVELOPMENT_CASES[0]['input'];p=self.oracle_params();p['rule.b1']=[-9]
  self.assertEqual(a.predict(p,q)['answer']['status'],'unknown')
  p=self.oracle_params();p['any.W1']=[[0]]
  self.assertEqual(a.predict(p,q)['answer']['status'],'unknown')
 def test_positive_historical_gate_preserved(self):
  self.assertEqual(a.BASELINE_PARAMS['rule.W1'][0][:2],[1.,-1.])
  self.assertEqual(a.BASELINE_PARAMS['rule.b1'],[0.5])
  q=a.DEVELOPMENT_CASES[0]['input']
  self.assertEqual({k:a.predict(a.BASELINE_PARAMS,q)[k] for k in ['answer','closure']},a.reference(q))
 def test_input_and_weight_fail_closed(self):
  q=copy.deepcopy(a.DEVELOPMENT_CASES[0]['input']);q['facts'][0]['polarity']=True
  with self.assertRaises(ValueError):a.predict(a.BASELINE_PARAMS,q)
  q=copy.deepcopy(a.DEVELOPMENT_CASES[0]['input']);q['horizon']=9
  with self.assertRaises(ValueError):a.reference(q)
  p=copy.deepcopy(a.BASELINE_PARAMS);p['rule.W1'][0][0]=float('nan')
  with self.assertRaises(ValueError):a.validate(p)
 def test_zero_rules_and_initial_contradiction(self):
  q={'facts':[a.lit('x'),a.lit('x',-1)],'rules':[],'query':'x','horizon':3}
  self.assertEqual(a.predict(self.oracle_params(),q)['answer'],{'status':'both','positive_round':0,'negative_round':0})
 def test_manifest_and_feedback_no_frozen_evaluation(self):
  self.assertFalse(a.MANIFEST['frozen_evaluated'])
  result=a.grade(a.BASELINE_CELL)
  self.assertEqual(set(result['outcomes']),set(a.MANIFEST['development_ids']))
  self.assertFalse(set(result['outcomes'])&set(a.MANIFEST['frozen_ids']))
  for trace in result['traces']:
   self.assertIn('input',trace);self.assertIn('expected',trace);self.assertIn('steps',trace)
 def test_crossover_exact_distinct_parentage(self):
  left=copy.deepcopy(a.BASELINE_CELL);right={'interface':a.INTERFACE,'params':self.oracle_params()}
  child=a.crossover(left,right,[2,4])
  self.assertEqual(len(child['inheritance']),9);self.assertEqual(child['authored_positions'],0)
  for item in child['inheritance']:
   p=left if item['parent']=='left' else right;v=p['params'][item['parameter']]
   for j in item['index']:v=v[j]
   self.assertEqual(v,item['value'])
  with self.assertRaises(ValueError):a.crossover(left,left,[0])
if __name__=='__main__':unittest.main()
