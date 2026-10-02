import pathlib,sys,copy,json,tempfile,unittest,runpy,types,hashlib
from unittest.mock import patch
import numpy as np
REPO=pathlib.Path(r'C:\Users\wkeif\Documents\GitHub\CellForge');sys.path.insert(0,str(REPO/'scripts'))
import arc3_tensor_effect_cells as T
P={'id':'A','params':{'effect':[0,3,-3,0,0,0,0,0],'recovery':[-2,0,6,0,0,0,0,0],'router':[1,2,.5,1],'rates':[2,2],'click':[3,5,7,11]}}
class AntiSubstitution(unittest.TestCase):
 def runner(self,payload,stale=False):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=pathlib.Path(tmp.name)/'pilot';calls=[]
  def ask(prompt,author,**kw):
   author.mkdir();(author/'response.json').write_text(json.dumps(payload));(author/'PROMPT.txt').write_text('STALE PROMPT' if stale else prompt)
   digest=hashlib.sha256(prompt.encode()).hexdigest()
   (author/'START.json').write_text(json.dumps({'prompt_sha256':digest,'model':'gpt-6.1-sol','reasoning':'low','auth':'ChatGPT subscription'}))
   (author/'RECEIPT.json').write_text(json.dumps({'prompt_sha256':'stale','parent_sha256':'wrong'} if stale else {'prompt_sha256':digest,'parent_sha256':None}));return copy.deepcopy(payload)
  def play(source,weights,game,budget,**kw):
   calls.append((source,copy.deepcopy(weights),game,kw.copy()));return {'levels':0,'error':None}
  m=types.ModuleType('maker');m.ask=ask;m.now=lambda:'test-fixture';e=types.ModuleType('arc3_env');e.play=play
  error=None
  with patch.dict(sys.modules,{'maker':m,'arc3_env':e}),patch.object(sys,'argv',['runner',str(root)]):
   try:runpy.run_path(str(REPO/'scripts/run_arc3_tensor_effect_cells.py'),run_name='__main__')
   except Exception as exc:error=exc
  return root,calls,error
 def pair(self):
  b=copy.deepcopy(P);b['id']='B';b['params']['effect'][0]=1;return [copy.deepcopy(P),b]
 def test_runner_rejects_program_payload_before_play(self):
  for key in ('program','code','source_lines','task_id','sample_key','lookup','targets','solver'):
   pair=self.pair();pair[0][key]='return hidden_target';root,calls,error=self.runner(pair)
   self.assertIsNotNone(error,key);self.assertEqual(calls,[],key)
 def test_runner_rejects_malformed_nonfinite_before_play(self):
  for value in (float('nan'),float('inf'),True,'1','exec(1)',[1],None):
   pair=self.pair();pair[0]['params']['effect'][0]=value;root,calls,error=self.runner(pair)
   self.assertIsNotNone(error,repr(value));self.assertEqual(calls,[])
 def test_runner_rejects_unchanged_candidate(self):
  pair=self.pair();pair[1]['params']=copy.deepcopy(pair[0]['params']);root,calls,error=self.runner(pair)
  self.assertIsNotNone(error);self.assertEqual(calls,[])
 def test_runner_must_reject_stale_prompt_parent_provenance(self):
  root,calls,error=self.runner(self.pair(),stale=True)
  self.assertIsNotNone(error,'runner accepted stale prompt/parent receipt');self.assertEqual(calls,[])
 def test_actual_runner_keeps_fixed_numeric_tensors_across_games_and_seals_report(self):
  root,calls,error=self.runner(self.pair());self.assertIsNone(error);self.assertEqual(len(calls),33)
  self.assertTrue(all(source==T.SOURCE for source,_,_,_ in calls))
  self.assertEqual([g for _,_,g,_ in calls[:3]],['ar25','lp85','sc25'])
  self.assertTrue(all(w==calls[0][1] for _,w,_,_ in calls[:3]))
  self.assertTrue(all(w==calls[3][1] for _,w,_,_ in calls[3:6]))
  self.assertEqual(json.loads((root/'SELECTION_SEALED.json').read_text())['chosen'],'B')
  self.assertTrue(all(w==calls[3][1] for _,w,_,_ in calls[6:9]))
  self.assertFalse(json.loads((root/'FINISH.json').read_text())['promoted'])
 def test_numeric_positive_action_fixture_and_zero_swap_ablation(self):
  params={k:np.asarray(v,float) for k,v in T.validate_proposal(P).items()};frame=np.zeros((64,64),int)
  def actions(p,info):
   mem={};return [T.act(p,mem,frame,[1,2],dict(info,step=i))[0] for i in range(2)]
  full=actions(params,{})
  self.assertEqual(full,[1,2]);self.assertNotEqual(actions({k:np.zeros_like(v) for k,v in params.items()},{}),full)
  self.assertNotEqual(actions(T.arm_params(params,'swap'),{}),full)
  self.assertEqual(actions(params,{'game_id':'ar25','task_id':'hidden','target':[2],'solver':lambda:2}),full)
 def test_actual_report_gate_blocks_unverified_pilot_and_forged_acceptance(self):
  import report_verification as R
  root,calls,error=self.runner(self.pair());self.assertIsNone(error)
  gate=R.read_gate(root);self.assertFalse(gate['scientific_headline_allowed']);self.assertFalse(gate['scientific_acceptance'])
  (root/'VERIFICATION.json').write_text(json.dumps({'scientific_standard':{'acceptance':True},'report_policy':{'scientific_headline_allowed':True}}))
  self.assertFalse(R.read_gate(root)['scientific_headline_allowed'])
if __name__=='__main__':unittest.main(verbosity=2)
