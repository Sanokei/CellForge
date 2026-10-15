import hashlib
import copy
import json
import pathlib
import tempfile
import unittest
import numpy as np
import arc3_tensor_effect_cells as T
import arc3cell as C

class TensorCellContract(unittest.TestCase):
    def setUp(self):
        self.proposal={'id':'A','params':{'effect':[0,3,-3,0,0,0,0,0],'recovery':[-2,0,6,0,0,0,0,0],'router':[1,2,.5,1],'rates':[2,2],'click':[3,5,7,11]}}
        self.params=T.validate_proposal(self.proposal)
    def test_numeric_positive_and_code_rejection(self):
        self.assertEqual(sum(np.asarray(v).size for v in self.params.values()),26)
        for extra in ('source_lines','program','task_id','sample_key','lookup','targets'):
            p=copy.deepcopy(self.proposal);p[extra]='return target'
            with self.assertRaises(ValueError):T.validate_proposal(p)
        p=copy.deepcopy(self.proposal);p['params']['effect'][0]='exec("solver")'
        with self.assertRaises(ValueError):T.validate_proposal(p)
    def test_nonfinite_shape_and_lookup_rejected(self):
        for value in (float('nan'),float('inf'),21):
            p=copy.deepcopy(self.proposal);p['params']['effect'][0]=value
            with self.assertRaises(ValueError):T.validate_proposal(p)
        p=copy.deepcopy(self.proposal);p['params']['effect']=[[1]*8]
        with self.assertRaises(ValueError):T.validate_proposal(p)
        p=copy.deepcopy(self.proposal);p['params']['ar25']=[1]
        with self.assertRaises(ValueError):T.validate_proposal(p)
    def test_runtime_has_no_task_identity_or_executable_payload(self):
        ok, why=C.validate(T.SOURCE);self.assertTrue(ok,why)
        for token in ('eval(', 'exec(', 'game_id', 'task_id', 'target', 'sample_key'):
            self.assertNotIn(token,T.SOURCE)
    def test_noop_observation_changes_memory_and_recovery_choice(self):
        params={k:np.asarray(v) for k,v in self.params.items()};mem={};frame=np.zeros((64,64),int)
        first=T.act(params,mem,frame,[1,2],{'step':0})[0]
        next_action=T.act(params,mem,frame,[1,2],{'step':1})[0]
        self.assertEqual(first,1);self.assertEqual(next_action,2)
        self.assertGreater(mem['stats'][1,1],0)
        self.assertEqual(mem['stats'][1,0],0)
    def test_meaningful_transition_and_numeric_ablations(self):
        params={k:np.asarray(v) for k,v in self.params.items()};mem={};frame=np.zeros((64,64),int)
        T.act(params,mem,frame,[1,2],{'step':0})
        changed=frame.copy();changed[:32]=1
        T.act(params,mem,changed,[1,2],{'step':1})
        self.assertGreater(mem['stats'][1,0],0);self.assertEqual(mem['stats'][1,1],0)
        feat=T.features(mem['stats'],1,1,mem['global'])
        full=T.predict(params,feat)
        zero=T.predict({k:np.zeros_like(v) for k,v in params.items()},feat)
        swap=T.predict(T.arm_params(params,'swap'),feat)
        self.assertNotEqual(full,zero);self.assertNotEqual(full,swap)
    def test_sandbox_executes_same_coefficients_and_recovery(self):
        policy=C.Policy(T.SOURCE,self.params)
        try:
            frame=np.zeros((64,64),int)
            a=policy.step(frame,[1,2],{'step':0})
            b=policy.step(frame,[1,2],{'step':1})
            self.assertIsNone(a[3]);self.assertIsNone(b[3])
            self.assertEqual((a[0],b[0]),(1,2))
        finally:policy.close()
    def test_generic_action_id_growth_sandbox(self):
        policy=C.Policy(T.SOURCE,self.params)
        try:
            frame=np.zeros((64,64),int)
            first=policy.step(frame,[1,2],{'step':0})
            grown=policy.step(frame,[1,7],{'step':1})
            self.assertIsNone(first[3]);self.assertIsNone(grown[3])
            self.assertIn(grown[0],[1,7])
        finally:policy.close()

    def test_reporting_cannot_select_and_seal_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            candidates={'A':self.params,'B':T.arm_params(self.params,'swap')}
            seal=T.seal_selection(root,candidates,{'A':0,'B':1},['dev:0'],['report:1'])
            self.assertEqual(seal['chosen'],'B')
            with self.assertRaises(FileExistsError):T.seal_selection(root,candidates,{'A':10,'B':0},['dev:0'],['report:1'])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):T.seal_selection(root,{'A':self.params},{'A':0},['same'],['same'])

    def test_provenance_response_and_numeric_revision_gate(self):
        raw=[self.proposal,{'id':'B','params':T.arm_params(self.params,'swap')}]
        candidates={p['id']:T.validate_proposal(p) for p in raw}
        prompt='generic numerical proposal'
        start={'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'model':'gpt-6.1-sol','reasoning':'low','auth':'ChatGPT subscription'}
        self.assertTrue(T.validate_release(raw,candidates,prompt,start,T.RUNTIME_SHA256))
        wrong=copy.deepcopy(candidates);wrong['A']['effect'][0]+=1
        with self.assertRaises(ValueError):T.validate_release(raw,wrong,prompt,start,T.RUNTIME_SHA256)
        with self.assertRaises(ValueError):T.validate_release(raw,candidates,'different',start,T.RUNTIME_SHA256)
        with self.assertRaises(ValueError):T.validate_release(raw,candidates,prompt,start,'other-runtime')
        parent_sha=hashlib.sha256(json.dumps(self.params,sort_keys=True).encode()).hexdigest()
        with self.assertRaises(ValueError):T.validate_revision(self.params,self.params,parent_sha,start['prompt_sha256'])
        child=copy.deepcopy(self.params);child['effect'][0]+=1
        self.assertEqual(T.validate_revision(self.params,child,parent_sha,start['prompt_sha256']),1)

if __name__=='__main__' :unittest.main(verbosity=2)
