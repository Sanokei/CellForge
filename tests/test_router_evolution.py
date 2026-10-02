import copy,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cellforge.construction.joint_weight_population import digest,predict_routed
try:
    from cellforge.construction.router_evolution import literal_router,assemble_snapshot,random_matched,MoECompetition,score_joint
except ModuleNotFoundError:
    from router_evolution import literal_router,assemble_snapshot,random_matched,MoECompetition,score_joint

class RouterBehavior(unittest.TestCase):
    def setUp(self):
        self.core=SimpleNamespace(SHAPES={'copy_or_flip':{'w':(1,)}},PREDICTORS={'copy_or_flip':lambda p,g,t=False:(g if p['w'][0]==0 else [list(reversed(r)) for r in g],{})})
        self.rows=[{'id':'real_fixture','task':{'train':[{'input':[[1,2]],'output':[[2,1]]}],'test':[{'input':[[3,4]],'output':[[4,3]]}]}}]
        self.parts={k:copy.deepcopy(self.rows) for k in ('fresh','archive','anchor')}
        cells=[]
        for value in (0,1):
            params={'w':[value]};cid=digest({'structure':'copy_or_flip','params':params})
            cells.append({'id':cid,'structure':'copy_or_flip','params':params,'state':'live','parents':[],'provenance':{'actual':'fixture'},'scores':{}})
        # Identity cell also earns a real separate development solve.
        self.parts['anchor'].append({'id':'identity','task':{'train':[{'input':[[1,2]],'output':[[1,2]]}],'test':[{'input':[[3,4]],'output':[[3,4]]}]}})
        self.snapshot={'cells':{c['id']:c for c in cells},'live':{'copy_or_flip':[c['id'] for c in cells]}}
        self.assembly={'cell_ids':[c['id'] for c in cells],'router_cell_order':[c['id'] for c in cells],'cells':cells,'router':{'route.W':[[1,0],[0,1]],'route.b':[0,0]}}
        self.joint=assemble_snapshot(self.snapshot,self.assembly,self.core,self.parts)
    def test_scale_is_numerical_but_neutral(self):
        scaled={'route.W':[[2,0],[0,2]],'route.b':[0,0]}
        v={'demos':self.rows[0]['task']['train'],'input':[[3,4]]}
        a,ta=predict_routed(self.joint,self.assembly['router'],v,self.core,True)
        b,tb=predict_routed(self.joint,scaled,v,self.core,True)
        self.assertEqual(a,b);self.assertEqual(ta['chosen_expert'],tb['chosen_expert'])
        self.assertNotEqual(ta['scores'],tb['scores'])
    def test_query_labels_forbidden(self):
        with self.assertRaises(ValueError):predict_routed(self.joint,self.assembly['router'],{'demos':self.rows[0]['task']['train'],'input':[[3,4]],'output':[[4,3]]},self.core)
    def test_slots_and_boolean_router_rejected(self):
        for router in ({'slot':'flip'}, {'route.W':[[True,0],[0,1]],'route.b':[0,0]}, {'route.W':[[float('nan'),0],[0,1]],'route.b':[0,0]}):
            with self.assertRaises(ValueError):literal_router(router,2)
    def test_retired_and_forged_cells_rejected(self):
        for variant in ('retired','hash'):
            s=copy.deepcopy(self.snapshot)
            if variant=='retired':s['cells'][self.assembly['cell_ids'][0]]['state']='retired'
            else:s['cells'][self.assembly['cell_ids'][0]]['params']['w']=[99]
            with self.assertRaises(ValueError):assemble_snapshot(s,self.assembly,self.core,self.parts)
    def test_claimed_competence_without_execution_rejected(self):
        parts={k:copy.deepcopy(self.rows) for k in self.parts}
        with self.assertRaisesRegex(ValueError,'executed development competence'):assemble_snapshot(self.snapshot,self.assembly,self.core,parts)
    def test_wrong_order_rejected(self):
        a=copy.deepcopy(self.assembly);a['router_cell_order'].reverse()
        with self.assertRaises(ValueError):assemble_snapshot(self.snapshot,a,self.core,self.parts)
    def test_real_uniform_is_distinct_from_zero_tie_first(self):
        s=score_joint(self.joint,self.rows,self.core)
        self.assertEqual(sum(s['outcomes'].values()),1)
        self.assertEqual(s['zero_weight_first_cell'],0)
        self.assertEqual(s['uniform_random_cell_expected_solves'],.5)
        self.assertEqual(s['oracle_query_upper_bound'],1)
    def test_matched_random_changes_only_authored_positions(self):
        candidate={'route.W':[[2,0],[0,2]],'route.b':[0,0]}
        r,pos=random_matched(self.assembly['router'],candidate,101)
        self.assertEqual(pos,{'route.W':[[0,0],[1,1]],'route.b':[]})
        self.assertEqual(r['route.b'],[0,0]);self.assertEqual(r['route.W'][0][1],0);self.assertEqual(r['route.W'][1][0],0)
    def test_fake_author_invalid_charged(self):
        c=MoECompetition(self.joint,self.core,self.parts,{'wall_seconds':100,'traced_peak_bytes':9999999})
        r=c.propose(self.assembly['router'],arm='llm',provenance={'prompt':'fake'},resource_measure=lambda f:(f(),{'wall_seconds':1,'traced_peak_bytes':10}))
        self.assertEqual(r['state'],'invalid');self.assertEqual(len(c.charges),1)
    def test_no_gain_never_promoted(self):
        c=MoECompetition(self.joint,self.core,self.parts,{'wall_seconds':100,'traced_peak_bytes':9999999})
        r=c.propose(self.assembly['router'],arm='random',provenance={'actual':'fixture'},resource_measure=lambda f:(f(),{'wall_seconds':1,'traced_peak_bytes':10}))
        self.assertEqual(r['state'],'rejected');self.assertEqual(len(c.history.records),1)
        self.assertIn({'gate':'strict_fresh_gain'},r['admission']['reasons'])

if __name__=='__main__':unittest.main()

