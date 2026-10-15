import copy,json,unittest
from pathlib import Path
from cellforge.construction.joint_weight_population import *
R=Path(r'C:\Users\wkeif\Documents\GitHub\CellForge')

class Tests(unittest.TestCase):
    def setUp(self):
        self.core=load_core(R/'runs/weights-pop1/pop1_core.py')
        self.parent=json.loads((R/'runs/weights-pop1-router-v1/raw-llm-responses.json').read_text())['founder']['tensors']
    def test_router_rejects_target_or_niche_features(self):
        with self.assertRaises(ValueError):router_features(self.parent,{'demos':[],'input':[[0]],'target':[[1]]},self.core)
    def test_exact_count_to_conv_inheritance_and_immutable_parents(self):
        old=copy.deepcopy(self.parent)
        child,receipt=inherit_neighbors(self.parent,self.parent,recipient_prefix='2:',donor_prefix='3:')
        self.assertEqual(self.parent,old)
        self.assertEqual(child['2:conv.W'][4],self.parent['2:conv.W'][4])
        self.assertEqual(child['2:paint'],self.parent['2:paint'])
        self.assertTrue(receipt['changed_scalars'])
        for i in (0,1,2,3,5,6,7,8):self.assertEqual(child['2:conv.W'][i],self.parent['3:cnt.W'][0])
    def test_nonisotropic_reverse_rejected(self):
        with self.assertRaises(ValueError):inherit_neighbors(self.parent,self.parent,recipient_prefix='3:',donor_prefix='2:')
    def test_reverse_exact_when_representable(self):
        conv,_=inherit_neighbors(self.parent,self.parent,recipient_prefix='2:',donor_prefix='3:')
        child,_=inherit_neighbors(self.parent,conv,recipient_prefix='3:',donor_prefix='2:')
        self.assertEqual(child['3:cnt.W'],self.parent['3:cnt.W'])
    def test_query_target_does_not_enter_router(self):
        tasks=self.core.sample_tasks('cmap_swap',0,734001,3)
        view={'demos':[{'input':t['input'],'output':t['expected']} for t in tasks[:2]],'input':tasks[2]['input']}
        identity={'route.W':np.eye(4).tolist(),'route.b':[0]*4}
        a,trace=predict_routed(self.parent,identity,view,self.core,True)
        self.assertEqual(a,tasks[2]['expected']);self.assertEqual(trace['chosen_expert'],1)

if __name__=='__main__':unittest.main()

