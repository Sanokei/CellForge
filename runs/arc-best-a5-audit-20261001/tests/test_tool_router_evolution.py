import unittest,copy
try:
    from cellforge.construction.tool_router_evolution import FEATURES,ACTIONS,features,route,inherited_parent,validate,safe_path,action_arguments,route_action
except ModuleNotFoundError:
    from tool_router_evolution import FEATURES,ACTIONS,features,route,inherited_parent,validate,safe_path,action_arguments,route_action
import numpy as np
class ToolRouterTests(unittest.TestCase):
    def setUp(self):self.t={'W':np.zeros((9,28)).tolist(),'b':[0,0,0,0,0,0,.5,0,0]}
    def test_original_collision_is_unrepairable_by_weights_alone(self):
        self.assertEqual(features('print the file notes.txt')[:18],features('what does b.py say')[:18])
        self.assertEqual(features('fix the build')[:18],features('what does b.py say')[:18])
    def test_features_have_no_target_or_file_reads(self):
        self.assertEqual(len(features('display x.txt')),28);self.assertEqual(features('display x.txt')[22],1)
        with self.assertRaises(ValueError):features({'request':'display x.txt','expected':'read_file'})
    def test_literal_only(self):
        for t in ({'slot':'write_file'},{'W':self.t['W'],'b':[True]*9},{'W':self.t['W'],'b':[float('nan')]*9}):
            with self.assertRaises(ValueError):validate(t)
    def test_weight_changes_really_choose_action(self):
        self.assertEqual(route(self.t,'write x.txt')['action'],'none')
        self.t['W'][7][23]=3;self.t['W'][7][22]=3;self.t['b'][7]=-4
        self.assertEqual(route(self.t,'write x.txt')['action'],'write_file')
        self.assertEqual(route(self.t,'write a poem')['action'],'none')
    def test_path_boundaries(self):
        for p in ('../notes.txt','/notes.txt','C:/notes.txt','C:\\notes.txt','x/../../y.txt','x.txt:stream','secret.key','x//y.txt','~/.ssh/key'):
            self.assertFalse(safe_path(p),p)
        self.assertTrue(safe_path('fixtures/memo.txt'))
    def test_typed_args_exact(self):
        self.assertEqual(action_arguments('write_file',{'path':'memo.txt','content':'hello'}),{'path':'memo.txt','content':'hello'})
        with self.assertRaises(ValueError):action_arguments('edit_in_place',{'path':'x.txt','old':'','new':'a','expected_matches':1})
        with self.assertRaises(ValueError):action_arguments('edit_in_place',{'path':'x.txt','old':'a','new':'b','expected_matches':True})
    def test_raw_and_gated_decisions_separate(self):
        self.t['W'][7][23]=3;self.t['W'][7][22]=3;self.t['b'][7]=-4
        r=route_action(self.t,'write x.txt',{'path':'../x.txt','content':'hello'})
        self.assertEqual(r['raw_action'],'write_file');self.assertEqual(r['action'],'none');self.assertFalse(r['arguments_valid'])
        r=route_action(self.t,'write x.txt',{'path':'x.txt','content':'hello'})
        self.assertEqual(r['action'],'write_file');self.assertTrue(r['arguments_valid'])
if __name__=='__main__':unittest.main()

