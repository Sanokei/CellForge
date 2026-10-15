import copy,json,unittest
from pathlib import Path
from cellforge.construction.tool_action_router_adapter import infer_compatible
ROOT=Path(__file__).resolve().parents[1]
class AdapterAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'tests/fixtures/tool_action_router_v2.json'
        cls.artifact=json.loads(p.read_text())
    def test_exact_label_mapping_and_dev_arguments(self):
        r=infer_compatible("replace 'milk' with 'tea' in 'notes.txt'",self.artifact)
        self.assertEqual(r['router_trace']['source_action'],'edit_in_place');self.assertEqual(r['action'],'edit_replace')
        self.assertEqual(r['packet']['name'],'edit_in_place')
        self.assertEqual(r['packet']['arguments'],{'path':'notes.txt','old':'milk','new':'tea','expected_matches':1})
    def test_router_ablation_changes_choice_without_host_recovery(self):
        a=copy.deepcopy(self.artifact);a['router']['W']=[[0]*28 for _ in range(9)];a['router']['b']=[0]*9
        r=infer_compatible("replace 'milk' with 'tea' in 'notes.txt'",a)
        self.assertEqual(r['action'],'grep_todo');self.assertIsNone(r['packet'])
    def test_each_argument_row_ablation_prevents_packet(self):
        for i in range(3):
            a=copy.deepcopy(self.artifact);a['argument_tensors']['slot.W'][i]=[0]*8
            self.assertIsNone(infer_compatible("replace 'milk' with 'tea' in 'notes.txt'",a)['packet'])
    def test_swapping_argument_weights_swaps_actual_copied_text(self):
        a=copy.deepcopy(self.artifact);w=a['argument_tensors']['slot.W'];w[0],w[1]=w[1],w[0]
        r=infer_compatible("replace 'milk' with 'tea' in 'notes.txt'",a)
        self.assertEqual(r['packet']['arguments']['old'],'tea');self.assertEqual(r['packet']['arguments']['new'],'milk')
    def test_nonperiod_path_uses_neural_file_port(self):
        r=infer_compatible("write 'version 1.2' to 'README'",self.artifact)
        self.assertEqual(r['router_trace']['features'][22],1);self.assertEqual(r['packet']['arguments']['path'],'README')
    def test_main_probe_router_choice_has_no_authority(self):
        a=copy.deepcopy(self.artifact);a['argument_probe_router']['W']=[[0]*25 for _ in range(9)];a['argument_probe_router']['b']=[0]*9
        r=infer_compatible("replace 'milk' with 'tea' in 'notes.txt'",a)
        self.assertEqual(r['action'],'edit_replace')
    def test_no_filename_no_host_answer_fill(self):
        r=infer_compatible("write 'v1.2'",self.artifact)
        self.assertIsNone(r['slots']['file']);self.assertIsNone(r['packet'])
    def test_version_and_port_order_required(self):
        a=copy.deepcopy(self.artifact);a['main_action_ports'].reverse()
        with self.assertRaises(ValueError):infer_compatible("write 'hello' to x.txt",a)
if __name__=='__main__':unittest.main()

