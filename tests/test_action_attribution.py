import unittest,tempfile,pathlib,json,copy
from cellforge.construction.action_argument_cell import infer,execute_packet
ROOT=pathlib.Path(__file__).resolve().parents[1]
S=json.loads((ROOT/"runs/baseline-logic-math-cells/tool_cells2.json").read_text())
W=[[6,-3,2,-5,0,-8,-8,-8],[-3,6,0,-5,0,-8,-8,-8],[-4,-4,0,-6,1,8,8,8]]
class TestActionAttribution(unittest.TestCase):
 def test_weights_control_action_and_every_argument(self):
  request="replace 'milk' with 'tea' in 'notes.txt'"
  result=infer(request,S["router"],W,[.5,.5,1.5])
  self.assertEqual(result["packet"]["arguments"],{"path":"notes.txt","old":"milk","new":"tea","expected_matches":1})
  for i in range(3):
   ablated=copy.deepcopy(W);ablated[i]=[0]*8
   self.assertIsNone(infer(request,S["router"],ablated,[.5,.5,1.5])["packet"])
  router=copy.deepcopy(S["router"]);router["W"]=[[0]*25 for _ in range(9)]
  self.assertIsNone(infer(request,router,W,[.5,.5,1.5])["packet"])
  altered=copy.deepcopy(W);altered[0],altered[1]=altered[1],altered[0]
  self.assertEqual(infer(request,S["router"],altered,[.5,.5,1.5])["packet"]["arguments"]["old"],"tea")
 def test_executor_has_no_intent_and_exact_effects(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);p=root/"notes.txt";p.write_text("milk")
   output=infer("replace 'milk' with 'tea' in 'notes.txt'",S["router"],W,[.5,.5,1.5])
   self.assertEqual(execute_packet(output["packet"],allowed_root=root)["status"],"executed")
   self.assertEqual(p.read_text(),"tea")
   self.assertEqual(execute_packet(output["packet"],allowed_root=root)["status"],"denied")
   self.assertEqual(p.read_text(),"tea")
 def test_missing_arguments_and_escape_never_execute(self):
  self.assertIsNone(infer("write 'v1.2'",S["router"],W,[.5,.5,1.5])["packet"])
  with tempfile.TemporaryDirectory() as d:
   packet=infer("write 'unsafe' to ../outside.txt",S["router"],W,[.5,.5,1.5])["packet"]
   self.assertEqual(execute_packet(packet,allowed_root=d)["status"],"denied")
   self.assertEqual(list(pathlib.Path(d).iterdir()),[])
 def test_no_execution_of_generated_code(self):
  with tempfile.TemporaryDirectory() as d:
   packet=infer("write 'raise RuntimeError()' to script.py",S["router"],W,[.5,.5,1.5])["packet"]
   self.assertEqual(execute_packet(packet,allowed_root=d)["status"],"executed")
   self.assertEqual((pathlib.Path(d)/"script.py").read_text(),"raise RuntimeError()")
 def test_invalid_tensor_rejected(self):
  with self.assertRaises(ValueError):infer("write 'x' to a.txt",S["router"],[[float("nan")]*8]*3,[.5]*3)


class TestJointActionAttribution(unittest.TestCase):
 def test_joint_cells_control_output_and_executor_cannot_fill(self):
  from cellforge.construction.action_argument_cell import infer_joint
  from cellforge.construction.tool_router_evolution import FEATURES,ACTIONS
  router={"W":[[0]*len(FEATURES) for _ in ACTIONS],"b":[0]*len(ACTIONS)}
  router["b"][ACTIONS.index("none")]=.5
  router["W"][ACTIONS.index("write_file")][FEATURES.index("write")]=3
  params={"argument.W":W,"argument.b":[.5,.5,1.5]}
  output=infer_joint("write 'payload' to 'new.txt'",params,router,S["router"])
  self.assertEqual(output["packet"]["arguments"],{"path":"new.txt","content":"payload"})
  zero=copy.deepcopy(params);zero["argument.W"]=[[0]*8]*3
  self.assertIsNone(infer_joint("write 'payload' to 'new.txt'",zero,router,S["router"])["packet"])
  router["W"]=[[0]*len(FEATURES) for _ in ACTIONS]
  self.assertIsNone(infer_joint("write 'payload' to 'new.txt'",params,router,S["router"])["packet"])
 def test_reporting_cannot_select_and_scope_is_not_arc(self):
  from cellforge.construction.action_population import ActionOpponentHistory
  section={"partition":"reporting","manifest":"f","source":"isolated-action-fixture-safety",
           "outcomes":{"a":True},"false_positive_given_consistency":0}
  scores={k:copy.deepcopy(section) for k in ("fresh","archive","anchor")}
  with self.assertRaises(ValueError):ActionOpponentHistory().admission(scores,scores)

 def test_adapter_matches_worker_v2_and_no_host_fill(self):
  from cellforge.construction.action_argument_cell import infer_joint
  from cellforge.construction.tool_action_router_adapter import infer_compatible
  artifact=json.loads((ROOT/"tests/fixtures/tool_action_router_v2.json").read_text())
  params={"argument.W":artifact["argument_tensors"]["slot.W"],"argument.b":artifact["argument_tensors"]["null.b"]}
  for request in ("write 'version 1.2' to README","replace 'milk' with 'tea' in notes.txt","write 'v1.2'"):
   a=infer_joint(request,params,artifact["router"],artifact["argument_probe_router"],interface_version="tool-action-joint/2")
   b=infer_compatible(request,artifact)
   self.assertEqual(a["packet"],b["packet"])
   self.assertEqual(a["slots"],b["slots"])
