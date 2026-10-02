import unittest,pathlib,tempfile,json,shutil,copy
from cellforge.construction.action_cell_continuation import initial_state,prompts_for,compute_phase,successor,atomic,append_charge,replay
from cellforge.construction.anchored_action_protocol import file_hash,REQUIRED
from cellforge.construction.joint_weight_population import digest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class TestCellContinuationReplay(unittest.TestCase):
 def fixture(self,p):
  source=ROOT/"runs/weights-action-growth-g2"
  shutil.copyfile(source/"GENERATION-2.json",p/"parent-result.json");shutil.copyfile(source/"baseline.json",p/"baseline.json")
  def row(i,text):
   return {"id":i,"request":"write '"+text+"' to LOG","initial":{},"expected":{"LOG":text},"action":"write_file"}
  atomic(p/"MANIFEST.json",{k:[row(k,k)] for k in ("fresh","archive","safety","reporting")})
  atomic(p/"AUTHORING_CONTRACT.json",{"schema":"cell-continuation-contract/1","seeds":[401,402,403],"generations":3,
   "positions":39,"range":[-9,9],"arms":["llm","random"],"interface":"tool-action-joint/2",
   "scope":"cell-only; inherited router pinned; no full-MoE architecture acceptance",
   "selected_router":"ea3d29b774e6dae1a04bb6dc54b44f3cb11d6dcd0f3fd7005dd9dbdc6cafc832"})
  atomic(p/"INTERFACES.json",{"feature_order":"fixed12"});(p/"run.py").write_text("# not executed")
  runtime="src/cellforge/construction/action_cell_continuation.py";(p/runtime).parent.mkdir(parents=True);shutil.copyfile(ROOT/runtime,p/runtime)
  atomic(p/"RUNTIME_SOURCES.json",[runtime])
  deps={n:file_hash(p/n) for n in [*REQUIRED,runtime]}
  atomic(p/"DECLARATION.json",{"schema":"anchored-action-inputs/1","dependencies":deps})
  return file_hash(p/"DECLARATION.json")
 def phase(self,p,anchor,state,n):
  prompts=prompts_for(state);responses=[]
  for item in prompts:
   q=item["prompt"];params=copy.deepcopy(q["parent_params"]);params["argument.W"][0][0]-=.1*n
   response={"model":"gpt-6.1-sol","reasoning":"low","parent_id":q["parent_id"],"parent_params":q["parent_params"],
    "tensors":params,"feature_order":q["feature_order"],"role_order":q["role_order"],"prompt_sha256":digest(q),
    "feedback_sha256":digest(q["development_evidence"]),"charged_positions":39,"interface":"tool-action-joint/2"}
   responses.append({"seed":item["seed"],"response":response})
  atomic(p/f"RESPONSES-{n}.json",responses)
  atomic(p/f"PHASE-{n}-DECLARATION.json",{"initial_anchor":anchor,"phase":n,"parent_state_hash":digest(state),
   "responses_sha256":file_hash(p/f"RESPONSES-{n}.json"),"prompts_sha256":digest(prompts)})
  phase_anchor=file_hash(p/f"PHASE-{n}-DECLARATION.json")
  result=compute_phase(p,anchor,n,phase_anchor,state,lambda e:append_charge(p,n,e))
  atomic(p/f"GENERATION-{n}.json",result);state=successor(state,result);atomic(p/f"STATE-{n}.json",state)
  return state,phase_anchor
 def test_two_sequential_phases_replay_real_literal_effects_and_independent_controls(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);anchor=self.fixture(p);state=initial_state(p,anchor);atomic(p/"STATE-0.json",state)
   anchors=[]
   for n in (1,2):state,a=self.phase(p,anchor,state,n);anchors.append(a)
   receipt=replay(p,anchor,anchors)
   self.assertTrue(receipt["cell_protocol_replay"]);self.assertEqual(receipt["generations"],2)
   self.assertFalse(receipt["scientific_acceptance"])
   for context in state["contexts"]:
    self.assertEqual([len(context["arms"][a]["charges"]) for a in ("llm","random")],[2,2])
 def test_rehashed_phase_parent_or_response_cannot_change_external_phase_anchor(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);anchor=self.fixture(p);state=initial_state(p,anchor);atomic(p/"STATE-0.json",state)
   state,a=self.phase(p,anchor,state,1);decl=json.loads((p/"PHASE-1-DECLARATION.json").read_text());decl["parent_state_hash"]="fake"
   atomic(p/"PHASE-1-DECLARATION.json",decl)
   with self.assertRaisesRegex(ValueError,"phase anchor mismatch"):replay(p,anchor,[a])
 def test_dropped_rejected_charge_rejected_by_independent_replay(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);anchor=self.fixture(p);state=initial_state(p,anchor);atomic(p/"STATE-0.json",state)
   state,a=self.phase(p,anchor,state,1)
   log=p/"CHARGES-1.jsonl";lines=log.read_text().splitlines();log.write_text("\n".join(lines[:-1])+"\n")
   with self.assertRaisesRegex(ValueError,"charged"):replay(p,anchor,[a])
 def test_substituted_router_author_is_not_consulted(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d);anchor=self.fixture(p);before=initial_state(p,anchor)
   atomic(p/"router-author.json",{"W":[[999]],"b":[999]})
   self.assertEqual(initial_state(p,anchor),before)
