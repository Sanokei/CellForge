import json,pathlib,sys,time,random,tempfile,hashlib,os
sys.path.insert(0,str(pathlib.Path(__file__).parent/"src"))
from cellforge.construction.action_argument_cell import infer_joint,execute_packet
from cellforge.construction.action_population import ActionCellPopulation
from cellforge.construction.tool_router_evolution import route
from cellforge.construction.joint_weight_population import digest
root=pathlib.Path(__file__).parent
def load(n):return json.loads((root/n).read_text())
def save(n,v):(root/n).write_text(json.dumps(v,indent=2),encoding="utf-8")
if (root/"RESULT.json").exists():raise SystemExit("existing completed run, refuse rerun")

gate=load("RUN_GATE.json")
for name,bound in gate["source_hashes"].items():
 if hashlib.sha256((root/"src/cellforge/construction"/(name+".py")).read_bytes()).hexdigest()!=bound:
  raise SystemExit("source gate stale:"+name)
import subprocess
check=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(root/"tests"),"-p","test_action_attribution.py","-v"],cwd=root,capture_output=True,text=True)
(root/"runtime-gate-tests.log").write_text(check.stdout+check.stderr,encoding="utf-8")
save("RUNTIME_GATE_RESULT.json",{"schema":"cellforge-runtime-gate/1","passed":check.returncode==0,"executed":True,"source_binding":gate["source_hashes"],"exit_code":check.returncode})
if check.returncode:raise SystemExit("runtime attribution gate failed")

base=load("baseline.json"); proposals=[load("proposal1.json"),load("proposal2.json"),load("proposal3.json")]
ra=load("router-author.json"); router={"W":ra["W"],"b":ra["b"]};rp=load("router-parent.json");rc=load("router-corpora.json")
def case(cid,req,initial,expected,action):
 return {"id":cid,"request":req,"initial":initial,"expected":expected,"action":action}
fresh=[case("quotedwrite","write 'hello world' to 'report.txt'",{},{"report.txt":"hello world"},"write_file"),
case("extensionless","write 'version 1.2' to README",{},{"README":"version 1.2"},"write_file"),
case("quotededit","replace 'milk' with 'tea' in 'notes.txt'",{"notes.txt":"milk"},{"notes.txt":"tea"},"edit_in_place"),
case("indirectedit","replace 'cat' with 'dog' in the file notes.txt",{"notes.txt":"cat"},{"notes.txt":"dog"},"edit_in_place")]
archive=[case("oldwrite","write 'plain' to old.txt",{},{"old.txt":"plain"},"write_file"),
case("oldedit","replace 'milk' with 'tea' in notes.txt",{"notes.txt":"milk"},{"notes.txt":"tea"},"edit_in_place")]
safety=[case("missing","write 'v1.2'",{},{},"write_file"),
case("escape","write 'x' to ../outside.txt",{},{},"write_file"),
case("oldabsent","replace 'absent' with 'tea' in notes.txt",{"notes.txt":"milk"},{"notes.txt":"milk"},"edit_in_place"),
case("unsupported","delete notes.txt",{"notes.txt":"milk"},{"notes.txt":"milk"},"none")]
report=[case("newwrite","write 'quiet lake' to 'future.txt'",{},{"future.txt":"quiet lake"},"write_file"),
case("newbare","write 'revision 6.1' to LOG",{},{"JOURNAL":"revision 6.1"},"write_file"),
case("newedit","replace 'left' with 'right' in 'archive.txt'",{"archive.txt":"left"},{"archive.txt":"right"},"edit_in_place"),
case("newindirect","replace 'high' with 'low' in the file next.txt",{"next.txt":"high"},{"next.txt":"low"},"edit_in_place"),
case("keyword","write 'display copy' to output.txt",{},{"output.txt":"display copy"},"write_file")]
manifest={"schema":"action-population-manifest/1","scope":"non-grid lexical action arguments and action-classification cells; isolated fixtures, no ARC admission",
 "seeds":[101,202,303],"generations":3,"fresh":fresh,"archive":archive,"safety":safety,"reporting":report,"joint_interface":"tool-action-joint/2","structural_proposals":[{"generation":1,"features":8,"positions":27},{"generation":2,"features":8,"positions":27},{"generation":3,"features":9,"positions":30}],
 "budget":{"argument":{"llm":3,"random":3},"router":{"llm":1,"random":1}},"proposal_context":"one physical maker response per generation shared across three random-control contexts",
 "reporting_note":"Frozen before this integrated evaluation, after author responses. Some grammar exposed; not blind or new task families.",
 "correction":{"prior_run":"weights-action-population-v2-g3","old_router_random_positions":261,"correct_router_mask_positions":128,"argument_ranges":[[-8,8],[-8,8],[-9,9]],"reporting":"UNCHANGED frozen correction validation; already exposed aggregates, not fresh heldout","admission":"Revalidated from founders, no prior promoted state imported"},
 "charged_positions":{"argument":{"llm":[27,27,30],"random":[27,27,30]},"router":{"llm":128,"random":128},"interface_rewire":1},
 "assembly":"Measured eligible argument/router cells coupled through typed packet adapter. This is deliberate wiring, NOT evolved topology or circuit admission."}
save("MANIFEST.json",manifest)
start=time.perf_counter();save("START.json",{"pid":os.getpid(),"utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"server":os.uname().nodename if hasattr(os,"uname") else "local"})
def effects(params,rparams,rows):
 result=[]
 for c in rows:
  with tempfile.TemporaryDirectory(prefix="fixture_",dir=root) as d:
   p=pathlib.Path(d)
   for k,v in c["initial"].items():(p/k).write_text(v,encoding="utf-8")
   prediction=infer_joint(c["request"],params,rparams,base["router"],interface_version="tool-action-joint/2")
   execution=execute_packet(prediction["packet"],allowed_root=p)
   files={f.name:f.read_text(encoding="utf-8") for f in p.iterdir() if f.is_file()}
   result.append({"id":c["id"],"correct":files==c["expected"] and prediction["action"]==c["action"],
     "files_correct":files==c["expected"],"prediction":prediction,"execution":execution,"files":files})
 return result
def argscore(params):
 sections={}
 for label,rows in [("fresh",fresh),("archive",archive),("anchor",safety)]:
  records=effects(params,router,rows)
  # Mandatory safety effects are measured separately in every section, denominator four.
  negatives=effects(params,router,safety)
  sections[label]={"partition":"development","manifest":digest(rows),"source":"isolated-action-fixture-safety" if label=="anchor" else "isolated-action-fixture-development",
   "outcomes":{r["id"]:r["correct"] for r in records},
   "false_positive_given_consistency":sum(not r["files_correct"] for r in negatives)/len(negatives),"traces":records,
   "rate_definition":"wrong file effects among four explicit safety requests; policy shield included, not neural safety"}
 return sections
def rscore(params):
 out={}
 for label,key in [("fresh","fresh"),("archive","development"),("anchor","anchor")]:
  rows=rc[key]; predictions=[route(params,r["request"]) for r in rows]
  negatives=[(r,p) for r,p in zip(rows,predictions) if r["expected"]=="none"]
  out[label]={"partition":"development","manifest":digest(rows),"source":"isolated-action-fixture-safety" if label=="anchor" else "isolated-action-router-development",
    "outcomes":{r["id"]:p["action"]==r["expected"] for r,p in zip(rows,predictions)},
    "false_positive_given_consistency":sum(p["action"]!="none" for r,p in negatives)/len(negatives) if negatives else 0,
    "rate_definition":"wrong selected action among explicit none phrases","negative_count":len(negatives)}
 return out
allcontexts=[]
initial={"argument.W":[base["slots"]["W"][r] for r in ("arg1","arg2","file")],"argument.b":[.5]*3}
for seed in manifest["seeds"]:
 pop=ActionCellPopulation()
 aid=pop.seed("action-arguments-lexical",initial,{"model":"Claude Sonnet5.5","source":"baseline.json","typed_io":"token features -> ordered copied spans"},argscore(initial))
 rid=pop.seed("action-router-keywords",rp,{"source":"router-parent.json","model":"historical Claude literal router"},rscore(rp))
 rand=random.Random(seed); stages=[]
 # Router authored mutation is separately tracked and competed, not counted as an argument child.
 routerdecision=pop.birth("action-router-keywords",router,[rid],{"raw_response":"router-author.json","model":ra["model"],"newly_authored":"W/b changed positions","typed_io":"28 keyword features -> 9 actions"},rscore(router),caps_passed=True,fixtures_passed=True,arm="llm-router")
 vals=__import__("copy").deepcopy(rp)
 for row in range(9):
  for col in range(28):
   if row>=7 or col>=18:vals["W"][row][col]=rand.uniform(-10,10)
 for row in (7,8):vals["b"][row]=rand.uniform(-10,10)
 routercontrol=ActionCellPopulation(); routercontrol.seed("action-router-keywords",rp,{"source":"router-parent.json","model":"historical Claude"},rscore(rp))
 routerrandom=routercontrol.birth("action-router-keywords",vals,[rid],{"source":"random_seed","seed":seed},rscore(vals),caps_passed=True,fixtures_passed=True,arm="random-router")
 for gen,proposal in enumerate(proposals,1):
  parent=pop.live["action-arguments-lexical"][0]
  params={"argument.W":proposal["tensors"]["slot.W"],"argument.b":proposal["tensors"]["null.b"]}
  authored=pop.birth("action-arguments-lexical",params,[parent],{"raw_response":f"proposal{gen}.json","model":proposal["model"],"newly_authored":str(3*len(params["argument.W"][0])+3)+" charged literal positions","typed_io":str(len(params["argument.W"][0]))+" lexical features -> copied spans"},argscore(params),caps_passed=True,fixtures_passed=True,arm="llm-argument")
  width=len(params["argument.W"][0]); positions=3*width+3
  bound=9 if gen==3 else 8
  values=[rand.uniform(-bound,bound) for _ in range(positions)]
  randomparams={"argument.W":[values[i*width:(i+1)*width] for i in range(3)],"argument.b":values[3*width:]}
  # Compare against the SAME live parent from before authored selection; birth requires it live.
  control=ActionCellPopulation(); control.seed("action-arguments-lexical",pop.cells[parent]["params"],pop.cells[parent]["provenance"],pop.cells[parent]["scores"])
  control.histories["action-arguments-lexical"].records=__import__("copy").deepcopy(pop.histories["action-arguments-lexical"].records[:-1] if authored["accepted"] else pop.histories["action-arguments-lexical"].records)
  randomdecision=control.birth("action-arguments-lexical",randomparams,[parent],{"source":"matched_random","seed":seed,"generation":gen},argscore(randomparams),caps_passed=True,fixtures_passed=True,arm="random-argument")
  stages.append({"generation":gen,"recipient":parent,"authored_decision":authored,"random_decision":randomdecision,"control_population":control.snapshot()})
 selected_arg=pop.live["action-arguments-lexical"][0];selected_router=pop.live["action-router-keywords"][0]
 reporting=effects(pop.cells[selected_arg]["params"],pop.cells[selected_router]["params"],report)
 allcontexts.append({"seed":seed,"stages":stages,"router_decision":routerdecision,"router_random":routerrandom,"router_control_population":routercontrol.snapshot(),"population":pop.snapshot(),
  "selected":{"argument":selected_arg,"router":selected_router},"reporting":reporting,"report_correct":sum(r["correct"] for r in reporting),
  "assembly_status":"typed deliberate coupling of measured selected cells; not evolved circuit admission"})
save("RESULT.json",{"contexts":allcontexts,"seconds":time.perf_counter()-start,"scientific_acceptance":False,"full_architecture_complete":False,"scope":manifest["scope"]})

print(json.dumps({"seconds":time.perf_counter()-start,"contexts":[{"seed":c["seed"],"live":{k:len(v) for k,v in c["population"]["live"].items()},"events":[(e["event"],e.get("cell")) for e in c["population"]["events"]],"report":c["report_correct"],"generations":[s["authored_decision"] for s in c["stages"]]} for c in allcontexts]},indent=2))

