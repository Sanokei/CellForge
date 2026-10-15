import pathlib,json,sys,copy,time,random,os,tempfile,hashlib
root=pathlib.Path(__file__).parent;sys.path.insert(0,str(root/"src"))
from cellforge.construction.action_argument_growth_cell import infer_joint,execute_packet
from cellforge.construction.action_population import ActionCellPopulation,ActionOpponentHistory
from cellforge.construction.joint_weight_population import digest
def load(n):return json.loads((root/n).read_text())
def save(n,v):(root/n).write_text(json.dumps(v,indent=2),encoding="utf-8")
phase=int(sys.argv[1]) if len(sys.argv)>1 else 1
if (root/f"GENERATION-{phase}.json").exists():raise SystemExit("already executed")
base=load("baseline.json");oldmanifest=load("old-manifest.json");prior=load("parent-result.json");ra=load("router-author.json");router={"W":ra["W"],"b":ra["b"]}
def c(i,q,path,text,initial=None):return {"id":i,"request":q,"initial":initial or {},"expected":{path:text},"action":"edit_in_place" if initial else "write_file"}
fresh=[c("write-plain","write hello to notes.txt","notes.txt","hello"),c("create-plain","create summary to LOG","LOG","summary"),
c("edit-plain","replace milk with tea in notes.txt","notes.txt","tea",{"notes.txt":"milk"}),c("edit-next","replace north with south in MAP","MAP","south",{"MAP":"north"})]
archive=oldmanifest["archive"]+oldmanifest["fresh"];safety=oldmanifest["safety"]
report=[c("report-write","write candle to report.txt","report.txt","candle"),c("report-create","create outline to JOURNAL","JOURNAL","outline"),
c("report-edit","replace morning with evening in plan.txt","plan.txt","evening",{"plan.txt":"morning"}),c("report-edit2","replace hot with cold in CACHE","CACHE","cold",{"CACHE":"hot"}),
c("report-quoted","write 'green field' to 'a.txt'","a.txt","green field"),c("report-functionpath","write done to file","file","done")]
if phase==1:
 save("MANIFEST.json",{"schema":"bounded-action-growth/1","phases":2,"seeds":[101,202,303],"fresh":fresh,"archive":archive,"safety":safety,"reporting":report,
 "capacity":[33,39],"edit_ranges":[[-9,9],[-9,9]],"budgets":{"llm":2,"random":2},"scope":"literal full-matrix reauthoring permitted; matching full-capacity random edits, copied span grammar only",
 "epoch_migration":"Re-evaluate all prior champions on declared new development pressure; preserve source checkpoint and exact archive expansion evidence. Original records remain in parent-result.json.",
 "reporting":"reserved before phase1; never evaluated until phase2 decisions freeze; samegrammar finite lexical challenge notARC"})
save(f"START-{phase}.json",{"pid":os.getpid(),"utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
def effects(params,rows):
 out=[]
 for item in rows:
  with tempfile.TemporaryDirectory(dir=root,prefix="fixture_") as d:
   p=pathlib.Path(d)
   for k,v in item["initial"].items():(p/k).write_text(v,encoding="utf-8")
   pred=infer_joint(item["request"],params,router,base["router"],interface_version="tool-action-joint/2")
   result=execute_packet(pred["packet"],allowed_root=p);files={f.name:f.read_text(encoding="utf-8") for f in p.iterdir() if f.is_file()}
   out.append({"id":item["id"],"prediction":pred,"execution":result,"files":files,"correct":files==item["expected"] and pred["action"]==item["action"],"files_correct":files==item["expected"]})
 return out
def scores(params):
 negative=effects(params,safety);out={}
 for label,rows in (("fresh",fresh),("archive",archive),("anchor",safety)):
  traces=effects(params,rows);out[label]={"partition":"development","manifest":digest(rows),"source":"isolated-action-fixture-safety" if label=="anchor" else "isolated-action-fixture-development","outcomes":{r["id"]:r["correct"] for r in traces},
    "false_positive_given_consistency":sum(not r["files_correct"] for r in negative)/len(negative),"traces":traces,"rate_definition":"false file effects among four safety fixtures, shield included"}
 return out
proposal=load("proposal"+str(phase+3)+".json");params={"argument.W":proposal["tensors"]["slot.W"],"argument.b":proposal["tensors"]["null.b"]};width=len(params["argument.W"][0]);positions=3*width+3
assert positions==(33 if phase==1 else 39)
start=time.perf_counter();contexts=[]
for idx,seed in enumerate((101,202,303)):
 snapshot=copy.deepcopy(prior["contexts"][idx]["population"]) if phase==1 else load("GENERATION-1.json")["contexts"][idx]["population"]
 pop=ActionCellPopulation();pop.cells=snapshot["cells"];pop.live=snapshot["live"];pop.events=snapshot["events"];pop.charges=snapshot["charges"]
 pop.histories={k:ActionOpponentHistory(v) for k,v in snapshot["histories"].items()}
 family="action-arguments-lexical"
 if phase==1:
  # New pressure epoch recomputes facts from exact weights, never relabels old reporting.
  for cell in pop.cells.values():
   if cell["structure"]==family:cell["scores"]=scores(cell["params"])
  oldhistory=pop.histories[family].records
  pop.histories[family]=ActionOpponentHistory([dict(r,scores=scores(pop.cells[r["graph_hash"]]["params"])) for r in oldhistory])
 recipient=pop.live[family][0];before=copy.deepcopy(pop.cells[recipient]["scores"]);priorhistory=copy.deepcopy(pop.histories[family].records)
 # Full literal revision vocabulary33/39 explicitly allows all positions to change.
 decision=pop.birth(family,params,[recipient],{"raw_response":f"proposal{phase+3}.json","model":proposal["model"],"newly_authored":str(positions)+" full-reauthoring positions","typed_io":str(width)+" lexical features -> copied spans"},scores(params),caps_passed=True,fixtures_passed=True,arm="llm-growth")
 rand=random.Random(seed+1000*phase);v=[rand.uniform(-9,9) for _ in range(positions)];randomparams={"argument.W":[v[i*width:(i+1)*width] for i in range(3)],"argument.b":v[3*width:]}
 control=ActionCellPopulation();control.seed(family,pop.cells[recipient]["params"],pop.cells[recipient]["provenance"],before);control.histories[family]=ActionOpponentHistory(priorhistory)
 randomdecision=control.birth(family,randomparams,[recipient],{"source":"matched_full_matrix","seed":seed,"phase":phase},scores(randomparams),caps_passed=True,fixtures_passed=True,arm="random-growth")
 contexts.append({"seed":seed,"parent":recipient,"decision":decision,"random_decision":randomdecision,"population":pop.snapshot(),"random_population":control.snapshot()})
save(f"GENERATION-{phase}.json",{"contexts":contexts,"seconds":time.perf_counter()-start,"numeric_capacity":positions,"scientific_acceptance":False})
save(f"FEEDBACK-{phase}.json",{"phase":phase,"contexts":[{"seed":c["seed"],"decision":c["decision"],"selected":c["population"]["live"][family][0],"development":c["population"]["cells"][c["population"]["live"][family][0]]["scores"]["fresh"]["traces"]} for c in contexts],"reporting_evaluated":False})
if phase==2:
 selected=[c["population"]["cells"][c["population"]["live"][family][0]]["params"] for c in contexts];save("REPORTING.json",{"rows":[effects(p,report) for p in selected],"scientific_acceptance":False})
print(json.dumps({"phase":phase,"seconds":time.perf_counter()-start,"contexts":[{"seed":c["seed"],"accepted":c["decision"]["accepted"],"random":c["random_decision"]["accepted"],"live":c["population"]["live"][family],"fresh":sum(c["population"]["cells"][c["population"]["live"][family][0]]["scores"]["fresh"]["outcomes"].values())} for c in contexts]},indent=2))
