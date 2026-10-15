import pathlib,json,copy,itertools,time,numpy as np,sys,os
ROOT=pathlib.Path(__file__).parent
sys.path.insert(0,str(ROOT/"src"))
from cellforge.construction.adhesion import AdhesionCandidate
from cellforge.construction.circuit_population import CircuitArchive
from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.joint_weight_population import digest
def load(n):return json.loads((ROOT/n).read_text())
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2),encoding="utf-8")
if (ROOT/"RESULT.json").exists():raise SystemExit("completed run; no duplicate")
manifest=load("PREDECLARED.json");author=load("RAW_AUTHOR_RESPONSE.json");source=load("cells.json")
catalogue={k:source["cells"][k] for k in ("cmp2","mux")}
def forward(cell,bits):
    v=list(bits)
    for layer in cell["layers"]:v=[int(sum(w*x for w,x in zip(row,v))+b>0) for row,b in zip(layer["W"],layer["b"])]
    return v
population=CellPopulation();founders={}
for name,cell in catalogue.items():
    rows=[]
    for bits in itertools.product((0,1),repeat=len(cell["inputs"])):
        # Reference belongs ONLY to evaluator, never to candidate execution.
        if name=="cmp2":
            a=bits[0]+2*bits[1];b=bits[2]+2*bits[3];expected=[int(a>b),int(a==b),int(a<b)]
        else:expected=[bits[2] if bits[0] else bits[1]]
        rows.append({"bits":list(bits),"predicted":forward(cell,bits),"expected":expected,"correct":forward(cell,bits)==expected})
    assert all(r["correct"] for r in rows)
    outcomes={str(i):r["correct"] for i,r in enumerate(rows)}
    scores={k:{"partition":"development","source":"exhaustive-primitive-logic-contract","manifest":digest(rows),"outcomes":outcomes,"false_positive_given_consistency":0} for k in ("fresh","archive","anchor")}
    params={f"layer{j}.{key}":l[key] for j,l in enumerate(cell["layers"]) for key in ("W","b")}
    fid=population.seed("threshold-"+name,params,{"source":"cells.json","model_metadata":source["author"],"ordered_ports":{"inputs":cell["inputs"],"outputs":cell["outputs"]}},scores)
    founders[name]={"id":fid,"cell_hash":digest(cell),"rows":rows,"eligible":True}
save("CELL_POPULATION.json",population.snapshot());save("FOUNDER_ELIGIBILITY.json",founders)
dev=[c for c in manifest["cases"] if c["partition"]=="development"];report=[c for c in manifest["cases"] if c["partition"]=="frozen-report"]
def evaluate(spec,cat,rows):
    try:
        c=AdhesionCandidate(spec,cat,forward);results=[]
        for r in rows:
            output=c.execute({"bits":r["bits"]});results.append({"id":r["id"],"actual":output,"correct":output["packet"]["bits"]==r["expected"]})
        return {"valid":True,"outcomes":{r["id"]:r["correct"] for r in results},"rows":results,"receipt":c.receipt}
    except ValueError as error:return {"valid":False,"outcomes":{r["id"]:False for r in rows},"error":str(error),"rows":[]}
def contribution(spec,base):
    if not base["valid"]:return False
    full=sum(base["outcomes"].values());scores=[]
    for intervention in ("cmp2","mux","glue"):
        cat=copy.deepcopy(catalogue);s=copy.deepcopy(spec)
        if intervention=="glue":
            node=next(n for n in s["nodes"] if n["kind"]=="glue");node["params"]={"W":[[0,0,0]],"b":[0]}
        else:
            for layer in cat[intervention]["layers"]:
                for key in ("W","b"):layer[key]=np.zeros_like(layer[key]).tolist()
        scores.append(sum(evaluate(s,cat,dev)["outcomes"].values()))
    return all(score<full for score in scores)
proposals=[p for p in author["proposals"] if p["arm"]=="llm-proposal"]
baseline=next(p for p in author["proposals"] if p["name"]=="unconnected-zero")
baseline_score=evaluate(baseline["spec"],catalogue,dev);assert baseline_score["valid"]
save("SCOPE.json",{"contexts":[601,602,603],"founder_families":["threshold-cmp2","threshold-mux"],"physical_founders":2,
"author_proposals":7,"author_glue_positions":27,"random_proposals_per_context":7,"random_glue_positions":27,"reporting":"REUSE worker frozen8cases for integration validation; not fresh heldout, never used for admission",
"caps":{"nodes":8,"new_positions_per_proposal":16},"limitations":"Finite4bit max task; requested modelmetadata gpt6.1sol low, backendidentity unobserved; notweightcrossover, ARCtransfer or fullarchitecture"})
save("START.json",{"pid":os.getpid(),"utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
start=time.perf_counter();contexts=[];charged_receipts=[]
def charge(seed,arm,proposal):
    charged_receipts.append({"ordinal":len(charged_receipts),"seed":seed,"arm":arm,"proposal":digest(proposal),"positions":proposal["new_numeric_positions"],"charged_before_validation":True})
    save("CHARGES.json",charged_receipts)
for seed in (601,602,603):
    charge(seed,"bootstrap-control",baseline)
    rng=np.random.default_rng(seed);llm=CircuitArchive(baseline["spec"],baseline_score["outcomes"],[f["id"] for f in founders.values()])
    random=CircuitArchive(baseline["spec"],baseline_score["outcomes"],[f["id"] for f in founders.values()]);trials=[]
    for proposal in proposals:
        # Charge is explicitly persisted before validation/execution.
        charge(seed,"llm",proposal)
        score=evaluate(proposal["spec"],catalogue,dev)
        decision=llm.propose(proposal,valid=score["valid"],dev_outcomes=score["outcomes"],contribution=contribution(proposal["spec"],score),arm="llm")
        control=copy.deepcopy(proposal);control["arm"]="random";node=next(n for n in control["spec"]["nodes"] if n["kind"]=="glue")
        for key in ("W","b"):node["params"][key]=rng.integers(-1,2,size=np.asarray(node["params"][key]).shape).tolist()
        charge(seed,"random",control)
        rscore=evaluate(control["spec"],catalogue,dev)
        rdecision=random.propose(control,valid=rscore["valid"],dev_outcomes=rscore["outcomes"],contribution=contribution(control["spec"],rscore),arm="random")
        trials.append({"proposal":proposal,"development":score,"decision":decision,"random_proposal":control,"random_development":rscore,"random_decision":rdecision})
    frozen={"llm":llm.snapshot(),"random":random.snapshot()};save(f"FROZEN-{seed}.json",frozen)
    # Only after decisions freeze do we execute reporting.
    reporting={arm:evaluate(archive.records[archive.live]["spec"],catalogue,report) for arm,archive in (("llm",llm),("random",random))}
    contexts.append({"seed":seed,"population":frozen,"trials":trials,"reporting":reporting})
save("RESULT.json",{"seconds":time.perf_counter()-start,"contexts":contexts,"live_founder_ids":{k:v["id"] for k,v in founders.items()},
"scientific_acceptance":False,"architecture_complete":False,"circuit_scope":"evolved authored topology/glue proposals selected on development against matched controls; not independent cellweight crossover or new ARC capability"})
print(json.dumps({"seconds":time.perf_counter()-start,"contexts":[{"seed":c["seed"],"admitted_llm":sum(e["event"]=="admitted" for e in c["population"]["llm"]["events"]),"admitted_random":sum(e["event"]=="admitted" for e in c["population"]["random"]["events"]),"reporting":{arm:sum(v["outcomes"].values()) for arm,v in c["reporting"].items()}} for c in contexts]},indent=2))
