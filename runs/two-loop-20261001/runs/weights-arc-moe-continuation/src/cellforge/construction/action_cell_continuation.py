"""Bounded prospective action-cell loop with independent surviving control arms.
No API/model invocation: fresh literal response files must be supplied by the maker.
The supervisor waiting for responses is not training or model authorship.
"""
import argparse,copy,json,os,pathlib,tempfile,time,hashlib
from .anchored_action_protocol import verify_input_anchor,validate_snapshot,resolve_live_router,file_hash
from .experiment_manifest import load_json_strict,assert_manifest_unchanged
from .action_phase_protocol import *
from .action_argument_growth_cell import infer_joint,execute_packet
from .tool_router_evolution import validate as validate_router

def atomic(path,value):
    path=pathlib.Path(path);pending=path.with_suffix(path.suffix+".pending")
    with pending.open("w",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(pending,path)

def runtime_binding(root,decl):
    for name,sha in decl["dependencies"].items():
        if name.startswith("src/") and name.endswith(".py"):
            actual=pathlib.Path(__file__).parents[2]/name[4:]
            if not actual.is_file() or file_hash(actual)!=sha:raise ValueError("loaded runtime differs:"+name)

def inputs(root,anchor):
    verify_input_anchor(root,anchor)
    decl=load_json_strict(root/"DECLARATION.json");runtime_binding(root,decl)
    manifest=load_json_strict(root/"MANIFEST.json");contract=load_json_strict(root/"AUTHORING_CONTRACT.json")
    required={"schema":"cell-continuation-contract/1","seeds":[401,402,403],"generations":3,
              "positions":39,"range":[-9,9],"arms":["llm","random"],"interface":"tool-action-joint/2",
              "scope":"cell-only; inherited router pinned; no full-MoE architecture acceptance"}
    if any(contract.get(k)!=v for k,v in required.items()):raise ValueError("prospective cell contract differs")
    parent=load_json_strict(root/"parent-result.json")
    snapshots=[c["population"] for c in parent["contexts"]]
    if not snapshots:raise ValueError("parent contexts required")
    # Current live tensors must match in every replicated context; legacy ancestry is not certified.
    selected=[]
    for s in snapshots:
        validate_snapshot(s)
        rid=contract["selected_router"];router=resolve_live_router(s,rid);validate_router(router)
        arg=s["live"][FAMILY]
        if len(arg)!=1:raise ValueError("exactly one selected argument founder")
        argument_caps(s["cells"][arg[0]]["params"])
        selected.append((arg[0],rid,s["cells"][arg[0]]["params"],router))
    if any(row!=selected[0] for row in selected):raise ValueError("replicated selected parents differ")
    return manifest,contract,snapshots[0],selected[0][3],load_json_strict(root/"baseline.json")["router"]

def evaluator(root,manifest,router,lexical):
    def effects(params,rows):
        argument_caps(params);out=[]
        for case in rows:
            with tempfile.TemporaryDirectory(prefix="cell_fixture_") as d:
                directory=pathlib.Path(d)
                for name,text in case["initial"].items():(directory/name).write_text(text,encoding="utf-8")
                prediction=infer_joint(case["request"],params,router,lexical,interface_version="tool-action-joint/2")
                execution=execute_packet(prediction["packet"],allowed_root=directory)
                files={p.name:p.read_text(encoding="utf-8") for p in directory.iterdir() if p.is_file()}
                out.append({"id":case["id"],"request":case["request"],"prediction":prediction,"execution":execution,
                            "files":files,"correct":files==case["expected"] and prediction["action"]==case["action"],
                            "files_correct":files==case["expected"]})
        return out
    def evaluate(params):
        safety=effects(params,manifest["safety"]);rate=sum(not r["files_correct"] for r in safety)/len(safety)
        scores={}
        for name,section in (("fresh","fresh"),("archive","archive"),("anchor","safety")):
            traces=effects(params,manifest[section])
            scores[name]={"partition":"development","manifest":digest(manifest[section]),
              "source":"isolated-action-fixture-safety" if name=="anchor" else "isolated-action-fixture-development",
              "outcomes":{r["id"]:r["correct"] for r in traces},"false_positive_given_consistency":rate,"traces":traces}
        validate_scores(scores);return scores
    return evaluate,effects

def initial_state(root,anchor):
    manifest,contract,source,router,lexical=inputs(root,anchor)
    evaluate,effects=evaluator(root,manifest,router,lexical)
    # Re-measured existing literal cell becomes a prospective founder; do not certify old ancestry.
    selected=source["live"][FAMILY][0];cell=source["cells"][selected]
    pop=ActionCellPopulation();pop.seed(FAMILY,cell["params"],
      {"author":"imported literal LLM tensor","source_parent_checkpoint":file_hash(root/"parent-result.json"),
       "source_selected_id":selected,"legacy_ancestry_certified":False},evaluate(cell["params"]))
    # All original argument rivals are independently remeasured, retained as archived weights.
    for cid,old in source["cells"].items():
        if old["structure"]!=FAMILY or cid==selected:continue
        if len(old["params"]["argument.W"][0])!=12:continue # Exact declared interface; other widths remain sealed source only.
        argument_caps(old["params"]);scores=evaluate(old["params"])
        pop.cells[cid]={"id":cid,"structure":FAMILY,"params":copy.deepcopy(old["params"]),"parents":[],
            "provenance":{"source_checkpoint":file_hash(root/"parent-result.json"),"legacy_ancestry_certified":False},
            "scores":scores,"state":"archived"}
        pop.histories[FAMILY].register(graph_hash=cid,family=FAMILY,structure_hash=digest(FAMILY),
                                     provenance=pop.cells[cid]["provenance"],scores=scores)
    validate_snapshot(pop.snapshot())
    return {"schema":"prospective-cell-state/1","initial_anchor":anchor,"phase":0,
      "manifest_sha256":file_hash(root/"MANIFEST.json"),"selected_router":contract["selected_router"],
      "router_tensor_hash":digest(router),"contexts":[{"seed":seed,"arms":{"llm":pop.snapshot(),"random":copy.deepcopy(pop.snapshot())}}
                                                    for seed in contract["seeds"]],"scientific_acceptance":False}

def prompts_for(state):
    result=[]
    for context in state["contexts"]:
        p=context["arms"]["llm"];cid=p["live"][FAMILY][0];cell=p["cells"][cid]
        result.append({"seed":context["seed"],"prompt":{"parent_id":cid,"parent_params":cell["params"],
          "development_evidence":cell["scores"],"partition":"development","feature_order":list(FEATURES),
          "role_order":["arg1","arg2","file"],"edit_contract":{"positions":39,"range":[-9,9],
                      "operator":"literal_full_matrix","interface":"tool-action-joint/2"}}})
    return result

def phase_inputs(root,initial_anchor,phase,phase_anchor,state):
    inputs(root,initial_anchor)
    file=root/f"PHASE-{phase}-DECLARATION.json"
    if file_hash(file)!=phase_anchor:raise ValueError("independent phase anchor mismatch")
    declaration=load_json_strict(file)
    if declaration.get("initial_anchor")!=initial_anchor or declaration.get("phase")!=phase:
        raise ValueError("phase declaration identity")
    if declaration.get("parent_state_hash")!=digest(state):raise ValueError("stale or fabricated parent state")
    prompts=prompts_for(state);responses=load_json_strict(root/f"RESPONSES-{phase}.json")
    if declaration.get("responses_sha256")!=file_hash(root/f"RESPONSES-{phase}.json"):
        raise ValueError("unbound maker responses")
    if declaration.get("prompts_sha256")!=digest(prompts):raise ValueError("maker prompt commitment")
    if len(responses)!=len(prompts):raise ValueError("per-context raw maker responses required")
    return prompts,responses

def compute_phase(root,initial_anchor,phase,phase_anchor,state,journal):
    if state["phase"]!=phase-1:raise ValueError("sequential survivor state required")
    manifest,contract,source,router,lexical=inputs(root,initial_anchor)
    assert_manifest_unchanged(root/"MANIFEST.json",state["manifest_sha256"])
    prompts,responses=phase_inputs(root,initial_anchor,phase,phase_anchor,state)
    evaluate,effects=evaluator(root,manifest,router,lexical)
    contexts=[]
    for context,prompt,response in zip(state["contexts"],prompts,responses):
        if prompt["seed"]!=context["seed"] or response["seed"]!=context["seed"]:raise ValueError("seed binding differs")
        arms=apply_phase(context["arms"],context["seed"],phase,response["response"],prompt["prompt"],
                         prompt["prompt"]["development_evidence"],evaluate,journal)
        contexts.append({"seed":context["seed"],"arms":arms})
    return {"phase":phase,"contexts":contexts,"initial_anchor":initial_anchor,"phase_anchor":phase_anchor,
      "selected_router":contract["selected_router"],"router_tensor_hash":digest(router),"scientific_acceptance":False}

def successor(state,result):
    return dict(state,phase=result["phase"],contexts=[{"seed":c["seed"],
      "arms":{a:row["population"] for a,row in c["arms"].items()}} for c in result["contexts"]])

def append_charge(root,phase,entry):
    path=root/f"CHARGES-{phase}.jsonl";previous="0"*64
    if path.exists():
        rows=path.read_text(encoding="utf-8").splitlines()
        if rows:previous=json.loads(rows[-1])["hash"]
    row={"previous":previous,"entry":entry};row["hash"]=digest(row)
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(row,sort_keys=True)+"\n");f.flush();os.fsync(f.fileno())

def read_charges(root,phase):
    rows=[];previous="0"*64
    for line in (root/f"CHARGES-{phase}.jsonl").read_text(encoding="utf-8").splitlines():
        row=json.loads(line);observed=row.pop("hash")
        if row["previous"]!=previous or digest(row)!=observed:raise ValueError("charge hash chain differs")
        previous=observed;rows.append(row["entry"])
    return rows

def replay(root,initial_anchor,phase_anchors):
    state=initial_state(root,initial_anchor)
    if state!=load_json_strict(root/"STATE-0.json"):raise ValueError("founder remeasurement differs")
    for phase,anchor in enumerate(phase_anchors,1):
        charges=[];expected=compute_phase(root,initial_anchor,phase,anchor,state,charges.append)
        stored=load_json_strict(root/f"GENERATION-{phase}.json")
        validate_phase_receipt(expected,stored,read_charges(root,phase))
        state=successor(state,expected)
        if state!=load_json_strict(root/f"STATE-{phase}.json"):raise ValueError("saved survivor checkpoint differs")
    return {"cell_protocol_replay":True,"generations":len(phase_anchors),"state":state,
      "scientific_acceptance":False,"full_architecture_acceptance":False,
      "scope":"Independent literal cell competition only; reporting not evaluated by this replay"}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("operation",choices=["initialize","phase","replay"])
    parser.add_argument("root",type=pathlib.Path);parser.add_argument("--anchor",required=True)
    parser.add_argument("--phase",type=int);parser.add_argument("--phase-anchor",action="append",default=[])
    args=parser.parse_args();root=args.root.resolve()
    if args.operation=="replay":print(json.dumps(replay(root,args.anchor,args.phase_anchor),indent=2));return
    lock=root/"EXECUTION.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        if args.operation=="initialize":
            if (root/"STATE-0.json").exists():raise ValueError("already initialized")
            state=initial_state(root,args.anchor);atomic(root/"STATE-0.json",state)
            atomic(root/"PROMPTS-1.json",prompts_for(state));print(json.dumps({"initialized":True,"pid":os.getpid(),"state_hash":digest(state)}))
        else:
            phase=args.phase
            if phase not in (1,2,3) or len(args.phase_anchor)!=1:raise ValueError("one prospective phase anchor required")
            if (root/f"GENERATION-{phase}.json").exists() or (root/f"CHARGES-{phase}.jsonl").exists():
                raise ValueError("already started; never rerun charged proposal")
            state=load_json_strict(root/f"STATE-{phase-1}.json")
            result=compute_phase(root,args.anchor,phase,args.phase_anchor[0],state,
                                  lambda entry:append_charge(root,phase,entry))
            atomic(root/f"GENERATION-{phase}.json",result);state=successor(state,result)
            atomic(root/f"STATE-{phase}.json",state)
            if phase<3:atomic(root/f"PROMPTS-{phase+1}.json",prompts_for(state))
            print(json.dumps({"phase":phase,"pid":os.getpid(),"state_hash":digest(state)}))
    finally:lock.unlink()
if __name__=="__main__":main()
