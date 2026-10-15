"""Bounded independent ARC MoE service. No API/client/provisioning route."""
import argparse,copy,datetime,json,os,pathlib,random,time
from .experiment_manifest import load_json_strict
from .joint_weight_population import digest
from .action_cell_continuation import atomic,append_charge,read_charges
from .anchored_action_protocol import file_hash
from .arc_weight_moe import make_core,score,measured_pool,assemble,admission,validate_weights,primitive_pool

def inputs(root,anchor):
    if file_hash(root/"DECLARATION.json")!=anchor:raise ValueError("independent MoE initial anchor mismatch")
    decl=load_json_strict(root/"DECLARATION.json")
    if decl["schema"]!="prospective-ARC-MoE/1":raise ValueError("MoE declaration schema")
    required={"CONFIG.json","TASKS.json","parent.json","pop1_core.py","ARC_FLOOR.json","arc_eval.py"}
    if not required.issubset(decl["dependencies"]):raise ValueError("MoE minimum dependency coverage")
    for name,expected in decl["dependencies"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path)!=expected:
            raise ValueError("MoE dependency differs:"+name)
        if name.startswith("src/") and name.endswith(".py"):
            installed=pathlib.Path(__file__).parents[2]/name[4:]
            if file_hash(installed)!=expected:raise ValueError("loaded MoE source differs:"+name)
    config=load_json_strict(root/"CONFIG.json")
    if config["seeds"]!=[501,502,503] or config["rounds"]!=3 or config["range"]!=[-9,9]:
        raise ValueError("predeclared MoE arms/seeds/budget")
    manifest=load_json_strict(root/"TASKS.json");seen=set();rows=[]
    for item in manifest["development"]:
        tid=item["id"]
        if tid in seen or len(tid)!=8 or any(c not in "0123456789abcdef" for c in tid):
            raise ValueError("unique official development task ID")
        seen.add(tid);path=root/"data/training"/(tid+".json")
        if file_hash(path)!=item["sha256"] or "data/training/"+tid+".json" not in decl["dependencies"]:
            raise ValueError("unbound full ARC task")
        rows.append({"id":tid,"task":load_json_strict(path)})
    if len(rows)!=100:raise ValueError("declared 100-task ARC development scope")
    return config,rows,make_core(root/"pop1_core.py"),load_json_strict(root/"ARC_FLOOR.json")["min_solved_fraction"]

def initial_state(root,anchor):
    config,rows,core,floor=inputs(root,anchor);tensors=load_json_strict(root/"parent.json")["tensors"]
    validate_weights(tensors,core);evidence=score(tensors,rows,core)
    if config.get("primitive_manifest"):
        name=config["primitive_manifest"]
        if name not in load_json_strict(root/"DECLARATION.json")["dependencies"]:raise ValueError("unbound primitive admission manifest")
        pool=primitive_pool(tensors,load_json_strict(root/name),core)
    else:pool=measured_pool(tensors,rows,core) # Historical protocol retained, never relabeled.

    champion={"tensors":tensors,"hash":digest(tensors),"evidence":evidence,"parents":[],"origin":"remeasured_existing_joint_baseline"}
    return {"initial_anchor":anchor,"round":0,"pool":pool,"pool_hash":digest(pool),
      "contexts":[{"seed":seed,"arms":{a:{"champion":copy.deepcopy(champion),"history":[copy.deepcopy(evidence)],"births":[]}
                                    for a in ("llm","random")}} for seed in config["seeds"]],
      "scope":"full-task ARC MoE competition separate from action-cell service","scientific_acceptance":False}

def prompts(state):
    return [{"seed":c["seed"],"prompt":{"parent_hash":c["arms"]["llm"]["champion"]["hash"],
              "pool":state["pool"],"pool_hash":state["pool_hash"],
              "development_evidence":c["arms"]["llm"]["champion"]["evidence"],"partition":"development",
              "assembly_rule":"Only earned live grid cells; copy actual source tensors. Author literal route.W NxN and route.b N.",
              "scope":"No held-out reporting or evaluation split. Oracle diagnostic cannot be actual routing input."}}
            for c in state["contexts"]]

def phase(root,anchor,round_anchor,state,journal):
    config,rows,core,floor=inputs(root,anchor);n=state["round"]+1
    if n>3:raise ValueError("declared MoE generation budget complete")
    path=root/f"ROUND-{n}-DECLARATION.json"
    if file_hash(path)!=round_anchor:raise ValueError("independent MoE round anchor mismatch")
    decl=load_json_strict(path);requests=prompts(state);responses=load_json_strict(root/f"RESPONSES-{n}.json")
    if decl!={"initial_anchor":anchor,"round":n,"parent_state_hash":digest(state),"prompts_hash":digest(requests),
              "responses_sha256":file_hash(root/f"RESPONSES-{n}.json")}:
        raise ValueError("MoE checkpoint/prompt/response binding")
    if len(responses)!=len(requests):raise ValueError("three actual context responses required")
    for prior in range(1,n):
        old=root/f"RESPONSES-{prior}.json"
        if old.exists() and digest(load_json_strict(old))==digest(responses):raise ValueError("previous raw author response reused")
    successor=copy.deepcopy(state);records=[]
    for context,request,response in zip(successor["contexts"],requests,responses):
        if response["seed"]!=context["seed"] or request["seed"]!=context["seed"]:raise ValueError("MoE seed binding")
        raw=response["response"];ncell=len(raw["cell_ids"]);positions=ncell*ncell+ncell
        for arm in ("llm","random"):
            population=context["arms"][arm];parent=population["champion"]
            charge={"arm":arm,"seed":context["seed"],"round":n,"parent":parent["hash"],"charged":True,
                    "positions":positions,"raw_response":digest(raw) if arm=="llm" else None}
            journal(charge) # Actual raw attempts charged before typed validation/evaluation.
            if arm=="llm":
                tensors=assemble(raw,request["prompt"],state["pool"],request["prompt"]["parent_hash"],core)
            else:
                # Same earned-cell assembly/vocabulary; controls retain their OWN independently selected champion/history.
                tensors=assemble(raw,request["prompt"],state["pool"],request["prompt"]["parent_hash"],core)
                rng=random.Random(context["seed"]+n*1000)
                tensors["route.W"]=[[rng.uniform(-9,9) for _ in range(ncell)] for _ in range(ncell)]
                tensors["route.b"]=[rng.uniform(-9,9) for _ in range(ncell)]
            evidence=score(tensors,rows,core);decision=admission(parent["evidence"],evidence,population["history"],floor)
            birth={"hash":digest(tensors),"tensors":tensors,"parents":[parent["hash"]],"evidence":evidence,
              "decision":decision,"origin":"LLM_literal_router_and_earned_assembly" if arm=="llm" else "matched_random_router",
              "cell_ids":raw["cell_ids"],"pool_snapshot_hash":state["pool_hash"],"charged_positions":positions}
            population["births"].append(birth)
            if decision["accepted"]:population["champion"]=birth;population["history"].append(copy.deepcopy(evidence))
            records.append({"seed":context["seed"],"arm":arm,"candidate":birth["hash"],"solved":evidence["solved"],
                            "oracle":evidence["oracle_solved_diagnostic_only"],"decision":decision})
    successor["round"]=n;return successor,records

def publish(root,state):
    # Conservative physical champion across contexts, not six independent authored experts.
    authored=[c["arms"]["llm"]["champion"] for c in state["contexts"]]
    champion=max(authored,key=lambda c:c["evidence"]["solved"]);tensors=champion["tensors"]
    atomic(root/"FINAL_WEIGHTS.json",{"tensors":tensors})
    config=load_json_strict(root/"CONFIG.json")
    zero={k:(__import__("numpy").zeros_like(v).tolist()) for k,v in tensors.items()}
    _,rows,core,floor=inputs(root,state["initial_anchor"]);zero_score=score(zero,rows,core)
    random_scores=[c["arms"]["random"]["champion"]["evidence"]["solved"] for c in state["contexts"]]
    history_path=root/"ARC_HISTORY.json";history=load_json_strict(history_path) if history_path.exists() else []
    entry={"iteration":state["round"],"solved":champion["evidence"]["solved"],"weights_hash":champion["hash"]}
    if not history or history[-1]["iteration"]!=state["round"]:history.append(entry)
    atomic(history_path,history)
    receipt={"split":"training","task_ids":[r["id"] for r in rows],"weights_file":"FINAL_WEIGHTS.json","evaluator":"arc_eval.py",
      "claimed":{"solved":champion["evidence"]["solved"],"total":len(rows),"per_task":champion["evidence"]["outcomes"]},
      "controls":{"zero_weights":{"solved":zero_score["solved"]},
                  "random_weights":{"solved":max(random_scores),"per_seed":random_scores,"seeds":config["seeds"],
                    "scope":"best surviving matched random-router arm, same earned-cell assembly"}},
      "history":history,"arc1_arc2_merged_by_task_id":True,
      "scope":"100 unique IDs from existing consolidated public ARC-AGI-2 training corpus; development only, not official evaluation",
      "scientific_acceptance":False}
    atomic(root/"ARC_SCORE.json",receipt)

def supervise(root,anchor,external_controller,max_seconds=1800,cadence=60):
    root=pathlib.Path(root).resolve();external_controller=pathlib.Path(external_controller).resolve()
    if external_controller.is_relative_to(root):raise ValueError("external round anchor controller required")
    if not 1<=max_seconds<=1800 or not 30<=cadence<=300:raise ValueError("declared slow-loop resources")
    lock=root/"SERVICE.lock";fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    deadline=time.monotonic()+max_seconds
    waiting=None
    def status(kind,**details):
        row={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),"status":kind,**details}
        atomic(root/"SERVICE_STATUS.json",row);print(json.dumps(row),flush=True)
    try:
        if not (root/"STATE-0.json").exists():
            started=time.monotonic();state=initial_state(root,anchor);atomic(root/"STATE-0.json",state)
            atomic(root/"PROMPTS-1.json",prompts(state));publish(root,state)
            status("ARC_baseline_measured",seconds=time.monotonic()-started,
                   solved=state["contexts"][0]["arms"]["llm"]["champion"]["evidence"]["solved"])
        n=max(i for i in range(4) if (root/f"STATE-{i}.json").exists());state=load_json_strict(root/f"STATE-{n}.json")
        while state["round"]<3 and time.monotonic()<deadline:
            n=state["round"]+1
            if not (root/f"PROMPTS-{n}.json").exists():atomic(root/f"PROMPTS-{n}.json",prompts(state))
            if not external_controller.exists() or not (root/f"RESPONSES-{n}.json").exists():
                waiting="literal_assembly_response"
                status("awaiting_literal_assembly_response",next_round=n);time.sleep(min(cadence,max(0,deadline-time.monotonic())));continue
            controller=load_json_strict(external_controller)
            if controller["initial_anchor"]!=anchor:raise ValueError("external MoE controller anchor mismatch")
            round_anchor=controller.get("round_anchors",{}).get(str(n))
            if not round_anchor:
                waiting="external_round_anchor"
                status("awaiting_external_round_anchor",next_round=n);time.sleep(cadence);continue
            committed=root/f"ROUND-{n}-COMMIT.json"
            if committed.exists():
                saved=load_json_strict(committed)
                if saved["round_anchor"]!=round_anchor or saved["parent_state_hash"]!=digest(state):raise ValueError("recovery commitment mismatch")
                state=saved["state"];atomic(root/f"STATE-{n}.json",state);atomic(root/f"ROUND-{n}.json",saved["records"]);publish(root,state)
                if n<3:atomic(root/f"PROMPTS-{n+1}.json",prompts(state))
                status("committed_round_recovered",round=n);continue
            if (root/f"CHARGES-{n}.jsonl").exists():raise ValueError("partial charged round has no committed result; never recharge")
            status("evaluating_ARC_MoE_round",round=n);started=time.monotonic()
            parent_state_hash=digest(state)
            state,records=phase(root,anchor,round_anchor,state,lambda e:append_charge(root,n,e))
            atomic(root/f"ROUND-{n}-COMMIT.json",{"parent_state_hash":parent_state_hash,"round_anchor":round_anchor,"state":state,"records":records})
            atomic(root/f"STATE-{n}.json",state);atomic(root/f"ROUND-{n}.json",records);publish(root,state)
            if n<3:atomic(root/f"PROMPTS-{n+1}.json",prompts(state))
            status("ARC_MoE_round_complete",round=n,seconds=time.monotonic()-started,records=records)
            time.sleep(min(cadence,max(0,deadline-time.monotonic())))
        status("declared_round_budget_complete" if state["round"]==3 else "bounded_service_deadline",round=state["round"],waiting_for=waiting if state["round"]<3 else None)
    except Exception as error:status("blocked",error=type(error).__name__+":"+str(error));raise
    finally:lock.unlink()

def main():
    p=argparse.ArgumentParser();p.add_argument("root",type=pathlib.Path);p.add_argument("--anchor",required=True)
    p.add_argument("--external-controller",type=pathlib.Path,required=True);p.add_argument("--max-seconds",type=int,default=1800)
    p.add_argument("--cadence",type=int,default=60);a=p.parse_args()
    supervise(a.root,a.anchor,a.external_controller,a.max_seconds,a.cadence)
if __name__=="__main__":main()
