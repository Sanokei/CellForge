"""Literal-weight ARC MoE backend and separate per-task admission.
Training answers are evaluator-only; no query output reaches routing/inference.
"""
import copy,math,pathlib,time
import numpy as np
from .joint_weight_population import digest,experts,predict_routed,expert_predictions,uniform_router
from .typed_population import PaletteExecutor,parity_row_stencil
from .component_count import SHAPES,predict_component_count
from ..runtime.arc_submission import validate_grid

def make_core(path):
    palette=PaletteExecutor(path,10);core=palette.core
    core.PREDICTORS={n:(lambda p,g,trace=False,n=n:(palette.predict(n,p,g),{})) for n in core.PREDICTORS}
    core.SHAPES["parity_row_stencil_v1"]={"stencil.W":(2,3),"foreground.W":(1,10),"activation.b":(1,)}
    core.PREDICTORS["parity_row_stencil_v1"]=lambda p,g,trace=False:(parity_row_stencil(p,g),{})
    core.SHAPES["component_count_unary_v1"]=SHAPES
    core.PREDICTORS["component_count_unary_v1"]=lambda p,g,trace=False:(predict_component_count(p,g),{})
    return core

def validate_weights(tensors,core):
    def walk(value):
        if isinstance(value,list):
            if not value:raise ValueError("nonempty tensors")
            for item in value:walk(item)
        elif type(value) not in (int,float) or not math.isfinite(value) or abs(value)>9:
            raise ValueError("finite literal weights bounded [-9,9]")
    if not isinstance(tensors,dict) or not tensors:raise ValueError("literal joint tensors required")
    for value in tensors.values():walk(value)
    n=len(experts(tensors,core))
    if np.asarray(tensors["route.W"]).shape!=(n,n) or np.asarray(tensors["route.b"]).shape!=(n,):
        raise ValueError("router/cell dimension mismatch")
    return n

def stripped_task(task):
    if set(task)!={"train","test"} or not task["train"] or not task["test"]:raise ValueError("ARC train/test task")
    return {"train":[{"input":validate_grid(p["input"]),"output":validate_grid(p["output"])} for p in task["train"]],
            "test":[{"input":validate_grid(p["input"])} for p in task["test"]]}

def predict(tensors,task,core):
    validate_weights(tensors,core);task=stripped_task(task)
    router={k:tensors[k] for k in ("route.W","route.b")}
    return [[validate_grid(predict_routed(tensors,router,{"demos":task["train"],"input":q["input"]},core))]
            for q in task["test"]]

def score(tensors,rows,core):
    validate_weights(tensors,core);outcomes={};details=[];oracle=uniform=0;started=time.monotonic()
    for row in rows:
        task=row["task"];query_view=stripped_task(task);hits=[];ohits=[];uhits=[];errors=[]
        for q,answer in zip(query_view["test"],task["test"]):
            view={"demos":query_view["train"],"input":q["input"]}
            try:
                pred=predict_routed(tensors,{k:tensors[k] for k in ("route.W","route.b")},view,core)
                hits.append(pred==answer["output"])
                uhits.append(predict_routed(tensors,uniform_router({k:tensors[k] for k in ("route.W","route.b")}),view,core)==answer["output"])
                ohits.append(any(p==answer["output"] for p in expert_predictions(tensors,q["input"],core)))
            except (ValueError,TypeError,IndexError,KeyError) as error:
                hits.append(False);uhits.append(False);ohits.append(False);errors.append(type(error).__name__+":"+str(error))
        outcomes[row["id"]]=all(hits);oracle+=all(ohits);uniform+=all(uhits)
        details.append({"id":row["id"],"correct":all(hits),"oracle_correct_diagnostic_only":all(ohits),
                        "uniform_correct":all(uhits),"errors":errors})
    return {"outcomes":outcomes,"solved":sum(outcomes.values()),"total":len(rows),"details":details,
            "oracle_solved_diagnostic_only":oracle,"uniform_solved":uniform,
            "seconds":time.monotonic()-started,"partition":"development"}

def measured_pool(tensors,rows,core):
    """Typed existing grid cells earn assembly through actual demo reconstruction."""
    pool={"cells":{},"live":{},"source_joint":digest(tensors),"kind":"measured-existing-grid-founders"}
    for prefix,structure,params in experts(tensors,core):
        cid=digest({"structure":structure,"params":params});outcomes={}
        for row in rows:
            for i,pair in enumerate(row["task"]["train"]):
                try:value=core.PREDICTORS[structure](params,pair["input"],False)[0];correct=value==pair["output"]
                except (ValueError,IndexError,TypeError):correct=False
                outcomes[row["id"]+":"+str(i)]=correct
        earned=any(outcomes.values())
        pool["cells"][cid]={"id":cid,"structure":structure,"params":copy.deepcopy(params),"parents":[],
            "state":"live" if earned else "failed-founder","scores":{"development_demonstrations":outcomes},
            "source_prefix":prefix,"source_joint":digest(tensors)}
        if earned:pool["live"].setdefault(structure,[]).append(cid)
    return pool

def primitive_pool(tensors,manifest,core):
    """Earn grid-primitive eligibility on declared niche tasks, not complete ARC."""
    if manifest.get("schema")!="grid-primitive-admission/1":raise ValueError("primitive manifest schema")
    pool={"cells":{},"live":{},"source_joint":digest(tensors),"kind":"declared-small-skill-founders","manifest_hash":digest(manifest)}
    for prefix,structure,params in experts(tensors,core):
        rows=manifest["development"].get(structure,[])
        outcomes={};zeros={};nontrivial=0
        zero={k:np.zeros_like(v).tolist() for k,v in params.items()}
        for row in rows:
            if row["partition"]!="development":raise ValueError("reporting cannot select primitive")
            grid=validate_grid(row["input"]);target=validate_grid(row["expected"])
            nontrivial+=target!=grid
            for weights,result in ((params,outcomes),(zero,zeros)):
                try:result[row["id"]]=core.PREDICTORS[structure](weights,grid,False)[0]==target
                except (ValueError,TypeError,IndexError):result[row["id"]]=False
        earned=len(rows)>=8 and nontrivial>=4 and sum(outcomes.values())/len(rows)>=.9 and sum(outcomes.values())>sum(zeros.values())
        cid=digest({"structure":structure,"params":params})
        pool["cells"][cid]={"id":cid,"structure":structure,"params":copy.deepcopy(params),"parents":[],"state":"live" if earned else "failed-founder",
          "scores":{"development_demonstrations":outcomes,"zero_control":zeros},"source_prefix":prefix,"source_joint":digest(tensors),
          "eligibility_scope":"declared grid primitives only; no ARC claim","manifest_hash":digest(manifest)}
        if earned:pool["live"].setdefault(structure,[]).append(cid)
    return pool

def assemble(response,prompt,pool,parent_hash,core):
    if response.get("parent_hash")!=parent_hash or response.get("pool_hash")!=digest(pool):
        raise ValueError("assembly parent/pool binding")
    if response.get("prompt_hash")!=digest(prompt):raise ValueError("assembly prompt binding")
    if response.get("model")!="gpt-6.1-sol" or response.get("reasoning")!="low":
        raise ValueError("literal task-maker provenance")
    if not response.get("rationale"):raise ValueError("justified LLM assembly decision")
    ids=response["cell_ids"]
    if not isinstance(ids,list) or len(ids)<2 or len(ids)!=len(set(ids)):raise ValueError("at least two distinct earned cells")
    tensors={}
    for index,cid in enumerate(ids):
        if cid not in pool["cells"] or pool["cells"][cid]["state"]!="live":raise ValueError("unearned/retired assembly cell")
        cell=pool["cells"][cid]
        if digest({"structure":cell["structure"],"params":cell["params"]})!=cid:raise ValueError("cell content identity")
        if not any(cell["scores"]["development_demonstrations"].values()):raise ValueError("untested cell cannot assemble")
        for name,value in cell["params"].items():tensors[str(index)+":"+name]=copy.deepcopy(value)
    tensors.update(copy.deepcopy(response["router"]))
    validate_weights(tensors,core)
    if response.get("charged_positions")!=len(ids)**2+len(ids):raise ValueError("full literal router authoring charge")
    return tensors

def admission(parent,candidate,history,floor):
    if parent["partition"]!="development" or candidate["partition"]!="development":raise ValueError("reporting cannot select MoE")
    if parent["outcomes"].keys()!=candidate["outcomes"].keys():raise ValueError("matched full task IDs")
    reasons=[]
    if candidate["solved"]<=parent["solved"]:reasons.append("strict_ARC_development_gain")
    if candidate["solved"]/max(1,candidate["total"])<floor:reasons.append("owner_ARC_floor")
    for old in [parent,*history]:
        if old["outcomes"].keys()!=candidate["outcomes"].keys():raise ValueError("ARC history task identities differ")
        if any(hit and not candidate["outcomes"][tid] for tid,hit in old["outcomes"].items()):
            reasons.append("historical_full_task_regression")
    return {"accepted":not reasons,"reasons":reasons,"scope":"official public-training task competition; no evaluation-split transfer claim"}
