"""Preregister literals/controls locally. This does not admit or play a study."""
import copy
import pathlib
import random
import arc3cell as C
import policy_transition as P
import scripted_explorer as S
from common import *

ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT/"prior-evidence"/"legacy"
if not OLD.is_dir(): OLD=ROOT.parent/"continuous-cellforge"/"evidence"
LAB_RUNS="/srv/cellforge-inbox/windows-migration-20261002/snapshot/repo/runs"
RESET_CONTRACT="E.play resets cost one action; per-game memory clears on public reset; level advance retains nodes and eligibility; blind hides frame and key but retains public level/reset; no hidden queries"

def arms(params,random_params,reference):
    source=P.SOURCE
    return [
      {"name":"full","source":source,"params":params},
      {"name":"matched_random","source":source,"params":random_params},
      {"name":"zero","source":source,"params":{k:[0]*n for k,n in SHAPES.items()}},
      {"name":"action_off","source":source,"params":dict(params,effect=[0]*8,action=[0]*8,lookahead=[0]*4)},
      {"name":"memory_off","source":source,"params":dict(params,memory_gate=[0])},
      {"name":"context_off","source":source,"params":dict(params,context_gate=[0])},
      {"name":"blind","source":source,"params":params,"blind":True},
      {"name":"frozen_spatial","source":reference,"params":{"action":[0,6,5,2,1,4,3,1],"pixel":PIXEL,"rates":[0,1]}},
      {"name":"scripted_explorer","source":S.SOURCE,"params":{"pixel":PIXEL}}]
def standard(founders):
    # Standard proposal count is distinct from the literal-position charge.
    entries=[]; proposals=[]; tensors={}
    for f in founders:
        for arm,key in (("llm","params"),("random","random_params")):
            p=f[key]; h=digest(p); filename="tensors/"+h+".json"; tensors[filename]=p
            entry={"hash":h,"niche":"arc3_shared_transition","origin":"llm_founder" if arm=="llm" else "random_mutation",
                   "parents":[] if arm=="llm" else [digest(f["params"])],"weights_file":filename,"accuracy":None,"generation":0 if arm=="llm" else 1,"arm":arm,"provenance":arm+"-"+h[:16]}
            entries.append(entry)
            proposals.append({"arm":arm,"seed":f["construction_seed"],"hash_or_null":h,"charged":True,
                              "outcome":"accepted","literal_positions_charged":32,"evaluation_replay_charge":0})
    fs=[dict(e,author="llm",actor="host Codex assistant",rationale=founders[i]["rationale"])
        for i,e in enumerate([x for x in entries if x["arm"]=="llm"])]
    manifest={"contract":"docs/ONE_STANDARD.md; schema-compatible unqualified development pilot",
      "evaluator":"standard_interactive_adapter","arms":{"llm":{"kind":"explicit host assistant literals"},
      "random":{"kind":"matched free-support replacement mutation of paired founder; frozen once"}},"seeds":SEEDS,
      "seeds_declared_before_run":True,"budget_per_arm":1,"niches":["arc3_shared_transition"],
      "founders":fs,"genomes":entries,"proposals":proposals,"eval_seeds":SEEDS,
      "qualified":False,"status":"preregistered_not_run","accuracy_metric":None,
      "construction_seed_semantics":"preregistered founder/random pairing IDs; one host assistant context, not three independent LLM draws",
      "evaluation_seed_semantics":"each frozen pair replayed at three environment seeds, no repeated proposal charge",
      "literal_positions_per_primary":32,"literal_positions_charged":{"llm":96,"random":96},
      "fixed_positions":["pixel[0:6]","memory_gate[0]","context_gate[0]"],
      "free_support":{"effect":[-8,8],"action":[-8,8],"lookahead":[-8,8],"rates":[-3,3],"mix":[-3,3]},
      "invalid_proposal_contract":"charged once then rejected, no clipping/redraw; validation failure prevents admission",
      "semantic_milestones_unmet":["four niches","100 distinct genomes","mutation/crossover lineage","router qualification",
         "accuracy recomputation interface is unavailable for interactive ARC3; adapter raises an explicit unsupported error"],
      "remote_model_calls":0,"host_context_hash":None,"maker_seen_seeds":SEEDS,
      "random_vocab":{"tensors":list(SHAPES),"low":-8,"high":8,"per_tensor_bounds":{"rates":[-3,3],"mix":[-3,3]},
                      "fixed_inherited_positions":["pixel[0:6]","memory_gate[0]","context_gate[0]"]},"ledger_file":"ledger.jsonl"}
    return manifest,tensors

def main():
    old=read(OLD/"PLAN.json"); environment=read(OLD/"ENVIRONMENT.json")
    reference=(ROOT/"reference_spatial.py").read_text().split("def act",1)[1]
    reference="def act"+reference
    values=[
      ("contextual_transition",[1,1,1,1,2,2,.5,.5],[0,3,6,5,4,2,2,1],[0,1],[2,0],[2,3,1,2]),
      ("global_effect_uncertainty",[1,1,1,1,2,2,.5,.5],[0,4,6,5,3,5,2,1],[1,0],[-2,0],[0,0,1,0]),
      ("cycle_recovery",[1,1,1,1,2,2,1,1],[0,2,8,6,6,3,5,2],[0,2],[2,1],[2,5,1,5])]
    founders=[]
    for i,(name,effect,action,rates,mix,look) in enumerate(values):
        params=dict(effect=effect,action=action,rates=rates,mix=mix,lookahead=look,
                    pixel=PIXEL,memory_gate=[1],context_gate=[1])
        rng=random.Random(2026101100+i)
        random_params={}
        for k,n in SHAPES.items():
            bound=3 if k in ("rates","mix") else 8
            random_params[k]=[rng.uniform(-bound,bound) for _ in range(n)]
        for k in ("pixel","memory_gate","context_gate"): random_params[k]=copy.deepcopy(params[k])
        founders.append({"name":name,"params":params,"random_params":random_params,
                         "construction_seed":SEEDS[i],"random_draw_seed":2026101100+i,
                         "rationale":name+" generic action-effect, context and delayed-progress coefficient hypothesis"})
    plan={"schema":"cellforge-plan/2","status":"PREPARED_ONLY","qualified":False,"study_cap":9,
          "limits":LIMITS,"remote_model_calls":0,"lab_runs_root":LAB_RUNS,
          "packages":environment["packages"],"dataset_sha256":old["dataset_sha256"],
          "dataset_files":old["dataset_files"],"protected_stops":old["protected_stops"],
          "founders":founders,"conditions":[],"acceptance":"same frozen founder, >=2/3 seeds gains on a multi-action game over zero/random/action_off/memory_off/context_off/spatial/scripted; no other-game regression vs spatial; complete matched initial states and budgets"}
    for f in founders:
        for seed in SEEDS:
            config={"label":f["name"],"arms":arms(f["params"],f["random_params"],reference),
               "games":GAMES,"seed":seed,"actions":1500,"dataset_sha256":old["dataset_sha256"],
               "packages":environment["packages"],"reset_contract":RESET_CONTRACT,
               "evaluator_ast":normalized_source((ROOT/"arc3_env.py").read_text())}
            config["fingerprint"]=science_identity(config); plan["conditions"].append(config)
    validate_plan(plan)
    for arm in plan["conditions"][0]["arms"]:
        ok,why=C.validate(arm["source"])
        if not ok: raise Blocked(arm["name"]+" sandbox validation: "+why)
    atomic(ROOT/"FOUNDERS.json",founders); atomic(ROOT/"PLAN.json",plan)
    atomic(ROOT/"SCIENCE_FINGERPRINTS.json",[{"label":c["label"],"seed":c["seed"],"fingerprint":c["fingerprint"]} for c in plan["conditions"]])
    state=import_legacy(fresh_registry(),OLD)
    atomic(ROOT/"LEGACY_REGISTRY_PREVIEW.json",state)
    manifest,tensors=standard(founders)
    preview=ROOT/"preregistration"/("weights-transition-"+digest(plan))
    preview.mkdir(parents=True,exist_ok=True); (preview/"tensors").mkdir(exist_ok=True)
    atomic(preview/"STANDARD.json",manifest)
    for filename,params in tensors.items(): atomic(preview/filename,params)
    (preview/"provenance").mkdir(exist_ok=True)
    events=[]; previous="GENESIS"
    for f in founders:
        for arm,key in (("llm","params"),("random","random_params")):
            params=f[key]; h=digest(params); pid=arm+"-"+h[:16]
            provenance={"id":pid,"kind":"founder" if arm=="llm" else "random_mutation","arm":arm,
              "seed":f["construction_seed"],"niche":"arc3_shared_transition",
              "actor":"host Codex assistant" if arm=="llm" else "Python random.Random frozen replacement operator",
              "ts_outbox":None,"ts_inbox":None,"ts_eval":None,
              "timestamp_status":"not fabricated: host platform outbox/inbox unavailable and evaluation not run",
              "shown":{"counterexample_inputs":[],"context_sha256":None},
              "response":{"tensors":params},"rationale":f["rationale"],
              "rng_seed":None if arm=="llm" else f["random_draw_seed"],"remote_model_calls":0}
            atomic(preview/"provenance"/(pid+".json"),provenance)
            record={"prev":previous,"event":{"type":"proposal","arm":arm,"seed":f["construction_seed"],
                    "hash":h,"provenance":pid,"charged":True,"literal_positions":32,"outcome":"accepted"}}
            line=canonical(record).decode(); events.append(line); previous=hashlib.sha256(line.encode()).hexdigest()
    (preview/"ledger.jsonl").write_text("\n".join(events)+"\n",encoding="utf-8")
    print("PREPARED_ONLY: 9 conditions, 27 plays/job, 364500 max actions; 96+96 positions once; zero ARC plays/model calls")
if __name__=="__main__": main()

