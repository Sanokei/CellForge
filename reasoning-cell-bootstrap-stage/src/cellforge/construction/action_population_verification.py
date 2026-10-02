"""Population replay from data, using installed typed inference and selection code."""
import copy,hashlib,json,pathlib,random,tempfile
from .action_argument_cell import infer_joint,execute_packet
from .action_population import ActionCellPopulation
from .joint_weight_population import digest
from .tool_router_evolution import route

def verify_population(root):
    errors=[];binding={};counts=[]
    def read(n):return json.loads((root/n).read_text(encoding="utf-8-sig"))
    def check(condition,name):
        if not condition:errors.append(name)
    try:
        manifest=read("MANIFEST.json");result=read("RESULT.json");base=read("baseline.json")
        ra=read("router-author.json");rp=read("router-parent.json");rc=read("router-corpora.json")
        gate=read("RUNTIME_GATE_RESULT.json");sources=read("RUN_GATE.json")["source_hashes"]
        check(gate.get("passed") is True and gate.get("executed") is True,"runtime_gate_not_executed_or_failed")
        check(gate.get("source_binding")==sources,"runtime_source_binding_differs")
        for name,expected in sources.items():
            installed=pathlib.Path(__file__).parent/(name+".py")
            check(installed.is_file() and hashlib.sha256(installed.read_bytes()).hexdigest()==expected,"installed_source_differs:"+name)
        # Final manifest is created AFTER stdout closes. Any missing/stale file fails.
        integrity=read("FINAL_INTEGRITY.json")
        for name,expected in integrity.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError("invalid integrity path")
            actual=hashlib.sha256(path.read_bytes()).hexdigest();binding[name]=actual
            check(actual==expected,"stale_or_tampered_artifact:"+name)
        for name in ("MANIFEST.json","RESULT.json","router-author.json","router-parent.json","router-corpora.json","RUN_GATE.json","RUNTIME_GATE_RESULT.json"):
            check(name in integrity,"missing_bound_artifact:"+name)
        check(manifest["charged_positions"]["router"]=={"llm":128,"random":128},"router_charged_capacity_not_equal")
        check(manifest["charged_positions"]["argument"]=={"llm":[27,27,30],"random":[27,27,30]},"argument_charged_capacity_not_equal")
        check(manifest["budget"]=={"argument":{"llm":3,"random":3},"router":{"llm":1,"random":1}},"proposal_budget_differs")
        check(len(set(manifest["seeds"]))>=3 and [c["seed"] for c in result["contexts"]]==manifest["seeds"],"seed_contexts_differs")
        router={"W":ra["W"],"b":ra["b"]}
        check(all(router["W"][r][c]==rp["W"][r][c] for r in range(7) for c in range(18)) and router["b"][:7]==rp["b"][:7],"router_inherited_material_changed")
        proposals=[read("proposal"+str(g)+".json") for g in (1,2,3)]
        for g in (1,2,3):check("proposal"+str(g)+".json" in integrity,"unbound_maker_response")
        cache={}
        def effects(params,rparams,rows):
            key=digest([params,rparams,rows])
            if key in cache:return copy.deepcopy(cache[key])
            out=[]
            for c in rows:
                with tempfile.TemporaryDirectory(prefix="population_verify_") as d:
                    p=pathlib.Path(d)
                    for n,v in c["initial"].items():
                        if pathlib.Path(n).name!=n or any(x in n for x in ("/","\\","..")):raise ValueError("unsafe initial fixture")
                        (p/n).write_text(v,encoding="utf-8")
                    prediction=infer_joint(c["request"],params,rparams,base["router"],interface_version=manifest["joint_interface"])
                    execution=execute_packet(prediction["packet"],allowed_root=p)
                    files={f.name:f.read_text(encoding="utf-8") for f in p.iterdir() if f.is_file()}
                    out.append({"id":c["id"],"correct":files==c["expected"] and prediction["action"]==c["action"],"files_correct":files==c["expected"],
                                "prediction":prediction,"execution":execution,"files":files})
            cache[key]=out;return copy.deepcopy(out)
        def argscore(params):
            out={};negative=effects(params,router,manifest["safety"])
            for label,key in (("fresh","fresh"),("archive","archive"),("anchor","safety")):
                rows=manifest[key];traces=effects(params,router,rows)
                out[label]={"partition":"development","manifest":digest(rows),"source":"isolated-action-fixture-safety" if label=="anchor" else "isolated-action-fixture-development",
                 "outcomes":{r["id"]:r["correct"] for r in traces},"false_positive_given_consistency":sum(not r["files_correct"] for r in negative)/len(negative),"traces":traces,
                 "rate_definition":"wrong file effects among four explicit safety requests; policy shield included, not neural safety"}
            return out
        def rscore(params):
            out={}
            for label,key in (("fresh","fresh"),("archive","development"),("anchor","anchor")):
                rows=rc[key];ps=[route(params,r["request"]) for r in rows];neg=[(r,p) for r,p in zip(rows,ps) if r["expected"]=="none"]
                out[label]={"partition":"development","manifest":digest(rows),"source":"isolated-action-fixture-safety" if label=="anchor" else "isolated-action-router-development",
                  "outcomes":{r["id"]:p["action"]==r["expected"] for r,p in zip(rows,ps)},
                  "false_positive_given_consistency":sum(p["action"]!="none" for r,p in neg)/len(neg) if neg else 0,
                  "rate_definition":"wrong selected action among explicit none phrases","negative_count":len(neg)}
            return out
        for context in result["contexts"]:
            seed=context["seed"];rand=random.Random(seed);p=ActionCellPopulation()
            initial={"argument.W":[base["slots"]["W"][r] for r in ("arg1","arg2","file")],"argument.b":[.5]*3}
            aid=p.seed("action-arguments-lexical",initial,{"model":"Claude Sonnet5.5","source":"baseline.json","typed_io":"token features -> ordered copied spans"},argscore(initial))
            rid=p.seed("action-router-keywords",rp,{"source":"router-parent.json","model":"historical Claude literal router"},rscore(rp))
            decision=p.birth("action-router-keywords",router,[rid],{"raw_response":"router-author.json","model":ra["model"],"newly_authored":"W/b changed positions","typed_io":"28 keyword features -> 9 actions"},
                rscore(router),caps_passed=True,fixtures_passed=True,arm="llm-router")
            check(decision==context["router_decision"],"router_selection_differs:"+str(seed))
            params=copy.deepcopy(rp)
            for r in range(9):
                for c in range(28):
                    if r>=7 or c>=18:params["W"][r][c]=rand.uniform(-10,10)
            for r in (7,8):params["b"][r]=rand.uniform(-10,10)
            control=ActionCellPopulation();control.seed("action-router-keywords",rp,{"source":"router-parent.json","model":"historical Claude"},rscore(rp))
            decision=control.birth("action-router-keywords",params,[rid],{"source":"random_seed","seed":seed},rscore(params),caps_passed=True,fixtures_passed=True,arm="random-router")
            check(decision==context["router_random"] and control.snapshot()==context["router_control_population"],"router_control_or_budget_differs:"+str(seed))
            check(len(context["stages"])==3,"generation_count_differs")
            for g,proposal,stage in zip((1,2,3),proposals,context["stages"]):
                recipient=p.live["action-arguments-lexical"][0];params={"argument.W":proposal["tensors"]["slot.W"],"argument.b":proposal["tensors"]["null.b"]}
                width=len(params["argument.W"][0]);positions=3*width+3
                check(positions==(30 if g==3 else 27),"argument_capacity_differs")
                check(stage["generation"]==g and stage["recipient"]==recipient,"live_parent_lineage_differs")
                decision=p.birth("action-arguments-lexical",params,[recipient],
                    {"raw_response":f"proposal{g}.json","model":proposal["model"],"newly_authored":str(positions)+" charged literal positions","typed_io":str(width)+" lexical features -> copied spans"},
                    argscore(params),caps_passed=True,fixtures_passed=True,arm="llm-argument")
                check(decision==stage["authored_decision"],"argument_selection_differs")
                bound=9 if g==3 else 8;values=[rand.uniform(-bound,bound) for _ in range(positions)]
                params={"argument.W":[values[i*width:(i+1)*width] for i in range(3)],"argument.b":values[3*width:]}
                control=ActionCellPopulation();control.seed("action-arguments-lexical",p.cells[recipient]["params"],p.cells[recipient]["provenance"],p.cells[recipient]["scores"])
                control.histories["action-arguments-lexical"].records=copy.deepcopy(p.histories["action-arguments-lexical"].records[:-1] if decision["accepted"] else p.histories["action-arguments-lexical"].records)
                randomdecision=control.birth("action-arguments-lexical",params,[recipient],{"source":"matched_random","seed":seed,"generation":g},
                    argscore(params),caps_passed=True,fixtures_passed=True,arm="random-argument")
                check(randomdecision==stage["random_decision"] and control.snapshot()==stage["control_population"],"argument_control_or_budget_differs")
            check(p.snapshot()==context["population"],"membership_birth_retirement_archive_differs:"+str(seed))
            chosen={"argument":p.live["action-arguments-lexical"][0],"router":p.live["action-router-keywords"][0]}
            check(chosen==context["selected"],"selected_ids_differs")
            reporting=effects(p.cells[chosen["argument"]]["params"],p.cells[chosen["router"]]["params"],manifest["reporting"])
            check(reporting==context["reporting"] and sum(r["correct"] for r in reporting)==context["report_correct"],"joint_reporting_differs")
            counts.append({"seed":seed,"live":{k:len(v) for k,v in p.live.items()},"admitted":sum(e["event"]=="admitted" for e in p.events),"retired":sum(e["event"]=="retired" for e in p.events)})
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:errors.append("missing_or_invalid_evidence:"+str(e))
    core_passed=not errors
    label_errors=[]
    try:
        import re
        for section in ("fresh","archive","safety","reporting"):
            for case in manifest[section]:
                tokens=[]
                for match in re.finditer(r"""'[^']*'|"[^"]*"|\S+""",case["request"]):
                    raw=match.group();tokens.append(raw[1:-1] if raw[0] in "'\"" else raw.strip("?!,;:").rstrip("."))
                for name,text in case["expected"].items():
                    if case["initial"].get(name)!=text and name not in tokens:
                        label_errors.append("fixture_filename_not_copyable:"+section+":"+case["id"]+":"+name)
    except (KeyError,TypeError,NameError) as e:label_errors.append("fixture_quality_invalid:"+str(e))
    errors.extend(label_errors)
    return {"schema":"cellforge-verified-action-trial/1","kind":"independent-population","artifact_binding":binding,
       "mechanism_replay":{"passed":not errors,"errors":errors},"evidence_quality":{"passed":not label_errors,"errors":label_errors},"population_replay":{"passed":core_passed,"counts":counts,"checks":["numeric maker response binding","matched edit masks/ranges/budgets","live parent lineage","executed selection and historical opponents","births/retirements/memberships","actual joint packets/file effects"]},
       "runtime_acceptance":not errors,"scientific_acceptance":False,"scientific_headline_allowed":False,
       "scope":"controlled lexical action niche only; not full architecture, evolved topology or ARC transfer"}
