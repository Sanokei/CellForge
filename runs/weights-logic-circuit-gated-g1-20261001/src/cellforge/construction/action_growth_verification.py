"""Independent two-phase action growth replay; artifact source is parsed, never run."""
import pathlib,json,hashlib,copy,random,tempfile,ast,re
from .action_argument_growth_cell import infer_joint,execute_packet
from .action_population import ActionCellPopulation,ActionOpponentHistory
from .joint_weight_population import digest

def recover_cases(source):
    tree=ast.parse(source.decode("utf-8-sig"))
    assignment=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="report" for t in n.targets))
    if not isinstance(assignment,ast.List):raise ValueError("literal report cases required")
    rows=[]
    for node in assignment.elts:
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!="c" or node.keywords:raise ValueError("only literal frozen case declarations supported")
        args=[ast.literal_eval(n) for n in node.args]
        if len(args) not in (4,5):raise ValueError("exact frozen case arity")
        initial=args[4] if len(args)==5 else {}
        rows.append({"id":args[0],"request":args[1],"initial":initial,"expected":{args[2]:args[3]},"action":"edit_in_place" if initial else "write_file"})
    return rows

def verify_growth(root):
    errors=[];quality=[];binding={};counts=[];report_scores=[]
    def read(n):return json.loads((root/n).read_text(encoding="utf-8-sig"))
    def check(test,reason):
        if not test:errors.append(reason)
    try:
        hashes=read("FINAL_INTEGRITY.json")
        for name,expected in hashes.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():raise ValueError("invalid bound artifact")
            binding[name]=hashlib.sha256(p.read_bytes()).hexdigest();check(binding[name]==expected,"stale_or_tampered:"+name)
        manifest=read("MANIFEST.json");recovery=read("REPORTING_MANIFEST_RECOVERY.json")
        report=recover_cases((root/"run.py").read_bytes())
        check(report==recovery["cases"] and hashlib.sha256((root/"run.py").read_bytes()).hexdigest()==recovery["source_sha256"],"reporting_recovery_differs")
        check(recovery["source_equals_original_prephase_bundle"] is True,"prephase_source_provenance_missing")
        for source in ("action_argument_growth_cell","action_argument_cell","action_population","cell_population","tool_router_evolution","action_packet"):
            current=pathlib.Path(__file__).parent/(source+".py");native=root/"src/cellforge/construction"/(source+".py")
            check(current.read_bytes()==native.read_bytes(),"installed_source_differs:"+source)
        check(manifest["capacity"]==[33,39] and manifest["edit_ranges"]==[[-9,9],[-9,9]],"capacity_or_edit_vocabulary_differs")
        check(manifest["budgets"]=={"llm":2,"random":2} and manifest["seeds"]==[101,202,303],"budget_or_seed_contract_differs")
        for section,rows in [(n,manifest[n]) for n in ("fresh","archive","safety")]+[("reporting",report)]:
            if not isinstance(rows,list):raise ValueError("case list required")
            for row in rows:
                tokens=[m.group()[1:-1] if m.group()[0] in "'\"" else m.group().strip("?!,;:").rstrip(".") for m in re.finditer(r"""'[^']*'|"[^"]*"|\S+""",row["request"])]
                for name,text in row["expected"].items():
                    if row["initial"].get(name)!=text and name not in tokens:quality.append("fixture_filename_not_copyable:"+section+":"+row["id"])
        base=read("baseline.json");prior=read("parent-result.json");routerauthor=read("router-author.json");router={"W":routerauthor["W"],"b":routerauthor["b"]}
        cache={}
        def effects(params,rows):
            key=digest([params,rows])
            if key in cache:return copy.deepcopy(cache[key])
            out=[]
            for row in rows:
                with tempfile.TemporaryDirectory(prefix="growth_replay_") as d:
                    p=pathlib.Path(d)
                    for name,text in row["initial"].items():
                        if pathlib.Path(name).name!=name or any(c in name for c in ("/","\\","..")):raise ValueError("unsafe initial path")
                        (p/name).write_text(text,encoding="utf-8")
                    prediction=infer_joint(row["request"],params,router,base["router"],interface_version="tool-action-joint/2")
                    result=execute_packet(prediction["packet"],allowed_root=p);files={f.name:f.read_text(encoding="utf-8") for f in p.iterdir() if f.is_file()}
                    out.append({"id":row["id"],"prediction":prediction,"execution":result,"files":files,"correct":files==row["expected"] and prediction["action"]==row["action"],"files_correct":files==row["expected"]})
            cache[key]=out;return copy.deepcopy(out)
        def scores(params):
            negative=effects(params,manifest["safety"]);out={}
            for label,key in (("fresh","fresh"),("archive","archive"),("anchor","safety")):
                rows=manifest[key];traces=effects(params,rows)
                out[label]={"partition":"development","manifest":digest(rows),"source":"isolated-action-fixture-safety" if label=="anchor" else "isolated-action-fixture-development","outcomes":{r["id"]:r["correct"] for r in traces},
                 "false_positive_given_consistency":sum(not r["files_correct"] for r in negative)/len(negative),"traces":traces,"rate_definition":"false file effects among four safety fixtures, shield included"}
            return out
        finalcontexts=None
        for phase in (1,2):
            actual=read(f"GENERATION-{phase}.json");proposal=read("proposal"+str(phase+3)+".json")
            params={"argument.W":proposal["tensors"]["slot.W"],"argument.b":proposal["tensors"]["null.b"]};width=len(params["argument.W"][0]);positions=3*width+3
            check(positions==manifest["capacity"][phase-1],"maker_shape_or_charged_capacity_differs")
            expectedcontexts=[]
            for idx,seed in enumerate(manifest["seeds"]):
                snapshot=copy.deepcopy(prior["contexts"][idx]["population"] if phase==1 else finalcontexts[idx]["population"])
                p=ActionCellPopulation();p.cells=snapshot["cells"];p.live=snapshot["live"];p.events=snapshot["events"];p.charges=snapshot["charges"];p.histories={k:ActionOpponentHistory(v) for k,v in snapshot["histories"].items()}
                family="action-arguments-lexical"
                if phase==1:
                    for cell in p.cells.values():
                        if cell["structure"]==family:cell["scores"]=scores(cell["params"])
                    p.histories[family]=ActionOpponentHistory([dict(r,scores=scores(p.cells[r["graph_hash"]]["params"])) for r in p.histories[family].records])
                recipient=p.live[family][0];before=copy.deepcopy(p.cells[recipient]["scores"]);oldhistory=copy.deepcopy(p.histories[family].records)
                decision=p.birth(family,params,[recipient],{"raw_response":f"proposal{phase+3}.json","model":proposal["model"],"newly_authored":str(positions)+" full-reauthoring positions","typed_io":str(width)+" lexical features -> copied spans"},
                  scores(params),caps_passed=True,fixtures_passed=True,arm="llm-growth")
                r=random.Random(seed+1000*phase);v=[r.uniform(-9,9) for _ in range(positions)];controlparams={"argument.W":[v[i*width:(i+1)*width] for i in range(3)],"argument.b":v[3*width:]}
                control=ActionCellPopulation();control.seed(family,p.cells[recipient]["params"],p.cells[recipient]["provenance"],before);control.histories[family]=ActionOpponentHistory(oldhistory)
                randomdecision=control.birth(family,controlparams,[recipient],{"source":"matched_full_matrix","seed":seed,"phase":phase},scores(controlparams),caps_passed=True,fixtures_passed=True,arm="random-growth")
                expectedcontexts.append({"seed":seed,"parent":recipient,"decision":decision,"random_decision":randomdecision,"population":p.snapshot(),"random_population":control.snapshot()})
            check(expectedcontexts==actual["contexts"],"phase_membership_lineage_selection_or_numeric_control_differs:"+str(phase))
            finalcontexts=expectedcontexts
        expectedrows=[]
        def depth(cid,cells):
            parents=cells[cid]["parents"];return 0 if not parents else 1+max(depth(p,cells) for p in parents)
        for context in finalcontexts:
            p=context["population"];cid=p["live"]["action-arguments-lexical"][0];rows=effects(p["cells"][cid]["params"],report);expectedrows.append(rows)
            report_scores.append({"seed":context["seed"],"correct":sum(r["correct"] for r in rows),"total":len(rows)})
            counts.append({"seed":context["seed"],"live":{k:len(v) for k,v in p["live"].items()},"selected_argument":cid,"ancestry_depth":depth(cid,p["cells"])})
        check(expectedrows==read("REPORTING.json")["rows"],"reserved_reporting_replay_differs")
    except (OSError,ValueError,KeyError,TypeError,IndexError,StopIteration) as e:errors.append("missing_or_invalid_evidence:"+str(e))
    errors.extend(quality)
    return {"schema":"cellforge-verified-action-trial/1","kind":"bounded-action-growth","artifact_binding":binding,
      "mechanism_replay":{"passed":not errors,"errors":errors},"evidence_quality":{"passed":not quality,"errors":quality},
      "protocol_quality":{"passed":False,"errors":["reporting declaration recovered from frozen source","growth pre-run gate not executed"]},
      "population_replay":{"counts":counts},"reporting":report_scores,"runtime_acceptance":False,
      "scientific_acceptance":False,"scientific_headline_allowed":False,
      "scope":"Independent replay of two literal-weight generations and first reporting; finite copied-span grammar, protocol recovery disclosed"}
