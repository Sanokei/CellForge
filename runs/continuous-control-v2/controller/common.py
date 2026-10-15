"""Canonical science identity, durable global registry, and bounded admission."""
import ast
import copy
import hashlib
import json
import math
import os
import pathlib
import time
from contextlib import contextmanager

GAMES = ["ar25", "lp85", "sc25"]
SEEDS = [0, 2, 3]
SHAPES = {"effect":8,"action":8,"rates":2,"mix":2,"lookahead":4,"pixel":6,"memory_gate":1,"context_gate":1}
PIXEL = [6,1,2,6,0,0]
LIMITS = {"cpu_quota":"100%","cpu_affinity":"0","memory_max":1073741824,
          "runtime_max_seconds":600,"play_wall_seconds":45,"disk_floor":5*1024**3,
          "run_bytes_max":64*1024**2,"total_bytes_max":256*1024**2}
FAILURES = {"failed","interrupted","resource_blocked"}
class Blocked(RuntimeError): pass

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(obj): return hashlib.sha256(canonical(obj)).hexdigest()
def filehash(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
def atomic(path,obj):
    path=pathlib.Path(path)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as stream:
        stream.write(canonical(obj)+b"\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,path)
    if os.name=="posix":
        fd=os.open(path.parent,os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
SEMANTIC_AST_FIELDS = {"AST":[],"Add":[],"And":[],"AnnAssign":["target","annotation","value","simple"],"Assert":["test","msg"],"Assign":["targets","value","type_comment"],"AsyncFor":["target","iter","body","orelse","type_comment"],"AsyncFunctionDef":["name","args","body","decorator_list","returns","type_comment"],"AsyncWith":["items","body","type_comment"],"Attribute":["value","attr","ctx"],"AugAssign":["target","op","value"],"AugLoad":[],"AugStore":[],"Await":["value"],"BinOp":["left","op","right"],"BitAnd":[],"BitOr":[],"BitXor":[],"BoolOp":["op","values"],"Break":[],"Bytes":["s"],"Call":["func","args","keywords"],"ClassDef":["name","bases","keywords","body","decorator_list"],"Compare":["left","ops","comparators"],"Constant":["value","kind"],"Continue":[],"Del":[],"Delete":["targets"],"Dict":["keys","values"],"DictComp":["key","value","generators"],"Div":[],"Ellipsis":[],"Eq":[],"ExceptHandler":["type","name","body"],"Expr":["value"],"Expression":["body"],"ExtSlice":[],"FloorDiv":[],"For":["target","iter","body","orelse","type_comment"],"FormattedValue":["value","conversion","format_spec"],"FunctionDef":["name","args","body","decorator_list","returns","type_comment"],"FunctionType":["argtypes","returns"],"GeneratorExp":["elt","generators"],"Global":["names"],"Gt":[],"GtE":[],"If":["test","body","orelse"],"IfExp":["test","body","orelse"],"Import":["names"],"ImportFrom":["module","names","level"],"In":[],"Index":[],"Interactive":["body"],"Invert":[],"Is":[],"IsNot":[],"JoinedStr":["values"],"LShift":[],"Lambda":["args","body"],"List":["elts","ctx"],"ListComp":["elt","generators"],"Load":[],"Lt":[],"LtE":[],"MatMult":[],"Match":["subject","cases"],"MatchAs":["pattern","name"],"MatchClass":["cls","patterns","kwd_attrs","kwd_patterns"],"MatchMapping":["keys","patterns","rest"],"MatchOr":["patterns"],"MatchSequence":["patterns"],"MatchSingleton":["value"],"MatchStar":["name"],"MatchValue":["value"],"Mod":[],"Module":["body","type_ignores"],"Mult":[],"Name":["id","ctx"],"NameConstant":["value","kind"],"NamedExpr":["target","value"],"Nonlocal":["names"],"Not":[],"NotEq":[],"NotIn":[],"Num":["n"],"Or":[],"Param":[],"Pass":[],"Pow":[],"RShift":[],"Raise":["exc","cause"],"Return":["value"],"Set":["elts"],"SetComp":["elt","generators"],"Slice":["lower","upper","step"],"Starred":["value","ctx"],"Store":[],"Str":["s"],"Sub":[],"Subscript":["value","slice","ctx"],"Suite":[],"Try":["body","handlers","orelse","finalbody"],"Tuple":["elts","ctx"],"TypeIgnore":["lineno","tag"],"UAdd":[],"USub":[],"UnaryOp":["op","operand"],"While":["test","body","orelse"],"With":["items","body","type_comment"],"Yield":["value"],"YieldFrom":["value"],"alias":["name","asname"],"arg":["arg","annotation","type_comment"],"arguments":["posonlyargs","args","vararg","kwonlyargs","kw_defaults","kwarg","defaults"],"boolop":[],"cmpop":[],"comprehension":["target","iter","ifs","is_async"],"excepthandler":[],"expr":[],"expr_context":[],"keyword":["arg","value"],"match_case":["pattern","guard","body"],"mod":[],"operator":[],"pattern":[],"slice":[],"stmt":[],"type_ignore":[],"unaryop":[],"withitem":["context_expr","optional_vars"]}

def normalized_source(source):
    """Frozen Python-3.10 grammar/field schema, independent of runtime ast.dump."""
    def encode(value):
        if isinstance(value, ast.AST):
            name=type(value).__name__
            fields=SEMANTIC_AST_FIELDS.get(name)
            if fields is None: raise Blocked("unsupported semantic AST node: "+name)
            extras=set(value._fields)-set(fields)
            if extras:
                # Python3.12 adds this field; an empty declaration has no semantics.
                if extras=={"type_params"} and name in {"FunctionDef","AsyncFunctionDef","ClassDef"} and value.type_params==[]:
                    pass
                else: raise Blocked("unsupported new semantic AST fields: "+name)
            result=[]
            for field in fields:
                if not hasattr(value,field): raise Blocked("missing frozen semantic field: "+name+"."+field)
                child=getattr(value,field)
                if field=="body" and name in {"Module","FunctionDef","AsyncFunctionDef","ClassDef"}:
                    if child and isinstance(child[0],ast.Expr) and isinstance(child[0].value,ast.Constant) and isinstance(child[0].value.value,str):
                        child=child[1:]
                result.append([field,encode(child)])
            return {"node":name,"fields":result}
        if isinstance(value,list): return [encode(x) for x in value]
        if isinstance(value,tuple): return {"tuple":[encode(x) for x in value]}
        if isinstance(value,bytes): return {"bytes_hex":value.hex()}
        if value is Ellipsis: return {"ellipsis":True}
        if value is None or type(value) in (str,int,float,bool): return value
        raise Blocked("unsupported semantic constant")
    tree=ast.parse(source,feature_version=(3,10))
    return canonical({"schema":"cellforge-semantic-ast/1","tree":encode(tree)}).decode()


def normalized_params(params):
    return {k:[0.0 if float(x)==0 else float(x) for x in v] for k,v in sorted(params.items())}
def science_identity(config):
    # Operational hashes, labels, paths and provenance are deliberately excluded.
    arms=[]
    for arm in config["arms"]:
        arms.append({"source_ast":normalized_source(arm["source"]),
                     "params":normalized_params(arm["params"]),
                     "blind":bool(arm.get("blind",False))})
    arms.sort(key=canonical)
    return digest({"schema":"cellforge-science/3","arms":arms,"games":config["games"],
                   "seed":config["seed"],"actions":config["actions"],"play_wall_seconds":45,
                   "evaluator_ast":config["evaluator_ast"],"dataset":config["dataset_sha256"],
                   "packages":config["packages"],"reset_contract":config["reset_contract"],
                   "score_contract":"E.play official completed-level result, reset costs one action"})
def validate_params(params,primary=True):
    if set(params)!=set(SHAPES): raise Blocked("tensor keys differ")
    for k,n in SHAPES.items():
        v=params[k]
        if len(v)!=n or any(type(x) not in (int,float) or not math.isfinite(x) for x in v):
            raise Blocked("invalid literal tensor: "+k)
        bound=3 if k in ("rates","mix") else 8
        if any(abs(x)>bound for x in v): raise Blocked("outside frozen proposal support: "+k)
    if primary and (params["pixel"]!=PIXEL or params["memory_gate"]!=[1] or params["context_gate"]!=[1]):
        raise Blocked("fixed proposal positions changed")
def validate_plan(plan):
    if plan["schema"]!="cellforge-plan/2" or plan["study_cap"]!=9: raise Blocked("unsupported study cap")
    if plan["limits"]!=LIMITS or plan["remote_model_calls"]!=0: raise Blocked("resource/model allowance differs")
    if plan["qualified"] is not False or len(plan["conditions"])!=9: raise Blocked("unapproved scope")
    ids=[]
    for c in plan["conditions"]:
        if c["games"]!=GAMES or c["seed"] not in SEEDS or c["actions"]!=1500:
            raise Blocked("unapproved development condition")
        if c["fingerprint"]!=science_identity(c): raise Blocked("science identity mismatch")
        names={"full","matched_random","zero","action_off","memory_off","context_off","blind","frozen_spatial","scripted_explorer"}
        if len(c["arms"])!=9 or {a["name"] for a in c["arms"]}!=names: raise Blocked("control matrix incomplete")
        f=next((x for x in plan["founders"] if x["name"]==c["label"]),None)
        if f is None: raise Blocked("founder binding missing")
        a={x["name"]:x for x in c["arms"]}; p=f["params"]
        expected={"full":p,"matched_random":f["random_params"],"blind":p,
                  "zero":{k:[0]*n for k,n in SHAPES.items()},
                  "action_off":dict(p,effect=[0]*8,action=[0]*8,lookahead=[0]*4),
                  "memory_off":dict(p,memory_gate=[0]),"context_off":dict(p,context_gate=[0])}
        for name,params in expected.items():
            if a[name]["params"]!=params or normalized_source(a[name]["source"])!=normalized_source(a["full"]["source"]):
                raise Blocked("control support/source binding differs")
        if any(bool(x.get("blind",False))!=(x["name"]=="blind") for x in c["arms"]):
            raise Blocked("blind-control binding differs")
        if c["packages"]!=plan["packages"] or c["dataset_sha256"]!=plan["dataset_sha256"]:
            raise Blocked("runtime/data binding differs")
        ids.append(c["fingerprint"])
    if len(set(ids))!=9: raise Blocked("duplicate science in plan")
    for f in plan["founders"]:
        validate_params(f["params"]); validate_params(f["random_params"])
    return True

@contextmanager
def registry_lock(root):
    # One lock for admission, terminal application and registry reads/writes.
    if os.name!="posix": raise Blocked("production registry requires POSIX flock")
    import fcntl
    stream=open(pathlib.Path(root)/"REGISTRY.lock","a+")
    try:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(stream,fcntl.LOCK_UN); stream.close()

def fresh_registry():
    return {"schema":"cellforge-registry/2","sequence":0,"seen_receipts":[],
            "conditions":{},"studies":{},"active":None,"consecutive_failures":0,
            "not_before":0,"status":"paused_no_approved_plan"}

def replay_terminal(state,receipt):
    rid=receipt["receipt_id"]
    if rid!=digest({k:v for k,v in receipt.items() if k!="receipt_id"}): raise Blocked("terminal receipt digest mismatch")
    if rid in state["seen_receipts"]: return False
    if receipt["previous_sequence"]!=state["sequence"] or receipt["sequence"]!=state["sequence"]+1:
        raise Blocked("terminal sequence gap/conflict")
    fp=receipt["fingerprint"]
    if fp not in state["conditions"]: raise Blocked("unknown terminal science")
    failures=state["consecutive_failures"]
    failures=failures+1 if receipt["outcome"] in FAILURES else (0 if receipt["outcome"]=="completed" else failures)
    deadline=receipt["terminal_epoch"]+min(300,30*2**(failures-1)) if receipt["outcome"] in FAILURES else state["not_before"]
    if receipt["failure_count_after"]!=failures or receipt["not_before_after"]!=deadline:
        raise Blocked("failure replay effect conflicts")
    state["sequence"]=receipt["sequence"]; state["seen_receipts"].append(rid)
    state["conditions"][fp]["status"]=receipt["outcome"]
    state["conditions"][fp]["receipt"]=rid
    state["active"]=None; state["consecutive_failures"]=failures; state["not_before"]=deadline
    return True

def make_terminal(state,fp,outcome,evidence,epoch=None):
    epoch=time.time() if epoch is None else epoch
    failures=state["consecutive_failures"]
    failures=failures+1 if outcome in FAILURES else (0 if outcome=="completed" else failures)
    deadline=epoch+min(300,30*2**(failures-1)) if outcome in FAILURES else state["not_before"]
    r={"schema":"cellforge-terminal/2","fingerprint":fp,"outcome":outcome,
       "previous_sequence":state["sequence"],"sequence":state["sequence"]+1,
       "terminal_epoch":epoch,"failure_count_after":failures,"not_before_after":deadline,
       "evidence":evidence}
    r["receipt_id"]=digest(r)
    return r

def admit(state,plan,approval,manifest_hash):
    validate_plan(plan)
    ph=digest(plan)
    required={"plan_sha256":ph,"bundle_manifest_sha256":manifest_hash,
              "decision":"permit_baseline_development","qualified":False,"remote_model_calls":0}
    if any(approval.get(k)!=v for k,v in required.items()) or not approval.get("audit_report_sha256"):
        raise Blocked("missing or mismatched independent approval")
    # A report digest is mandatory; caller also verifies its actual bytes.
    if state["active"] or state["consecutive_failures"]>=3: raise Blocked("global active/failure gate")
    if ph in state["studies"]: raise Blocked("plan already admitted")
    for c in plan["conditions"]:
        if c["fingerprint"] in state["conditions"]: raise Blocked("science already queued or consumed globally")
    state["studies"][ph]={"cap":9,"attempted":0,"condition_ids":[c["fingerprint"] for c in plan["conditions"]]}
    for c in plan["conditions"]:
        state["conditions"][c["fingerprint"]]={"study":ph,"status":"pending"}
    return ph

def reserve(state,fp):
    entry=state["conditions"][fp]; study=state["studies"][entry["study"]]
    if state["active"] or entry["status"]!="pending": raise Blocked("not a new pending science condition")
    if study["attempted"]>=study["cap"]: raise Blocked("frozen study cap exhausted")
    if state["consecutive_failures"]>=3 or time.time()<state["not_before"]: raise Blocked("global failure/backoff gate")
    study["attempted"]+=1; entry["status"]="launch_intent"; state["active"]=fp
    return entry["study"]

def import_legacy(state,root):
    # Read-only source, normalized original comparison semantics. Never rewrites receipts.
    root=pathlib.Path(root)
    plan=read(root/"PLAN.json")
    environment=read(root/"ENVIRONMENT.json")
    for c in plan["experiments"]:
        run=root/"runs"/c["fingerprint"]
        r=read(run/"TERMINAL.json"); arms=read(run/"ARMS.json")
        if r["fingerprint"]!=c["fingerprint"] or r["config_sha256"]!=filehash(run/"CONFIG.json") or r["results_sha256"]!=filehash(run/"RESULTS.json"):
            raise Blocked("legacy receipt verification failed")
        if c["policy_source_sha256"]!=hashlib.sha256(arms["source"].encode()).hexdigest():
            raise Blocked("legacy source verification failed")
        condition={"arms":[{"source":arms["source"],"params":v,"blind":k=="blind"}
                            for k,v in arms["params"].items()],
                   "games":c["games"],"seed":c["seed"],"actions":c["actions"],
                   "evaluator_ast":normalized_source((root/"arc3_env.py").read_text()),
                   "dataset_sha256":plan["dataset_sha256"],"packages":environment["packages"],
                   "reset_contract":"E.play public reset costs one action; per-game policy memory"}
        fp=science_identity(condition)
        entry={"status":r["outcome"],"legacy_unqualified":True,"original_fingerprint":c["fingerprint"],
               "original_receipt_sha256":filehash(run/"TERMINAL.json"),"source_path":str(run)}
        if fp in state["conditions"] and state["conditions"][fp]!=entry: raise Blocked("legacy identity conflict")
        state["conditions"][fp]=entry
    return state

