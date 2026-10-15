"""Tensor-selected actions and copied spans; executor never receives the request."""
import re
import numpy as np
from .action_packet import validate_file_action

ROLES=("arg1","arg2","file")
def infer(request, router, W, thresholds, *, legacy=False):
    if not isinstance(request,str) or len(request)>256 or not request.isascii():
        raise ValueError("bounded ASCII request required")
    w=np.asarray(W,dtype=float); b=np.asarray(thresholds,dtype=float)
    width=5 if legacy else 8
    if w.shape!=(3,width) or b.shape!=(3,) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("literal finite slot tensors required")
    rw=np.asarray(router["W"],dtype=float); rb=np.asarray(router["b"],dtype=float)
    if rw.shape!=(len(router["actions"]),len(router["vocab"])) or rb.shape!=(len(router["actions"]),) or not np.isfinite(rw).all() or not np.isfinite(rb).all():
        raise ValueError("literal finite router tensors required")
    words=set(re.findall(r"[a-z]+",request.lower()))
    features=np.array([int((v in words) if v!="<quote>" else ("'" in request or '"' in request)) for v in router["vocab"]])
    logits=rw@features+rb
    action=router["actions"][int(np.argmax(logits))]
    toks=[]; nq=0
    for m in re.finditer(r"""'[^']*'|"[^"]*"|\S+""",request):
        raw=m.group()
        if raw[0] in "'\"":
            nq+=1; toks.append({"text":raw[1:-1],"q":nq})
        else: toks.append({"text":raw.strip("?!,;:").rstrip("."),"q":0})
    matrix=[]
    for i,t in enumerate(toks):
        prev=toks[i-1]["text"].lower() if i and toks[i-1]["q"]==0 else ""
        f=[int(t["q"]==1),int(t["q"]==2),int(prev=="for"),
           int(t["q"]==0 and t["text"].lower() in router["vocab"]),
           int("." in t["text"] and (not legacy or t["q"]==0))]
        if not legacy: f += [int(prev=="to"),int(prev=="in"),int(prev in ("show","open","read","run","execute"))]
        matrix.append(f)
    chosen={}; scores={}
    for role,row,threshold in zip(ROLES,w,b):
        ss=[float(row@np.asarray(f)) for f in matrix]; scores[role]=ss
        idx=max(range(len(ss)),key=lambda j:(ss[j],-j)) if ss else None
        chosen[role]=None if idx is None or ss[idx]<=threshold else toks[idx]["text"]
    packet=None
    if action=="write_file" and chosen["file"] is not None and chosen["arg1"] is not None:
        packet={"type":"call","name":"write_file","arguments":{"path":chosen["file"],"content":chosen["arg1"]}}
    elif action=="edit_replace" and all(chosen[x] is not None for x in ROLES):
        packet={"type":"call","name":"edit_in_place","arguments":{"path":chosen["file"],"old":chosen["arg1"],"new":chosen["arg2"],"expected_matches":1}}
    return {"action":action,"slots":chosen,"packet":packet,"trace":{"tokens":toks,"token_features":matrix,"slot_scores":scores,"router_logits":logits.tolist()}}

def execute_packet(packet, *, allowed_root):
    """Only mechanical validation and literal effects. No request or model access."""
    if packet is None: return {"status":"abstained"}
    try:
        checked=validate_file_action(packet,allowed_root=allowed_root)
    except ValueError as e: return {"status":"denied","reason":str(e)}
    from pathlib import Path
    target=Path(checked["resolved_path"]); args=packet["arguments"]
    if packet["name"]=="write_file":
        target.write_text(args["content"],encoding="utf-8")
    else:
        before=target.read_text(encoding="utf-8")
        if before.count(args["old"])!=args["expected_matches"]:
            return {"status":"denied","reason":"literal expected match count differs"}
        target.write_text(before.replace(args["old"],args["new"]),encoding="utf-8")
    return {"status":"executed","packet":packet}


ACTION_PACKET_VERSION="file-action-packet/1"
def infer_joint(request, slot_params, router_params, lexical_router):
    """Typed adapter only: tensor-selected router label plus tensor-selected spans."""
    from .tool_router_evolution import route
    slots=infer(request,lexical_router,slot_params["argument.W"],slot_params["argument.b"],
                legacy=len(slot_params["argument.W"][0])==5)
    selected=route(router_params,request)
    v=slots["slots"]; packet=None
    if selected["action"]=="write_file" and v["file"] is not None and v["arg1"] is not None:
        packet={"type":"call","name":"write_file","arguments":{"path":v["file"],"content":v["arg1"]}}
    elif selected["action"]=="edit_in_place" and all(v[x] is not None for x in ROLES):
        packet={"type":"call","name":"edit_in_place","arguments":{"path":v["file"],"old":v["arg1"],"new":v["arg2"],"expected_matches":1}}
    return {"contract_version":ACTION_PACKET_VERSION,"action":selected["action"],
            "slots":v,"packet":packet,"argument_trace":slots["trace"],"router_trace":selected}
