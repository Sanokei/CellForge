"""Concatenate fragment ops verbatim, add expert/router/root bundles. Weights are typed in a config json.
usage: python integrate.py cfg.json out_proposal.json"""
import json, sys, os
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
cfg = json.load(open(sys.argv[1]))
CIRC = ["cl_router","cl_qstate","cl_bigram","cl_trigram","cl_words_fn","cl_words_open","cl_suffix","cl_wordstart",
        "cl_colloc","cl_endpunct","cl_quote","cl_apos","cl_name_write","cl_name_read"]
NONAME = cfg.get("noname", False)
used = {"cl_router","cl_qstate"} | (set() if NONAME else {"cl_name_write"}) | {k for e in cfg["experts"].values() for k,w in e.items() if w}
V2={"cl_words_fn","cl_trigram","cl_endpunct","cl_apos","cl_colloc","cl_name_write"}
ops = []
for c in [c for c in CIRC if c in used]:
    ops += json.load(open(os.path.join(T, ("c2-" if c in V2 else "c-")+c, "proposal.json")))["ops"]
ARGS = {"cl_endpunct": ("ctx","q"), "cl_quote": ("ctx","q"), "cl_name_read": ("ctx","nm")}
def circuit_node(name):
    a = ARGS.get(name, ("ctx",))
    return {"id": "i_"+name, "use": name, "in": {k: "$"+k for k in a}}
EXPERTS = ["ex_bos","ex_wordstart","ex_inword_short","ex_inword_long","ex_punct","ex_quote","ex_apos"]
for ex in EXPERTS:
    nodes, terms = [], []
    for name, w in cfg["experts"][ex].items():
        if w == 0: continue
        nodes.append(circuit_node(name))
        if w == 1: terms.append("i_"+name)
        else:
            nodes.append({"id": "s_"+name, "use": "op:scale", "in": {"x": "i_"+name}, "args": {"c": w}}); terms.append("s_"+name)
    acc = terms[0]
    for k, t in enumerate(terms[1:]):
        nodes.append({"id": f"a{k}", "use": "op:add", "in": {"a": acc, "b": t}}); acc = f"a{k}"
    ops.append({"op": "put_def", "name": ex, "def": {"kind": "bundle", "role": "expert", "inputs": ["ctx","q","nm"],
                "nodes": nodes, "outputs": {"logits": acc}}})
ops.append({"op": "put_def", "name": "router", "def": {"kind": "bundle", "role": "router", "inputs": ["ctx"],
    "nodes": [{"id": "r", "use": "cl_router", "in": {"ctx": "$ctx"}}], "outputs": {"scores": "r"}}})
root = [
 {"id": "q0", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": 0, "stop": 1}},
 {"id": "nm0", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": 1, "stop": 225}},
 {"id": "qn", "use": "cl_qstate", "in": {"prev": "$prev", "q": "q0"}},
 {"id": "nmn", "use": "cl_name_write", "in": {"ctx": "$ctx", "nm": "nm0"}} if not NONAME else {"id": "nmn", "use": "op:scale", "in": {"x": "nm0"}, "args": {"c": 1}},
 {"id": "rt", "use": "router", "in": {"ctx": "$ctx"}},
 {"id": "d", "use": "dispatch", "in": {"scores": "rt.scores", "ctx": "$ctx", "q": "qn", "nm": "nmn"},
  "args": {"branches": EXPERTS, "k": 1}},
 {"id": "st", "use": "op:concat", "in": {"x0": "qn", "x1": "nmn"}, "args": {"axis": 0}},
]
ops.append({"op": "put_def", "name": "step_root", "def": {"kind": "bundle", "inputs": ["ctx","prev","step","state"],
    "nodes": root, "outputs": {"logits": "d.logits", "state": "st"}}})
ops.append({"op": "set_state_dim", "value": 225})
json.dump({"author": "a1-architect", "rationale": cfg.get("rationale", "integration"), "ops": ops}, open(sys.argv[2], "w"))
print(len(ops), "ops")
