"""Expand contract data.words (architect-typed word list) into a trie circuit. Does not read the corpus."""
import json, math, os
here = os.path.dirname(os.path.abspath(__file__))
task = os.path.dirname(here)
con = json.load(open(os.path.join(task, "contract.json")))
words = con["data"]["words"]; cap = con["data"]["cap"]; gain = con["data"]["gain"]
NS = "cl_words_open."
IDX = lambda ch: 10 + ord(ch) - 97
# trie nodes: prefixes w (len 1..7)
nodes = {}
for w in words:
    for L in range(1, min(len(w), 7) + 1):
        p = w[:L]
        v = nodes.setdefault(p, [0.0] * 36)
        if len(w) > L:
            v[IDX(w[L])] += words[w]
ops, params, ids = [], [], []
N = []          # cluster nodes
have = set()
def add(n):
    N.append(n); return n["id"]
def op(id_, o, ins, args=None):
    n = {"id": id_, "use": "op:" + o, "in": ins}
    if args is not None: n["args"] = args
    add(n); return id_
# row slices and boundaries
for r in range(0, 8):
    op(f"row{r}", "slice", {"x": "$ctx"}, {"axis": 0, "start": r, "stop": r + 1})
for r in range(0, 7):
    op(f"al{r}", "slice", {"x": f"row{r}"}, {"axis": 1, "start": 10, "stop": 36})
    op(f"as{r}", "sum", {"x": f"al{r}"}, {"axis": 1, "keepdims": True})
    op(f"ap{r}", "slice", {"x": f"row{r}"}, {"axis": 1, "start": 8, "stop": 9})
    op(f"nb{r}", "add", {"a": f"as{r}", "b": f"ap{r}"})
    op(f"ng{r}", "neg", {"x": f"nb{r}"})
    op(f"B{r}", "bias", {"x": f"ng{r}"}, {"c": 1.0})
def ind(r, c):
    id_ = f"i{r}_{c}"
    if id_ not in have:
        have.add(id_)
        op(id_, "slice", {"x": f"row{r}"}, {"axis": 1, "start": IDX(c), "stop": IDX(c) + 1})
    return id_
def chain(s, pre):
    # match of letters pre at rows s.. with boundary at row s-1
    id_ = f"t{s}_{pre}"
    if id_ in have: return id_
    if pre == "":
        return f"B{s-1}"
    prev = chain(s, pre[:-1])
    a = ind(s + len(pre) - 1, pre[-1])
    have.add(id_)
    op(id_, "mul", {"a": prev, "b": a})
    return id_
acc = None
for p in sorted(nodes):
    L = len(p); v = nodes[p]
    out = [0.0] * 36
    for c in range(36):
        if v[c] > 0: out[c] = min(gain * math.log(1 + v[c]), cap)
    if p in words:
        out[3] = min(gain * math.log(1 + words[p]), cap)
    pn = NS + "v_" + p
    params.append({"op": "set_param", "name": pn, "value": out})
    m = chain(8 - L, p)
    add({"id": "pv_" + p, "use": "param:" + pn})
    op("c_" + p, "mul", {"a": m, "b": "pv_" + p})
    if acc is None: acc = "c_" + p
    else: acc = op("s_" + p, "add", {"a": acc, "b": "c_" + p})
op("yout", "reshape", {"x": acc}, {"shape": [36]})
cl = {"kind": "bundle", "role": "cluster", "inputs": ["ctx"], "nodes": N, "outputs": {"y": "yout"}}
prop = {"author": "c-cl_words_open", "rationale": "Trie of prefix-match products (boundary row x letter-indicator chain per length) times per-prefix 36-vector; summed.",
        "ops": params + [{"op": "put_def", "name": "cl_words_open", "def": cl}]}
json.dump(prop, open(os.path.join(task, "proposal.json"), "w"))
print(len(nodes), "prefix nodes", len(N), "graph nodes")
