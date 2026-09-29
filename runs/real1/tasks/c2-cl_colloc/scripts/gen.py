"""Expand contract data.phrases (typed by the architect) into proposal.json. Reads only contract.json phrases."""
import json, collections
ROOT = "runs/real1/tasks/c2-cl_colloc/"
phr = json.load(open(ROOT + "contract.json"))["data"]["phrases"]
CH = ["<pad>", "<bos>", "<eos>", " ", ",", ".", "!", "?", "'", '"'] + [chr(97 + i) for i in range(26)]
IDX = {c: i for i, c in enumerate(CH)}
NS = "cl_colloc."
nodes = []
made = {}

def add(id_, use, inp, args=None):
    n = {"id": id_, "use": use, "in": inp}
    if args is not None: n["args"] = args
    nodes.append(n); made[id_] = 1; return id_

def row(r):
    i = f"r{r}"
    if i not in made: add(i, "op:slice", {"x": "$ctx"}, {"axis": 0, "start": r, "stop": r + 1})
    return i

def sel(r, c):
    i = f"s{r}_{c}"
    if i not in made: add(i, "op:slice", {"x": row(r)}, {"axis": 1, "start": c, "stop": c + 1})
    return i

def bnd(r):
    i = f"b{r}"
    if i not in made:
        L = add(f"bl{r}", "op:sum", {"x": add(f"bls{r}", "op:slice", {"x": row(r)}, {"axis": 1, "start": 10, "stop": 36})}, {"axis": 1, "keepdims": True})
        t = add(f"bt{r}", "op:add", {"a": L, "b": sel(r, 8)})
        add(i, "op:bias", {"x": add(f"bn{r}", "op:scale", {"x": t}, {"c": -1.0})}, {"c": 1.0})
    return i

def chain(key):  # key = tuple of (row,char) from row7 leftwards
    i = "m_" + "_".join(f"{r}x{c}" for r, c in key)
    if i in made: return i
    if len(key) == 1: made[i] = 1; nodes.append({"id": i, "use": "op:mul", "in": {"a": sel(*key[0]), "b": sel(*key[0])}}); return i
    prev = chain(key[:-1])
    return add(i, "op:mul", {"a": prev, "b": sel(*key[-1])})

rules = collections.defaultdict(float)  # (target, fire node) -> strength
for P, s in phr.items():
    for j in range(3, len(P)):
        m = min(j, 8); cs = P[j - m:j]
        key = tuple((7 - k, IDX[cs[m - 1 - k]]) for k in range(m))
        f = chain(key)
        if j < 8:
            f2 = f"f_{f}_b{7-j}"
            if f2 not in made: add(f2, "op:mul", {"a": f, "b": bnd(7 - j)})
            f = f2
        rules[(IDX[P[j]], f)] += s

by_t = collections.defaultdict(list)
for (t, f), s in sorted(rules.items()): by_t[t].append((f, s))
ops = []
ops.append({"op": "set_param", "name": NS + "b0", "value": [0.0]})
zero = add("zero", "op:scale", {"x": "s7_3"}, {"c": 0.0})
outs = []
for t in range(36):
    if t not in by_t: outs.append(zero); continue
    items = by_t[t]; parts = []
    for ci in range(0, len(items), 39):
        ch = items[ci:ci + 39]; nm = f"t{t}_{ci//39}"
        ops.append({"op": "set_param", "name": f"{NS}W_{nm}", "value": [[s for _, s in ch]]})
        ops.append({"op": "put_def", "name": f"{NS}c_{nm}", "def": {"kind": "cell", "act": "identity", "params": {"W1": f"{NS}W_{nm}", "b1": NS + "b0"}}})
        cat = add(f"cat_{nm}", "op:concat", {f"x{k}": f for k, (f, _) in enumerate(ch)}, {"axis": 1}) if len(ch) > 1 else ch[0][0]
        parts.append(add(f"o_{nm}", f"{NS}c_{nm}", {"x": cat}))
    acc = parts[0]
    for k, p in enumerate(parts[1:]): acc = add(f"acc{t}_{k}", "op:add", {"a": acc, "b": p})
    outs.append(acc)
cat = add("ycat", "op:concat", {f"x{k}": o for k, o in enumerate(outs)}, {"axis": 1})
add("y", "op:reshape", {"x": cat}, {"shape": [36]})
ops.append({"op": "put_def", "name": "cl_colloc", "def": {"kind": "bundle", "role": "cluster", "inputs": ["ctx"], "nodes": nodes, "outputs": {"y": "y"}}})
json.dump({"author": "c2-cl_colloc", "rationale": "Prefix-shared exact-match product chains on ctx rows per phrase position, boundary gating, per-target 40-slot strength cells.", "ops": ops}, open(ROOT + "proposal.json", "w"))
print(len(nodes), "nodes", sum(1 for o in ops if o["op"] == "set_param"), "params")
