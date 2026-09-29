"""Expand the contract's word list (architect-typed data) into a trie-lookup proposal. Does not read any corpus."""
import json, math, os
here = os.path.dirname(os.path.abspath(__file__))
words = json.load(open(os.path.join(here, "..", "contract.json")))["data"]["words"]
NS = "cl_words_fn."
def mc(x): return min(0.7 * math.log(1 + x), 5.0)
# trie nodes: prefixes of length 1..7
nodes = {}
for w, f in words.items():
    for j in range(1, min(len(w), 7) + 1):
        p = w[:j]
        n = nodes.setdefault(p, {})
    # continuation weights
    for j in range(1, len(w)):
        if j <= 7:
            c = w[j]
            nodes[w[:j]][c] = nodes[w[:j]].get(c, 0) + f
    if len(w) <= 7:
        nodes[w]["_"] = f
ops = []
ops.append({"op": "set_param", "name": NS + "zero", "value": [0.0] * 36})
nd = []       # cluster nodes
def add(id, use, inn=None, args=None):
    n = {"id": id, "use": use}
    if inn: n["in"] = inn
    if args: n["args"] = args
    nd.append(n)
for r in range(8):
    add(f"row{r}", "op:slice", {"x": "$ctx"}, {"axis": 0, "start": r, "stop": r + 1})
probe = {}
def P(r, c):
    k = (r, c)
    if k not in probe:
        add(f"p{r}_{c}", "op:slice", {"x": f"row{r}"}, {"axis": 1, "start": c, "stop": c + 1})
        probe[k] = f"p{r}_{c}"
    return probe[k]
for r in range(7):
    add(f"L{r}s", "op:slice", {"x": f"row{r}"}, {"axis": 1, "start": 10, "stop": 36})
    add(f"L{r}", "op:sum", {"x": f"L{r}s"}, {"axis": 1, "keepdims": True})
    add(f"LA{r}", "op:add", {"a": f"L{r}", "b": P(r, 8)})
    add(f"nLA{r}", "op:neg", {"x": f"LA{r}"})
    add(f"B{r}", "op:bias", {"x": f"nLA{r}"}, {"c": 1.0})
# row 7 must be a letter is implied by the probes of the last letter
acc = None
cells = 0
for p in sorted(nodes):
    j = len(p)
    y = [0.0] * 36
    for c, f in nodes[p].items():
        if c == "_": y[3] = mc(f)
        else: y[10 + ord(c) - 97] = mc(f)
    tag = p
    cur = f"B{7 - j}"
    for i, ch in enumerate(p, 1):
        nid = f"a_{tag}_{i}"
        add(nid, "op:mul", {"a": cur, "b": P(7 - j + i, 10 + ord(ch) - 97)})
        cur = nid
    ops.append({"op": "set_param", "name": NS + f"Y_{tag}", "value": [round(v, 6) for v in y]})
    ops.append({"op": "put_def", "name": NS + f"c_{tag}", "def": {"kind": "cell", "act": "identity",
        "params": {"W1": NS + f"Yc_{tag}", "b1": NS + "zero"}}})
    ops.append({"op": "set_param", "name": NS + f"Yc_{tag}", "value": [[v] for v in [round(v, 6) for v in y]]})
    # drop the unused Y_ param
    ops.pop(-3)
    add(f"o_{tag}", NS + f"c_{tag}", {"x": cur})
    cells += 1
    if acc is None: acc = f"o_{tag}"
    else:
        add(f"s_{tag}", "op:add", {"a": acc, "b": f"o_{tag}"})
        acc = f"s_{tag}"
add("yv", "op:reshape", {"x": acc}, {"shape": [36]})
ops.append({"op": "put_def", "name": "cl_words_fn", "def": {"kind": "bundle", "role": "cluster",
    "inputs": ["ctx"], "nodes": nd, "outputs": {"y": "yv"}}})
json.dump({"author": "c-cl_words_fn", "rationale": "Trie of prefix detectors (boundary row * letter probes, exact ops) each driving a one-node lookup cell [36,1] that emits the evidence vector; outputs summed. Generated from contract word list by scripts/gen.py.", "ops": ops},
          open(os.path.join(here, "..", "proposal.json"), "w"))
print(len(nodes), "cells")
