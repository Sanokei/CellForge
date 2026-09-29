"""Expands the hand-typed design of cl_name_read into proposal.json (all constants typed here)."""
import json
NS = "cl_name_read."
G = {1: 1.5, 2: 2.5, 3: 3.0, 4: 3.0, 5: 3.0}
G0 = 1.2
END = {3: 1.5, 4: 0.7, 5: 0.7, 6: 0.3, 7: 0.2, 8: 0.6}
nodes = []
def n(id, use, ins=None, args=None):
    d = {"id": id, "use": use}
    if ins: d["in"] = ins
    if args: d["args"] = args
    nodes.append(d); return id
def op(id, o, args=None, **ins): return n(id, "op:" + o, ins, args)
def sl(id, x, axis, a, b): return op(id, "slice", {"axis": axis, "start": a, "stop": b, "step": 1}, x=x)
def rs(id, x, shape): return op(id, "reshape", {"shape": shape}, x=x)

# buffer and length
sl("bf", "$nm", 0, 0, 216); rs("br", "bf", [6, 36]); op("B", "onehot_argmax", {"axis": 1}, x="br")
sl("lr", "$nm", 0, 217, 224); op("Lo", "onehot_argmax", {"axis": 0}, x="lr")
sl("cap", "$nm", 0, 216, 217)
op("ncap0", "scale", {"c": -1.0}, x="cap"); op("ncap1", "bias", {"c": 0.5}, x="ncap0"); op("ncap", "step", x="ncap1")
sl("l26", "Lo", 0, 2, 7); op("lge2", "sum", {"axis": 0, "keepdims": True}, x="l26")
op("valid", "mul", a="lge2", b="ncap")
# dot products buffer rows x ctx rows
op("CT", "transpose", {"perm": [1, 0]}, x="$ctx"); op("D", "matmul", a="B", b="CT")
# boundary flags
sl("cl", "$ctx", 1, 10, 36); op("lm", "sum", {"axis": 1}, x="cl")
sl("ap", "$ctx", 1, 8, 9); rs("ap1", "ap", [8])
op("bd0", "add", a="lm", b="ap1"); op("bd1", "scale", {"c": -1.0}, x="bd0"); op("bd", "bias", {"c": 1.0}, x="bd1")
have = {}
def bds(r):
    k = "bd%d" % r
    if k not in have:
        sl("bs%d" % r, "bd", 0, r, r + 1); have[k] = "bs%d" % r
    return have[k]
def dd(j, r):
    k = "d%d_%d" % (j, r)
    if k not in have:
        sl("dr%d_%d" % (j, r), "D", 0, j, j + 1)
        sl("dc%d_%d" % (j, r), "dr%d_%d" % (j, r), 1, r, r + 1)
        rs(k, "dc%d_%d" % (j, r), [1]); have[k] = k
    return k
# word-start
sl("c7", "$ctx", 0, 7, 8); rs("c7v", "c7", [36])
sl("sp", "c7v", 0, 3, 4); sl("dq", "c7v", 0, 9, 10); op("spdq", "add", a="sp", b="dq")
op("st1", "mul", a="spdq", b="valid"); op("st2", "scale", {"c": G0}, x="st1")
sl("b0r", "B", 0, 0, 1); rs("b0", "b0r", [36])
op("start", "mul", a="st2", b="b0")
terms = ["start"]
# param end vector
n("endv", "param:" + NS + "endvec")
for k in range(1, 7):
    prev = None
    for j in range(k):
        d = dd(j, 8 - k + j)
        prev = d if prev is None else op("mp%d_%d" % (k, j), "mul", a=prev, b=d)
    m = op("m%d" % k, "mul", a=prev, b=bds(7 - k))
    if k <= 5:
        sl("lg%d" % k, "Lo", 0, k + 1, 7); op("lgt%d" % k, "sum", {"axis": 0, "keepdims": True}, x="lg%d" % k)
        op("ca%d" % k, "mul", a=m, b="lgt%d" % k); op("cb%d" % k, "mul", a="ca%d" % k, b="ncap")
        op("cc%d" % k, "scale", {"c": G[k]}, x="cb%d" % k)
        sl("bkr%d" % k, "B", 0, k, k + 1); rs("bk%d" % k, "bkr%d" % k, [36])
        terms.append(op("cont%d" % k, "mul", a="cc%d" % k, b="bk%d" % k))
    if k >= 2:
        sl("le%d" % k, "Lo", 0, k, k + 1)
        op("ea%d" % k, "mul", a=m, b="le%d" % k); op("eb%d" % k, "mul", a="ea%d" % k, b="ncap")
        terms.append(op("endt%d" % k, "mul", a="eb%d" % k, b="endv"))
acc = terms[0]
for i, t in enumerate(terms[1:]):
    acc = op("acc%d" % i, "add", a=acc, b=t)
endvec = [0.0] * 36
for i, v in END.items(): endvec[i] = v
prop = {"author": "c-cl_name_read", "rationale": "Exact ops: dot products of one-hot buffer rows with ctx rows give letter matches; boundary and length gates; one 36-vector param for the word-end profile.",
 "ops": [{"op": "set_param", "name": NS + "endvec", "value": endvec},
         {"op": "put_def", "name": "cl_name_read", "def": {"kind": "bundle", "role": "cluster", "inputs": ["ctx", "nm"], "nodes": nodes, "outputs": {"y": acc}}}]}
json.dump(prop, open("proposal.json", "w"), indent=1)
print(len(nodes), "nodes")
