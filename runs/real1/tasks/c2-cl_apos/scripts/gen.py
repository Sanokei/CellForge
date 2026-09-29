"""Expands hand-typed word lists / values into proposal.json (no corpus access)."""
import json
NS = "cl_apos."
CH = ["<pad>", "<bos>", "<eos>", " ", ",", ".", "!", "?", "'", '"'] + [chr(97+i) for i in range(26)]
ID = {c: i for i, c in enumerate(CH)}
STEMS = [("didn",3.5),("don",3.0),("wasn",3.5),("isn",3.5),("couldn",3.5),("wouldn",3.5),("doesn",3.5),
         ("aren",3.5),("weren",3.5),("haven",3.5),("hasn",3.5),("can",0.4),("won",1.5),("let",1.2),("it",0.8),
         ("that",0.8),("what",0.8),("i",1.0),("he",0.3),("she",0.3),("there",0.5),("you",0.3),("we",0.3),("they",0.3)]
S1 = "it that what let he she there here who where how".split()
S2 = "you we they".split()
nodes = []; ops = []
def node(i, use, inp=None, args=None):
    d = {"id": i, "use": use}
    if inp is not None: d["in"] = inp
    if args is not None: d["args"] = args
    nodes.append(d); return i
def op(i, name, inp, **args): return node(i, "op:"+name, inp, args or None)
def vec(d):
    v = [0.0]*36
    for c, x in d.items(): v[ID[c]] = x
    return v
flat = op("flat", "reshape", {"x": "$ctx"}, shape=[288])
def sl(i, src, a, b): return op(i, "slice", {"x": src}, axis=0, start=a, stop=b)
Ls, As, Bs = [], [], []
for r in range(8):
    l = op(f"L{r}", "sum", {"x": sl(f"lr{r}", flat, r*36+10, r*36+36)}, axis=0, keepdims=True)
    a = sl(f"A{r}", flat, r*36+8, r*36+9)
    t = op(f"la{r}", "add", {"a": l, "b": a})
    b = op(f"B{r}", "bias", {"x": op(f"nla{r}", "neg", {"x": t})}, c=1.0)
    Ls.append(l); As.append(a); Bs.append(b)
Lv = op("Lv", "concat", {f"x{r}": Ls[r] for r in range(8)}, axis=0)
Bv = op("Bv", "concat", {f"x{r}": Bs[r] for r in range(8)}, axis=0)
def word(name, w, end):
    k = len(w)
    parts = [sl(f"{name}_s{j}", flat, (end-k+1+j)*36+ID[c], (end-k+1+j)*36+ID[c]+1) for j, c in enumerate(w)]
    parts.append(Bs[end-k])
    c = op(f"{name}_c", "concat", {f"x{j}": p for j, p in enumerate(parts)}, axis=0)
    s = op(f"{name}_sum", "sum", {"x": c}, axis=0, keepdims=True)
    return op(f"{name}_m", "step", {"x": op(f"{name}_b", "bias", {"x": s}, c=-(k+0.5))})
# case B
ind = [word("st_"+w, w, 7) for w, _ in STEMS]
cat = op("stemcat", "concat", {f"x{i}": n for i, n in enumerate(ind)}, axis=0)
node("stemval", NS+"c_stem", {"x": cat})
node("place_p", "param:"+NS+"place_v")
op("stemy", "mul", {"a": "stemval", "b": "place_p"})
# case A
def run_exact(k):
    e = 6
    s = op(f"E{k}_ls", "sum", {"x": op(f"E{k}_sl", "slice", {"x": Lv}, axis=0, start=e-k+1, stop=e+1)}, axis=0, keepdims=True)
    t = op(f"E{k}_t", "add", {"a": s, "b": sl(f"E{k}_b", Bv, e-k, e-k+1)})
    return op(f"E{k}", "step", {"x": op(f"E{k}_bb", "bias", {"x": t}, c=-(k+0.5))})
E = {k: run_exact(k) for k in range(2, 7)}
def addall(name, xs):
    cur = xs[0]
    for j, x in enumerate(xs[1:]):
        cur = op(f"{name}_{j}", "add", {"a": cur, "b": x})
    return cur
gi = word("wi", "i", 6)
g1 = addall("g1", [word("w1_"+w, w, 6) for w in S1])
g2 = addall("g2", [word("w2_"+w, w, 6) for w in S2])
T = addall("T", [E[k] for k in range(2, 7)])
N = op("N", "mul", {"a": sl("n6", flat, 6*36+23, 6*36+24), "b": addall("E3p", [E[k] for k in range(3, 7)])})
P = op("P", "sub", {"a": op("P0", "sub", {"a": op("P00", "sub", {"a": T, "b": g1}), "b": g2}), "b": N})
A7 = As[7]
params = {
 "av_i": vec({"m":2.5,"l":1.2,"v":0.8,"d":0.3}),
 "av_s1": vec({"s":2.4,"l":0.6}),
 "av_s2": vec({"r":2.0,"l":1.2,"v":1.0,"d":0.4}),
 "av_n": vec({"t":2.6,"s":-1.0}),
 "av_p": vec({"s":2.0}),
 "stem_w": [v for _, v in STEMS],
 "stem_b": [0.0],
 "place_v": vec({"'":1.0}),
}
outs = []
for nm, g in [("i", gi), ("s1", g1), ("s2", g2), ("n", N), ("p", P)]:
    gg = op(f"gate_{nm}", "mul", {"a": g, "b": A7})
    p = node(f"pv_{nm}", "param:"+NS+"av_"+nm)
    outs.append(op(f"out_{nm}", "mul", {"a": gg, "b": p}))
outs.append("stemy")
y = addall("ysum", outs)
pops = []
for k, v in params.items():
    pops.append({"op": "set_param", "name": NS+k, "value": v} if k != "stem_w" and k != "place_W"
                else {"op": "set_param", "name": NS+k, "pattern": {"fill": 0, "shape": [1,24]}} if False else {"op": "set_param", "name": NS+k, "value": v})
# reshape matrices: W1 [1,24] and [36,1]
for o in pops:
    if o["name"] == NS+"stem_w": o["value"] = [list(o["value"])]
    if o["name"] == NS+"stem_b" or o["name"] == NS+"place_b": pass
pops += [
 {"op": "put_def", "name": NS+"c_stem", "def": {"kind":"cell","act":"identity","params":{"W1":NS+"stem_w","b1":NS+"stem_b"}}},
 {"op": "put_def", "name": "cl_apos", "def": {"kind":"bundle","role":"cluster","inputs":["ctx"],"nodes":nodes,"outputs":{"y":y}}},
]
json.dump({"author": "c2-cl_apos", "rationale": "Apostrophe evidence via exact slice/step word matchers over ctx rows; hand-typed values from the contract.", "ops": pops}, open("proposal.json","w"), indent=1)
print(len(nodes), "nodes")
