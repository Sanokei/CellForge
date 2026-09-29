"""Expand hand-typed lexicon, grammar rules and probabilities into a ladder/1 proposal.

Nothing here reads the corpus. Every word list and probability below was typed by
hand (from reading samples of dev_train.json); this script only turns them into
arrays (match units, lookup tables, piecewise-linear log).
"""
import json, math, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", sys.argv[1] if len(sys.argv) > 1 else "proposal.json")

# ---------------------------------------------------------------- hand-chosen numbers
P = dict(
    adj=0.31,        # P(adjective | after determiner)
    prep_s=0.15, that_s=0.125,   # modifier after subject head noun (depth 0)
    prep_o=0.08, that_o=0.16,    # modifier after object head noun (depth 0)
    prep_1=0.10, that_1=0.12,    # modifier after a depth-1 noun (raised after eval 004 worst positions)
    prep_2=0.002, that_2=0.002,  # depth-2 nouns: never seen with a modifier; tiny safety mass
    iv=0.58,         # P(intransitive | main verb)
    adv=0.28,        # P(adverb | clause end reached)
    comma=0.19,      # P(", and" | clause end, no 'and' yet)
    the_sg=0.6,      # P(singular noun | det = the)
    eos_early=0.001, eos_mid=0.03, eos_late=0.99,   # P(eos | '.') by length bucket
    L1=139.5, L2=149.5,                               # length bucket edges (step index)
    eps=1e-6,        # relative floor added to every char
    impossible=-50.0,
)
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    P[k] = float(v)

VOCAB = ["<pad>", "<bos>", "<eos>", " ", ",", "."] + [chr(97 + i) for i in range(26)]
V = 32
def ci(ch):
    return VOCAB.index(ch)
SP, BOS, EOS, COM, DOT = ci(" "), 1, 2, ci(","), ci(".")

DET_SG = ["every", "this", "a"]
DET_PL = ["these", "many", "some"]
DET_ANY = ["the"]
DETS = DET_SG + DET_PL + DET_ANY
ADJ = ["old", "brave", "tired", "red", "small", "happy", "quiet", "young"]
NOUNS = [("teacher", "teachers"), ("dog", "dogs"), ("bird", "birds"), ("king", "kings"),
         ("farmer", "farmers"), ("fox", "foxes"), ("boy", "boys"), ("child", "children"),
         ("horse", "horses"), ("cat", "cats"), ("girl", "girls"), ("mouse", "mice")]
IV = [("waits", "wait"), ("sings", "sing"), ("laughs", "laugh"), ("sleeps", "sleep"),
      ("jumps", "jump"), ("runs", "run")]
TV = [("helps", "help"), ("likes", "like"), ("finds", "find"), ("follows", "follow"),
      ("chases", "chase"), ("sees", "see")]
PREP = ["with", "behind", "near", "under"]
ADV = ["never", "often", "quickly", "slowly", "again"]

# ---------------------------------------------------------------- helpers
def zeros(*shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [zeros(*shape[1:]) for _ in range(shape[0])]

def ctx_index(row, tok):
    return row * V + tok

def match_row(conds):
    """conds: list of (row, [tokens]) ; returns (w[256], b) for a step unit."""
    w = [0.0] * 256
    for row, toks in conds:
        for t in toks:
            w[ctx_index(row, t)] = 1.0
    return w, -(len(conds) - 0.5)

def prefix_conds(prefix):
    """current word prefix ends at row 7, with boundary before (if visible)."""
    j = len(prefix)
    conds = [(8 - j + i, [ci(prefix[i])]) for i in range(j)]
    if 7 - j >= 0:
        conds.append((7 - j, [SP, BOS]))
    return conds

def completed_conds(word):
    """word ended at row 6, row 7 is a space."""
    L = len(word)
    conds = [(7, [SP])]
    start = 7 - L
    for i in range(L):
        r = start + i
        if r >= 0:
            conds.append((r, [ci(word[i])]))
    if 6 - L >= 0:
        conds.append((6 - L, [SP, BOS]))
    return conds

ops = []
def set_param(name, value):
    ops.append({"op": "set_param", "name": name, "value": value})
def put_def(name, d):
    ops.append({"op": "put_def", "name": name, "def": d})

def cell(name, act, W1, b1, W2=None, b2=None):
    params = {"W1": name + ".W1", "b1": name + ".b1"}
    set_param(name + ".W1", W1)
    set_param(name + ".b1", b1)
    if W2 is not None:
        params["W2"] = name + ".W2"
        params["b2"] = name + ".b2"
        set_param(name + ".W2", W2)
        set_param(name + ".b2", b2)
    put_def(name, {"kind": "cell", "act": act, "params": params})

def cluster(name, inputs, nodes, outputs):
    put_def(name, {"kind": "bundle", "role": "cluster", "inputs": inputs, "nodes": nodes,
                   "outputs": outputs})

# ---------------------------------------------------------------- state layout
SLOTS = ["ADJN", "NOUN", "POST", "VERB", "ADV", "PC", "PP", "AND"]   # DET = none set
S_SLOT = {s: i for i, s in enumerate(SLOTS)}
S_NUM = {"sg": 8, "pl": 9, "any": 10}
S_RN, S_MD, S_AU, S_V, S_D, S_DD = 11, 12, 13, 14, 15, 16   # S_DD: modifier depth of current NP
S = 17

CATS = ["DSG", "DPL", "DANY", "ADJ", "NSG", "NPL", "IV", "TV", "PREP", "THAT", "AND",
        "COMMA", "PERIOD", "PCEV", "PPEV"]
C = {c: i for i, c in enumerate(CATS)}
NC = len(CATS)

# ---------------------------------------------------------------- trunk: classifier
cls_units = []   # (conds, cat)
for w in DET_SG: cls_units.append((completed_conds(w), "DSG"))
for w in DET_PL: cls_units.append((completed_conds(w), "DPL"))
for w in DET_ANY: cls_units.append((completed_conds(w), "DANY"))
for w in ADJ: cls_units.append((completed_conds(w), "ADJ"))
for sg, pl in NOUNS:
    cls_units.append((completed_conds(sg), "NSG"))
    cls_units.append((completed_conds(pl), "NPL"))
for sg, pl in IV:
    cls_units.append((completed_conds(sg), "IV")); cls_units.append((completed_conds(pl), "IV"))
for sg, pl in TV:
    cls_units.append((completed_conds(sg), "TV")); cls_units.append((completed_conds(pl), "TV"))
for w in PREP: cls_units.append((completed_conds(w), "PREP"))
cls_units.append((completed_conds("that"), "THAT"))
cls_units.append((completed_conds("and"), "AND"))
cls_units.append(([(7, [SP]), (6, [COM])], "COMMA"))
cls_units.append(([(7, [SP]), (6, [DOT])], "PERIOD"))
cls_units.append(([(7, [COM])], "PCEV"))
cls_units.append(([(7, [DOT])], "PPEV"))
W1, b1 = [], []
W2 = zeros(NC, len(cls_units))
for k, (conds, cat) in enumerate(cls_units):
    w, b = match_row(conds)
    W1.append(w); b1.append(b)
    W2[C[cat]][k] = 1.0
cell("c_wordcls", "step", W1, b1, W2, [0.0] * NC)
cluster("cl_wordcls", ["x"], [{"id": "c", "use": "c_wordcls", "in": {"x": "$x"}}], {"y": "c"})

# ---------------------------------------------------------------- trunk: stack reader
# input [v, d] -> [top_sg, top_pl, topval, d_eq0, d_eq1, d_ge2]
d_ = 1e-4
cell("c_stack", "step",
     [[3.0, 0.0, 0], [3.0, 0.0, 0], [0.0, 1.0, 0], [0.0, 1.0, 0], [0, 0, 1.0], [0, 0, 1.0]],
     [-1.0 + d_, -2.0 + d_, -0.5, -1.5, -0.5, -1.5],
     [[1, -1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0],
      [0, 0, 1, -1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 1, -1],
      [0, 0, 0, 0, 0, 1]],
     [0, 0, 0, 1, 0, 0, 1, 0, 0])
cluster("cl_stack", ["s"], [
    {"id": "vd", "use": "op:slice", "in": {"x": "$s"}, "args": {"axis": 0, "start": S_V, "stop": S_DD + 1}},
    {"id": "c", "use": "c_stack", "in": {"x": "vd"}}], {"y": "c"})
Z = {"top_sg": 0, "top_pl": 1, "topval": 2, "d_eq0": 3, "d_eq1": 4, "d_ge2": 5,
     "D_eq0": 6, "D_eq1": 7, "D_ge2": 8}
NZ = len(Z)

# ---------------------------------------------------------------- trunk: discrete update
# input X = concat[cat(NC), s_old(S), z_old(6)]
XI = {}
for c in CATS: XI["cat:" + c] = C[c]
for i in range(S): XI["s:%d" % i] = NC + i
for z, i in Z.items(): XI["z:" + z] = NC + S + i
NX = NC + S + NZ
def unit(pos, neg=(), extra_bias=0.0):
    """step unit: fires iff sum(pos) - sum(neg) > len(pos)-0.5 (pos terms are groups: list of names -> OR-sum)."""
    w = [0.0] * NX
    npos = 0
    for p in pos:
        names = p if isinstance(p, (list, tuple)) else [p]
        for n in names:
            w[XI[n]] += 1.0
        npos += 1
    for n in neg:
        w[XI[n]] -= 1.0
    return w, -(npos - 0.5) + extra_bias

ALLC = ["cat:" + c for c in CATS]
upd_units = []   # (w, b, [outputs])
OUTS = SLOTS + ["sg", "pl", "any", "rn", "md", "au"]
O = {o: i for i, o in enumerate(OUTS)}
def add(u, outs):
    upd_units.append((u[0], u[1], outs))
# keep units
for k, s in enumerate(SLOTS):
    add(unit(["s:%d" % k], ALLC), [s])
NUMC = ["cat:DSG", "cat:DPL", "cat:DANY", "cat:NSG", "cat:NPL"]
for nm in ["sg", "pl", "any"]:
    add(unit(["s:%d" % S_NUM[nm]], NUMC), [nm])
add(unit(["s:%d" % S_RN], ["cat:THAT", "cat:AND", "cat:PERIOD"]), ["rn"])
add(unit(["s:%d" % S_MD], ["cat:AND", "cat:PERIOD"]), ["md"])
add(unit(["s:%d" % S_AU], ["cat:PERIOD"]), ["au"])
# set units
add(unit([["cat:DSG", "cat:DPL", "cat:DANY"]]), ["ADJN"])
add(unit(["cat:ADJ"]), ["NOUN"])
add(unit([["cat:NSG", "cat:NPL"]]), ["POST"])
add(unit(["cat:DSG"]), ["sg"]); add(unit(["cat:NSG"]), ["sg"])
add(unit(["cat:DPL"]), ["pl"]); add(unit(["cat:NPL"]), ["pl"])
add(unit(["cat:DANY"]), ["any"])
add(unit(["cat:IV", "z:d_eq1"], ["s:%d" % S_MD]), ["ADV", "md"])          # IV main
add(unit(["cat:TV", "z:d_eq1"], ["s:%d" % S_MD]), ["rn", "md"])           # TV main -> object NP
add(unit([["cat:IV", "cat:TV"], "z:d_ge2"]), ["VERB"])                   # RC verb, more pending
add(unit([["cat:IV", "cat:TV"], "z:d_eq1", "s:%d" % S_MD]), ["ADV"])     # last RC verb in object area
add(unit([["cat:IV", "cat:TV"], "z:d_eq0"]), ["ADV"])                    # safety
add(unit(["cat:PREP"]), ["rn"])
add(unit(["cat:AND"]), ["au"])
add(unit(["cat:COMMA"]), ["AND"])
add(unit(["cat:PCEV"]), ["PC"])
add(unit(["cat:PPEV"]), ["PP"])
W1 = [u[0] for u in upd_units]; b1 = [u[1] for u in upd_units]
W2 = zeros(len(OUTS), len(upd_units))
for k, u in enumerate(upd_units):
    for o in u[2]:
        W2[O[o]][k] = 1.0
cell("c_upd", "step", W1, b1, W2, [-0.5] * len(OUTS))
# gates for v,d: [push1, push2, pop, reset]
g = [unit(["cat:NSG"], ["s:%d" % S_RN]), unit(["cat:NPL"], ["s:%d" % S_RN]),
     unit([["cat:IV", "cat:TV"]]), unit([["cat:AND", "cat:PERIOD"]])]
cell("c_gate", "step", [u[0] for u in g], [u[1] for u in g])
# deltas: input [v, topval, d]
cell("c_dv", "identity",
     [[-2 / 3, 0, 0], [-2 / 3, 0, 0], [2.0, -1.0, 0], [-1.0, 0, 0]], [1 / 3, 2 / 3, 0.0, 0.0])
# depth counter gates: [inc (prep/that), reset (and/period/TV-main)]
g2 = [unit([["cat:PREP", "cat:THAT"]]),
      unit([["cat:AND", "cat:PERIOD"]])]
tvm = unit(["cat:TV", "z:d_eq1"], ["s:%d" % S_MD])
cell("c_gate2", "step", [g2[0][0], g2[1][0], tvm[0]], [g2[0][1], g2[1][1], tvm[1]])
cell("c_dD", "identity", [[0.0], [-1.0], [-1.0]], [1.0, 0.0, 0.0])
cell("c_dd", "identity",
     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, -1.0]], [1.0, 1.0, -1.0, 0.0])
cluster("cl_update", ["cat", "s", "z"], [
    {"id": "X", "use": "op:concat", "in": {"x0": "$cat", "x1": "$s", "x2": "$z"}, "args": {"axis": 0}},
    {"id": "disc0", "use": "c_upd", "in": {"x": "X"}},
    {"id": "disc", "use": "op:step", "in": {"x": "disc0"}},
    {"id": "G", "use": "c_gate", "in": {"x": "X"}},
    {"id": "v", "use": "op:slice", "in": {"x": "$s"}, "args": {"axis": 0, "start": S_V, "stop": S_V + 1}},
    {"id": "d", "use": "op:slice", "in": {"x": "$s"}, "args": {"axis": 0, "start": S_D, "stop": S_D + 1}},
    {"id": "tv", "use": "op:slice", "in": {"x": "$z"}, "args": {"axis": 0, "start": 2, "stop": 3}},
    {"id": "vtd", "use": "op:concat", "in": {"x0": "v", "x1": "tv", "x2": "d"}, "args": {"axis": 0}},
    {"id": "Dv", "use": "c_dv", "in": {"x": "vtd"}},
    {"id": "Dd", "use": "c_dd", "in": {"x": "vtd"}},
    {"id": "gv", "use": "op:mul", "in": {"a": "G", "b": "Dv"}},
    {"id": "gd", "use": "op:mul", "in": {"a": "G", "b": "Dd"}},
    {"id": "sv", "use": "op:sum", "in": {"x": "gv"}, "args": {"axis": 0, "keepdims": True}},
    {"id": "sd", "use": "op:sum", "in": {"x": "gd"}, "args": {"axis": 0, "keepdims": True}},
    {"id": "v2", "use": "op:add", "in": {"a": "v", "b": "sv"}},
    {"id": "d2", "use": "op:add", "in": {"a": "d", "b": "sd"}},
    {"id": "D", "use": "op:slice", "in": {"x": "$s"}, "args": {"axis": 0, "start": S_DD, "stop": S_DD + 1}},
    {"id": "G2", "use": "c_gate2", "in": {"x": "X"}},
    {"id": "DD", "use": "c_dD", "in": {"x": "D"}},
    {"id": "gD", "use": "op:mul", "in": {"a": "G2", "b": "DD"}},
    {"id": "sD", "use": "op:sum", "in": {"x": "gD"}, "args": {"axis": 0, "keepdims": True}},
    {"id": "D2", "use": "op:add", "in": {"a": "D", "b": "sD"}},
    {"id": "snew", "use": "op:concat", "in": {"x0": "disc", "x1": "v2", "x2": "d2", "x3": "D2"}, "args": {"axis": 0}},
], {"y": "snew"})

# ---------------------------------------------------------------- trunk: features phi
# input concat[s_new(S), z_new(6), step(1)]
FI = {}
for i in range(S): FI["s:%d" % i] = i
for z, i in Z.items(): FI["z:" + z] = S + i
FI["step"] = S + NZ
NF_IN = S + NZ + 1
PHI = ["slot_noun", "sg", "pl", "any", "top_sg", "top_pl", "top_none", "endreg", "main",
       "rclast", "rcinner", "au", "rn", "slot_AND", "slot_PC", "slot_PP", "L_mid_or_late", "L_late", "D0", "D1", "D2"]
PH = {p: i for i, p in enumerate(PHI)}
def funit(pos, neg=(), thr=None):
    w = [0.0] * NF_IN
    for n in pos: w[FI[n]] += 1.0
    for n in neg: w[FI[n]] -= 1.0
    return w, -(len(pos) - 0.5)
fu = [funit(["s:%d" % S_SLOT["NOUN"]]), funit(["s:8"]), funit(["s:9"]), funit(["s:10"]),
      funit(["z:top_sg"]), funit(["z:top_pl"]), funit(["z:d_eq0"]),
      funit(["s:%d" % S_RN, "z:d_eq0"]),
      funit(["z:d_eq1"], ["s:%d" % S_MD]), funit(["z:d_eq1", "s:%d" % S_MD]), funit(["z:d_ge2"]),
      funit(["s:%d" % S_AU]), funit(["s:%d" % S_RN]), funit(["s:%d" % S_SLOT["AND"]]), funit(["s:%d" % S_SLOT["PC"]]),
      funit(["s:%d" % S_SLOT["PP"]])]
w = [0.0] * NF_IN; w[FI["step"]] = 1.0; fu.append((w, -P["L1"]))
w = [0.0] * NF_IN; w[FI["step"]] = 1.0; fu.append((w, -P["L2"]))
fu += [funit(["z:D_eq0"]), funit(["z:D_eq1"]), funit(["z:D_ge2"])]
cell("c_feat", "step", [u[0] for u in fu], [u[1] for u in fu])
cluster("cl_feat", ["s", "z", "step"], [
    {"id": "X", "use": "op:concat", "in": {"x0": "$s", "x1": "$z", "x2": "$step"}, "args": {"axis": 0}},
    {"id": "c", "use": "c_feat", "in": {"x": "X"}}], {"y": "c"})

# ---------------------------------------------------------------- router
EXPERTS = ["ex_det", "ex_adjn", "ex_post", "ex_verb", "ex_adv", "ex_func"]
RW = zeros(len(EXPERTS), S)
RW[1][S_SLOT["ADJN"]] = RW[1][S_SLOT["NOUN"]] = 1.0
RW[2][S_SLOT["POST"]] = 1.0
RW[3][S_SLOT["VERB"]] = 1.0
RW[4][S_SLOT["ADV"]] = 1.0
RW[5][S_SLOT["PC"]] = RW[5][S_SLOT["PP"]] = RW[5][S_SLOT["AND"]] = 1.0
cell("c_route", "identity", RW, [0.5, 0, 0, 0, 0, 0])
cluster("cl_route", ["s"], [{"id": "c", "use": "c_route", "in": {"x": "$s"}}], {"y": "c"})
put_def("router", {"kind": "bundle", "role": "router", "inputs": ["s"],
                   "nodes": [{"id": "r", "use": "cl_route", "in": {"s": "$s"}}],
                   "outputs": {"scores": "r"}})

# ---------------------------------------------------------------- shared log / output cells
knots = [0.0] + [2.0 ** (-k / 2.0) for k in range(70, -3, -1)]   # 0, 2^-35 ... 2^1
yk = [math.log(knots[1]) - 3.0] + [math.log(t) for t in knots[1:]]
slopes = [(yk[i + 1] - yk[i]) / (knots[i + 1] - knots[i]) for i in range(len(knots) - 1)]
coef = [slopes[0]] + [slopes[i] - slopes[i - 1] for i in range(1, len(slopes))]
cell("c_log", "relu", [[1.0] for _ in coef], [-t for t in knots[:-1]], [coef], [yk[0]])
mask = [0.0] * V; mask[0] = -60.0; mask[1] = -60.0
set_param("outmask", mask)

def situ_unit(pos, neg=()):
    w = [0.0] * len(PHI)
    for n in pos: w[PH[n]] += 1.0
    for n in neg: w[PH[n]] -= 1.0
    return w, -(len(pos) - 0.5)

def build_expert(name, units, sits, prior):
    """units: list of (conds, char);  sits: list of (pos,neg);  prior[u][s] logit."""
    W1 = []; b1 = []
    for conds, ch in units:
        w, b = match_row(conds); W1.append(w); b1.append(b)
    cell(name + "_m", "step", W1, b1)
    su = [situ_unit(*s) for s in sits]
    cell(name + "_s", "step", [u[0] for u in su], [u[1] for u in su])
    cell(name + "_p", "identity", prior, [0.0] * len(units))
    cm = zeros(V, len(units))
    for k, (conds, ch) in enumerate(units):
        cm[ch][k] = 1.0
    cell(name + "_c", "identity", cm, [0.0] * V)
    cluster("cl_" + name + "_match", ["x"], [{"id": "c", "use": name + "_m", "in": {"x": "$x"}}], {"y": "c"})
    cluster("cl_" + name + "_prior", ["phi"], [
        {"id": "s", "use": name + "_s", "in": {"x": "$phi"}},
        {"id": "p", "use": name + "_p", "in": {"x": "s"}},
        {"id": "e", "use": "op:exp", "in": {"x": "p"}}], {"y": "e"})
    cluster("cl_" + name + "_mix", ["m", "w"], [
        {"id": "u", "use": "op:mul", "in": {"a": "$m", "b": "$w"}},
        {"id": "c", "use": name + "_c", "in": {"x": "u"}}], {"y": "c"})
    put_def(name, {"kind": "bundle", "role": "expert", "inputs": ["x", "phi"], "nodes": [
        {"id": "m", "use": "cl_" + name + "_match", "in": {"x": "$x"}},
        {"id": "w", "use": "cl_" + name + "_prior", "in": {"phi": "$phi"}},
        {"id": "p", "use": "cl_" + name + "_mix", "in": {"m": "m", "w": "w"}},
        {"id": "lg", "use": "cl_logout", "in": {"p": "p"}}], "outputs": {"logits": "lg"}})

cluster("cl_logout", ["p"], [
    {"id": "s", "use": "op:sum", "in": {"x": "$p"}, "args": {"axis": 0, "keepdims": True}},
    {"id": "f", "use": "op:scale", "in": {"x": "s"}, "args": {"c": P["eps"]}},
    {"id": "q", "use": "op:add", "in": {"a": "$p", "b": "f"}},
    {"id": "q2", "use": "op:reshape", "in": {"x": "q"}, "args": {"shape": [V, 1]}},
    {"id": "l", "use": "c_log", "in": {"x": "q2"}},
    {"id": "l2", "use": "op:reshape", "in": {"x": "l"}, "args": {"shape": [V]}},
    {"id": "mk", "use": "param:outmask"},
    {"id": "o", "use": "op:add", "in": {"a": "l2", "b": "mk"}}], {"y": "o"})

LOG0 = P["impossible"]
def lg(p):
    return math.log(p) if p > 0 else LOG0

def word_units(words):
    """returns units list and index map: for word w: (prefix unit ids list, end unit dict char->id)."""
    units = []; idx = {}
    for w in words:
        pre = []
        for j in range(len(w)):
            units.append((prefix_conds(w[:j]), ci(w[j]))); pre.append(len(units) - 1)
        ends = {}
        for ch in [" ", ",", "."]:
            units.append((prefix_conds(w), ci(ch))); ends[ch] = len(units) - 1
        idx[w] = (pre, ends)
    return units, idx

def fill_prior(units, idx, nsit, table):
    """table: list over sits of dict word -> (pword, {endch: p})."""
    pr = [[LOG0] * nsit for _ in units]
    for s, d in enumerate(table):
        for w, (pw, ends) in d.items():
            pre, eu = idx[w]
            for u in pre:
                pr[u][s] = lg(pw)
            for ch, uid in eu.items():
                pr[uid][s] = lg(pw * ends.get(ch, 0.0)) if pw > 0 else LOG0
    return pr

def endchars(au, pspace):
    """clause could end here: P(' ')=pspace, rest split into ',' and '.'."""
    rest = 1.0 - pspace
    if au:
        return {" ": pspace, ".": rest}
    return {" ": pspace, ",": rest * P["comma"], ".": rest * (1 - P["comma"])}
SPONLY = {" ": 1.0}

# ---- DET expert
units, idx = word_units(DETS)
DETP = {"every": 0.13, "this": 0.135, "a": 0.115, "these": 0.10, "many": 0.105, "some": 0.095, "the": 0.32}
table = [{w: (DETP[w], SPONLY) for w in DETS}]
build_expert("ex_det", units, [([], [])], fill_prior(units, idx, 1, table))

# ---- ADJN expert
def modp(D, obj):
    if D == 2: return P["prep_2"], P["that_2"]
    if D == 1: return P["prep_1"], P["that_1"]
    return (P["prep_o"], P["that_o"]) if obj else (P["prep_s"], P["that_s"])
def pspace_end(D):
    pp, pt = modp(D, True); pm = pp + pt
    return pm + (1 - pm) * P["adv"]
nounwords = [n for pair in NOUNS for n in pair]
units, idx = word_units(ADJ + nounwords)
sits, table = [], []
ENDK = [("P", None, None)] + [("E", D, au) for D in range(3) for au in (0, 1)]
for sn in [0, 1]:
    for num in ["sg", "pl", "any"]:
        for endk, D, au in ENDK:
            pos = [num]; neg = []
            (pos if sn else neg).append("slot_noun")
            if endk == "P": neg.append("endreg")
            else:
                pos += ["endreg", "D%d" % D]; (pos if au else neg).append("au")
            sits.append((pos, neg))
            d = {}
            padj = 0.0 if sn else P["adj"]
            for a in ADJ: d[a] = (padj / len(ADJ), SPONLY)
            ne = SPONLY if endk == "P" else endchars(au, pspace_end(D))
            for sg, pl in NOUNS:
                if num == "sg": d[sg] = ((1 - padj) / 12, ne)
                elif num == "pl": d[pl] = ((1 - padj) / 12, ne)
                else:
                    d[sg] = ((1 - padj) * P["the_sg"] / 12, ne); d[pl] = ((1 - padj) * (1 - P["the_sg"]) / 12, ne)
            table.append(d)
build_expert("ex_adjn", units, sits, fill_prior(units, idx, len(sits), table))

# ---- verb situations shared by POST and VERB
def verb_table(top, kind, au, pverb_total):
    d = {}
    si = 0 if top == "sg" else 1
    if kind == "main":
        e_iv = endchars(au, P["adv"])
        for pair in IV: d[pair[si]] = (pverb_total * P["iv"] / len(IV), e_iv)
        for pair in TV: d[pair[si]] = (pverb_total * (1 - P["iv"]) / len(TV), SPONLY)
    elif kind == "rclast":
        e = endchars(au, P["adv"])
        for pair in TV: d[pair[si]] = (pverb_total / len(TV), e)
    else:
        for pair in TV: d[pair[si]] = (pverb_total / len(TV), SPONLY)
    return d
VSITS = []
for top in ["sg", "pl"]:
    for kind, au in [("main", 0), ("main", 1), ("rclast", 0), ("rclast", 1), ("rcinner", 0)]:
        pos = ["top_" + top, kind]; neg = []
        if kind != "rcinner": (pos if au else neg).append("au")
        VSITS.append((top, kind, au, pos, neg))
verbwords = [v for pair in IV + TV for v in pair]

# ---- POST expert
units, idx = word_units(PREP + ["that"] + verbwords + ADV)
sits, table = [], []
for top, kind, au, pos, neg in VSITS:
    for D in ([0, 1, 2] if kind == "main" else [1, 2]):
        sits.append((pos + ["D%d" % D], neg))
        pp, pt = modp(D, False)
        d = verb_table(top, kind, au, 1 - pp - pt)
        for p in PREP: d[p] = (pp / len(PREP), SPONLY)
        d["that"] = (pt, SPONLY)
        table.append(d)
for D in range(3):
    for au in [0, 1]:
        sits.append((["top_none", "D%d" % D] + (["au"] if au else []), [] if au else ["au"]))
        pp, pt = modp(D, True); pm = pp + pt
        pmod = pm / pspace_end(D)
        d = {p: (pmod * pp / pm / len(PREP) if pm > 0 else 0.0, SPONLY) for p in PREP}
        d["that"] = (pmod * pt / pm if pm > 0 else 0.0, SPONLY)
        for a in ADV: d[a] = ((1 - pmod) / len(ADV), endchars(au, 0.0))
        table.append(d)
build_expert("ex_post", units, sits, fill_prior(units, idx, len(sits), table))

# ---- VERB expert
units, idx = word_units(verbwords)
sits, table = [], []
for top, kind, au, pos, neg in VSITS:
    sits.append((pos, neg)); table.append(verb_table(top, kind, au, 1.0))
build_expert("ex_verb", units, sits, fill_prior(units, idx, len(sits), table))

# ---- ADV expert
units, idx = word_units(ADV)
sits = [([], ["au"]), (["au"], [])]
table = [{a: (1.0 / len(ADV), endchars(0, 0.0)) for a in ADV},
         {a: (1.0 / len(ADV), endchars(1, 0.0)) for a in ADV}]
build_expert("ex_adv", units, sits, fill_prior(units, idx, 2, table))

# ---- FUNC expert: 'and' after ", ", ' ' after ',', ' '/eos after '.'
units, idx = word_units(["and"])
nw = len(units)
units.append(([(7, [COM])], SP))              # after comma
units.append(([(7, [DOT])], SP))              # after period: continue
units.append(([(7, [DOT])], EOS))             # after period: end
sits = [(["slot_AND"], []), (["slot_PC"], []),
        (["slot_PP"], ["L_mid_or_late"]), (["slot_PP", "L_mid_or_late"], ["L_late"]),
        (["slot_PP", "L_late"], [])]
pr = fill_prior(units[:nw], idx, len(sits), [{"and": (1.0, SPONLY)}, {}, {}, {}, {}])
pr += [[LOG0] * len(sits) for _ in range(3)]
pr[nw][1] = 0.0
for s, pe in [(2, P["eos_early"]), (3, P["eos_mid"]), (4, P["eos_late"])]:
    pr[nw + 1][s] = lg(1 - pe); pr[nw + 2][s] = lg(pe)
build_expert("ex_func", units, sits, pr)

# ---------------------------------------------------------------- root
put_def("step_root", {"kind": "bundle", "inputs": ["ctx", "prev", "step", "state"], "nodes": [
    {"id": "x", "use": "op:reshape", "in": {"x": "$ctx"}, "args": {"shape": [256]}},
    {"id": "cat", "use": "cl_wordcls", "in": {"x": "x"}},
    {"id": "z0", "use": "cl_stack", "in": {"s": "$state"}},
    {"id": "s1", "use": "cl_update", "in": {"cat": "cat", "s": "$state", "z": "z0"}},
    {"id": "z1", "use": "cl_stack", "in": {"s": "s1"}},
    {"id": "phi", "use": "cl_feat", "in": {"s": "s1", "z": "z1", "step": "$step"}},
    {"id": "r", "use": "router", "in": {"s": "s1"}},
    {"id": "d", "use": "dispatch", "in": {"scores": "r.scores", "x": "x", "phi": "phi"},
     "args": {"branches": EXPERTS, "k": 1}}],
    "outputs": {"logits": "d.logits", "state": "s1"}})
ops.append({"op": "set_state_dim", "value": S})
ops.append({"op": "delete_param", "name": "boot_logits"})

prop = {"author": "r1-L1",
        "rationale": ("Grammar-state MoE: a trunk classifies each completed word from ctx and updates a "
                      "16-d recurrent parse state (slot, NP number, subject-role flag, main-verb-done, "
                      "and-used, pending-verb stack encoded base-3 in one scalar, stack depth). A router "
                      "maps the slot to one of 6 experts (det, adj/noun, post-noun, verb, adverb, "
                      "punctuation/and). Each expert matches the current word prefix against its own "
                      "lexicon, weights words by a hand-set prior chosen by parse situation, sums per "
                      "next char, and takes a piecewise-linear log."),
        "ops": ops}
with open(OUT, "w") as f:
    json.dump(prop, f)
print("wrote", OUT, "ops", len(ops), "params", P)
