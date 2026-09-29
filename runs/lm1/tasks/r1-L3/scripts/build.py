"""Expand my hand-written grammar (grammar.py) into a ladder/1 proposal.

No corpus is read here.  All numbers come from grammar.py or constants below.
usage: python build.py <out.json>
"""
import json
import math
import sys

import grammar as g

V = 32
W = len(g.WORDS)
K = 9
STATES = g.reachable()
N = len(STATES)
SIDX = {s: i for i, s in enumerate(STATES)}
EXPERTS = ["np", "post", "tail"]

# ---- hand-chosen constants ----
WORD_FLOOR = 0.00002       # every word keeps this mass in every state
CHAR_FLOOR = 1e-6          # added to every char before the log
LOG_KNOTS = 49             # knots at 2^-24 .. 2^0, ratio sqrt(2)
EOS_SLOPE = 6.0            # eos logit ramp per step
EOS_CENTER = 149.5         # docs end at the first period at/after char 150 (my manual counts)
EOS_CLIP = 14.0
ROUTE_PUNCT = 10.0


def sparse2(entries, shape, default=0.0):
    return {"sparse": [[i, j, float(v)] for (i, j), v in sorted(entries.items())],
            "shape": list(shape), "default": default}


def sparse1(entries, n, default=0.0):
    return {"sparse": [[i, float(v)] for i, v in sorted(entries.items())],
            "shape": [n], "default": default}


def sp(name, v):
    key = "pattern" if isinstance(v, dict) else "value"
    return {"op": "set_param", "name": name, key: v}


def cellp(name, W1, b1, W2=None, b2=None):
    ops = [sp(name + ".W1", W1), sp(name + ".b1", b1)]
    if W2 is not None:
        ops += [sp(name + ".W2", W2), sp(name + ".b2", b2)]
    return ops


def celldef(name, act, two=False):
    pn = name[2:] if name.startswith("c_") else name
    p = {"W1": pn + ".W1", "b1": pn + ".b1"}
    if two:
        p.update({"W2": pn + ".W2", "b2": pn + ".b2"})
    return {"op": "put_def", "name": name, "def": {"kind": "cell", "act": act, "params": p}}


def put(name, d):
    return {"op": "put_def", "name": name, "def": d}


ops = []

# ---------- lexical prefix units: u[w,k] = current partial word == w[:k] ----------
Wu, bu = {}, {}
Ch, Em = {}, {}
for wi, (w, _) in enumerate(g.WORDS):
    L = len(w)
    for k in range(K):
        r = wi * K + k
        if k > L:
            bu[r] = -1.0
            continue
        m = 0
        for j in range(k):
            row = 8 - k + j
            Wu[(r, row * V + g.VI[w[j]])] = 1.0
            m += 1
        brow = 7 - k
        if brow >= 0:
            Wu[(r, brow * V + g.VI[" "])] = 1.0
            Wu[(r, brow * V + g.VI["<bos>"])] = 1.0
            m += 1
        bu[r] = -(m - 0.5)
        if k < L:
            Ch[(wi, k * V + g.VI[w[k]])] = 1.0
        else:
            Em[(wi, k)] = 1.0
ops += cellp("lexu", sparse2(Wu, [W * K, 8 * V]), sparse1(bu, W * K))
ops.append({"op": "set_param", "name": "lex.chars",
            "pattern": dict(sparse2(Ch, [W, K * V]), reshape=[W, K, V])})
ops.append({"op": "set_param", "name": "lex.endmask", "pattern": sparse2(Em, [W, K])})
ops.append(celldef("c_lexu", "step"))
ops.append(put("cl_lex", {"kind": "bundle", "role": "cluster", "inputs": ["ctx"], "nodes": [
    {"id": "f", "use": "op:reshape", "in": {"x": "$ctx"}, "args": {"shape": [8 * V]}},
    {"id": "u", "use": "c_lexu", "in": {"x": "f"}},
    {"id": "U", "use": "op:reshape", "in": {"x": "u"}, "args": {"shape": [W, K]}},
    {"id": "U3", "use": "op:expand", "in": {"x": "U"}, "args": {"axis": 2}},
    {"id": "ch", "use": "param:lex.chars"},
    {"id": "m", "use": "op:mul", "in": {"a": "U3", "b": "ch"}},
    {"id": "C", "use": "op:sum", "in": {"x": "m"}, "args": {"axis": 1}},
    {"id": "em", "use": "param:lex.endmask"},
    {"id": "e", "use": "op:mul", "in": {"a": "U", "b": "em"}},
    {"id": "ue", "use": "op:sum", "in": {"x": "e"}, "args": {"axis": 1}}],
    "outputs": {"C": "C", "uend": "ue"}}))

# ---------- completed-word detector -> class ----------
Wz, bz, Wc = {}, {}, {}
for wi, (w, c) in enumerate(g.WORDS):
    L = len(w)
    m = 0
    Wz[(wi, 7 * V + g.VI[" "])] = 1.0
    m += 1
    for j, ch in enumerate(w):
        row = 7 - L + j
        if row < 0:
            continue
        Wz[(wi, row * V + g.VI[ch])] = 1.0
        m += 1
    brow = 6 - L
    if brow >= 0:
        Wz[(wi, brow * V + g.VI[" "])] = 1.0
        Wz[(wi, brow * V + g.VI["<bos>"])] = 1.0
        m += 1
    bz[wi] = -(m - 0.5)
    Wc[(g.WCLS.index(c), wi)] = 1.0
NW = len(g.WCLS)
NC = len(g.CLS)
ops += cellp("detz", sparse2(Wz, [W, 8 * V]), sparse1(bz, W),
             sparse2(Wc, [NW, W]), sparse1({}, NW))
Wk = {}
for i in range(NW):
    Wk[(i, i)] = 1.0
    Wk[(NC - 1, i)] = -1.0
for cname, tok in (("PERIOD", "."), ("COMMA", ","), ("BOS", "<bos>")):
    Wk[(g.CLS.index(cname), NW + g.VI[tok])] = 1.0
    Wk[(NC - 1, NW + g.VI[tok])] = -1.0
ops += cellp("detc", sparse2(Wk, [NC, NW + V]), sparse1({NC - 1: 1.0}, NC))
ops.append(celldef("c_detz", "step", two=True))
ops.append(celldef("c_detc", "identity"))
ops.append(put("cl_detect", {"kind": "bundle", "role": "cluster", "inputs": ["ctx", "prev"], "nodes": [
    {"id": "f", "use": "op:reshape", "in": {"x": "$ctx"}, "args": {"shape": [8 * V]}},
    {"id": "z", "use": "c_detz", "in": {"x": "f"}},
    {"id": "zc", "use": "op:concat", "in": {"x0": "z", "x1": "$prev"}, "args": {"axis": 0}},
    {"id": "c", "use": "c_detc", "in": {"x": "zc"}}],
    "outputs": {"cls": "c"}}))

# ---------- state transition (factored: phase register x stack register) ----------
def phi(s):
    return (s[0], s[1], s[2])


def sig(s):
    return (s[3], s[4])


def summ(s):  # what the phase update needs to know about the stack
    return 2 * int(bool(s[4])) + int(s[3] is None)


def rho(s):  # what the stack update needs to know about the phase
    return int(s[0] in ("A", "N") and s[1] == "S")


PH = sorted(set(map(phi, STATES)), key=str)
SG = sorted(set(map(sig, STATES)), key=str)
NP_, NS_ = len(PH), len(SG)
PI = {p: i for i, p in enumerate(PH)}
SI = {q: i for i, q in enumerate(SG)}
NST = NP_ + NS_
Tp, Ts, Msum, Mrho = {}, {}, {}, {}
for s in STATES:
    for ci, c in enumerate(g.CLS):
        if c == "BOS":
            continue
        t = g.trans(s, c)
        Tp[(PI[phi(t)], (PI[phi(s)] * NC + ci) * 4 + summ(s))] = 1.0
        Ts[(SI[sig(t)], (SI[sig(s)] * NC + ci) * 2 + rho(s))] = 1.0
    Msum[(summ(s), SI[sig(s)])] = 1.0
    Mrho[(rho(s), PI[phi(s)])] = 1.0
Tp[(PI[phi(g.START)], NP_ * NC * 4 + g.CLS.index("BOS"))] = 1.0
Ts[(SI[sig(g.START)], NS_ * NC * 2 + g.CLS.index("BOS"))] = 1.0
ops += cellp("tphase", sparse2(Tp, [NP_, NP_ * NC * 4 + NC]), sparse1({}, NP_))
ops += cellp("tstack", sparse2(Ts, [NS_, NS_ * NC * 2 + NC]), sparse1({}, NS_))
ops += cellp("ssum", sparse2(Msum, [4, NS_]), sparse1({}, 4))
ops += cellp("prho", sparse2(Mrho, [2, NP_]), sparse1({}, 2))
for nm in ("tphase", "tstack", "ssum", "prho"):
    ops.append(celldef("c_" + nm, "identity"))


def outer(prefix, a, b, na, nb):
    return [
        {"id": prefix + "a", "use": "op:expand", "in": {"x": a}, "args": {"axis": 1}},
        {"id": prefix + "b", "use": "op:expand", "in": {"x": b}, "args": {"axis": 0}},
        {"id": prefix + "m", "use": "op:mul", "in": {"a": prefix + "a", "b": prefix + "b"}},
        {"id": prefix + "f", "use": "op:reshape", "in": {"x": prefix + "m"}, "args": {"shape": [na * nb]}}]


# clause-2 flag: set by the comma before "and", cleared by period / bos
Tc = {}
for ci, c in enumerate(g.CLS):
    for old in (0, 1):
        if c == "BOS":
            continue
        new = 0 if c == "PERIOD" else (1 if c in ("COMMA", "AND") else old)
        Tc[(new, old * NC + ci)] = 1.0
Tc[(0, 2 * NC + g.CLS.index("BOS"))] = 1.0
ops += cellp("tc2", sparse2(Tc, [2, 2 * NC + NC]), sparse1({}, 2))
ops.append(celldef("c_tc2", "identity"))
FIRST_COMMA, FIRST_PERIOD = 0.225 / 0.18, 0.775 / 0.82   # table split is 0.18/0.82 overall
Ma = {(0, 0): 1.0, (1, 2): FIRST_COMMA, (2, 4): FIRST_PERIOD,     # c2 = 0 (first clause)
      (0, 1): 1.0, (2, 3): 1.0, (2, 5): 1.0}                      # c2 = 1: comma mass -> period
ops += cellp("c2adj", sparse2(Ma, [3, 6]), sparse1({}, 3))
ops.append(celldef("c_c2adj", "identity"))

ops.append(put("cl_trans", {"kind": "bundle", "role": "cluster", "inputs": ["state", "cls"], "nodes": [
    {"id": "ph", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": 0, "stop": NP_}},
    {"id": "sg", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": NP_, "stop": NST}},
    {"id": "sm", "use": "c_ssum", "in": {"x": "sg"}},
    {"id": "rh", "use": "c_prho", "in": {"x": "ph"}},
    {"id": "cc", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": NST, "stop": NST + 2}}]
    + outer("c1", "cc", "$cls", 2, NC) + [
    {"id": "cx", "use": "op:concat", "in": {"x0": "c1f", "x1": "$cls"}, "args": {"axis": 0}},
    {"id": "nc", "use": "c_tc2", "in": {"x": "cx"}}]
    + outer("p1", "ph", "$cls", NP_, NC) + outer("p2", "p1f", "sm", NP_ * NC, 4)
    + outer("s1", "sg", "$cls", NS_, NC) + outer("s2", "s1f", "rh", NS_ * NC, 2) + [
    {"id": "px", "use": "op:concat", "in": {"x0": "p2f", "x1": "$cls"}, "args": {"axis": 0}},
    {"id": "sx", "use": "op:concat", "in": {"x0": "s2f", "x1": "$cls"}, "args": {"axis": 0}},
    {"id": "np", "use": "c_tphase", "in": {"x": "px"}},
    {"id": "ns", "use": "c_tstack", "in": {"x": "sx"}},
    {"id": "st", "use": "op:concat", "in": {"x0": "np", "x1": "ns", "x2": "nc"}, "args": {"axis": 0}},
    {"id": "key", "use": "op:expand", "in": {"x": "np"}, "args": {"axis": 1}},
    {"id": "kb", "use": "op:expand", "in": {"x": "ns"}, "args": {"axis": 0}},
    {"id": "km", "use": "op:mul", "in": {"a": "key", "b": "kb"}},
    {"id": "kf", "use": "op:reshape", "in": {"x": "km"}, "args": {"shape": [NP_ * NS_]}}],
    "outputs": {"st": "st", "phase": "np", "key": "kf", "c2": "nc"}}))
NKEY = NP_ * NS_


def keyidx(s):
    return PI[phi(s)] * NS_ + SI[sig(s)]


# ---------- router ----------
Wr = {}
for p in PH:
    s = next(x for x in STATES if phi(x) == p)
    Wr[(EXPERTS.index(g.group(s)), PI[p])] = 1.0
Wr[(3, NP_ + g.VI["."])] = ROUTE_PUNCT
Wr[(3, NP_ + g.VI[","])] = ROUTE_PUNCT
ops += cellp("route", sparse2(Wr, [4, NP_ + V]), sparse1({}, 4))
ops.append(celldef("c_route", "identity"))
ops.append(put("cl_route", {"kind": "bundle", "role": "cluster", "inputs": ["phase", "prev"], "nodes": [
    {"id": "x", "use": "op:concat", "in": {"x0": "$phase", "x1": "$prev"}, "args": {"axis": 0}},
    {"id": "s", "use": "c_route", "in": {"x": "x"}}],
    "outputs": {"scores": "s"}}))
ops.append(put("router", {"kind": "bundle", "role": "router", "inputs": ["phase", "prev"], "nodes": [
    {"id": "r", "use": "cl_route", "in": {"phase": "$phase", "prev": "$prev"}}],
    "outputs": {"scores": "r.scores"}}))

# ---------- log approximation (shared by lexical experts) ----------
knots = [0.0] + [2.0 ** (i / 2.0 - 24.0) for i in range(LOG_KNOTS)]
vals = [math.log(t + CHAR_FLOOR) for t in knots]
slopes = [(vals[i + 1] - vals[i]) / (knots[i + 1] - knots[i]) for i in range(len(knots) - 1)]
a = [slopes[0]] + [slopes[i] - slopes[i - 1] for i in range(1, len(slopes))]
nk = len(slopes)
ops.append({"op": "set_param", "name": "log.W1", "value": [[1.0]] * nk})
ops.append({"op": "set_param", "name": "log.b1", "value": [-t for t in knots[:nk]]})
ops.append({"op": "set_param", "name": "log.W2", "value": [a]})
ops.append({"op": "set_param", "name": "log.b2", "value": [vals[0]]})
mask = [0.0] * V
mask[0] = mask[1] = -30.0
mask[2] = -20.0
ops.append({"op": "set_param", "name": "lex.mask", "value": mask})
ops.append(celldef("c_log", "relu", two=True))
ops.append(put("cl_log", {"kind": "bundle", "role": "cluster", "inputs": ["p"], "nodes": [
    {"id": "a", "use": "op:reshape", "in": {"x": "$p"}, "args": {"shape": [V, 1]}},
    {"id": "l", "use": "c_log", "in": {"x": "a"}},
    {"id": "b", "use": "op:reshape", "in": {"x": "l"}, "args": {"shape": [V]}},
    {"id": "mk", "use": "param:lex.mask"},
    {"id": "o", "use": "op:add", "in": {"a": "b", "b": "mk"}}],
    "outputs": {"logits": "o"}}))

# ---------- lexical experts ----------
WIDX = {w: i for i, (w, _) in enumerate(g.WORDS)}
WCLS_OF = {w: c for w, c in g.WORDS}


def end_split(s_next):
    d = g.dist(s_next)
    return d.get("END,", 0.0), d.get("END.", 0.0)


for e in EXPERTS:
    Pw, Tw = {}, {}
    for si, s in enumerate(STATES):
        if g.group(s) != e:
            continue
        d = g.dist(s)
        cont = 1.0 - d.get("END.", 0.0) - d.get("END,", 0.0)
        for w, p in d.items():
            if w.startswith("END"):
                continue
            pc = p / cont
            wi = WIDX[w]
            Pw[(wi, keyidx(s))] = pc
            ec, ep = end_split(g.trans(s, WCLS_OF[w]))
            for j, q in enumerate((1.0 - ec - ep, ec, ep)):
                if pc * q > 0:
                    Tw[(wi * 3 + j, keyidx(s))] = pc * q
    ops += cellp("P_" + e, sparse2(Pw, [W, NKEY]), {"fill": WORD_FLOOR, "shape": [W]})
    ops += cellp("T_" + e, sparse2(Tw, [W * 3, NKEY]),
                 {"tile": [WORD_FLOOR * 0.8, WORD_FLOOR * 0.1, WORD_FLOOR * 0.1], "reps": [W]})
    ops.append(celldef("c_P_" + e, "identity"))
    ops.append(celldef("c_T_" + e, "identity"))
    ops.append(put("cl_mix_" + e, {"kind": "bundle", "role": "cluster", "inputs": ["st", "C", "uend", "c2"], "nodes": [
        {"id": "P", "use": "c_P_" + e, "in": {"x": "$st"}},
        {"id": "Pr", "use": "op:reshape", "in": {"x": "P"}, "args": {"shape": [1, W]}},
        {"id": "v", "use": "op:matmul", "in": {"a": "Pr", "b": "$C"}},
        {"id": "vr", "use": "op:reshape", "in": {"x": "v"}, "args": {"shape": [V]}},
        {"id": "T", "use": "c_T_" + e, "in": {"x": "$st"}},
        {"id": "Tr", "use": "op:reshape", "in": {"x": "T"}, "args": {"shape": [W, 3]}},
        {"id": "ue", "use": "op:expand", "in": {"x": "$uend"}, "args": {"axis": 1}},
        {"id": "tm", "use": "op:mul", "in": {"a": "Tr", "b": "ue"}},
        {"id": "ts", "use": "op:sum", "in": {"x": "tm"}, "args": {"axis": 0}},
        {"id": "to", "use": "op:expand", "in": {"x": "ts"}, "args": {"axis": 1}},
        {"id": "tc", "use": "op:expand", "in": {"x": "$c2"}, "args": {"axis": 0}},
        {"id": "tq", "use": "op:mul", "in": {"a": "to", "b": "tc"}},
        {"id": "tf", "use": "op:reshape", "in": {"x": "tq"}, "args": {"shape": [6]}},
        {"id": "ta", "use": "c_c2adj", "in": {"x": "tf"}},
        {"id": "tp", "use": "op:pad", "in": {"x": "ta"}, "args": {"axis": 0, "before": 3, "after": V - 6, "value": 0.0}},
        {"id": "p", "use": "op:add", "in": {"a": "vr", "b": "tp"}}],
        "outputs": {"p": "p"}}))
    ops.append(put("ex_" + e, {"kind": "bundle", "role": "expert", "inputs": ["st", "C", "uend", "prev", "step", "c2"], "nodes": [
        {"id": "m", "use": "cl_mix_" + e, "in": {"st": "$st", "C": "$C", "uend": "$uend", "c2": "$c2"}},
        {"id": "l", "use": "cl_log", "in": {"p": "m.p"}}],
        "outputs": {"logits": "l.logits"}}))

# ---------- punctuation expert ----------
Wp = {}
for i in range(V):
    for col in (g.VI["."], g.VI[","]):
        Wp[(i, col)] = -15.0
Wp[(g.VI[" "], g.VI["."])] = 0.0
Wp[(g.VI["<eos>"], g.VI["."])] = 0.0
Wp[(g.VI[" "], g.VI[","])] = 0.0
ops += cellp("punct", sparse2(Wp, [V, V]), sparse1({0: -30.0, 1: -30.0}, V))
ops.append(celldef("c_punct", "identity"))
ops += cellp("eosramp", [[EOS_SLOPE], [EOS_SLOPE]],
             [-EOS_SLOPE * EOS_CENTER + EOS_CLIP, -EOS_SLOPE * EOS_CENTER - EOS_CLIP],
             [[1.0, -1.0]], [-EOS_CLIP])
ops.append(celldef("c_eosramp", "relu", two=True))
ops.append(put("cl_punct", {"kind": "bundle", "role": "cluster", "inputs": ["prev", "step"], "nodes": [
    {"id": "l", "use": "c_punct", "in": {"x": "$prev"}},
    {"id": "r", "use": "c_eosramp", "in": {"x": "$step"}},
    {"id": "pd", "use": "op:slice", "in": {"x": "$prev"}, "args": {"axis": 0, "start": 5, "stop": 6}},
    {"id": "m", "use": "op:mul", "in": {"a": "r", "b": "pd"}},
    {"id": "mp", "use": "op:pad", "in": {"x": "m"}, "args": {"axis": 0, "before": 2, "after": V - 3, "value": 0.0}},
    {"id": "o", "use": "op:add", "in": {"a": "l", "b": "mp"}}], "outputs": {"logits": "o"}}))
ops.append(put("ex_punct", {"kind": "bundle", "role": "expert", "inputs": ["st", "C", "uend", "prev", "step", "c2"], "nodes": [
    {"id": "l", "use": "cl_punct", "in": {"prev": "$prev", "step": "$step"}}], "outputs": {"logits": "l.logits"}}))

# ---------- root ----------
ops.append({"op": "set_state_dim", "value": NST + 2})
ops.append(put("step_root", {"kind": "bundle", "inputs": ["ctx", "prev", "step", "state"], "nodes": [
    {"id": "lex", "use": "cl_lex", "in": {"ctx": "$ctx"}},
    {"id": "det", "use": "cl_detect", "in": {"ctx": "$ctx", "prev": "$prev"}},
    {"id": "tr", "use": "cl_trans", "in": {"state": "$state", "cls": "det.cls"}},
    {"id": "r", "use": "router", "in": {"phase": "tr.phase", "prev": "$prev"}},
    {"id": "d", "use": "dispatch", "in": {"scores": "r.scores", "st": "tr.key", "C": "lex.C",
                                           "uend": "lex.uend", "prev": "$prev", "step": "$step", "c2": "tr.c2"},
     "args": {"branches": ["ex_np", "ex_post", "ex_tail", "ex_punct"], "k": 1}}],
    "outputs": {"logits": "d.logits", "state": "tr.st"}}))
ops.append({"op": "delete_param", "name": "boot_logits"})

prop = {"author": "r1-L3",
        "rationale": ("Hand-written grammar LM as a hierarchical MoE. Trunk clusters: lexical prefix "
                      "units over ctx (which lexicon words the current partial word could be), a "
                      "completed-word class detector, and a one-hot pushdown-style clause-state "
                      "transition (phase, NP role, determiner number, pending main-verb number, "
                      "pending relative-clause verb numbers). Router sends each position to an NP "
                      "expert (det/adj/noun), a post-noun expert (verbs/preps/that), a clause-tail "
                      "expert (adverbs/and/clause end) or a punctuation expert. Lexical experts mix "
                      "hand-set word probabilities through the prefix units and a piecewise-linear log."),
        "ops": ops}
json.dump(prop, open(sys.argv[1], "w"))
print("states", N, "phases", NP_, "stacks", NS_, "words", W, "ops", len(ops))
