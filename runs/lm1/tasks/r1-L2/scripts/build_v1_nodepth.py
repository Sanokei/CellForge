"""Expand my hand-typed grammar + probabilities into a ladder/1 proposal.

Nothing here reads data/tinylm/*. Word lists, grammar rules and every
probability below are typed by me (from reading dev_train.json by eye and
general English knowledge). The script only does exact bookkeeping:
enumerating the parser states my rules reach, and marginalising my word
distributions into per-prefix next-character tables.
"""
import json, math, sys, itertools
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "proposal.json"
CFG = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}

VOCAB = ["<pad>", "<bos>", "<eos>", " ", ",", "."] + [chr(ord('a') + i) for i in range(26)]
IX = {c: i for i, c in enumerate(VOCAB)}
V = 32
PAD, BOS, EOS, SP, CM, PD = 0, 1, 2, 3, 4, 5

# ---------------------------------------------------------------- my lexicon
DET_SG = ["every", "this", "a"]
DET_PL = ["these", "many", "some"]
DET_THE = ["the"]
ADJ = ["old", "brave", "tired", "red", "happy", "quiet", "small", "young"]
NOUNS = [("teacher", "teachers"), ("dog", "dogs"), ("bird", "birds"), ("king", "kings"),
         ("farmer", "farmers"), ("fox", "foxes"), ("boy", "boys"), ("child", "children"),
         ("horse", "horses"), ("cat", "cats"), ("girl", "girls"), ("mouse", "mice")]
PREP = ["with", "behind", "near", "under"]
TV = [("helps", "help"), ("follows", "follow"), ("finds", "find"), ("likes", "like"),
      ("chases", "chase"), ("sees", "see")]
IV = [("waits", "wait"), ("sings", "sing"), ("laughs", "laugh"), ("sleeps", "sleep"),
      ("jumps", "jump"), ("runs", "run")]
ADV = ["never", "often", "quickly", "slowly", "again"]

CLASSES = ["DET_SG", "DET_PL", "DET_THE", "ADJ", "N_SG", "N_PL", "PREP", "THAT",
           "TV_S", "TV_P", "IV_S", "IV_P", "ADV", "AND"]
CI = {c: i for i, c in enumerate(CLASSES)}
C = len(CLASSES)
WORDS = {}
for w in DET_SG: WORDS[w] = "DET_SG"
for w in DET_PL: WORDS[w] = "DET_PL"
WORDS["the"] = "DET_THE"
for w in ADJ: WORDS[w] = "ADJ"
for s, p in NOUNS: WORDS[s] = "N_SG"; WORDS[p] = "N_PL"
for w in PREP: WORDS[w] = "PREP"
WORDS["that"] = "THAT"
for s, p in TV: WORDS[s] = "TV_S"; WORDS[p] = "TV_P"
for s, p in IV: WORDS[s] = "IV_S"; WORDS[p] = "IV_P"
for w in ADV: WORDS[w] = "ADV"
WORDS["and"] = "AND"

# ---------------------------------------------------------------- my probabilities
P = dict(
    adj=0.30,            # adjective after a determiner
    mod_subj=0.33, that_subj=0.65,   # subject NP head noun -> modifier (RC share)
    mod_obj=0.35, that_obj=0.64,     # first-clause object NP
    mod_obj_conj=0.0,                # object NP of the clause after ", and" (route counts: never modified)
    mod_rs=0.15, that_rs=0.60,       # subject NP inside a relative clause
    mod_pp=0.125, that_pp=0.30,       # NP inside a prepositional phrase
    mod_end=0.02, that_end=0.30,     # right after a relative clause closes
    tv=0.58,             # main verb transitive
    adv=0.33,            # clause-final adverb
    comma=0.25,          # first clause joined by ", and"
    the_pl=0.5, det_the=0.25,         # 'the' followed by plural noun
    e_generic=0.001,     # backoff mass to a state-free prefix model
    e_uniform=0.0002,    # uniform floor
    eos_T=148.5, eos_k=0.6, eos_A=6.0,  # hand-counted doc lengths: continue at <=146, stop at >=151
)
P.update(CFG.get("P", {}))


def uni(ws, mass=1.0):
    return {w: mass / len(ws) for w in ws}


def dets(mass=1.0):
    # reading: 'the' ~ twice as common as each other determiner (27/107 subjects)
    pt = P["det_the"]
    d = uni(DET_SG + DET_PL, mass * (1 - pt))
    d["the"] = mass * pt
    return d


# ---------------------------------------------------------------- parser FSM
# state = (phase, dn, stack, conj, pp)
INIT = ("S0", None, (), 0, 0)
UNK = ("UNK", None, (), 0, 0)
MAXSTACK = 3


def dnum(cls):
    return {"DET_SG": "s", "DET_PL": "p", "DET_THE": "?"}[cls]


def step(s, c):
    ph, dn, st, conj, pp = s
    if ph in ("S0", "PERIOD"):
        if c.startswith("DET"):
            return ("DET", dnum(c), ("MS?",), conj if ph == "S0" else 0, 0)
        return UNK
    if ph in ("VT", "PREP", "THAT"):
        if c.startswith("DET"):
            return ("DET", dnum(c), st, conj, 1 if ph == "PREP" else 0)
        return UNK
    if ph in ("DET", "ADJ"):
        if c == "ADJ" and ph == "DET":
            return ("ADJ", dn, st, conj, pp)
        if c in ("N_SG", "N_PL"):
            top = st[-1]
            if top.endswith("?"):
                st = st[:-1] + (top[:-1] + ("s" if c == "N_SG" else "p"),)
            return ("N", None, st, conj, pp)
        return UNK
    if ph in ("N", "END"):
        top = st[-1] if st else ""
        if c == "PREP":
            return ("PREP", None, st, conj, 0)
        if c == "THAT":
            if len(st) >= MAXSTACK:
                st = st[:-1]
            return ("THAT", None, st + ("RS?",), conj, 0)
        if c in ("TV_S", "TV_P"):
            if top.startswith("RS"):
                return ("END", None, st[:-1], conj, 0)
            if top.startswith("MS"):
                return ("VT", None, st[:-1] + ("MO",), conj, 0)
            return UNK
        if c in ("IV_S", "IV_P"):
            if top.startswith("MS"):
                return ("CLE", None, (), conj, 0)
            return UNK
        if c == "ADV":
            if top == "MO":
                return ("ADV", None, (), conj, 0)
            return UNK
        return UNK
    if ph == "CLE":
        if c == "ADV":
            return ("ADV", None, (), conj, 0)
        return UNK
    if ph == "COMMA":
        if c == "AND":
            return ("S0", None, (), 1, 0)
        return UNK
    return UNK


COMMA_S = ("COMMA", None, (), 1, 0)
PERIOD_S = ("PERIOD", None, (), 0, 0)


def clause_end(conj, mass):
    """ADV / ',' / '.' at the end of a clause."""
    out = uni(ADV, mass * P["adv"])
    pc = 0.0 if conj else P["comma"]
    out[","] = mass * (1 - P["adv"]) * pc
    out["."] = mass * (1 - P["adv"]) * (1 - pc)
    return out


def punct(conj, mass=1.0):
    pc = 0.0 if conj else P["comma"]
    return {",": mass * pc, ".": mass * (1 - pc)}


def nouns(dn, mass):
    out = {}
    for s_, p_ in NOUNS:
        if dn == "s":
            out[s_] = mass / 12
        elif dn == "p":
            out[p_] = mass / 12
        else:
            out[s_] = mass * (1 - P["the_pl"]) / 12
            out[p_] = mass * P["the_pl"] / 12
    return out


def add(d, e):
    for k, v in e.items():
        d[k] = d.get(k, 0.0) + v
    return d


def follow(s):
    """My distribution over the next token (word, ',' or '.') in state s."""
    ph, dn, st, conj, pp = s
    if ph in ("S0", "VT", "PREP", "THAT", "PERIOD"):
        return dets()
    if ph == "DET":
        d = uni(ADJ, P["adj"])
        return add(d, nouns(dn, 1 - P["adj"]))
    if ph == "ADJ":
        return nouns(dn, 1.0)
    if ph in ("N", "END"):
        top = st[-1] if st else ""
        if ph == "END":
            pm, ts = P["mod_end"], P["that_end"]
        elif pp:
            pm, ts = P["mod_pp"], P["that_pp"]
        elif top.startswith("RS"):
            pm, ts = P["mod_rs"], P["that_rs"]
        elif top.startswith("MS"):
            pm, ts = P["mod_subj"], P["that_subj"]
        else:
            pm, ts = (P["mod_obj_conj"] if conj else P["mod_obj"]), P["that_obj"]
        d = {"that": pm * ts}
        add(d, uni(PREP, pm * (1 - ts)))
        rest = 1 - pm
        if top.startswith("RS"):
            n = top[2]
            if n == "s": d["tvs"] = rest
            elif n == "p": d["tvp"] = rest
            else: d["tvs"] = rest / 2; d["tvp"] = rest / 2
        elif top.startswith("MS"):
            n = top[2]
            ws = [0.5, 0.5] if n == "?" else ([1, 0] if n == "s" else [0, 1])
            for k, (ts, tp) in enumerate(TV):
                add(d, {ts: rest * P["tv"] * ws[0] / 6, tp: rest * P["tv"] * ws[1] / 6})
            for k, (is_, ip) in enumerate(IV):
                add(d, {is_: rest * (1 - P["tv"]) * ws[0] / 6, ip: rest * (1 - P["tv"]) * ws[1] / 6})
        elif top == "MO":
            add(d, clause_end(conj, rest))
        else:
            return None
        if "tvs" in d:
            m = d.pop("tvs"); add(d, {ts: m / 6 for ts, _ in TV})
        if "tvp" in d:
            m = d.pop("tvp"); add(d, {tp: m / 6 for _, tp in TV})
        return d
    if ph == "CLE":
        return clause_end(conj, 1.0)
    if ph == "ADV":
        return punct(conj)
    if ph == "COMMA":
        return {"and": 1.0}
    return None  # UNK


def after(s, tok):
    if tok == ",":
        return COMMA_S
    if tok == ".":
        return PERIOD_S
    return step(s, WORDS[tok])


# enumerate reachable states
states = [INIT]
seen = {INIT: 0}
q = [INIT]
while q:
    s = q.pop(0)
    nxt = [step(s, c) for c in CLASSES] + [COMMA_S, PERIOD_S, UNK]
    for t in nxt:
        if t not in seen:
            seen[t] = len(states); states.append(t); q.append(t)
K = len(states)
print("states", K, file=sys.stderr)

# ---------------------------------------------------------------- prefix detectors
allw = list(WORDS)
prefixes = []
for w in allw:
    for L in range(1, len(w) + 1):
        p = w[:L]
        if p not in prefixes:
            prefixes.append(p)
PFX = ["<bos>", " ", ",", "."] + prefixes
NPF = len(PFX)
print("prefix detectors", NPF, file=sys.stderr)


def detector_rows(p):
    """(weights over flattened ctx [8*32], bias) for an anchored prefix detector."""
    w = np.zeros((8, V))
    if p in ("<bos>", " ", ",", "."):
        w[7, IX[p]] = 1
        return w.reshape(-1), -0.5
    L = len(p)
    n = 0
    for i, ch in enumerate(p[-8:]):
        w[8 - min(L, 8) + i, IX[ch]] = 1; n += 1
    if L <= 7:
        w[7 - L, IX[" "]] = 1; w[7 - L, BOS] = 1; n += 1
    return w.reshape(-1), -(n - 0.5)


PW1 = np.zeros((NPF, 256)); PB1 = np.zeros(NPF)
for i, p in enumerate(PFX):
    PW1[i], PB1[i] = detector_rows(p)

# word-event detectors: word followed by ' ', ',' or '.'
WW1 = np.zeros((len(allw), 256)); WB1 = np.zeros(len(allw)); WW2 = np.zeros((C, len(allw)))
for i, wd in enumerate(allw):
    w = np.zeros((8, V)); n = 0
    vis = wd[-7:]
    for j, ch in enumerate(vis):
        w[7 - len(vis) + j, IX[ch]] = 1; n += 1
    w[7, SP] = w[7, CM] = w[7, PD] = 1; n += 1
    if len(wd) <= 6:
        w[6 - len(wd), SP] = 1; w[6 - len(wd), BOS] = 1; n += 1
    WW1[i] = w.reshape(-1); WB1[i] = -(n - 0.5)
    WW2[CI[WORDS[wd]], i] = 1

# ---------------------------------------------------------------- char tables
VALID = [i for i in range(V) if i not in (PAD, BOS)]


def term_dist(s2):
    f = follow(s2)
    out = np.zeros(V)
    if f is None:
        out[SP] = 0.85; out[CM] = 0.05; out[PD] = 0.10
        return out
    for k, v in f.items():
        out[CM if k == "," else PD if k == "." else SP] += v
    return out / out.sum()


def generic_table():
    T = np.zeros((NPF, V))
    wts = {w: 1.0 for w in allw}
    for i, p in enumerate(PFX):
        if p == "<bos>":
            for w in dets(): T[i, IX[w[0]]] += 1
        elif p == " ":
            for w in allw: T[i, IX[w[0]]] += 1
        elif p == ",":
            T[i, SP] = 1
        elif p == ".":
            T[i, SP] = 0.75; T[i, EOS] = 0.25
        else:
            for w in allw:
                if w.startswith(p):
                    if len(w) > len(p): T[i, IX[w[len(p)]]] += 1
                    else: T[i, SP] += 0.85; T[i, CM] += 0.05; T[i, PD] += 0.10
        T[i] /= T[i].sum()
    return T


GEN = generic_table()


def state_table(s):
    T = np.zeros((NPF, V))
    f = follow(s)
    words = {k: v for k, v in (f or {}).items() if k not in (",", ".")}
    Z = sum(words.values())
    for i, p in enumerate(PFX):
        row = np.zeros(V)
        if p == "<bos>":
            for w, v in dets().items(): row[IX[w[0]]] += v
        elif p == ",":
            row[SP] = 1
        elif p == ".":
            row[SP] = 0.5; row[EOS] = 0.5
        elif Z > 0:
            if p == " ":
                for w, v in words.items(): row[IX[w[0]]] += v
            else:
                for w, v in words.items():
                    if w.startswith(p):
                        if len(w) > len(p): row[IX[w[len(p)]]] += v
                        else: row += v * term_dist(after(s, w))
        if row.sum() <= 0:
            row = GEN[i].copy()
        row /= row.sum()
        mix = (1 - P["e_generic"] - P["e_uniform"]) * row + P["e_generic"] * GEN[i]
        mix[VALID] += P["e_uniform"] / len(VALID)
        lg = np.full(V, -30.0)
        lg[VALID] = np.log(mix[VALID])
        T[i] = lg
    return T


tables = [state_table(s) for s in states]
# group states whose tables are identical -> one expert each
groups = {}
for k, s in enumerate(states):
    key = np.round(tables[k], 6).tobytes()
    groups.setdefault(key, []).append(k)
glist = list(groups.values())
E = len(glist)
print("experts", E, file=sys.stderr)


def sname(s):
    ph, dn, st, conj, pp = s
    return f"{ph}{'_'+dn if dn else ''}{'_'+'.'.join(st) if st else ''}{'_c' if conj else ''}{'_pp' if pp else ''}"


def ename(g):
    ss = [states[k] for k in g]
    phs = sorted({s[0] for s in ss})
    tops = sorted({(s[2][-1] if s[2] else "-") for s in ss})
    return "ex_" + "-".join(phs) + "__" + "-".join(t.replace("?", "u") for t in tops)


enames = []
for g in glist:
    n = ename(g)
    base = n; j = 2
    while n in enames:
        n = f"{base}_{j}"; j += 1
    enames.append(n)

state_expert = np.zeros(K, dtype=int)
for e, g in enumerate(glist):
    for k in g: state_expert[k] = e
period_e = state_expert[seen[PERIOD_S]]

# ---------------------------------------------------------------- factored FSM
# joint state = phase part (ph, dn, pp) x stack part (stack, conj)
def phase_of(s): return (s[0], s[1], s[4])
def stack_of(s): return (s[2], s[3])
def topinfo(sk):
    st = sk[0]
    return "none" if not st else st[-1][:2]
def summary(sk):
    st, conj = sk
    top = st[-1] if st else "-"
    below = st[-2][:2] if len(st) > 1 else "-"
    return (top, below, conj)

OPS = ["push_MS", "set_s", "set_p", "push_RS", "pop", "MS2MO", "clear", "reset"]
def apply_op(op, sk):
    st, conj = sk
    if op == "id": return sk
    if op == "push_MS": return (("MS?",), conj)
    if op in ("set_s", "set_p"):
        if st and st[-1].endswith("?"):
            return (st[:-1] + (st[-1][:-1] + op[-1],), conj)
        return sk
    if op == "push_RS":
        if len(st) >= MAXSTACK: st = st[:-1]
        return (st + ("RS?",), conj)
    if op == "pop": return (st[:-1], conj)
    if op == "MS2MO": return (st[:-1] + ("MO",), conj)
    if op == "clear": return ((), conj)
    if op == "reset": return ((), 0)

PHASES = sorted({phase_of(s) for s in states}, key=lambda p: (p != phase_of(INIT), str(p)))
STACKS = sorted({stack_of(s) for s in states}, key=lambda p: (p != stack_of(INIT), str(p)))
PI = {p: i for i, p in enumerate(PHASES)}; SI = {p: i for i, p in enumerate(STACKS)}
TOPS = ["none", "MS", "RS", "MO"]; TI = {t: i for i, t in enumerate(TOPS)}
NP_, NT_ = len(PHASES), len(TOPS)
Tp = {}; To = {}
for s in states:
    for c in CLASSES:
        t = step(s, c)
        key = (PI[phase_of(s)], CI[c], TI[topinfo(stack_of(s))])
        pt = PI[phase_of(t)]
        assert Tp.setdefault(key, pt) == pt, ("phase inconsistency", s, c)
        op = "reset" if t == UNK else next(o for o in ["id"] + OPS if apply_op(o, stack_of(s)) == stack_of(t))
        prev_op = To.setdefault(key, op)
        if prev_op != op:
            assert apply_op(prev_op, stack_of(s)) == stack_of(t), ("op inconsistency", s, c, prev_op, op)
# close stacks under ops
i = 0
while i < len(STACKS):
    for o in OPS:
        r = apply_op(o, STACKS[i])
        if r not in SI:
            SI[r] = len(STACKS); STACKS.append(r)
    i += 1
NS_ = len(STACKS)
SUMS = []; UI = {}
for sk in STACKS:
    u = summary(sk)
    if u not in UI: UI[u] = len(SUMS); SUMS.append(u)
NU_ = len(SUMS)
route = {}
for k, s in enumerate(states):
    key = (PI[phase_of(s)], UI[summary(stack_of(s))])
    assert route.setdefault(key, state_expert[k]) == state_expert[k], ("route inconsistency", s)
unk_e = state_expert[seen[UNK]]
print("phases", NP_, "stacks", NS_, "summaries", NU_, file=sys.stderr)
# ---------------------------------------------------------------- proposal
ops = []


def sp(name, arr):
    arr = np.asarray(arr, dtype=float)
    nz = np.argwhere(arr != 0)
    if nz.shape[0] * (arr.ndim + 1) < arr.size:
        ops.append({"op": "set_param", "name": name, "pattern": {
            "sparse": [[int(x) for x in idx] + [round(float(arr[tuple(idx)]), 6)] for idx in nz],
            "shape": list(arr.shape)}})
    else:
        ops.append({"op": "set_param", "name": name, "value": np.round(arr, 6).tolist()})


def cell(name, params, act="identity"):
    ops.append({"op": "put_def", "name": name, "def": {"kind": "cell", "act": act, "params": params}})


def bundle(name, role, inputs, nodes, outputs):
    d = {"kind": "bundle", "inputs": inputs, "nodes": nodes, "outputs": outputs}
    if role: d["role"] = role
    ops.append({"op": "put_def", "name": name, "def": d})


# features cluster
sp("pfx.W1", PW1); sp("pfx.b1", PB1)
sp("wrd.W1", WW1); sp("wrd.b1", WB1); sp("wrd.W2", WW2); sp("wrd.b2", np.zeros(C))
G = np.zeros((2, V)); G[0, CM] = 1; G[1, PD] = 1
sp("gate.W1", G); sp("gate.b1", np.zeros(2))
cell("c_prefix", {"W1": "pfx.W1", "b1": "pfx.b1"}, "step")
cell("c_word", {"W1": "wrd.W1", "b1": "wrd.b1", "W2": "wrd.W2", "b2": "wrd.b2"}, "step")
cell("c_gate", {"W1": "gate.W1", "b1": "gate.b1"})
bundle("cl_features", "cluster", ["ctx"], [
    {"id": "flat", "use": "op:reshape", "in": {"x": "$ctx"}, "args": {"shape": [256]}},
    {"id": "last", "use": "op:slice", "in": {"x": "$ctx"}, "args": {"axis": 0, "start": 7, "stop": 8}},
    {"id": "lastf", "use": "op:reshape", "in": {"x": "last"}, "args": {"shape": [32]}},
    {"id": "pf", "use": "c_prefix", "in": {"x": "flat"}},
    {"id": "cls", "use": "c_word", "in": {"x": "flat"}},
    {"id": "g", "use": "c_gate", "in": {"x": "lastf"}},
], {"pf": "pf", "cls": "cls", "g": "g"})

# FSM cluster (factored: phase one-hot ++ stack one-hot)
def aug_mats(n):
    A = np.eye(n); A[0, :] -= 1
    b = np.zeros(n); b[0] = 1
    return A, b
A, bb = aug_mats(NP_); sp("fsm.paug.W1", A); sp("fsm.paug.b1", bb)
A, bb = aug_mats(NS_); sp("fsm.saug.W1", A); sp("fsm.saug.b1", bb)
Mt = np.zeros((NT_, NS_))
for sk, i in SI.items(): Mt[TI[topinfo(sk)], i] = 1
sp("fsm.top.W1", Mt); sp("fsm.top.b1", np.zeros(NT_))
Mp = np.zeros((NP_, NP_ * C * NT_)); Mo = np.zeros((len(OPS), NP_ * C * NT_))
unkp = PI[phase_of(UNK)]
for p in range(NP_):
    for c in range(C):
        for tp in range(NT_):
            col = (p * C + c) * NT_ + tp
            key = (p, c, tp)
            if key in Tp:
                Mp[Tp[key], col] = 1
                if To[key] != "id": Mo[OPS.index(To[key]), col] = 1
            else:
                Mp[unkp, col] = 1; Mo[OPS.index("reset"), col] = 1
sp("fsm.Tp", Mp); sp("fsm.Tp.b", np.zeros(NP_))
sp("fsm.To", Mo); sp("fsm.To.b", np.zeros(len(OPS)))
Ms = np.zeros((NS_, NS_ * len(OPS)))
for sk, i in SI.items():
    for oi, o in enumerate(OPS):
        Ms[SI[apply_op(o, sk)], i * len(OPS) + oi] = 1
sp("fsm.Ts", Ms); sp("fsm.Ts.b", np.zeros(NS_))
sp("fsm.none.W1", -np.ones((1, C))); sp("fsm.none.b1", [1.0])
sp("fsm.opnone.W1", -np.ones((1, len(OPS)))); sp("fsm.opnone.b1", [1.0])
sp("fsm.keep.W1", -np.ones((1, 2))); sp("fsm.keep.b1", [1.0])
Op_ = np.zeros((NP_, 2)); Op_[PI[phase_of(COMMA_S)], 0] = 1; Op_[PI[phase_of(PERIOD_S)], 1] = 1
Os_ = np.zeros((NS_, 2)); Os_[SI[stack_of(COMMA_S)], 0] = 1; Os_[SI[stack_of(PERIOD_S)], 1] = 1
sp("fsm.povr.W1", Op_); sp("fsm.povr.b1", np.zeros(NP_))
sp("fsm.sovr.W1", Os_); sp("fsm.sovr.b1", np.zeros(NS_))
for nm, W, b_ in [("paug", "fsm.paug.W1", "fsm.paug.b1"), ("saug", "fsm.saug.W1", "fsm.saug.b1"),
                  ("top", "fsm.top.W1", "fsm.top.b1"), ("Tp", "fsm.Tp", "fsm.Tp.b"), ("To", "fsm.To", "fsm.To.b"),
                  ("Ts", "fsm.Ts", "fsm.Ts.b"), ("none", "fsm.none.W1", "fsm.none.b1"),
                  ("opnone", "fsm.opnone.W1", "fsm.opnone.b1"), ("keep", "fsm.keep.W1", "fsm.keep.b1"),
                  ("povr", "fsm.povr.W1", "fsm.povr.b1"), ("sovr", "fsm.sovr.W1", "fsm.sovr.b1")]:
    cell("c_" + nm, {"W1": W, "b1": b_})
bundle("cl_fsm_in", "cluster", ["state"], [
    {"id": "ps", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": 0, "stop": NP_}},
    {"id": "ss", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": NP_, "stop": NP_ + NS_}},
    {"id": "pa", "use": "c_paug", "in": {"x": "ps"}},
    {"id": "sa", "use": "c_saug", "in": {"x": "ss"}},
    {"id": "top", "use": "c_top", "in": {"x": "sa"}},
], {"pa": "pa", "sa": "sa", "top": "top"})
bundle("cl_fsm_phase", "cluster", ["pa", "top", "cls"], [
    {"id": "p3", "use": "op:reshape", "in": {"x": "$pa"}, "args": {"shape": [NP_, 1, 1]}},
    {"id": "c3", "use": "op:reshape", "in": {"x": "$cls"}, "args": {"shape": [1, C, 1]}},
    {"id": "t3", "use": "op:reshape", "in": {"x": "$top"}, "args": {"shape": [1, 1, NT_]}},
    {"id": "pc", "use": "op:mul", "in": {"a": "p3", "b": "c3"}},
    {"id": "pct", "use": "op:mul", "in": {"a": "pc", "b": "t3"}},
    {"id": "key", "use": "op:reshape", "in": {"x": "pct"}, "args": {"shape": [NP_ * C * NT_]}},
    {"id": "np", "use": "c_Tp", "in": {"x": "key"}},
    {"id": "op", "use": "c_To", "in": {"x": "key"}},
    {"id": "none", "use": "c_none", "in": {"x": "$cls"}},
    {"id": "stay", "use": "op:mul", "in": {"a": "$pa", "b": "none"}},
    {"id": "newp", "use": "op:add", "in": {"a": "np", "b": "stay"}},
], {"phase": "newp", "op": "op"})
bundle("cl_fsm_stack", "cluster", ["sa", "op"], [
    {"id": "s2", "use": "op:expand", "in": {"x": "$sa"}, "args": {"axis": 1}},
    {"id": "o2", "use": "op:expand", "in": {"x": "$op"}, "args": {"axis": 0}},
    {"id": "so", "use": "op:mul", "in": {"a": "s2", "b": "o2"}},
    {"id": "sof", "use": "op:reshape", "in": {"x": "so"}, "args": {"shape": [NS_ * len(OPS)]}},
    {"id": "ns", "use": "c_Ts", "in": {"x": "sof"}},
    {"id": "none", "use": "c_opnone", "in": {"x": "$op"}},
    {"id": "stay", "use": "op:mul", "in": {"a": "$sa", "b": "none"}},
    {"id": "news", "use": "op:add", "in": {"a": "ns", "b": "stay"}},
], {"stack": "news"})
bundle("cl_fsm_punct", "cluster", ["phase", "stack", "g"], [
    {"id": "keep", "use": "c_keep", "in": {"x": "$g"}},
    {"id": "pk", "use": "op:mul", "in": {"a": "$phase", "b": "keep"}},
    {"id": "sk", "use": "op:mul", "in": {"a": "$stack", "b": "keep"}},
    {"id": "po", "use": "c_povr", "in": {"x": "$g"}},
    {"id": "so", "use": "c_sovr", "in": {"x": "$g"}},
    {"id": "p", "use": "op:add", "in": {"a": "pk", "b": "po"}},
    {"id": "s", "use": "op:add", "in": {"a": "sk", "b": "so"}},
    {"id": "st", "use": "op:concat", "in": {"x0": "p", "x1": "s"}, "args": {"axis": 0}},
], {"phase": "p", "stack": "s", "state": "st"})

# router: (phase x stack-summary) -> expert
Mu = np.zeros((NU_, NS_))
for sk, i in SI.items(): Mu[UI[summary(sk)], i] = 1
sp("route.sum.W1", Mu); sp("route.sum.b1", np.zeros(NU_))
R = np.zeros((E, NP_ * NU_))
for p in range(NP_):
    for u in range(NU_):
        R[route.get((p, u), unk_e), p * NU_ + u] = 1
sp("route.W1", R); sp("route.b1", np.zeros(E))
cell("c_sum", {"W1": "route.sum.W1", "b1": "route.sum.b1"})
cell("c_route", {"W1": "route.W1", "b1": "route.b1"})
bundle("cl_route", "cluster", ["phase", "stack"], [
    {"id": "u", "use": "c_sum", "in": {"x": "$stack"}},
    {"id": "p2", "use": "op:expand", "in": {"x": "$phase"}, "args": {"axis": 1}},
    {"id": "u2", "use": "op:expand", "in": {"x": "u"}, "args": {"axis": 0}},
    {"id": "pu", "use": "op:mul", "in": {"a": "p2", "b": "u2"}},
    {"id": "puf", "use": "op:reshape", "in": {"x": "pu"}, "args": {"shape": [NP_ * NU_]}},
    {"id": "s", "use": "c_route", "in": {"x": "puf"}},
], {"y": "s"})
bundle("router", "router", ["phase", "stack"], [{"id": "s", "use": "cl_route", "in": {"phase": "$phase", "stack": "$stack"}}],
       {"scores": "s.y"})

# experts
for e, g in enumerate(glist):
    tab = tables[g[0]]
    nm = enames[e]
    fb = np.full(V, -30.0); fb[VALID] = np.log(1.0 / len(VALID))
    sp(f"{nm}.W1", tab.T - fb[:, None]); sp(f"{nm}.b1", fb)
    cell(f"c_{nm}", {"W1": f"{nm}.W1", "b1": f"{nm}.b1"})
    cnodes = [{"id": "t", "use": f"c_{nm}", "in": {"x": "$pf"}}]
    bundle(f"cl_{nm}", "cluster", ["pf"], cnodes, {"y": "t"})
    enodes = [{"id": "lex", "use": f"cl_{nm}", "in": {"pf": "$pf"}}]
    out = "lex.y"
    if e == period_e:
        sp("eos.W1", [[P["eos_k"]]]); sp("eos.b1", [-P["eos_k"] * P["eos_T"]])  # tanh-bounded
        ew = np.zeros((V, 1)); ew[EOS, 0] = P["eos_A"]
        sp("eos.W2", ew); sp("eos.b2", np.zeros(V))
        cell("c_eos", {"W1": "eos.W1", "b1": "eos.b1", "W2": "eos.W2", "b2": "eos.b2"}, "tanh")
        bundle("cl_eos", "cluster", ["step", "g"], [
            {"id": "e", "use": "c_eos", "in": {"x": "$step"}},
            {"id": "dot", "use": "op:slice", "in": {"x": "$g"}, "args": {"axis": 0, "start": 1, "stop": 2}},
            {"id": "m", "use": "op:mul", "in": {"a": "e", "b": "dot"}},
        ], {"y": "m"})
        enodes.append({"id": "eos", "use": "cl_eos", "in": {"step": "$step", "g": "$g"}})
        enodes.append({"id": "sum", "use": "op:add", "in": {"a": "lex.y", "b": "eos.y"}})
        out = "sum"
    bundle(nm, "expert", ["pf", "g", "step"], enodes, {"logits": out})

bundle("step_root", None, ["ctx", "prev", "step", "state"], [
    {"id": "f", "use": "cl_features", "in": {"ctx": "$ctx"}},
    {"id": "fi", "use": "cl_fsm_in", "in": {"state": "$state"}},
    {"id": "fp", "use": "cl_fsm_phase", "in": {"pa": "fi.pa", "top": "fi.top", "cls": "f.cls"}},
    {"id": "fs", "use": "cl_fsm_stack", "in": {"sa": "fi.sa", "op": "fp.op"}},
    {"id": "fx", "use": "cl_fsm_punct", "in": {"phase": "fp.phase", "stack": "fs.stack", "g": "f.g"}},
    {"id": "r", "use": "router", "in": {"phase": "fx.phase", "stack": "fx.stack"}},
    {"id": "d", "use": "dispatch", "in": {"scores": "r.scores", "pf": "f.pf", "g": "f.g", "step": "$step"},
     "args": {"branches": enames, "k": 1}},
], {"logits": "d.logits", "state": "fx.state"})
ops.append({"op": "set_state_dim", "value": NP_ + NS_})
ops.append({"op": "delete_param", "name": "boot_logits"})

prop = {"author": "r1-L2", "rationale": CFG.get("rationale",
        "Hand-built parser-state MoE: trunk detects words and runs a hand-typed pushdown-like FSM "
        "(one-hot recurrent state); router maps parser state to an expert; each expert is a "
        "prefix-trie next-char table for one syntactic situation; period expert adds a length-based EOS term."),
        "ops": ops}
json.dump(prop, open(OUT, "w"))
info = {"K": K, "E": E, "NPF": NPF, "experts": {enames[e]: [sname(states[k]) for k in g] for e, g in enumerate(glist)}}
json.dump(info, open(OUT.replace(".json", "_info.json"), "w"), indent=1)
print(json.dumps({"K": K, "E": E, "NPF": NPF}), file=sys.stderr)
