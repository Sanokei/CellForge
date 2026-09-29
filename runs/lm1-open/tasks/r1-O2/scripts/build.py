"""Expand a hand-typed word grammar into CellForge proposal params (task r1-O2).

Nothing here reads the corpus. Word lists and every probability below were
typed by hand after reading samples of data/tinylm/dev_train.json.
The script only turns those typed choices into lookup tables:
  * phase / agreement-stack / clause-index transition tables (driven by word events)
  * word-completion detectors and word-prefix detectors over the 8-token ctx window
  * log P(next char | distribution-class, prefix) table
Usage: python build.py out.json
"""
import json, math, sys

# ------------------------------------------------------------------ vocab
V = ['<pad>', '<bos>', '<eos>', ' ', ',', '.'] + [chr(c) for c in range(97, 123)]
VI_ = {c: i for i, c in enumerate(V)}
PAD, BOS, EOS, SP, COM, PER = 0, 1, 2, 3, 4, 5

# ------------------------------------------------------------------ typed word lists
DET = {'every': 's', 'this': 's', 'a': 's', 'these': 'p', 'many': 'p', 'some': 'p', 'the': 'u'}
ADJ = ['old', 'brave', 'tired', 'red', 'small', 'quiet', 'happy', 'young']
NOUN = [('teacher', 'teachers'), ('dog', 'dogs'), ('bird', 'birds'), ('king', 'kings'),
        ('farmer', 'farmers'), ('fox', 'foxes'), ('boy', 'boys'), ('child', 'children'),
        ('horse', 'horses'), ('cat', 'cats'), ('girl', 'girls'), ('mouse', 'mice')]
VI = [('waits', 'wait'), ('sings', 'sing'), ('laughs', 'laugh'), ('sleeps', 'sleep'),
      ('jumps', 'jump'), ('runs', 'run')]
VT = [('helps', 'help'), ('follows', 'follow'), ('finds', 'find'), ('likes', 'like'),
      ('chases', 'chase'), ('sees', 'see')]
PREP = ['with', 'behind', 'near', 'under']
ADV = ['never', 'often', 'quickly', 'slowly', 'again']

# ------------------------------------------------------------------ typed probabilities
P = dict(
    adj=0.34,                 # P(adjective | after determiner)
    # at most one modifier per NP; (P(prep), P(that)) by NP type
    mod={'MS': (0.14, 0.17),  # main-clause subject
         'OB': (0.08, 0.17),  # object of main verb
         'PPN': (0.10, 0.20), # object of a preposition
         'RS': (0.08, 0.10)}, # subject of a relative clause
    mod_decay=1.0,            # per extra level of relative-clause nesting
    vi=0.52,                  # intransitive share of main verbs
    adv=0.27,                 # adverb at clause end
    and_=0.22,                # ", and" instead of "." after first clause
    det_w={'every': 1.2, 'this': 1, 'a': 1, 'these': 1, 'many': 1, 'some': 1, 'the': 2.0},
    the_pl=0.5,               # P(plural noun | 'the')
    eps_word=0.003,           # word-level mix with "any word" fallback
    eps_char=0.0005,          # char-level mix with uniform
)
MAXDEPTH = 4
EOS_SLOPE = 0.5      # logit per char
EOS_THR = 147.5      # text length where P(eos | '.') = 0.5

CLASSES = ['DET_S', 'DET_P', 'DET_U', 'ADJ', 'N_S', 'N_P', 'VI_S', 'VI_P', 'VT_S', 'VT_P',
           'PREP', 'ADV', 'THAT', 'AND']
TERMS = [' ', ',', '.']
WORDCLASS = {}
for w, n in DET.items():
    WORDCLASS[w] = {'s': 'DET_S', 'p': 'DET_P', 'u': 'DET_U'}[n]
for w in ADJ: WORDCLASS[w] = 'ADJ'
for s, p in NOUN: WORDCLASS[s] = 'N_S'; WORDCLASS[p] = 'N_P'
for s, p in VI: WORDCLASS[s] = 'VI_S'; WORDCLASS[p] = 'VI_P'
for s, p in VT: WORDCLASS[s] = 'VT_S'; WORDCLASS[p] = 'VT_P'
for w in PREP: WORDCLASS[w] = 'PREP'
for w in ADV: WORDCLASS[w] = 'ADV'
WORDCLASS['that'] = 'THAT'; WORDCLASS['and'] = 'AND'
WORDS = list(WORDCLASS)
EVENTS = [None] + [(c, t) for c in CLASSES for t in TERMS]

# ------------------------------------------------------------------ registers
# phase: (name, typ, n).  typ = NP type for DET/ADJ/NOUN (NOUN typ DONE = NP finished)
TYPES = ['MS', 'OB', 'PPN', 'RS']
OPS_ = ['none', 'pushM', 'pushR', 'set_s', 'set_p', 'pop', 'clear']
TOPK = ['none', 'M', 'R']


def all_phases():
    out = [('CS', '-', '-'), ('PERIOD', '-', '-'), ('COMMA', '-', '-'), ('PREP', '-', '-'),
           ('THAT', '-', '-'), ('VT', '-', '-'), ('VI', '-', '-'), ('ADV', '-', '-')]
    for t in TYPES + ['DONE']:
        out.append(('NOUN', t, '-'))
    for ph in ('DET', 'ADJ'):
        for t in TYPES:
            for n in ('s', 'p', 'u'):
                out.append((ph, t, n))
    return out


def all_stacks():
    res = {()}
    level = [()]
    for depth in range(MAXDEPTH):
        nxt = []
        for st in level:
            if st and st[-1][1] == 'u':
                continue
            for kd in (['M', 'R'] if depth == 0 else ['R']):
                for num in ('s', 'p', 'u'):
                    ns = st + ((kd, num),)
                    if ns not in res:
                        res.add(ns); nxt.append(ns)
        level = nxt
    return sorted(res, key=lambda x: (len(x), x))


def op_of(ph, cls, term):
    name, typ, n = ph
    if term in (',', '.') or cls == 'AND':
        return 'clear'
    if cls.startswith('DET') and name in ('CS', 'PERIOD', 'COMMA'):
        return 'pushM'
    if cls in ('N_S', 'N_P') and name in ('DET', 'ADJ') and typ in ('MS', 'RS'):
        return 'set_s' if cls == 'N_S' else 'set_p'
    if cls.startswith('VI') or cls.startswith('VT'):
        return 'pop'
    if cls == 'THAT':
        return 'pushR'
    return 'none'


def apply_op(st, op):
    st = list(st)
    if op == 'pushM':
        return (('M', 'u'),)
    if op == 'pushR':
        if len(st) < MAXDEPTH:
            st.append(('R', 'u'))
        return tuple(st)
    if op in ('set_s', 'set_p'):
        if st:
            st[-1] = (st[-1][0], op[-1])
        return tuple(st)
    if op == 'pop':
        return tuple(st[:-1])
    if op == 'clear':
        return ()
    return tuple(st)


def fix_stack(st, SI):
    if st in SI:
        return SI[st]
    st = tuple((k, 's' if (n == 'u' and i < len(st) - 1) else n) for i, (k, n) in enumerate(st))
    return SI.get(st, SI[()])


def summary(st):
    if not st:
        return ('none',)
    return (st[-1][0], st[-1][1], len(st) == 1, len(st) >= MAXDEPTH)


def topkind(st):
    return st[-1][0] if st else 'none'


def trans_phase(ph, cls, term, tk):
    name, typ, n = ph
    if term == ',':
        return ('COMMA', '-', '-')
    if term == '.':
        return ('PERIOD', '-', '-')
    if cls.startswith('DET'):
        dn = {'DET_S': 's', 'DET_P': 'p', 'DET_U': 'u'}[cls]
        t = {'CS': 'MS', 'PERIOD': 'MS', 'COMMA': 'MS', 'THAT': 'RS', 'PREP': 'PPN',
             'VT': 'OB'}.get(name, 'OB')
        return ('DET', t, dn)
    if cls == 'ADJ':
        if name == 'DET':
            return ('ADJ', typ, n)
        return ('ADJ', 'OB', 'u')
    if cls in ('N_S', 'N_P'):
        return ('NOUN', typ if name in ('DET', 'ADJ') else 'DONE', '-')
    if cls.startswith('VI'):
        return ('VI', '-', '-')
    if cls.startswith('VT'):
        return ('NOUN', 'DONE', '-') if tk == 'R' else ('VT', '-', '-')
    if cls == 'PREP':
        return ('PREP', '-', '-')
    if cls == 'THAT':
        return ('THAT', '-', '-')
    if cls == 'ADV':
        return ('ADV', '-', '-')
    if cls == 'AND':
        return ('CS', '-', '-')
    return ph


def trans_ci(ci, cls, term):
    if term == '.':
        return 0
    if cls == 'AND':
        return 1
    return ci


# ------------------------------------------------------------------ word-unit distributions
def end_terms(ci):
    if ci == 0:
        return {',': P['and_'], '.': 1 - P['and_']}
    return {'.': 1.0}


def done_terms(stack_after, ci):
    """terminator after the last word of a finished NP (noun w/o modifier, or RC verb)."""
    if stack_after:
        return {' ': 1.0}
    d = {' ': P['adv']}
    for t, q in end_terms(ci).items():
        d[t] = (1 - P['adv']) * q
    return d


def modp(typ, stack):
    """modifier probs, decayed by nesting depth beyond the type's usual depth."""
    pp, pt = P['mod'][typ]
    nR = sum(1 for k, _ in stack if k == 'R')
    base = {'PPN': 0, 'RS': 1}.get(typ)
    if base is not None and nR > base:
        f = P['mod_decay'] ** (nR - base)
        pp, pt = pp * f, pt * f
    return pp, pt


def det_dist():
    tot = sum(P['det_w'].values())
    return {(w, ' '): q / tot for w, q in P['det_w'].items()}


def noun_dist(ph, stack, ci, adj_ok):
    name, typ, n = ph
    out = {}
    pa = P['adj'] if adj_ok else 0.0
    for a in ADJ:
        out[(a, ' ')] = pa / len(ADJ)
    nums = {'s': {'s': 1.0}, 'p': {'p': 1.0}, 'u': {'s': 1 - P['the_pl'], 'p': P['the_pl']}}[n]
    pm = sum(modp(typ, stack))
    dt = done_terms(stack, ci)
    terms = {t: (1 - pm) * q for t, q in dt.items()}
    terms[' '] = terms.get(' ', 0) + pm
    for k, qk in nums.items():
        for s, p in NOUN:
            w = s if k == 's' else p
            for t, qt in terms.items():
                out[(w, t)] = out.get((w, t), 0) + (1 - pa) * qk / len(NOUN) * qt
    return out


def after_noun_dist(ph, stack, ci):
    """next word after noun+' ' (typ given) or after closing RC verb+' ' (typ DONE)."""
    typ = ph[1]
    pp, pt = modp(typ, stack) if typ != 'DONE' else (0.0, 0.0)
    if len(stack) >= MAXDEPTH:
        pt = 0.0
    out = {}
    if not stack:
        pa = (1 - pp - pt) * P['adv']
        z = pp + pt + pa
        for w in PREP: out[(w, ' ')] = pp / z / len(PREP)
        out[('that', ' ')] = pt / z
        for w in ADV:
            for t, q in end_terms(ci).items():
                out[(w, t)] = pa / z / len(ADV) * q
        return out
    kind, k = stack[-1]
    for w in PREP: out[(w, ' ')] = pp / len(PREP)
    out[('that', ' ')] = pt
    pv = 1 - pp - pt
    ks = [k] if k in ('s', 'p') else ['s', 'p']
    for kk in ks:
        qk = 1.0 / len(ks)
        if kind == 'R':
            vt_terms = done_terms(stack[:-1], ci)
            pvt = pv
        else:
            vt_terms = {' ': 1.0}
            pvt = pv * (1 - P['vi'])
        for s, p in VT:
            w = s if kk == 's' else p
            for t, q in vt_terms.items():
                out[(w, t)] = out.get((w, t), 0) + pvt * qk / len(VT) * q
        if kind == 'M':
            vi_terms = done_terms((), ci)
            for s, p in VI:
                w = s if kk == 's' else p
                for t, q in vi_terms.items():
                    out[(w, t)] = out.get((w, t), 0) + pv * P['vi'] * qk / len(VI) * q
    return out


def adv_dist(ci):
    out = {}
    for w in ADV:
        for t, q in end_terms(ci).items():
            out[(w, t)] = q / len(ADV)
    return out


def unit_dist(ph, stack, ci):
    name = ph[0]
    if name in ('CS', 'PREP', 'THAT', 'VT', 'PERIOD'):
        return det_dist()
    if name == 'DET':
        return noun_dist(ph, stack, ci, True)
    if name == 'ADJ':
        return noun_dist(ph, stack, ci, False)
    if name == 'NOUN':
        return after_noun_dist(ph, stack, ci)
    if name in ('VI', 'ADV'):
        return adv_dist(ci)
    if name == 'COMMA':
        return {('and', ' '): 1.0}
    raise ValueError(name)


ANY = {(w, t): 1.0 / (len(WORDS) * 3) for w in WORDS for t in TERMS}

prefixes = {''}
for w in WORDS:
    for i in range(1, len(w) + 1):
        prefixes.add(w[:i])
TNODES = sorted(prefixes, key=lambda s: (len(s), s)) + [',', '.']
VALID = [EOS, SP, COM, PER] + list(range(6, 32))


def char_logits(name, ud, t):
    probs = [0.0] * 32
    if t == ',':
        probs[SP] = 1.0
    elif t == '.':
        if name == 'PERIOD':
            probs[SP] = 0.5; probs[EOS] = 0.5   # step-dependent eos term added in graph
        else:
            probs[SP] = 1.0
    else:
        mix = {}
        for u, q in ud.items(): mix[u] = mix.get(u, 0) + (1 - P['eps_word']) * q
        for u, q in ANY.items(): mix[u] = mix.get(u, 0) + P['eps_word'] * q
        L = len(t)
        for (w, term), q in mix.items():
            if not w.startswith(t):
                continue
            if len(w) > L:
                probs[VI_[w[L]]] += q
            else:
                probs[VI_[term]] += q
        z = sum(probs)
        probs = [p / z for p in probs]
    e = P['eps_char']
    out = []
    for c in range(32):
        if c in (PAD, BOS):
            out.append(-30.0)
        else:
            out.append(math.log((1 - e) * probs[c] + e / len(VALID)))
    if t == '.' and name == 'PERIOD':
        out[EOS] = out[SP]
    return out


# ------------------------------------------------------------------ build tables
def build():
    PH = all_phases(); PI = {p: i for i, p in enumerate(PH)}; nP = len(PH)
    SK = all_stacks(); SI = {s: i for i, s in enumerate(SK)}; nK = len(SK)
    E1 = len(EVENTS); nO = len(OPS_)
    op_tab, ph_tab, ci_tab = [], [], []
    for pi, ph in enumerate(PH):
        for ei, ev in enumerate(EVENTS):
            col = pi * E1 + ei
            if ev is None:
                op_tab.append([0, col, 1.0])
                for ki in range(3):
                    ph_tab.append([pi, col * 3 + ki, 1.0])
                continue
            op_tab.append([OPS_.index(op_of(ph, *ev)), col, 1.0])
            for ki, tk in enumerate(TOPK):
                ph_tab.append([PI[trans_phase(ph, ev[0], ev[1], tk)], col * 3 + ki, 1.0])
    for ci in (0, 1):
        for ei, ev in enumerate(EVENTS):
            c2 = ci if ev is None else trans_ci(ci, *ev)
            ci_tab.append([c2, ci * E1 + ei, 1.0])
    sk_tab, tk_tab = [], []
    for si, st in enumerate(SK):
        for oi, op in enumerate(OPS_):
            sk_tab.append([fix_stack(apply_op(st, op), SI), si * nO + oi, 1.0])
        tk_tab.append([TOPK.index(topkind(st)), si, 1.0])
    # stack summary (all that the next-word distribution needs from the stack)
    SUM = []
    for st in SK:
        sm = summary(st)
        if sm not in SUM:
            SUM.append(sm)
    nS = len(SUM)
    sm_tab = [[SUM.index(summary(st)), si, 1.0] for si, st in enumerate(SK)]
    REP = {}
    for st in SK:
        REP.setdefault(summary(st), st)
    # distribution classes over (phase, summary, ci)
    tables, dmap, tbl_index = [], [], {}
    for pi, ph in enumerate(PH):
        for si, sm in enumerate(SUM):
            st = REP[sm]
            for ci in (0, 1):
                ud = unit_dist(ph, st, ci)
                key = (ph[0] if ph[0] in ('PERIOD', 'COMMA') else 'X',
                       tuple(sorted((k, round(v, 12)) for k, v in ud.items())))
                if key not in tbl_index:
                    tbl_index[key] = len(tables)
                    tables.append([char_logits(ph[0], ud, t) for t in TNODES])
                dmap.append([tbl_index[key], (pi * nS + si) * 2 + ci, 1.0])
    nD = len(tables); nT = len(TNODES)
    L = []
    for d, tbl in enumerate(tables):
        for ti, lg in enumerate(tbl):
            for c, v in enumerate(lg):
                if v != 0.0:
                    L.append([c, d * nT + ti, round(v, 5)])
    det, detb = [], []
    for ti, t in enumerate(TNODES):
        if t in (',', '.'):
            det.append([ti, 7 * 32 + VI_[t], 1.0]); detb.append(-0.5)
            continue
        Lt = len(t)
        for i, ch in enumerate(t):
            det.append([ti, (8 - Lt + i) * 32 + VI_[ch], 1.0])
        need = Lt
        if Lt <= 7:
            r = 7 - Lt
            det.append([ti, r * 32 + SP, 1.0]); det.append([ti, r * 32 + BOS, 1.0])
            need += 1
        detb.append(-need + 0.5)
    ev1, ev1b, ev2 = [], [], []
    k = 0
    for w in WORDS:
        for ti, term in enumerate(TERMS):
            s = w[-7:]
            Ls = len(s)
            for i, ch in enumerate(s):
                ev1.append([k, (7 - Ls + i) * 32 + VI_[ch], 1.0])
            need = Ls + 1
            ev1.append([k, 7 * 32 + VI_[term], 1.0])
            if len(w) <= 6:
                r = 6 - len(w)
                ev1.append([k, r * 32 + SP, 1.0]); ev1.append([k, r * 32 + BOS, 1.0])
                need += 1
            ev1b.append(-need + 0.5)
            ev2.append([CLASSES.index(WORDCLASS[w]) * 3 + ti, k, 1.0])
            k += 1
    return dict(nS=nS, sm_tab=sm_tab, nP=nP, nK=nK, E1=E1, nO=nO, op_tab=op_tab, ph_tab=ph_tab, ci_tab=ci_tab,
                sk_tab=sk_tab, tk_tab=tk_tab, dmap=dmap, nD=nD, nT=nT, L=L, det=det, detb=detb,
                ev1=ev1, ev1b=ev1b, ev2=ev2, nW=k,
                init_ph=[[PI[('CS', '-', '-')], BOS, 1.0]], init_sk=[[SI[()], BOS, 1.0]],
                init_ci=[[0, BOS, 1.0]])


def outer(nodes, a, b, na, nb, out):
    nodes += [
        {"id": out + "_a", "use": "op:expand", "in": {"x": a}, "args": {"axis": 1}},
        {"id": out + "_b", "use": "op:expand", "in": {"x": b}, "args": {"axis": 0}},
        {"id": out + "_m", "use": "op:matmul", "in": {"a": out + "_a", "b": out + "_b"}},
        {"id": out, "use": "op:reshape", "in": {"x": out + "_m"}, "args": {"shape": [na * nb]}}]


def proposal(B):
    nP, nK, E1, nO, nD, nT, nW = B['nP'], B['nK'], B['E1'], B['nO'], B['nD'], B['nT'], B['nW']
    E = E1 - 1
    sp = lambda entries, shape: {"sparse": entries, "shape": shape, "default": 0.0}
    z = lambda n: {"fill": 0.0, "shape": [n]}
    ops = []
    def P_(name, pat):
        ops.append({"op": "set_param", "name": name, "pattern": pat})
    P_("gr.initp.W", sp(B['init_ph'], [nP, 32])); P_("gr.initp.b", z(nP))
    P_("gr.inits.W", sp(B['init_sk'], [nK, 32])); P_("gr.inits.b", z(nK))
    P_("gr.initc.W", sp(B['init_ci'], [2, 32])); P_("gr.initc.b", z(2))
    P_("gr.op.W", sp(B['op_tab'], [nO, nP * E1])); P_("gr.op.b", z(nO))
    P_("gr.ph.W", sp(B['ph_tab'], [nP, nP * E1 * 3])); P_("gr.ph.b", z(nP))
    P_("gr.ci.W", sp(B['ci_tab'], [2, 2 * E1])); P_("gr.ci.b", z(2))
    P_("gr.sk.W", sp(B['sk_tab'], [nK, nK * nO])); P_("gr.sk.b", z(nK))
    P_("gr.tk.W", sp(B['tk_tab'], [3, nK])); P_("gr.tk.b", z(3))
    P_("ev.W1", sp(B['ev1'], [nW, 256]))
    P_("ev.b1", sp([[i, v] for i, v in enumerate(B['ev1b'])], [nW]))
    P_("ev.W2", sp(B['ev2'], [E, nW])); P_("ev.b2", z(E))
    P_("pf.W", sp(B['det'], [nT, 256]))
    P_("pf.b", sp([[i, v] for i, v in enumerate(B['detb'])], [nT]))
    nS = B['nS']
    P_("gr.sm.W", sp(B['sm_tab'], [nS, nK])); P_("gr.sm.b", z(nS))
    P_("dm.W", sp(B['dmap'], [nD, nP * nS * 2])); P_("dm.b", z(nD))
    h = (nD + 1) // 2
    LA = [[c, j, v] for c, j, v in B['L'] if j < h * nT]
    LB = [[c, j - h * nT, v] for c, j, v in B['L'] if j >= h * nT]
    P_("outA.W", sp(LA, [32, h * nT])); P_("outA.b", z(32))
    P_("outB.W", sp(LB, [32, (nD - h) * nT])); P_("outB.b", z(32))
    P_("eos.lin.W", sp([[0, 0, EOS_SLOPE]], [1, 1]))
    P_("eos.lin.b", sp([[0, -EOS_SLOPE * EOS_THR]], [1]))
    P_("eos.put.W", sp([[EOS, 0, 1.0]], [32, 1])); P_("eos.put.b", z(32))
    P_("one", {"fill": 1.0, "shape": [1]})

    def cell(name, act, W1, b1, W2=None, b2=None):
        p = {"W1": W1, "b1": b1}
        if W2: p["W2"] = W2; p["b2"] = b2
        ops.append({"op": "put_def", "name": name, "def": {"kind": "cell", "act": act, "params": p}})
    for nm, pr in [("c_initp", "gr.initp"), ("c_inits", "gr.inits"), ("c_initc", "gr.initc"),
                   ("c_op", "gr.op"), ("c_phase", "gr.ph"), ("c_ci", "gr.ci"),
                   ("c_stack", "gr.sk"), ("c_topk", "gr.tk"), ("c_sum", "gr.sm"), ("c_dclass", "dm"),
                   ("c_outA", "outA"), ("c_outB", "outB"), ("c_eoslin", "eos.lin"), ("c_eosput", "eos.put")]:
        cell(nm, "identity", pr + ".W", pr + ".b")
    cell("c_event", "step", "ev.W1", "ev.b1", "ev.W2", "ev.b2")
    cell("c_prefix", "step", "pf.W", "pf.b")

    ops.append({"op": "put_def", "name": "word_events", "def": {"kind": "bundle",
        "inputs": ["cf"], "nodes": [
            {"id": "ev", "use": "c_event", "in": {"x": "$cf"}},
            {"id": "evs", "use": "op:sum", "in": {"x": "ev"}, "args": {"axis": 0, "keepdims": True}},
            {"id": "one", "use": "param:one"},
            {"id": "none", "use": "op:sub", "in": {"a": "one", "b": "evs"}},
            {"id": "evf", "use": "op:concat", "in": {"x0": "none", "x1": "ev"}, "args": {"axis": 0}}],
        "outputs": {"ev": "evf"}}})
    gn = [
        {"id": "p0", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": 0, "stop": nP}},
        {"id": "s0", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": nP, "stop": nP + nK}},
        {"id": "c0", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": nP + nK, "stop": nP + nK + 2}},
        {"id": "pi", "use": "c_initp", "in": {"x": "$prev"}},
        {"id": "si", "use": "c_inits", "in": {"x": "$prev"}},
        {"id": "cI", "use": "c_initc", "in": {"x": "$prev"}},
        {"id": "ph", "use": "op:add", "in": {"a": "p0", "b": "pi"}},
        {"id": "sk", "use": "op:add", "in": {"a": "s0", "b": "si"}},
        {"id": "ci", "use": "op:add", "in": {"a": "c0", "b": "cI"}},
        {"id": "we", "use": "word_events", "in": {"cf": "$cf"}}]
    outer(gn, "ph", "we.ev", nP, E1, "pe")
    outer(gn, "ci", "we.ev", 2, E1, "ce")
    gn += [{"id": "op", "use": "c_op", "in": {"x": "pe"}},
           {"id": "tk", "use": "c_topk", "in": {"x": "sk"}}]
    outer(gn, "sk", "op", nK, nO, "so")
    outer(gn, "pe", "tk", nP * E1, 3, "pet")
    gn += [{"id": "sk2", "use": "c_stack", "in": {"x": "so"}},
           {"id": "ph2", "use": "c_phase", "in": {"x": "pet"}},
           {"id": "ci2", "use": "c_ci", "in": {"x": "ce"}},
           {"id": "st", "use": "op:concat", "in": {"x0": "ph2", "x1": "sk2", "x2": "ci2"}, "args": {"axis": 0}}]
    ops.append({"op": "put_def", "name": "grammar", "def": {"kind": "bundle",
        "inputs": ["cf", "prev", "state"], "nodes": gn,
        "outputs": {"phase": "ph2", "stack": "sk2", "ci": "ci2", "state": "st"}}})
    spn = []
    spn.append({"id": "sm", "use": "c_sum", "in": {"x": "$stack"}})
    outer(spn, "$phase", "sm", nP, nS, "ps")
    outer(spn, "ps", "$ci", nP * nS, 2, "psc")
    spn += [{"id": "d", "use": "c_dclass", "in": {"x": "psc"}},
            {"id": "t", "use": "c_prefix", "in": {"x": "$cf"}}]
    spn += [{"id": "dA", "use": "op:slice", "in": {"x": "d"}, "args": {"axis": 0, "start": 0, "stop": h}},
            {"id": "dB", "use": "op:slice", "in": {"x": "d"}, "args": {"axis": 0, "start": h, "stop": nD}}]
    outer(spn, "dA", "t", h, nT, "dtA")
    outer(spn, "dB", "t", nD - h, nT, "dtB")
    spn += [{"id": "lA", "use": "c_outA", "in": {"x": "dtA"}},
            {"id": "lB", "use": "c_outB", "in": {"x": "dtB"}},
            {"id": "lg", "use": "op:add", "in": {"a": "lA", "b": "lB"}}]
    ops.append({"op": "put_def", "name": "speller", "def": {"kind": "bundle",
        "inputs": ["phase", "stack", "ci", "cf"], "nodes": spn, "outputs": {"logits": "lg"}}})
    ops.append({"op": "put_def", "name": "eos_timer", "def": {"kind": "bundle",
        "inputs": ["step", "prev"], "nodes": [
            {"id": "lin", "use": "c_eoslin", "in": {"x": "$step"}},
            {"id": "isp", "use": "op:slice", "in": {"x": "$prev"}, "args": {"axis": 0, "start": 5, "stop": 6}},
            {"id": "m", "use": "op:mul", "in": {"a": "lin", "b": "isp"}},
            {"id": "v", "use": "c_eosput", "in": {"x": "m"}}],
        "outputs": {"delta": "v"}}})
    ops.append({"op": "put_def", "name": "step_root", "def": {"kind": "bundle",
        "inputs": ["ctx", "prev", "step", "state"], "nodes": [
            {"id": "cf", "use": "op:reshape", "in": {"x": "$ctx"}, "args": {"shape": [256]}},
            {"id": "gr", "use": "grammar", "in": {"cf": "cf", "prev": "$prev", "state": "$state"}},
            {"id": "sp", "use": "speller", "in": {"phase": "gr.phase", "stack": "gr.stack", "ci": "gr.ci", "cf": "cf"}},
            {"id": "eo", "use": "eos_timer", "in": {"step": "$step", "prev": "$prev"}},
            {"id": "lg", "use": "op:add", "in": {"a": "sp.logits", "b": "eo.delta"}}],
        "outputs": {"logits": "lg", "state": "gr.state"}}})
    ops.append({"op": "set_state_dim", "value": nP + nK + 2})
    ops.append({"op": "delete_param", "name": "boot_logits"})
    return {"author": "r1-O2", "rationale": RATIONALE, "ops": ops}


RATIONALE = ("Hand-built grammar LM. Recurrent one-hot registers (phase incl. NP type, agreement stack "
             "of pending verb numbers, clause index) are updated by word-completion events detected in "
             "the ctx window; a prefix detector bank identifies the partial word; next-char logits are "
             "a lookup over (distribution class x prefix) computed from hand-typed word lists and "
             "hand-estimated probabilities (one modifier max per NP); eos after '.' uses a length ramp.")

if __name__ == '__main__':
    B = build()
    print('P', B['nP'], 'K', B['nK'], 'S', B['nS'], 'D', B['nD'], 'T', B['nT'], 'W', B['nW'], 'Lnnz', len(B['L']),
          file=sys.stderr)
    with open(sys.argv[1], 'w') as f:
        json.dump(proposal(B), f)
