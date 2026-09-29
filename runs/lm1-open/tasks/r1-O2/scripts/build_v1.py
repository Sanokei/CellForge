"""Expand a hand-typed word grammar into CellForge proposal params.

Nothing here reads the corpus. Word lists and every probability below were
typed by hand after reading samples of data/tinylm/dev_train.json.
The script only turns those typed choices into lookup tables:
  * grammar-state transition table (state x word-event -> state)
  * word-prefix detectors over the 8-token ctx window
  * log P(next char | distribution-class, prefix) table
Usage: python build.py [out.json]
"""
import json, math, sys
from itertools import product

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
    adj=0.30,          # P(adjective | after determiner)
    prep_M=0.12, that_M=0.12,   # modifiers after a noun while main verb pending
    prep_R=0.08, that_R=0.05,   # modifiers after a noun while an RC verb pending
    prep_O=0.10, that_O=0.12,   # modifiers after an object-side noun (no verb pending)
    vi=0.50,           # intransitive share of main verbs
    adv=0.30,          # adverb at clause end
    and_=0.30,         # ", and" instead of "." after first clause
    det_w={'every': 1, 'this': 1, 'a': 1, 'these': 1, 'many': 1, 'some': 1, 'the': 1},
    the_pl=0.5,        # P(plural noun | 'the')
    eps_word=0.01,     # word-level mix with "any word" fallback
    eps_char=0.002,    # char-level mix with uniform
)
MAXDEPTH = 4

# classes of word events
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

# ------------------------------------------------------------------ grammar states
# state = (phase, stack, ci, role, n)
#   phase in CS, DET, ADJ, NOUN, PREP, THAT, VT, VI, ADV, COMMA, PERIOD
#   stack: tuple of (kind, num) with kind M (main verb) / R (relative verb), num u/s/p
#   ci: clause index 0/1 ; role H (subject head) / O (other) for DET/ADJ ; n det number
START = ('CS', (), 0, '-', '-')


def norm(phase, stack, ci, role='-', n='-'):
    return (phase, tuple(stack), ci, role, n)


def trans(g, cls, term):
    phase, stack, ci, role, n = g
    stack = list(stack)
    if term == ',':
        return norm('COMMA', (), 0)
    if term == '.':
        return norm('PERIOD', (), 0)
    if cls in ('DET_S', 'DET_P', 'DET_U'):
        dn = {'DET_S': 's', 'DET_P': 'p', 'DET_U': 'u'}[cls]
        if phase in ('CS', 'PERIOD', 'COMMA'):
            return norm('DET', [('M', 'u')], ci if phase == 'CS' else 0, 'H', dn)
        if phase == 'THAT':
            return norm('DET', stack, ci, 'H', dn)
        return norm('DET', stack, ci, 'O', dn)
    if cls == 'ADJ':
        if phase == 'DET':
            return norm('ADJ', stack, ci, role, n)
        return norm('ADJ', stack, ci, 'O', 'u')
    if cls in ('N_S', 'N_P'):
        k = 's' if cls == 'N_S' else 'p'
        if phase in ('DET', 'ADJ') and role == 'H' and stack:
            stack[-1] = (stack[-1][0], k)
        return norm('NOUN', stack, ci)
    if cls in ('VI_S', 'VI_P', 'VT_S', 'VT_P'):
        popped = stack.pop() if stack else ('M', 'u')
        if cls.startswith('VI'):
            return norm('VI', stack, ci)
        if popped[0] == 'M':
            return norm('VT', stack, ci)
        return norm('NOUN', stack, ci)
    if cls == 'PREP':
        return norm('PREP', stack, ci)
    if cls == 'THAT':
        if len(stack) < MAXDEPTH:
            stack.append(('R', 'u'))
        return norm('THAT', stack, ci)
    if cls == 'ADV':
        return norm('ADV', stack, ci)
    if cls == 'AND':
        return norm('CS', (), 1)
    return g


def enumerate_states():
    """BFS over events that the typed grammar gives nonzero probability."""
    seen = {START: 0}
    order = [START]
    i = 0
    while i < len(order):
        g = order[i]; i += 1
        for (w, term), q in unit_dist(g).items():
            if q <= 0:
                continue
            h = trans(g, WORDCLASS[w], term)
            if h not in seen:
                seen[h] = len(order); order.append(h)
    return order


def resolve(h, GI):
    """map an off-grammar successor onto an enumerated state."""
    if h in GI:
        return GI[h]
    phase, stack, ci, role, n = h
    for cand in [(phase, stack, 0, role, n), (phase, (), ci, role, n), (phase, (), 0, role, n),
                 (phase, (('M', 'u'),), 0, 'H', n), (phase, (), 0, 'O', n),
                 (phase, (), 0, 'O', 'u'), (phase, (('M', 'u'),), 0, 'H', 'u')]:
        if cand in GI:
            return GI[cand]
    return GI[START]


# ------------------------------------------------------------------ word-unit distributions
def end_terms(ci):
    """terminator distribution when a clause may end here (after optional adv)."""
    if ci == 0:
        return {',': P['and_'], '.': 1 - P['and_']}
    return {'.': 1.0}


def obj_noun_terms(stack_after, ci):
    """terminator after an object-side noun / closing RC verb."""
    if stack_after:
        return {' ': 1.0}
    cont = P['prep_O'] + P['that_O'] + (1 - P['prep_O'] - P['that_O']) * P['adv']
    d = {' ': cont}
    for t, q in end_terms(ci).items():
        d[t] = (1 - cont) * q
    return d


def det_dist():
    tot = sum(P['det_w'].values())
    return {(w, ' '): q / tot for w, q in P['det_w'].items()}


def noun_dist(g, adj_ok):
    phase, stack, ci, role, n = g
    out = {}
    pa = P['adj'] if adj_ok else 0.0
    for a in ADJ:
        out[(a, ' ')] = pa / len(ADJ)
    nums = {'s': {'s': 1.0}, 'p': {'p': 1.0}, 'u': {'s': 1 - P['the_pl'], 'p': P['the_pl']}}[n]
    if role == 'H':
        terms = {' ': 1.0}
    else:
        terms = obj_noun_terms(stack, ci)
    for k, qk in nums.items():
        for s, p in NOUN:
            w = s if k == 's' else p
            for t, qt in terms.items():
                out[(w, t)] = out.get((w, t), 0) + (1 - pa) * qk / len(NOUN) * qt
    return out


def after_noun_dist(g):
    phase, stack, ci, role, n = g
    out = {}
    if not stack:
        pp, pt = P['prep_O'], P['that_O']
        pa = (1 - pp - pt) * P['adv']
        z = pp + pt + pa
        for w in PREP: out[(w, ' ')] = pp / z / len(PREP)
        out[('that', ' ')] = pt / z
        for w in ADV:
            for t, q in end_terms(ci).items():
                out[(w, t)] = pa / z / len(ADV) * q
        return out
    kind, k = stack[-1]
    if k == 'u':
        k = None
    if kind == 'M':
        pp, pt = P['prep_M'], P['that_M']
    else:
        pp, pt = P['prep_R'], P['that_R']
    if len(stack) >= MAXDEPTH:
        pt = 0.0
    for w in PREP: out[(w, ' ')] = pp / len(PREP)
    out[('that', ' ')] = pt
    pv = 1 - pp - pt
    ks = [k] if k else ['s', 'p']
    for kk in ks:
        qk = 1.0 / len(ks)
        vt_terms = obj_noun_terms(stack[:-1], ci) if kind == 'R' else {' ': 1.0}
        pvt = pv * (1 - P['vi']) if kind == 'M' else pv
        for s, p in VT:
            w = s if kk == 's' else p
            for t, q in vt_terms.items():
                out[(w, t)] = out.get((w, t), 0) + pvt * qk / len(VT) * q
        if kind == 'M':
            vi_terms = {' ': P['adv']}
            for t, q in end_terms(ci).items():
                vi_terms[t] = (1 - P['adv']) * q
            for s, p in VI:
                w = s if kk == 's' else p
                for t, q in vi_terms.items():
                    out[(w, t)] = out.get((w, t), 0) + pv * P['vi'] * qk / len(VI) * q
    return out


def adv_dist(g):
    out = {}
    for w in ADV:
        for t, q in end_terms(g[2]).items():
            out[(w, t)] = q / len(ADV)
    return out


def unit_dist(g):
    phase = g[0]
    if phase in ('CS', 'PREP', 'THAT', 'VT', 'PERIOD'):
        return det_dist()
    if phase == 'DET':
        return noun_dist(g, True)
    if phase == 'ADJ':
        return noun_dist(g, False)
    if phase == 'NOUN':
        return after_noun_dist(g)
    if phase in ('VI', 'ADV'):
        return adv_dist(g)
    if phase == 'COMMA':
        return {('and', ' '): 1.0}
    raise ValueError(phase)


ANY = {}
for w in WORDS:
    for t in TERMS:
        ANY[(w, t)] = 1.0 / (len(WORDS) * 3)

# ------------------------------------------------------------------ prefixes (trie nodes)
prefixes = {''}
for w in WORDS:
    for i in range(1, len(w) + 1):
        prefixes.add(w[:i])
TNODES = sorted(prefixes, key=lambda s: (len(s), s)) + [',', '.']
TIDX = {t: i for i, t in enumerate(TNODES)}
VALID = [EOS, SP, COM, PER] + list(range(6, 32))


def char_logits(g, t):
    phase = g[0]
    probs = [0.0] * 32
    if t == ',':
        probs[SP] = 1.0
    elif t == '.':
        if phase == 'PERIOD':
            probs[SP] = 0.5; probs[EOS] = 0.5   # step-dependent eos term added in graph
        else:
            probs[SP] = 1.0
    else:
        ud = unit_dist(g)
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
        if z <= 0:
            probs = [0.0] * 32
            for c in VALID: probs[c] = 1.0
            z = len(VALID)
        probs = [p / z for p in probs]
    e = P['eps_char']
    out = []
    for c in range(32):
        if c in (PAD, BOS):
            out.append(-30.0)
        else:
            p = (1 - e) * probs[c] + e / len(VALID)
            out.append(math.log(p))
    if t == '.' and phase == 'PERIOD':
        out[EOS] = out[SP]   # equal; step term decides
    return out


# ------------------------------------------------------------------ factored state
# phase register: (phase, ci, role, n) ; stack register: tuple of (kind, num)
OPS_ = ['none', 'pushM', 'pushR', 'set_s', 'set_p', 'pop', 'clear']
TOPK = ['none', 'M', 'R']


def all_phases():
    out = [('COMMA', 0, '-', '-'), ('PERIOD', 0, '-', '-')]
    for ci in (0, 1):
        for ph in ('CS', 'NOUN', 'PREP', 'THAT', 'VT', 'VI', 'ADV'):
            out.append((ph, ci, '-', '-'))
        for ph in ('DET', 'ADJ'):
            for role in ('H', 'O'):
                for n in ('s', 'p', 'u'):
                    out.append((ph, ci, role, n))
    return out


def all_stacks():
    res = {()}
    level = [()]
    for depth in range(MAXDEPTH):
        nxt = []
        for st in level:
            if st and st[-1][1] == 'u':
                continue   # only the top entry may be unset
            kinds = ['M', 'R'] if depth == 0 else ['R']
            for kd in kinds:
                for num in ('s', 'p', 'u'):
                    ns = st + ((kd, num),)
                    if ns not in res:
                        res.add(ns); nxt.append(ns)
        level = nxt
    return sorted(res, key=lambda x: (len(x), x))


def op_of(ph, cls, term):
    phase, ci, role, n = ph
    if term in (',', '.') or cls == 'AND':
        return 'clear'
    if cls in ('DET_S', 'DET_P', 'DET_U') and phase in ('CS', 'PERIOD', 'COMMA'):
        return 'pushM'
    if cls in ('N_S', 'N_P') and phase in ('DET', 'ADJ') and role == 'H':
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
    """off-grammar stacks (unset entry below top) -> mark unset as plural-agnostic 'u' top only."""
    if st in SI:
        return SI[st]
    st = tuple((k, 's' if (n == 'u' and i < len(st) - 1) else n) for i, (k, n) in enumerate(st))
    return SI.get(st, SI[()])


def topkind(st):
    return st[-1][0] if st else 'none'


def trans_phase(ph, cls, term, tk):
    phase, ci, role, n = ph
    fake = (('M', 'u'),) if tk == 'M' else ((('R', 'u'),) if tk == 'R' else ())
    h = trans((phase, fake, ci, role, n), cls, term)
    return (h[0], h[2], h[3], h[4])


def build():
    PH = all_phases(); PI = {p: i for i, p in enumerate(PH)}; nP = len(PH)
    SK = all_stacks(); SI = {s: i for i, s in enumerate(SK)}; nK = len(SK)
    E1 = len(CLASSES) * 3 + 1
    nO = len(OPS_)
    events = [None] + [(c, t) for c in CLASSES for t in TERMS]
    # consistency check against the joint grammar on reachable joint states
    for g in enumerate_states():
        ph = (g[0], g[2], g[3], g[4]); st = g[1]
        for ev in events[1:]:
            h = trans(g, *ev)
            ph2 = trans_phase(ph, ev[0], ev[1], topkind(st))
            st2 = apply_op(st, op_of(ph, *ev))
            assert (h[0], h[2], h[3], h[4]) == ph2 and h[1] == st2, (g, ev, h, ph2, st2)
    op_tab = []   # [nO, nP*E1]
    ph_tab = []   # [nP, nP*E1*3]
    for pi, ph in enumerate(PH):
        for ei, ev in enumerate(events):
            col = pi * E1 + ei
            if ev is None:
                op_tab.append([OPS_.index('none'), col, 1.0])
                for ki in range(3):
                    ph_tab.append([pi, col * 3 + ki, 1.0])
                continue
            op_tab.append([OPS_.index(op_of(ph, *ev)), col, 1.0])
            for ki, tk in enumerate(TOPK):
                ph2 = trans_phase(ph, ev[0], ev[1], tk)
                ph_tab.append([PI.get(ph2, PI[('CS', 0, '-', '-')]), col * 3 + ki, 1.0])
    sk_tab = []   # [nK, nK*nO]
    tk_tab = []   # [3, nK]
    for si, st in enumerate(SK):
        for oi, op in enumerate(OPS_):
            sk_tab.append([fix_stack(apply_op(st, op), SI), si * nO + oi, 1.0])
        tk_tab.append([TOPK.index(topkind(st)), si, 1.0])
    # distribution classes over (phase, stack)
    tables = []
    dmap = []
    tbl_index = {}
    for pi, ph in enumerate(PH):
        for si, st in enumerate(SK):
            g = (ph[0], st, ph[1], ph[2], ph[3])
            ud = unit_dist(g)
            key = (ph[0], tuple(sorted((k, round(v, 12)) for k, v in ud.items())))
            if ph[0] not in ('PERIOD', 'COMMA'):
                key = ('X', key[1])
            if key not in tbl_index:
                tbl_index[key] = len(tables)
                tables.append(tuple(tuple(char_logits(g, t)) for t in TNODES))
            dmap.append([tbl_index[key], pi * nK + si, 1.0])
    nD = len(tables); nT = len(TNODES)
    L = []
    for d, tbl in enumerate(tables):
        for ti, lg in enumerate(tbl):
            for c, v in enumerate(lg):
                if v != 0.0:
                    L.append([c, d * nT + ti, round(v, 5)])
    det = []; detb = []
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
    ev1 = []; ev1b = []; ev2 = []
    k = 0
    for w in WORDS:
        cls = WORDCLASS[w]
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
            ev2.append([CLASSES.index(cls) * 3 + ti, k, 1.0])
            k += 1
    return dict(nP=nP, nK=nK, E1=E1, nO=nO, op_tab=op_tab, ph_tab=ph_tab, sk_tab=sk_tab,
                tk_tab=tk_tab, dmap=dmap, nD=nD, nT=nT, L=L, det=det, detb=detb, ev1=ev1,
                ev1b=ev1b, ev2=ev2, nW=k,
                init_ph=[[PI[('CS', 0, '-', '-')], BOS, 1.0]], init_sk=[[SI[()], BOS, 1.0]])


EOS_SLOPE = 0.4      # logit per char
EOS_THR = 145.0      # length where P(eos | '.') = 0.5


def outer(ops_list, a, b, na, nb, out):
    ops_list += [
        {"id": out + "_a", "use": "op:expand", "in": {"x": a}, "args": {"axis": 1}},
        {"id": out + "_b", "use": "op:expand", "in": {"x": b}, "args": {"axis": 0}},
        {"id": out + "_m", "use": "op:matmul", "in": {"a": out + "_a", "b": out + "_b"}},
        {"id": out, "use": "op:reshape", "in": {"x": out + "_m"}, "args": {"shape": [na * nb]}}]


def proposal(B):
    nP, nK, E1, nO, nD, nT, nW = B['nP'], B['nK'], B['E1'], B['nO'], B['nD'], B['nT'], B['nW']
    E = E1 - 1
    sp = lambda entries, shape, default=0.0: {"sparse": entries, "shape": shape, "default": default}
    z = lambda n: {"fill": 0.0, "shape": [n]}
    ops = []
    def P_(name, pat):
        ops.append({"op": "set_param", "name": name, "pattern": pat})
    P_("gr.initp.W", sp(B['init_ph'], [nP, 32])); P_("gr.initp.b", z(nP))
    P_("gr.inits.W", sp(B['init_sk'], [nK, 32])); P_("gr.inits.b", z(nK))
    P_("gr.op.W", sp(B['op_tab'], [nO, nP * E1])); P_("gr.op.b", z(nO))
    P_("gr.ph.W", sp(B['ph_tab'], [nP, nP * E1 * 3])); P_("gr.ph.b", z(nP))
    P_("gr.sk.W", sp(B['sk_tab'], [nK, nK * nO])); P_("gr.sk.b", z(nK))
    P_("gr.tk.W", sp(B['tk_tab'], [3, nK])); P_("gr.tk.b", z(3))
    P_("ev.W1", sp(B['ev1'], [nW, 256]))
    P_("ev.b1", sp([[i, v] for i, v in enumerate(B['ev1b'])], [nW]))
    P_("ev.W2", sp(B['ev2'], [E, nW])); P_("ev.b2", z(E))
    P_("pf.W", sp(B['det'], [nT, 256]))
    P_("pf.b", sp([[i, v] for i, v in enumerate(B['detb'])], [nT]))
    P_("dm.W", sp(B['dmap'], [nD, nP * nK])); P_("dm.b", z(nD))
    P_("out.W", sp(B['L'], [32, nD * nT])); P_("out.b", z(32))
    P_("eos.lin.W", sp([[0, 0, EOS_SLOPE]], [1, 1]))
    P_("eos.lin.b", sp([[0, -EOS_SLOPE * EOS_THR]], [1]))
    P_("eos.put.W", sp([[EOS, 0, 1.0]], [32, 1])); P_("eos.put.b", z(32))
    P_("one", {"fill": 1.0, "shape": [1]})

    def cell(name, act, W1, b1, W2=None, b2=None):
        p = {"W1": W1, "b1": b1}
        if W2: p["W2"] = W2; p["b2"] = b2
        ops.append({"op": "put_def", "name": name, "def": {"kind": "cell", "act": act, "params": p}})
    cell("c_initp", "identity", "gr.initp.W", "gr.initp.b")
    cell("c_inits", "identity", "gr.inits.W", "gr.inits.b")
    cell("c_op", "identity", "gr.op.W", "gr.op.b")
    cell("c_phase", "identity", "gr.ph.W", "gr.ph.b")
    cell("c_stack", "identity", "gr.sk.W", "gr.sk.b")
    cell("c_topk", "identity", "gr.tk.W", "gr.tk.b")
    cell("c_event", "step", "ev.W1", "ev.b1", "ev.W2", "ev.b2")
    cell("c_prefix", "step", "pf.W", "pf.b")
    cell("c_dclass", "identity", "dm.W", "dm.b")
    cell("c_out", "identity", "out.W", "out.b")
    cell("c_eoslin", "identity", "eos.lin.W", "eos.lin.b")
    cell("c_eosput", "identity", "eos.put.W", "eos.put.b")

    ev_nodes = [
        {"id": "ev", "use": "c_event", "in": {"x": "$cf"}},
        {"id": "evs", "use": "op:sum", "in": {"x": "ev"}, "args": {"axis": 0, "keepdims": True}},
        {"id": "one", "use": "param:one"},
        {"id": "none", "use": "op:sub", "in": {"a": "one", "b": "evs"}},
        {"id": "evf", "use": "op:concat", "in": {"x0": "none", "x1": "ev"}, "args": {"axis": 0}}]
    ops.append({"op": "put_def", "name": "word_events", "def": {"kind": "bundle",
        "inputs": ["cf"], "nodes": ev_nodes, "outputs": {"ev": "evf"}}})
    gn = [
        {"id": "p0", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": 0, "stop": nP}},
        {"id": "s0", "use": "op:slice", "in": {"x": "$state"}, "args": {"axis": 0, "start": nP, "stop": nP + nK}},
        {"id": "pi", "use": "c_initp", "in": {"x": "$prev"}},
        {"id": "si", "use": "c_inits", "in": {"x": "$prev"}},
        {"id": "ph", "use": "op:add", "in": {"a": "p0", "b": "pi"}},
        {"id": "sk", "use": "op:add", "in": {"a": "s0", "b": "si"}},
        {"id": "we", "use": "word_events", "in": {"cf": "$cf"}}]
    outer(gn, "ph", "we.ev", nP, E1, "pe")
    gn += [{"id": "op", "use": "c_op", "in": {"x": "pe"}},
           {"id": "tk", "use": "c_topk", "in": {"x": "sk"}}]
    outer(gn, "sk", "op", nK, nO, "so")
    outer(gn, "pe", "tk", nP * E1, 3, "pet")
    gn += [{"id": "sk2", "use": "c_stack", "in": {"x": "so"}},
           {"id": "ph2", "use": "c_phase", "in": {"x": "pet"}},
           {"id": "st", "use": "op:concat", "in": {"x0": "ph2", "x1": "sk2"}, "args": {"axis": 0}}]
    ops.append({"op": "put_def", "name": "grammar", "def": {"kind": "bundle",
        "inputs": ["cf", "prev", "state"], "nodes": gn,
        "outputs": {"phase": "ph2", "stack": "sk2", "state": "st"}}})
    spn = []
    outer(spn, "$phase", "$stack", nP, nK, "ps")
    spn += [{"id": "d", "use": "c_dclass", "in": {"x": "ps"}},
            {"id": "t", "use": "c_prefix", "in": {"x": "$cf"}}]
    outer(spn, "d", "t", nD, nT, "dt")
    spn += [{"id": "lg", "use": "c_out", "in": {"x": "dt"}}]
    ops.append({"op": "put_def", "name": "speller", "def": {"kind": "bundle",
        "inputs": ["phase", "stack", "cf"], "nodes": spn, "outputs": {"logits": "lg"}}})
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
            {"id": "sp", "use": "speller", "in": {"phase": "gr.phase", "stack": "gr.stack", "cf": "cf"}},
            {"id": "eo", "use": "eos_timer", "in": {"step": "$step", "prev": "$prev"}},
            {"id": "lg", "use": "op:add", "in": {"a": "sp.logits", "b": "eo.delta"}}],
        "outputs": {"logits": "lg", "state": "gr.state"}}})
    ops.append({"op": "set_state_dim", "value": nP + nK})
    ops.append({"op": "delete_param", "name": "boot_logits"})
    return {"author": "r1-O2", "rationale": RATIONALE, "ops": ops}


RATIONALE = ("Hand-built grammar model: a recurrent one-hot grammar state (phase x agreement stack "
             "of pending verb numbers x clause index) is updated by word-completion events detected "
             "from the ctx window; a prefix detector bank identifies the partial word; next-char "
             "logits are a lookup over (distribution class x prefix) computed from hand-typed word "
             "lists and hand-estimated structural probabilities; eos after '.' uses a step-length ramp.")

if __name__ == '__main__':
    B = build()
    print('P', B['nP'], 'K', B['nK'], 'D', B['nD'], 'T', B['nT'], 'W', B['nW'], 'Lnnz', len(B['L']), file=sys.stderr)
    out = sys.argv[1] if len(sys.argv) > 1 else 'proposal.json'
    with open(out, 'w') as f:
        json.dump(proposal(B), f)
