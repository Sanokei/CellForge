"""Hand-written probabilistic grammar for the tinylm/1 text (all numbers chosen by me).

Grammar (my hypothesis from reading dev_train by eye):
  DOC  -> CL CEND1
  CEND1 -> "," "and" CL "." SEP   (P_AND)   |  "." SEP
  SEP  -> SPACE CL CEND1 (1-P_EOS)  |  EOS
  CL   -> NP_n VP_n                 n in {s,p}, P_SG
  NP_n -> DET_n ADJ? N_n MOD?
  MOD? -> PREP NP_m (P_PP) | "that" NP_m VTRC_m (P_RC, only if RC depth < DMAX) | eps
  VP_n -> VI_n ADV? (P_VI) | VT_n NP_m ADV?
  ADV? -> ADV (P_ADV) | eps
This module expands the grammar into a deterministic token-level automaton over
belief states (sets of weighted parser stacks).  A token is a word plus its
following separator (' ', ',', '.'), or one of the specials ' and ', ' ', <eos>.
"""
from __future__ import annotations

# ----------------------------------------------------------------- my numbers
P_SG = 0.5
P_ADJ = 0.35
# modifier probabilities by NP embedding depth d (0 = subject/object of a clause)
P_PP = [0.14, 0.08, 0.04, 0.02, 0.0]
P_RC = [0.17, 0.10, 0.05, 0.025, 0.0]
P_VI = 0.45
P_ADV = 0.27
P_AND = 0.21
P_EOS = 0.5       # neutral; the graph multiplies the <eos> odds by r(step) (length rule)

DET = {"s": ["every", "this", "a", "the"], "p": ["these", "many", "some", "the"]}
ADJ = ["old", "brave", "tired", "small", "red", "happy", "quiet", "young"]
NOUN = {"s": ["teacher", "dog", "bird", "king", "farmer", "fox", "boy", "child", "horse", "cat", "girl", "mouse"],
        "p": ["teachers", "dogs", "birds", "kings", "farmers", "foxes", "boys", "children", "horses", "cats",
              "girls", "mice"]}
VI = {"s": ["waits", "sings", "laughs", "sleeps", "runs", "jumps"],
      "p": ["wait", "sing", "laugh", "sleep", "run", "jump"]}
VT = {"s": ["helps", "finds", "follows", "likes", "chases", "sees"],
      "p": ["help", "find", "follow", "like", "chase", "see"]}
PREP = ["with", "behind", "near", "under"]
ADV = ["never", "often", "quickly", "slowly", "again"]

SPECIAL = {"and": "and ", "SPACE": " ", "EOS": "\x02"}   # \x02 stands for <eos>


def uni(words, p=1.0):
    return [(p / len(words), (w,)) for w in words]


def expand(sym, rest):
    """Return list of (prob, replacement tuple) for nonterminal sym; None if terminal."""
    if sym == "CL":
        return [(P_SG, ("NP_s_0", "VP_s")), (1 - P_SG, ("NP_p_0", "VP_p"))]
    if sym == "CEND1":
        return [(P_AND, (",", "and", "CL", ".", "SEP")), (1 - P_AND, (".", "SEP"))]
    if sym == "CEND2":
        return [(1.0, (".", "SEP"))]
    if sym == "SEP":
        return [(1 - P_EOS, ("SPACE", "CL", "CEND1")), (P_EOS, ("EOS",))]
    if sym.startswith("NP_"):
        _, n, d = sym.split("_")
        return [(1.0, ("DET_" + n, "ADJ?", "N_" + n, "MOD?_" + d))]
    if sym.startswith("DET_"):
        return uni(DET[sym[4:]])
    if sym == "ADJ?":
        return uni(ADJ, P_ADJ) + [(1 - P_ADJ, ())]
    if sym.startswith("N_"):
        return uni(NOUN[sym[2:]])
    if sym.startswith("MOD?_"):
        d = int(sym[5:])
        ppp, prc, e = P_PP[d], P_RC[d], str(d + 1)
        out = []
        if ppp > 0:
            out += [(ppp * P_SG, ("PREP", "NP_s_" + e)), (ppp * (1 - P_SG), ("PREP", "NP_p_" + e))]
        if prc > 0:
            out += [(prc * P_SG, ("that", "NP_s_" + e, "VTRC_s")), (prc * (1 - P_SG), ("that", "NP_p_" + e, "VTRC_p"))]
        out.append((1 - ppp - prc, ()))
        return out
    if sym == "PREP":
        return uni(PREP)
    if sym.startswith("VP_"):
        n = sym[3:]
        out = [(P_VI / 6, (v, "ADV?")) for v in VI[n]]
        for m, pm in (("s", P_SG), ("p", 1 - P_SG)):
            out += [((1 - P_VI) / 6 * pm, (v, "NP_" + m + "_0", "ADV?")) for v in VT[n]]
        return out
    if sym.startswith("VTRC_"):
        return uni(VT[sym[5:]])
    if sym == "ADV?":
        return uni(ADV, P_ADV) + [(1 - P_ADV, ())]
    return None


def is_word(t):
    return t not in (",", ".") and t not in SPECIAL


def firsts(stack, p=1.0):
    """All leftmost derivations to the first terminal: list of (prob, firm_stack) with terminal on top."""
    if not stack:
        return []
    top, rest = stack[0], stack[1:]
    ex = expand(top, rest)
    if ex is None:
        return [(p, stack)]
    out = []
    for q, rep in ex:
        if q > 0:
            out += firsts(tuple(rep) + rest, p * q)
    return out


def merge(pairs):
    d = {}
    for w, s in pairs:
        d[s] = d.get(s, 0.0) + w
    return d


def norm(d):
    z = sum(d.values())
    return frozenset((s, round(w / z, 12)) for s, w in d.items())


def emissions(state):
    """state: frozenset of (firm_stack, weight). Returns {token: (prob, next_state_dict)}."""
    out = {}
    for stack, w in state:
        top, rest = stack[0], stack[1:]
        if not is_word(top):
            tok = SPECIAL[top]
            nxt = firsts(rest, 1.0)
            e = out.setdefault(tok, [0.0, {}])
            e[0] += w
            for q, s in nxt:
                e[1][s] = e[1].get(s, 0.0) + w * q
            continue
        for q, s2 in firsts(rest, 1.0):
            t2 = s2[0]
            if is_word(t2):
                tok, nstack = top + " ", [(1.0, s2)]
            elif t2 == ",":
                tok, nstack = top + ", ", firsts(s2[1:], 1.0)
            elif t2 == ".":
                tok, nstack = top + ".", firsts(s2[1:], 1.0)
            else:
                raise ValueError(f"unexpected {t2} after word")
            e = out.setdefault(tok, [0.0, {}])
            e[0] += w * q
            for r, s in nstack:
                e[1][s] = e[1].get(s, 0.0) + w * q * r
    return out


def build():
    init = norm(merge([(q, s) for q, s in firsts(("CL", "CEND1"))]))
    states = [init]
    index = {init: 0}
    trans = []   # per state: {token: (prob, next_index)}
    i = 0
    while i < len(states):
        em = emissions(states[i])
        row = {}
        for tok, (p, nd) in em.items():
            if not nd:        # after <eos>
                row[tok] = (p, None)
                continue
            ns = norm(nd)
            if ns not in index:
                index[ns] = len(states)
                states.append(ns)
            row[tok] = (p, index[ns])
        trans.append(row)
        i += 1
    return states, trans


if __name__ == "__main__":
    states, trans = build()
    toks = sorted({t for r in trans for t in r})
    print("states", len(states), "tokens", len(toks))
    print("max belief size", max(len(s) for s in states))


def minimize(trans):
    """Partition refinement: merge states with identical emission probs and equivalent successors."""
    n = len(trans)
    sig0 = [tuple(sorted((t, round(p, 9)) for t, (p, _) in r.items())) for r in trans]
    ids = {}
    part = [ids.setdefault(s, len(ids)) for s in sig0]
    while True:
        sig = [(part[i], tuple(sorted((t, -1 if j is None else part[j]) for t, (p, j) in trans[i].items())))
               for i in range(n)]
        ids = {}
        new = [ids.setdefault(s, len(ids)) for s in sig]
        if len(ids) == len(set(part)):
            part = new
            break
        part = new
    G = max(part) + 1
    mtrans = [None] * G
    for i in range(n):
        if mtrans[part[i]] is None:
            mtrans[part[i]] = {t: (p, None if j is None else part[j]) for t, (p, j) in trans[i].items()}
    return part, mtrans


def token_classes(mtrans):
    toks = sorted({t for r in mtrans for t in r})
    cols = {}
    for t in toks:
        col = tuple(r[t][1] if t in r else -2 for r in mtrans)
        cols.setdefault(col, []).append(t)
    return toks, list(cols.values())
