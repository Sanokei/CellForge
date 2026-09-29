"""Hand-written grammar for the tinylm text (r1-L3).

Everything here is typed by hand: the word list (read off dev_train by eye),
the word classes, the pushdown-style clause state machine and every
probability.  Nothing is computed from the corpus.
"""

VOCAB = ["<pad>", "<bos>", "<eos>", " ", ",", "."] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
VI = {t: i for i, t in enumerate(VOCAB)}

# (word, class, number)   number: s / p / - ; classes below
NOUNS = [("dog", "dogs"), ("bird", "birds"), ("king", "kings"), ("farmer", "farmers"),
         ("fox", "foxes"), ("boy", "boys"), ("child", "children"), ("teacher", "teachers"),
         ("cat", "cats"), ("horse", "horses"), ("mouse", "mice"), ("girl", "girls")]
VINTR = [("sings", "sing"), ("waits", "wait"), ("laughs", "laugh"), ("sleeps", "sleep"),
         ("jumps", "jump"), ("runs", "run")]
VTR = [("helps", "help"), ("follows", "follow"), ("finds", "find"), ("likes", "like"),
       ("chases", "chase"), ("sees", "see")]
DETS = [("the", "DET_A"), ("a", "DET_S"), ("this", "DET_S"), ("every", "DET_S"),
        ("these", "DET_P"), ("many", "DET_P"), ("some", "DET_P")]
ADJS = ["happy", "tired", "brave", "red", "old", "small", "young", "quiet"]
PREPS = ["with", "near", "under", "behind"]
ADVS = ["never", "often", "quickly", "slowly", "again"]

WORDS = []  # list of (word, cls)
for w, c in DETS:
    WORDS.append((w, c))
for w in ADJS:
    WORDS.append((w, "ADJ"))
for s, p in NOUNS:
    WORDS.append((s, "N_S"))
    WORDS.append((p, "N_P"))
for s, p in VINTR:
    WORDS.append((s, "VI_S"))
    WORDS.append((p, "VI_P"))
for s, p in VTR:
    WORDS.append((s, "VT_S"))
    WORDS.append((p, "VT_P"))
for w in PREPS:
    WORDS.append((w, "PREP"))
WORDS.append(("that", "THAT"))
WORDS.append(("and", "AND"))
for w in ADVS:
    WORDS.append((w, "ADV"))

WCLS = ["DET_S", "DET_P", "DET_A", "ADJ", "N_S", "N_P", "VI_S", "VI_P", "VT_S", "VT_P",
        "PREP", "THAT", "AND", "ADV"]
CLS = WCLS + ["PERIOD", "COMMA", "BOS", "NONE"]

# state = (phase, np_role, det_number, main, rcs, clause2)
#   main: number of the pending main verb ('x' unknown yet, 's', 'p') or None once seen
#   rcs : tuple of numbers of pending relative-clause verbs (innermost last)
START = ("D", "S", "-", "x", (), False)
MAXRC = 2


def trans(s, c):
    ph, role, nd, main, rcs, c2 = s
    if c == "NONE":
        return s
    if c in ("BOS", "PERIOD"):
        return START
    if c == "COMMA":
        return ("AND", "-", "-", None, (), False)
    if c == "AND":
        return ("D", "S", "-", "x", (), False)
    if c.startswith("DET"):
        n = {"DET_S": "s", "DET_P": "p", "DET_A": "?"}[c]
        if ph == "D":
            return ("A", role, n, main, rcs, c2)
        return ("A", "O", n, main, rcs, c2)
    if c == "ADJ":
        if ph in ("A", "N"):
            return ("N", role, nd, main, rcs, c2)
        return ("N", "O", "?", main, rcs, c2)
    if c in ("N_S", "N_P"):
        num = "s" if c == "N_S" else "p"
        if ph in ("A", "N") and role == "S":
            if rcs:
                rcs = rcs[:-1] + (num,)
            elif main is not None:
                main = num
        # an unresolved number (should not happen) defaults to the noun's number
        if main == "x":
            main = num
        rcs = tuple(num if x == "x" else x for x in rcs)
        return ("PN", "-", "-", main, rcs, c2)
    if c == "PREP":
        return ("D", "O", "-", main, rcs, c2)
    if c == "THAT":
        r = rcs + ("x",)
        if len(r) > MAXRC:
            r = r[-MAXRC:]
        return ("D", "S", "-", main, r, c2)
    if c.startswith("VI") or c.startswith("VT"):
        if rcs:
            return ("RV", "-", "-", main, rcs[:-1], c2)
        if main is not None and c.startswith("VT"):
            return ("D", "O", "-", None, (), c2)
        return ("V", "-", "-", None, (), c2)
    if c == "ADV":
        return ("ADV", "-", "-", None, (), c2)
    raise ValueError(c)


def words_of(cls):
    return [w for w, c in WORDS if c == cls]


def spread(d, words, mass):
    if not words or mass <= 0:
        return
    for w in words:
        d[w] = d.get(w, 0.0) + mass / len(words)


def ends(c2, mass):
    """Split clause-end mass between '.' and ', and' (only in first clause)."""
    if c2:
        return {"END.": mass, "END,": 0.0}
    return {"END.": mass * 0.82, "END,": mass * 0.18}


def dist(s):
    """My hand-set next-item distribution in grammar state s."""
    ph, role, nd, main, rcs, c2 = s
    pend = rcs if rcs else ((main,) if main is not None else ())
    d = {}
    if ph == "D":
        d.update({"the": 0.22, "a": 0.13, "this": 0.13, "every": 0.13,
                  "these": 0.13, "many": 0.13, "some": 0.13})
    elif ph in ("A", "N"):
        adj = 0.36 if ph == "A" else 0.0
        spread(d, ADJS, adj)
        nm = 1.0 - adj
        if nd == "s":
            spread(d, words_of("N_S"), nm)
        elif nd == "p":
            spread(d, words_of("N_P"), nm)
        else:
            spread(d, words_of("N_S"), nm * 0.5)
            spread(d, words_of("N_P"), nm * 0.5)
    elif ph in ("PN", "RV") and pend:
        top = pend[-1]
        main = not rcs
        if ph == "PN":
            pv, pp, pt = (0.72, 0.14, 0.14) if main else (0.82, 0.09, 0.09)
        else:
            pv, pp, pt = 0.97, 0.015, 0.015
        nums = ["s", "p"] if top == "x" else [top]
        for n in nums:
            m = pv / len(nums)
            S = "S" if n == "s" else "P"
            if main:
                spread(d, words_of("VI_" + S), m * 0.5)
                spread(d, words_of("VT_" + S), m * 0.5)
            else:
                spread(d, words_of("VT_" + S), m)
        spread(d, PREPS, pp)
        d["that"] = pt
    elif ph == "PN":  # object noun, clause may end
        d.update(ends(c2, 0.58))
        spread(d, ADVS, 0.17)
        spread(d, PREPS, 0.12)
        d["that"] = 0.13
    elif ph == "RV":  # relative clause in object finished
        d.update(ends(c2, 0.75))
        spread(d, ADVS, 0.20)
        spread(d, PREPS, 0.025)
        d["that"] = 0.025
    elif ph == "V":
        d.update(ends(c2, 0.65))
        spread(d, ADVS, 0.35)
    elif ph == "ADV":
        d.update(ends(c2, 1.0))
    elif ph == "AND":
        d["and"] = 1.0
    else:
        raise ValueError(s)
    return d


def reachable():
    seen = [START]
    idx = {START: 0}
    i = 0
    while i < len(seen):
        s = seen[i]
        i += 1
        for c in CLS:
            t = trans(s, c)
            if t not in idx:
                idx[t] = len(seen)
                seen.append(t)
    return seen


def group(s):
    ph = s[0]
    if ph in ("D", "A", "N"):
        return "np"
    if ph in ("PN", "RV"):
        return "post"
    return "tail"  # V, ADV, AND
