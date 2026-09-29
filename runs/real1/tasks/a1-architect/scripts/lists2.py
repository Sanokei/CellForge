"""Hand-typed tables (part 2): phrases, punctuation words, quote rules, apostrophe rules, name cache, router."""
import numpy as np
from lists import *  # noqa

PHRASES = {
    "once upon a time": 2.5, "one day": 1.5, "there was a": 1.8, "there was": 1.5,
    "little girl": 1.0, "little boy": .7, "girl named": 1.5, "boy named": 1.5, "dog named": 1.0,
    "cat named": 1.0, "bird named": 1.0, "wanted to": 2.0, "want to": 1.8, "went to the": 1.5,
    "went to": 1.5, "had a": 1.0, "was very": 1.2, "very happy": 1.0, "so much fun": 2.0,
    "much fun": 1.5, "best friends": 2.0, "lots of fun": 2.0, "lots of": 1.5, "in the": .8,
    "on the": .8, "of the": .8, "all day": 1.5, "every day": 1.5, "the park": 1.0,
    "to play": 1.0, "play with": 1.5, "played with": 1.5, "with his": .6, "thank you": 2.0,
    "i love you": 2.0, "i want to": 2.0, "can i": 1.0, "lived in": 1.5, "lived in a": 1.5,
    "big tree": 1.0, "happy and": 1.0, "did not": 1.5, "could not": 1.5, "they were": 1.0,
    "she was": 1.0, "he was": 1.0, "she said": 1.0, "he said": 1.0, "said, ": 1.0,
    "asked her mom": 1.5, "her mom": 1.0, "his mom": 1.0, "mom and dad": 1.5,
    "became friends": 2.0, "they all": .8, "and they": .8, "felt sad": .8, "because": 1.2,
    "then, ": 1.2, "thought, ": 1.5, "one day, ": 1.5, "upon a time, ": 1.5, "every day, ": 1.0,
    "look at": 1.2, "so happy": 1.0, "too, ": .8, "the end": 1.0, "the sun": .6, "next to": 1.5,
    "out of": 1.5, "a lot of": 1.5, "all the": 1.0, "in the sky": 1.0, "the little": .8,
}


def colloc_eval(ctx):
    v = np.zeros(V)
    for P, s in PHRASES.items():
        for j in range(3, len(P)):
            cs = P[max(0, j - 8):j]
            m = len(cs)
            rows = ctx[8 - m:]
            if any(r < 0 for r in rows):
                continue
            if "".join(CH[r] for r in rows) != cs:
                continue
            if j < 8 and not isb(ctx[7 - j]):
                continue
            v[IDX[P[j]]] += s
    return v


COMMA_W = {"day": 2.3, "time": 2.3, "then": 1.5, "so": 1.3, "yes": 2.0, "no": 2.0, "oh": 1.8,
           "wow": .8, "hello": 1.5, "hi": 1.5, "well": 1.5, "okay": 1.0, "said": 1.7, "asked": 1.4,
           "soon": 1.0, "finally": 1.5, "later": 1.0, "inside": 1.3, "outside": .8, "thought": 1.2,
           "but": .3, "too": .8, "ok": .8, "please": .6}
PERIOD_W = {"day": .8, "happy": 1.0, "fun": 1.4, "home": 1.0, "too": 1.0, "friends": 1.0, "friend": .6,
            "again": 1.0, "it": .8, "them": .8, "him": .7, "her": .3, "me": 1.0, "you": .6,
            "there": .8, "away": 1.0, "out": .5, "up": .3, "back": .8, "here": .5, "more": .6,
            "time": 1.0, "sky": .8, "tree": .5, "park": .7, "sad": .7, "together": 1.0, "ever": .8,
            "all": .5, "food": .5, "play": .6, "much": .5, "good": .4, "scared": .5,
            "excited": .5, "proud": .5, "toy": .5, "ball": .5, "water": .5, "house": .5, "mom": .4,
            "dad": .4, "dog": .4, "cat": .4, "bird": .4, "sun": .4}
EXCL_W = {"wow": 2.0, "yay": 2.5, "help": 1.2, "look": 1.0, "come": .8, "stop": 1.5, "oops": 2.0,
          "hooray": 2.5, "surprise": 1.5, "ouch": 2.0, "hello": .8, "hi": .8, "thanks": 1.0,
          "please": .6, "no": .8, "yes": .8, "oh": .6, "again": .3, "fun": .4, "wait": 1.5, "go": .6}


def endpunct_eval(ctx, q):
    v = np.zeros(V)
    if not isl(ctx[7]):
        return v
    w, ok = run_letters(ctx, 7)
    if ok and len(w) <= 7:
        v[COMMA] += COMMA_W.get(w, 0)
        v[PERIOD] += PERIOD_W.get(w, 0)
        v[EXCL] += EXCL_W.get(w, 0)
    if q >= .5:
        v[EXCL] += .9
        v[QMARK] += .6
    else:
        v[EXCL] -= .4
        v[QMARK] -= .6
    return v


QOPEN = {"i": .6, "l": .4, "y": .5, "w": .4, "c": .3, "h": .4, "o": .4, "n": .3, "t": .2, "a": -.2, "m": .2, "s": .2, "p": .2}


def quote_eval(ctx, q):
    v = np.zeros(V)
    p = ctx[7]
    inq = q >= .5
    if inq:
        v[EOS] += -6
    if inq and p == DQ:
        v += vec(QOPEN)
        v[DQ] += -3; v[SP] += -3; v[COMMA] += -2; v[PERIOD] += -2
    if (not inq) and p == DQ:
        v += vec({" ": .8, ",": .3, ".": .4, '"': -3}, -1.5)
    if inq and p in (COMMA, PERIOD, EXCL, QMARK):
        v[DQ] += {COMMA: 1.0, PERIOD: .6, EXCL: 1.6, QMARK: 1.6}[p]
    if (not inq) and p == SP and ctx[6] == COMMA:
        v[DQ] += .4
        if ctx[5] == IDX["d"]:
            v[DQ] += 2.1
        if ctx[5] == IDX["m"]:
            v[DQ] += .6
    if (not inq) and p == SP and ctx[6] in (PERIOD, EXCL, QMARK):
        v[DQ] += .8
    if inq and p == SP:
        v[DQ] += -2.5
    return v


APOS_STEM = {"didn": 3.5, "don": 3.0, "wasn": 3.5, "isn": 3.5, "couldn": 3.5, "wouldn": 3.5,
             "doesn": 3.5, "aren": 3.5, "weren": 3.5, "haven": 3.5, "hasn": 3.5, "can": .4,
             "won": 1.5, "let": 1.2, "it": .8, "that": .8, "what": .8, "i": 1.0, "he": .3,
             "she": .3, "there": .5, "you": .3, "we": .3, "they": .3}
S_GROUP = {"it", "that", "what", "let", "he", "she", "there", "here", "who", "where", "how"}
R_GROUP = {"you", "we", "they"}


def apos_eval(ctx):
    v = np.zeros(V)
    p = ctx[7]
    if p == APOS:
        w, ok = run_letters(ctx, 6)
        if not w or not ok or len(w) > 6:
            return v
        if w == "i":
            return vec({"m": 2.5, "l": 1.2, "v": .8, "d": .3})
        if w in S_GROUP:
            return vec({"s": 2.4, "l": .6})
        if w in R_GROUP:
            return vec({"r": 2.0, "l": 1.2, "v": 1.0, "d": .4})
        if w.endswith("n") and len(w) >= 3:
            return vec({"t": 2.6, "s": -1.0})
        if len(w) >= 2:
            return vec({"s": 2.0})
        return v
    if isl(p):
        w, ok = run_letters(ctx, 7)
        if ok and w in APOS_STEM:
            v[APOS] = APOS_STEM[w]
    return v


NM = 216 + 1 + 7
G = {1: 1.5, 2: 2.5, 3: 3.0, 4: 3.0, 5: 3.0}
END = {" ": 1.5, ",": .7, ".": .7, "'": .6, "!": .3, "?": .2}


def nm_state(name, cap=0):
    s = np.zeros(NM)
    for i, c in enumerate(name[:6]):
        s[i * 36 + IDX[c]] = 1
    s[216] = cap
    s[217 + min(len(name), 6)] = 1
    return s


def nm_write(ctx, s):
    s = s.copy()
    named = [IDX[c] for c in "named "]
    if ctx[2:8] == named:
        s[:] = 0
        s[216] = 1
        s[217] = 1
        return s
    if s[216] > .5:
        if isl(ctx[7]):
            L = int(np.argmax(s[217:224]))
            if L < 6:
                s[L * 36 + ctx[7]] = 1
                s[217 + L] = 0
                s[217 + L + 1] = 1
        else:
            s[216] = 0
    return s


def nm_read(ctx, s):
    v = np.zeros(V)
    if s[216] > .5:
        return v
    L = int(np.argmax(s[217:224]))
    if L < 2:
        return v
    buf = s[:216].reshape(6, 36)
    name = [int(np.argmax(buf[i])) for i in range(L)]
    p = ctx[7]
    if p in (SP, DQ):
        v[name[0]] += 1.2
        return v
    if not isl(p):
        return v
    w, ok = run_letters(ctx, 7)
    k = len(w)
    if not ok or k > L:
        return v
    if [IDX[c] for c in w] != name[:k]:
        return v
    if k < L:
        if k in G:
            v[name[k]] += G[k]
    else:
        v += vec(END)
    return v


def router_eval(ctx):
    p = ctx[7]
    y = np.zeros(7)
    if p == 1:
        y[0] = 3
    elif p == SP:
        y[1] = 3
    elif isl(p):
        y[3 if (isl(ctx[5]) and isl(ctx[6])) else 2] = 3
    elif p in (COMMA, PERIOD, EXCL, QMARK):
        y[4] = 3
    elif p == DQ:
        y[5] = 3
    elif p == APOS:
        y[6] = 3
    else:
        y[1] = 1
    return y
