"""Hand-typed tables and lists of the architect (part 1: vocabulary helpers, word lists, pairs, suffix, wordstart).

Every number was typed by the architect from general knowledge of English children's stories plus reading a sample
of dev_train.json by eye. Nothing here counts or estimates anything from a corpus.
"""
import math
import numpy as np

V = 36
CH = ["<pad>", "<bos>", "<eos>", " ", ",", ".", "!", "?", "'", '"'] + [chr(97 + i) for i in range(26)]
IDX = {c: i for i, c in enumerate(CH)}
LETTERS = range(10, 36)
SP, COMMA, PERIOD, EXCL, QMARK, APOS, DQ, EOS = 3, 4, 5, 6, 7, 8, 9, 2


def ctx_from(text):
    """last 8 chars of text as token ids; '^' = <bos>, '_' = empty row (-1); short text is left-padded."""
    t = text[-8:]
    rows = [-1] * (8 - len(t))
    for c in t:
        rows.append(-1 if c == "_" else 1 if c == "^" else IDX[c])
    return rows


def isl(i):
    return 10 <= i <= 35


def isb(i):
    return i == -1 or (not isl(i) and i != APOS)


def run_letters(ctx, end):
    k = 0
    r = end
    while r >= 0 and isl(ctx[r]):
        k += 1
        r -= 1
    if k == 0:
        return "", False
    word = "".join(CH[ctx[r + 1 + i]] for i in range(k))
    ok = r >= 0 and isb(ctx[r])
    return word, ok


def vec(d, rest_letters=0.0):
    v = np.zeros(V)
    for c, x in d.items():
        v[IDX[c]] += x
    if rest_letters:
        for i in LETTERS:
            v[i] += rest_letters
    return v


def mcap(x):
    return min(0.7 * math.log(1 + x), 5.0)


WORDS_FN = {
    "the": 550, "and": 380, "a": 330, "to": 300, "was": 250, "he": 160, "she": 150, "it": 130,
    "they": 120, "his": 90, "her": 100, "of": 70, "in": 90, "on": 55, "with": 75, "said": 80,
    "but": 90, "had": 80, "i": 80, "you": 60, "we": 45, "that": 70, "for": 60, "at": 30,
    "up": 25, "all": 40, "so": 55, "not": 40, "very": 50, "were": 50, "is": 40, "are": 20,
    "my": 30, "me": 25, "out": 30, "as": 25, "be": 20, "do": 15, "can": 25, "will": 20,
    "then": 40, "there": 40, "when": 35, "what": 20, "how": 10, "this": 30, "them": 25,
    "their": 15, "from": 15, "into": 15, "too": 25, "no": 15, "yes": 12, "if": 15, "have": 15,
    "has": 8, "him": 25, "an": 15, "one": 50, "because": 25, "just": 15, "would": 25,
    "could": 40, "did": 30, "your": 15, "its": 10, "our": 8, "some": 15, "who": 8,
    "after": 20, "every": 25, "again": 15, "back": 20, "down": 15, "over": 15, "around": 15,
}
WORDS_OPEN = {
    "day": 90, "time": 60, "little": 80, "girl": 65, "boy": 45, "named": 45, "lily": 60,
    "tim": 55, "mom": 40, "dog": 35, "cat": 35, "bird": 30, "play": 35, "played": 50,
    "playing": 15, "park": 30, "tree": 30, "friend": 30, "friends": 60, "happy": 75, "sad": 30,
    "big": 70, "home": 25, "saw": 50, "went": 45, "found": 40, "want": 20, "wanted": 50,
    "love": 20, "loved": 35, "like": 25, "liked": 30, "look": 20, "looked": 30, "see": 20,
    "make": 15, "made": 25, "get": 20, "got": 25, "go": 20, "came": 15, "come": 15, "ran": 20,
    "fun": 40, "together": 30, "upon": 35, "once": 35, "mia": 25, "ben": 25, "sue": 20,
    "sam": 25, "anna": 20, "tom": 30, "bob": 15, "toy": 25, "toys": 15, "ball": 25,
    "flower": 12, "flowers": 12, "sun": 15, "water": 20, "food": 15, "house": 20, "room": 15,
    "mommy": 12, "daddy": 8, "dad": 20, "mother": 8, "family": 10, "smile": 15, "smiled": 30,
    "laughed": 15, "new": 30, "good": 25, "great": 12, "nice": 15, "kind": 12, "best": 20,
    "old": 15, "small": 15, "red": 20, "blue": 12, "green": 8, "beautiful": 20, "pretty": 15,
    "special": 15, "magic": 15, "box": 12, "door": 12, "never": 12, "always": 10, "felt": 30,
    "learned": 20, "told": 20, "asked": 20, "help": 25, "helped": 30, "thought": 12,
    "know": 12, "knew": 15, "take": 12, "took": 20, "gave": 15, "give": 12, "lot": 12,
    "excited": 25, "scared": 10, "proud": 10, "sky": 10, "forest": 10, "garden": 12,
    "night": 10, "morning": 8, "wish": 8, "heart": 10, "hugged": 12, "jumped": 15,
    "flew": 15, "walked": 12, "tried": 20, "cried": 10, "decided": 10,
    "everyone": 12, "something": 12, "outside": 12, "inside": 10, "school": 10, "story": 8,
    "ever": 8, "much": 20, "many": 12, "more": 15, "soon": 12, "way": 10, "eat": 12,
    "fly": 10, "run": 12, "jump": 10, "swim": 8, "fish": 8, "bear": 8, "rabbit": 6,
}
assert not (set(WORDS_FN) & set(WORDS_OPEN)), set(WORDS_FN) & set(WORDS_OPEN)


def trie_eval(words, ctx):
    v = np.zeros(V)
    w, ok = run_letters(ctx, 7)
    if not w or not ok or len(w) > 7:
        return v
    j = len(w)
    for c in LETTERS:
        F = sum(x for ww, x in words.items() if len(ww) > j and ww.startswith(w + CH[c]))
        if F:
            v[c] = mcap(F)
    if w in words:
        v[SP] = mcap(words[w])
    return v


PAIRS = {
    "th": ({"e": .9, "a": .4, "i": .5, "o": .3, "r": .2, " ": -2}, 0),
    "sh": ({"e": 1.0, "o": .4, " ": -2}, 0),
    "ch": ({"i": .5, "e": .4, " ": .4}, 0),
    "wh": ({"e": .9, "a": .8, "o": .3, "i": .3, " ": -2}, 0),
    "qu": ({"i": .8, "e": .6, "a": .6, " ": -3}, 0),
    "ng": ({" ": 1.0, ",": .7, ".": .6, "!": .2, "e": .2, "s": .3, "l": .3}, 0),
    "ck": ({" ": .8, "e": .3, "i": .3, ",": .3, ".": .2}, 0),
    "ll": ({" ": .5, "y": .4, "e": .3, ".": .3, ",": .3, "o": .3}, 0),
    "ss": ({" ": .3, "e": .3, ".": .2}, 0),
    "nd": ({" ": .7, "e": .3, ".": .3, ",": .3, "s": .3}, 0),
    "nt": ({" ": .3, "e": .4, "i": .3, "s": .3, ".": .2}, 0),
    "st": ({" ": .3, "e": .3, "a": .3, "o": .3, "r": .3, "i": .3}, 0),
    "ed": ({" ": 1.0, ",": .6, ".": .5}, -0.4),
    "ly": ({" ": 1.0, ",": .7, ".": .6, "!": .2}, -0.4),
    "er": ({" ": .5, ",": .3, ".": .3, "s": .3, "y": .3, "e": .3}, 0),
    "re": ({" ": .4, "a": .4, "d": .4, "s": .2, ",": .2, ".": .2}, 0),
    "in": ({"g": 1.2, " ": .5, "t": .3, "s": .3, "d": .3}, 0),
    "an": ({"d": .9, "t": .3, " ": .5, "y": .4, "g": .3}, 0),
    "ou": ({"l": .5, "r": .4, "d": .4, "t": .4, "n": .4, "s": .3, "g": .3, " ": .8}, 0),
    "ea": ({"r": .4, "t": .4, "s": .3, "d": .3, "l": .3, " ": .2}, 0),
    "oo": ({"d": .4, "k": .5, "n": .4, " ": .3, "l": .4, "t": .3, "m": .3}, 0),
    "ow": ({" ": .6, "n": .4, "e": .3, ",": .3, ".": .3}, 0),
    "ay": ({" ": .8, ",": .8, ".": .6, "!": .2, "s": .3}, -0.6),
    "ey": ({" ": .6, ",": .4, ".": .4}, -0.6),
    "ie": ({"n": .6, "d": .5, "s": .5, " ": .2, "l": .3}, 0),
    "ai": ({"n": .5, "d": .5, "l": .4, "r": .3, "t": .3}, 0),
    "al": ({"l": .4, " ": .3, "s": .3, "k": .3, "w": .3}, 0),
    "il": ({"l": .4, "d": .3, "e": .2}, 0),
    "ig": ({" ": .8, "h": .5, "g": .3, ",": .3, ".": .2}, 0),
    "of": ({" ": .8}, 0),
    "on": ({" ": .5, "e": .5, "g": .3}, 0),
    "me": ({" ": .5, ",": .3, ".": .3, "t": .3}, 0),
    "ha": ({"d": .5, "s": .4, "v": .5, "p": .6, "n": .3, "t": .4, "r": .3}, 0),
    "pp": ({"y": .8, "e": .5, "i": .3, "l": .3, " ": -1}, 0),
    "ee": ({" ": .3, "d": .3, "n": .3, "t": .3, "p": .3, "l": .3, "m": .3}, 0),
    "ar": ({"k": .3, "t": .4, "d": .3, "e": .4, " ": .3, "y": .3, "s": .3}, 0),
    "or": ({" ": .4, "e": .3, "d": .3, "n": .2, "k": .3, "t": .3}, 0),
    "is": ({" ": .8, "h": .3}, 0),
    "as": ({" ": .8, "k": .4, "t": .3, "s": .2}, 0),
    "at": ({" ": .5, "e": .3, "c": .3, "h": .3, ".": .2, ",": .2}, 0),
    "it": ({" ": .5, "t": .5, "h": .3, "e": .2, ".": .2, ",": .2}, 0),
}


def pairs_eval(ctx):
    v = np.zeros(V)
    if isl(ctx[6]) and isl(ctx[7]):
        p = CH[ctx[6]] + CH[ctx[7]]
        if p in PAIRS:
            d, rest = PAIRS[p]
            v = vec(d, rest)
    return v


def suffix_eval(ctx):
    w, _ = run_letters(ctx, 7)
    n = len(w)
    v = np.zeros(V)
    if n == 0:
        return v
    if n >= 4 and w.endswith("ing"):
        v += vec({" ": 1.6, ",": .9, ".": .8, "!": .3, "?": .2}, -0.5)
    if n >= 3 and w.endswith("ed"):
        v += vec({" ": 1.0, ",": .7, ".": .6, "!": .2}, -0.3)
    if n >= 3 and w.endswith("ly"):
        v += vec({" ": 1.0, ",": .7, ".": .5}, -0.5)
    if n >= 5 and w.endswith("ful"):
        v += vec({" ": 1.2, ",": .6, ".": .5, "y": .5}, -0.3)
    if n >= 4 and w.endswith("est"):
        v += vec({" ": .7, ",": .3, ".": .3})
    if n >= 4 and w.endswith("s") and w[-2] not in "suia":
        v += vec({" ": .5, ",": .3, ".": .3})
    if n >= 4 and w.endswith("er"):
        v += vec({" ": .4, ",": .3, ".": .3})
    if n >= 4 and w.endswith("y") and w[-2] not in "aeiouy" and not w.endswith("ly"):
        v += vec({" ": .5, ",": .3, ".": .5})
    return v


WS = {
    "@sent": {"t": .6, "i": .5, "o": .5, "h": .3, "s": .2, "l": .3, "m": .3, "a": -.6, "w": -.4, "u": -.5, "g": -.4, "c": -.2, "r": -.8, "p": -.5},
    "@comma": {"a": .7, "b": .6, "s": .5, "t": .2, "h": .2, "w": .2, "i": -.2, "c": -.4},
    "the": {"b": .5, "l": .3, "g": .2, "p": .2, "s": .2, "t": -.5, "i": -.6, "u": -.4},
    "a": {"l": .9, "b": .7, "g": .5, "s": .3, "d": .2, "n": .3, "a": -.6, "t": -.4, "w": -.4, "i": -1.0, "u": -.5},
    "to": {"p": .7, "g": .4, "s": .4, "h": .3, "b": .2, "t": .3, "a": -.3, "w": -.2, "i": -.6, "m": .2, "f": .3, "e": .2, "l": .3},
    "was": {"h": .7, "v": .6, "s": .4, "n": .4, "a": .3, "t": .3, "b": .3, "w": -.3},
    "and": {"s": .5, "t": .5, "h": .5, "w": .3, "l": .2, "p": .4, "a": .1, "i": -.2},
    "he": {"w": .7, "s": .5, "l": .4, "h": .3, "f": .3, "c": .3, "d": .2, "g": .3, "t": .2, "m": .2, "a": -.3, "b": -.5, "p": -.3, "i": -.6},
    "she": {"w": .7, "s": .5, "l": .4, "h": .3, "f": .3, "c": .3, "d": .2, "g": .3, "t": .2, "m": .2, "a": -.3, "b": -.5, "p": -.3, "i": -.6},
    "they": {"p": .6, "w": .6, "a": .3, "h": .4, "s": .4, "l": .3, "d": .2, "t": .2, "f": .2},
    "it": {"w": .6, "h": .3, "s": .1},
    "his": {"m": .7, "f": .6, "d": .5, "b": .3, "p": .3, "c": .3, "h": .2, "t": .3, "s": .2, "r": -.2},
    "her": {"m": .7, "f": .6, "d": .5, "b": .3, "p": .3, "c": .3, "h": .2, "t": .3, "s": .2, "r": -.2},
    "of": {"t": .6, "a": .3, "h": .3, "s": .3, "m": .2, "f": .3},
    "in": {"t": 1.0, "a": .3, "h": .4, "i": -.3, "s": .2, "m": .1},
    "on": {"t": .9, "a": .4, "d": .4, "h": .2, "m": .1},
    "with": {"h": .5, "t": .6, "a": .4, "m": .3, "s": .2, "f": .4, "l": .3},
    "very": {"h": .7, "s": .4, "b": .6, "p": .4, "e": .5, "l": .2, "t": .3, "c": .3, "g": .2, "n": .2, "k": .3},
    "so": {"h": .6, "m": .4, "t": .4, "s": .4, "l": .3, "i": .3, "a": .2, "b": .3},
    "named": {"l": .5, "t": .5, "m": .5, "b": .5, "s": .5, "a": .3, "j": .3, "c": .2, "k": .2, "d": .2, "r": .2, "e": .2, "n": .1, "u": -.8, "w": -.3, "i": -.2},
    "one": {"d": 1.6, "o": .4, "t": .2, "s": .1},
    "once": {"u": 1.8},
    "upon": {"a": 1.8},
    "there": {"w": 1.2, "i": .4, "l": .2, "a": .2},
    "but": {"s": .4, "h": .4, "t": .4, "i": .4, "n": .2, "w": .2},
    "not": {"w": .4, "h": .3, "s": .3, "g": .2, "t": .3, "f": .2, "a": .2},
    "did": {"n": 1.3},
    "could": {"n": .8, "s": .2, "h": .2, "g": .2, "p": .2, "b": .2, "f": .3},
    "would": {"n": .8, "s": .2, "h": .2, "g": .2, "p": .2, "b": .2, "f": .3},
    "can": {"i": .5, "w": .3, "y": .4, "s": .2, "h": .2},
    "i": {"w": .7, "a": .5, "c": .3, "l": .5, "h": .2, "d": .1},
    "all": {"t": .6, "d": .5, "h": .3, "a": .2, "m": .2},
    "up": {"t": .4, "i": .3, "a": .2},
    "for": {"t": .5, "a": .3, "h": .3, "m": .3, "y": .1, "e": .2},
    "at": {"t": .5, "h": .4, "s": .2, "a": .2, "n": .3},
    "little": {"g": .7, "b": .6, "d": .3, "c": .3, "m": .2, "r": .1},
    "big": {"d": .3, "t": .3, "b": .3, "r": .2, "h": .2, "s": .2},
    "went": {"t": 1.0, "b": .3, "h": .3, "o": .3, "u": .2, "i": .2, "a": .3, "d": .2, "f": .3},
    "got": {"t": .3, "u": .5, "o": .3, "a": .3, "b": .3, "h": .3, "i": .2, "s": .2},
    "played": {"w": .7, "t": .5, "a": .4, "i": .3, "o": .3, "f": .3, "b": .2, "h": .2},
    "wanted": {"t": 1.3, "a": .2},
    "want": {"t": 1.2, "a": .2, "s": .1},
    "liked": {"t": .4, "h": .3, "p": .3, "s": .2, "m": .3, "a": .2, "e": .2},
    "loved": {"t": .4, "h": .3, "p": .3, "s": .2, "m": .3, "a": .2, "e": .2},
    "love": {"t": .4, "h": .3, "p": .3, "s": .2, "m": .3, "a": .2, "e": .2},
    "like": {"t": .4, "h": .3, "p": .3, "s": .2, "m": .3, "a": .2, "e": .2},
    "looked": {"a": .5, "f": .3, "u": .4, "d": .3, "i": .3, "o": .2, "t": .3, "b": .2},
    "saw": {"a": .6, "t": .5, "h": .3, "s": .3, "l": .2, "b": .1},
    "found": {"a": .6, "t": .4, "h": .3, "s": .3, "m": .2, "b": .1},
    "had": {"a": .6, "f": .3, "t": .3, "s": .3, "n": .3},
    "were": {"h": .5, "s": .4, "a": .4, "t": .3, "v": .3, "p": .3, "e": .2},
    "are": {"h": .5, "s": .4, "a": .4, "t": .3, "v": .3, "p": .3, "e": .2},
    "is": {"a": .5, "s": .3, "v": .3, "n": .3, "t": .2, "h": .3},
    "as": {"s": .4, "t": .4, "h": .3, "b": .3, "a": .3, "l": .1, "m": .2},
    "when": {"s": .5, "h": .5, "t": .5, "i": .4, "w": .3, "l": .3, "m": .3, "a": .2, "y": .2},
    "my": {"m": .6, "f": .6, "d": .4, "b": .3, "t": .3, "p": .3, "s": .3, "h": .2},
    "we": {"w": .4, "c": .3, "a": .3, "h": .3, "l": .3, "s": .3, "m": .2, "d": .2, "g": .3, "p": .3, "f": .2},
    "then": {"t": .3, "h": .5, "s": .5, "i": .4, "a": .3, "w": .3, "l": .3, "m": .3},
}


def wordstart_eval(ctx):
    v = np.zeros(V)
    if ctx[7] != SP:
        return v
    r6 = ctx[6]
    if r6 in (PERIOD, EXCL, QMARK):
        return vec(WS["@sent"])
    if r6 == COMMA:
        return vec(WS["@comma"])
    if isl(r6):
        w, ok = run_letters(ctx, 6)
        if ok and w in WS:
            return vec(WS[w])
    return v
