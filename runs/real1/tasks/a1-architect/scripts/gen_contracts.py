"""Expands the architect's hand-typed tables (lists.py, lists2.py) into contract JSON files.

Counts nothing from any corpus: it only evaluates the formulas stated in each contract's spec on typed tables
to produce expected vectors for hand-picked cases.  Run: python gen_contracts.py
"""
import json, os
import numpy as np
from lists2 import *  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "contracts")
os.makedirs(OUT, exist_ok=True)


def onehots(rows):
    return {"onehots": rows, "n": V}


STD = ("Vocabulary ids: 0 <pad>, 1 <bos>, 2 <eos>, 3 ' ', 4 ',', 5 '.', 6 '!', 7 '?', 8 \"'\", 9 '\"', "
       "10..35 = a..z (a=10, e=14, i=18, o=24, u=30, y=34). Input ctx [8,36]: one-hot rows, oldest first, row 7 = most "
       "recent token; an all-zero row means 'before document start'. LETTER row = token in 10..35. BOUNDARY row = an "
       "all-zero row, or a row whose token is not a letter and is not the apostrophe (id 8) [so bos, space, , . ! ? and the "
       "double quote are boundaries; the apostrophe is NOT]. 'word so far' = the maximal run of LETTER rows ending at row 7 "
       "(or at the row named in the rule); it is 'anchored' when the row just before the run exists and is a BOUNDARY row. "
       "Evidence output y [36] is added to next-token logits by the parent, in nats; 0 means 'no opinion'. Entries for "
       "ids 0,1 must be exactly 0 unless a rule says otherwise; id 2 (<eos>) is 0 unless a rule says otherwise. Case 'note' "
       "fields explain the case; 'expect' vectors are index 0..35 in vocabulary order and must be matched within tol. "
       "Inputs are exact one-hot / 0-1 values. Tables in 'data' are authoritative and must not be changed. You may write a "
       "helper script that expands 'data' into proposal JSON. Each cell/param holds at most 40 numbers (spread lookups over "
       "many cells and combine with ops; ops on ctx rows such as slice/mul/sum are exact and free of parameters).")


def write(name, spec, inputs, data, cases):
    assert len(cases) >= 6, name
    c = {"component": name, "version": 1, "inputs": inputs, "output": "y", "spec": STD + "\n\n" + spec, "data": data, "cases": cases}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(c, f, indent=1)
    print(name, len(cases), "cases")


def ecase(ctx_text, fn, q=None, tol=0.02, note=""):
    rows = ctx_from(ctx_text)
    ins = {"ctx": onehots(rows)}
    if q is None:
        y = fn(rows)
    else:
        ins["q"] = [float(q)]
        y = fn(rows, q)
    return {"note": (note or "") + f" [ctx={ctx_text!r}" + (f" q={q}" if q is not None else "") + "]", "in": ins,
            "expect": [round(float(x), 4) for x in y], "tol": tol}


CTXIN = {"ctx": {"shape": [8, 36]}}
CTXQ = {"ctx": {"shape": [8, 36]}, "q": {"shape": [1]}}


def bg(prev_txt, gt=None, lt=None, argmax=None, note=""):
    rows = ctx_from(prev_txt)
    g = np.full(V, -12.0); l = np.full(V, 1000.0)
    g[0] = g[1] = -1000; l[0] = l[1] = -10.0
    for c, x in (gt or {}).items():
        g[IDX[c]] = x
    for c, x in (lt or {}).items():
        l[IDX[c]] = x
    c = {"note": note + f" [ctx={prev_txt!r}]", "in": {"ctx": onehots(rows)}, "expect_gt": g.tolist(), "expect_lt": l.tolist()}
    if argmax:
        c["expect_argmax"] = IDX[argmax]
    return c


def main():
    # ---- router
    cases = []
    for t, n in [("^", "bos"), ("the big ", "space"), ("_______l", "1 letter"), ("s. he wa", "2 letters"), ("d he was", "3+ letters"),
                 ("one day,", "comma"), ("at home.", "period"), ("help me!", "excl"), ("you ok?", "qmark"), ('said, "', "dquote"),
                 ("___didn'", "apostrophe"), ("____it's", "letter after apostrophe = length 1"), ("________", "all-empty ctx"),
                 ("_____ he", "2 letters with empty rows"), ("ab'cd", "run of 2 letters after apostrophe even though row 5 is a letter")]:
        cases.append(ecase(t, router_eval, note=n))
    write("cl_router", """Router scores. y [7]: scores for experts in this order: 0 ex_bos, 1 ex_wordstart, 2 ex_inword_short, 3 ex_inword_long,
4 ex_punct, 5 ex_quote, 6 ex_apos. Class of the most recent token (row 7 = prev): bos(1) -> 0; space(3) -> 1; a letter whose word-so-far has length 1 or 2 -> 2;
a letter whose word-so-far has length >= 3 (rows 5,6,7 ALL letters) -> 3; one of , . ! ? (ids 4..7) -> 4; double quote (9) -> 5; apostrophe (8) -> 6.
y[k] = 3.0 for the class, 0.0 for the others. Any other prev (<eos>, <pad>, all-zero row 7) -> y = [0,1,0,0,0,0,0]. Only rows 5,6,7 of ctx matter.
Word-so-far length counts letters only: an apostrophe or any non-letter ends the run (so in "it's", the 's' has length 1; in "b'cd" the 'd' has length 2).""",
          CTXIN, {}, cases)

    # ---- qstate
    def qc(prev, q, exp, note):
        return {"note": note, "in": {"prev": {"onehot": prev, "n": V}, "q": [q]}, "expect": [exp], "tol": 0.02}
    cases = [qc(9, 0.0, 1.0, "open quote"), qc(9, 1.0, 0.0, "close quote"), qc(IDX["a"], 1.0, 1.0, "letter inside quote"),
             qc(IDX["a"], 0.0, 0.0, "letter outside"), qc(SP, 1.0, 1.0, "space inside"), qc(PERIOD, 1.0, 1.0, "period inside"),
             qc(1, 1.0, 0.0, "bos resets"), qc(1, 0.0, 0.0, "bos at 0"), qc(APOS, 0.0, 0.0, "apostrophe does not toggle"),
             qc(APOS, 1.0, 1.0, "apostrophe does not toggle inside")]
    write("cl_qstate", "Inside-quote flag. Inputs prev [36] one-hot of the token just emitted, q [1] the flag before seeing prev (0 or 1). "
          "Output y [1] = the new flag: if prev is the double quote (id 9) then 1-q; else if prev is <bos> (id 1) then 0; else q. "
          "Meaning: 1 iff an odd number of double quotes have occurred since <bos> (we are inside a quotation). "
          "This circuit has its own input names (prev, q), not ctx.",
          {"prev": {"shape": [36]}, "q": {"shape": [1]}}, {}, cases)

    # ---- bigram
    cases = [
        bg("^", {"o": -1.0, "t": -3.5}, {" ": -6, "z": -6}, "o", "start of story: 'once upon'/'one day' dominate"),
        bg("thing.", {" ": -0.5, "<eos>": -3.4}, {"<eos>": -1.3, "a": -6}, " ", "after period: space; eos moderately rare"),
        bg("one day,", {" ": -0.6}, {"a": -6, "e": -6}, " ", "after comma: space"),
        bg("me!", {" ": -1.5, '"': -3.0}, {}, None, "after '!': space or closing quote"),
        bg("okay?", {" ": -1.5, '"': -3.0}, {}, None, "after '?'"),
        bg("the big ", {"t": -2.2, "a": -3.0, "s": -3.2}, {"q": -5, "x": -5, ".": -5, ",": -5, " ": -5}, "t", "after space: t most likely word-initial letter"),
        bg("a q", {"u": -0.2}, {"a": -3, " ": -6}, "u", "q -> u"),
        bg("ple", {" ": -1.3}, {"q": -5}, None, "after e: word end common"),
        bg("goin", {"g": -1.6}, {"q": -6}, None, "after n: g (ing) very common"),
        bg("going", {" ": -1.3}, {"b": -5, "z": -5}, None, "after g: word end (-ing)"),
        bg("t", {"h": -1.6}, {"q": -5}, None, "after t"),
        bg("h", {"e": -1.6}, {"q": -5}, "e", "after h: e most likely"),
        bg("don'", {"t": -1.6, "s": -1.3}, {"q": -6, " ": -6}, None, "after apostrophe: t and s dominate"),
        bg("her ", {"t": -2.2}, {".": -5}, None, "after space again with different left context (must ignore rows 0..6)"),
        bg("zzzq", {}, {}, None, "all entries finite in [-12,0]-ish; pad/bos <= -10 on odd contexts"),
    ]
    write("cl_bigram", """Character bigram backbone. y[c] = natural-log probability of next token c given ONLY the previous token (row 7 of ctx; rows 0..6 are ignored),
for simple children's-story text (lowercase; characters space , . ! ? ' " and a-z; whole story then <eos>). Every entry is a log-probability estimate in [-12, 0]
except ids 0 and 1 (<pad>,<bos>) which must be <= -10 (use -20). Think of the unigram/bigram statistics of stories like 'once upon a time, there was a little
girl named lily. she loved to play in the park with her friends.' Requirements are the bounds in the cases (expect_gt / expect_lt / expect_argmax): all must hold. Outside them choose the numbers
from your knowledge of English. Rows should be roughly normalized: sum of exp(y) within 0.3 of 1 for prev in a-z, space, and , . ! ? (not tested, but the parent relies on it).
What should be true: after a letter the next token is another letter, a space, or , or . (word ends are common after e, s, d, y, g, t, n, r); after a space a word-initial letter (t, a, s, h, w, b, l most common;
almost never q, x, z, j, punctuation, or another space); after '.' a space (or <eos> about 10%, or a closing quote); after ',' a space (or closing quote); after '!' or '?' a space or a closing quote;
after the apostrophe s, t, m, l, r, v, d; after the double quote (ambiguous open/close) a space, a word-initial letter, or , . ; after <bos>: 'o' (once/one) dominates, then t, l, m, s, a, b.
Impossible events (e.g. space after space, ',' after ',') get about -9.""", CTXIN, {}, cases)

    # ---- trigram
    cases = []
    for t, n in [("sh", "sh"), ("wath", "th anywhere in a word"), ("thing", "ng"), ("qu", "qu"), ("played", "ed"), ("xy", "unlisted pair"), (" t", "row 6 space -> zeros"),
                 ("th ", "row 7 non-letter -> zeros"), ("a day", "ay"), ("oun", "'un' unlisted"), ("_____ou", "ou after empty rows"), ("happ", "pp"),
                 ("d,g", "row 6 comma -> zeros")]:
        cases.append(ecase(t, pairs_eval, note=n))
    write("cl_trigram", """Letter-pair (trigram) corrections. y is nonzero only when rows 6 and 7 of ctx are BOTH letters (the two most recent tokens form a letter pair, anywhere in a word, no anchor needed) AND that pair is a key of data.pairs.
Then y = the listed additive deltas (nats) for the listed next tokens (keys are literal next characters; ' ' is the space token), plus data.rest_letters[pair] (if present) added to ALL 26 letter entries
(listed letter deltas are added on top of the rest value), 0 elsewhere. If the pair is not listed, or either row is a non-letter or empty, y is all zeros. Rows 0..5 are ignored. Ids 0,1,2 are 0.""",
          CTXIN, {"pairs": {k: v[0] for k, v in PAIRS.items()}, "rest_letters": {k: v[1] for k, v in PAIRS.items() if v[1]}}, cases)

    # ---- tries
    def trie_cases(words, texts):
        return [ecase(t, lambda r: trie_eval(words, r), note=n) for t, n in texts]
    fn_texts = [("d and th", "prefix th (anchored by space)"), ("and he t", "prefix t"), ("nd the", "word 'the' complete plus continuations"),
                ("____bathe", "th inside a word: not anchored -> zeros"), ("zoo z", "prefix z unlisted"), ("______sh", "anchored by empty rows"),
                ('said, "th', "anchored by double quote"), ("____it's", "prefix s after apostrophe: not anchored -> zeros"), ("and the ", "prev is space -> zeros"),
                ("and a", "single-letter word a plus continuations"), ("ate. i", "i complete plus continuations"), ("abcdefgh", "8 letters: anchor invisible -> zeros"),
                ("and bec", "prefix bec -> because"), ("so sho", "prefix sho (unlisted) -> zeros"), ("__ was", "was complete: only END"), ("ing, t", "prefix t after comma-space")]
    write("cl_words_fn", """Word-completion trie, function words. Given data.words (word -> integer weight F0), compute for the anchored word-so-far w (length j, 1 <= j <= 7, anchored by a BOUNDARY row; if the
run is longer than 7 letters, or not anchored, or row 7 is not a letter, y = all zeros):
  for each letter c: F(w,c) = sum of weights of listed words that are longer than w and start with w+c;  y[c] = min(0.7*ln(1+F(w,c)), 5.0)  (0 if F = 0).
  if w itself is a listed word: y[' '] (id 3) = min(0.7*ln(1+weight(w)), 5.0); no other non-letter entry is ever set.
All other entries 0. Words are lowercase letters only. You may generate the trie from data.words with a helper script you write (the list is given to you; do not change the weights). Because one cell holds at most 40 numbers,
spread the lookup over many cells (e.g. one output-letter cell per group of <=40 trie nodes) and add them with ops. Prefix detection must use ops on ctx rows (slice / mul / sum); letter probes are exact.
At most one trie node is active at a time (the anchored word-so-far), so node detectors are exclusive and the lookup is linear.""",
          CTXIN, {"words": WORDS_FN, "cap": 5.0, "gain": 0.7}, trie_cases(WORDS_FN, fn_texts))
    op_texts = [("a litt", "prefix litt"), (" happ", "prefix happ"), ("girl", "girl: row before is empty; complete"), ("a girl", "girl complete"), ("as ba", "prefix ba"),
                ("togethe", "7-letter prefix"), ("a friend", "friend complete plus friends"), ("d lo", "lo -> love loved look..."), ("____ing", "ing not listed"),
                ("xbig", "not anchored"), ("one d", "prefix d"), ('said, "m', "prefix m after quote"), ("to be", "'be' is in the fn list, not here"),
                ("named ", "prev space -> zeros"), ("_tim", "tim complete"), ("a fl", "prefix fl")]
    write("cl_words_open", """Word-completion trie, content words and names. Identical mechanism and formula to cl_words_fn (repeated here): for the anchored word-so-far w
(length 1..7, anchored by a BOUNDARY row, row 7 must be a letter; else all zeros): y[c] = min(0.7*ln(1+F(w,c)),5.0) with F(w,c) = sum of data.words weights of listed words longer than w that start with w+c;
if w is itself a listed word then y[' '] (id 3) = min(0.7*ln(1+weight(w)),5.0). All else 0. The word list here is different (nouns, verbs, adjectives, names), ~170 words; the 8-letter word
'together' and 9-letter 'beautiful' only contribute through their prefixes of length <= 7 (they never trigger the ' ' entry). Spread across many <=40-element cells; you may generate the JSON with a
helper script that expands data.words. Do not change weights.""",
          CTXIN, {"words": WORDS_OPEN, "cap": 5.0, "gain": 0.7}, trie_cases(WORDS_OPEN, op_texts))

    # ---- suffix
    sfx = [("e played", "ed"), ("was red", "red is 3 letters: ed applies"), (" ed", "only 2 letters: no rule"), ("e singing", "ing"), (" ing", "3 letters: no ing rule"),
           ("really", "ly"), ("wonderful", "ful"), ("fastest", "est"), ("friends", "s plural, s[-2]=d"), ("us", "too short"), ("this", "s after i: excluded"),
           ("happy", "y after consonant"), ("very", "y after r, 4 letters"), ("boy", "y after o, short"), ("slowly", "ly not y"), ("bigger", "er"), ("was ", "row 7 non-letter -> zeros"),
           ("yes", "3 letters s: no (n>=4)"), ("plays", "s after y (4+ letters)")]
    write("cl_suffix", """Word-ending evidence from the LAST letters of the word so far (no anchor needed). w = maximal run of letter rows ending at row 7 (up to 8 letters, n = its length; n = 0 if row 7 is not a letter -> zeros).
y is the SUM of the deltas of every rule below that fires (each delta vector has zeros elsewhere; 'rest' means: add that value to all 26 letter entries):
 ING:  n>=4 and w ends 'ing':   ' ' +1.6, ',' +0.9, '.' +0.8, '!' +0.3, '?' +0.2, rest -0.5
 ED:   n>=3 and w ends 'ed':    ' ' +1.0, ',' +0.7, '.' +0.6, '!' +0.2, rest -0.3
 LY:   n>=3 and w ends 'ly':    ' ' +1.0, ',' +0.7, '.' +0.5, rest -0.5
 FUL:  n>=5 and w ends 'ful':   ' ' +1.2, ',' +0.6, '.' +0.5, 'y' +0.5, rest -0.3
 EST:  n>=4 and w ends 'est':   ' ' +0.7, ',' +0.3, '.' +0.3
 S:    n>=4 and w ends 's' and the second-to-last letter is not one of s,u,i,a:  ' ' +0.5, ',' +0.3, '.' +0.3
 ER:   n>=4 and w ends 'er':    ' ' +0.4, ',' +0.3, '.' +0.3
 Y:    n>=4 and w ends 'y', second-to-last letter not in a,e,i,o,u,y, and w does not end 'ly':  ' ' +0.5, ',' +0.3, '.' +0.5
Rules may overlap (e.g. 'wonderfully' ends in ly so only LY fires; a word ending 'ests' fires only S). Ids 0,1,2 are 0.""", CTXIN, {}, [ecase(t, suffix_eval, note=n) for t, n in sfx])

    # ---- wordstart
    wst = [("park. ", "sentence start"), ("day, ", "after comma"), ("_on the ", "the"), ("_was ", "was (anchor is an empty row)"), ("_a ", "a"), ("went to ", "to"),
           ("_named ", "named"), ("one ", "one"), ("_once ", "once"), ("xwanted ", "7-letter run: anchor invisible -> zeros"), ("zzz ", "unlisted word -> zeros"),
           ("wow! ", "after !"), ("what? ", "after ?"), ("_little ", "little"), ("he saw", "row 7 not space -> zeros"), ("she ", "she"), ('said "', "after quote char -> zeros"), ("it's ", "s after apostrophe: not anchored -> zeros")]
    write("cl_wordstart", """First-letter evidence at a word start, keyed on the previous word/punctuation. Only active when row 7 is the space token; otherwise y = zeros.
Let r6 = the token in row 6. If r6 is '.', '!' or '?': y = data.table['@sent']. If r6 is ',': y = data.table['@comma']. If r6 is a letter: let u = the maximal run of letter rows ending at row 6
(length k; it must be anchored, meaning row 6-k exists and is a BOUNDARY row, so k<=6) - if u is a key of data.table then y = data.table[u]; otherwise zeros. Any other r6 (quote, apostrophe, empty, etc.): zeros.
Table values are additive nats for the listed first letters (keys are letters; unlisted letters and all non-letter entries are 0). Ids 0,1,2 are 0.""",
          CTXIN, {"table": WS}, [ecase(t, wordstart_eval, note=n) for t, n in wst])

    # ---- colloc
    col = [("upon a t", "upon a time -> i"), ("_once up", "once upon -> o"), ("_once ", "once -> u"), ("one", "one -> space (j=3)"), ("_one da", "one day -> y"),
           ("there wa", "there was -> s"), ("and there was", "there was a: ' '"), ("wanted t", "wanted to -> o"), ("bathe", "'the end'/'the sun' contexts unanchored: nothing"), ("__so muc", "so much fun -> h"),
           ("_a lot o", "a lot of -> f"), ("in the s", "in the sky / the sun"), ("mom and d", "mom and dad"), ("th", "j<3 never fires"),
           ("_tim'", "nothing"), ("the end", "the end"), ("she said", "said -> ','?"), ("said", "said -> ','")]
    write("cl_colloc", """Fixed-phrase (multi-word) evidence. data.phrases maps phrase -> strength s (nats). Each phrase P may contain spaces and punctuation from the vocabulary. For each phrase P and each position j with 3 <= j <= len(P)-1 (predicting the character P[j]):
  let cs = P[max(0,j-8):j] (the up to 8 characters just before P[j]); m = len(cs).
  The rule FIRES iff the last m rows of ctx (rows 8-m..7) are all non-empty and spell cs exactly, AND (if j < 8, so that cs starts at the phrase start) the row 7-j (just before cs) is a BOUNDARY row
  (empty rows count as boundary; the apostrophe does not). If j >= 8 no boundary condition is applied.
  When it fires, y[id(P[j])] += s. y is the SUM over all phrases and positions. Ids 0,1,2 are 0. Rows to the left of the matched window are ignored. The table is data.phrases; do not change strengths.
  Implement with ops on ctx rows (exact one-hot products) feeding <=40-element cells (spread the phrase-position detectors over many cells grouped by target character).""",
          CTXIN, {"phrases": PHRASES}, [ecase(t, colloc_eval, note=n) for t, n in col])

    # ---- endpunct
    ep = [("one day", 0, "day comma"), ("one day", 1, "day inside quote"), ("she wow", 0, "wow outside"), ("she wow", 1, "wow inside"), ("was fun", 0, "fun"), ("so ok", 0, "ok"),
          ("the dog", 1, "dog in quote"), ("xthen", 0, "then unanchored: only q terms"), ("abcdefgh", 0, "8-letter run: only q terms"), ("a big ", 0, "row 7 space -> zeros"),
          ("a big ", 1, "row 7 space, q=1 -> zeros"), ("home.", 0, "row 7 non-letter"), ("yay", 1, "yay in quote"), ("in time", 0, "time: comma and period")]
    write("cl_endpunct", """Punctuation choice at a possible word end. Inputs ctx [8,36] and q [1] (1 = inside a quotation). Only active when row 7 is a letter; if row 7 is not a letter y = all zeros.
Let w = the anchored word-so-far (run of letter rows ending at row 7, anchored by a BOUNDARY row, length <= 7); if it is not anchored or longer than 7, only the q terms apply.
  y[','] = data.comma[w] (0 if absent); y['.'] = data.period[w]; y['!'] = data.excl[w]   (lookups use only anchored w).
  q terms (added whenever row 7 is a letter): if q >= 0.5: y['!'] += 0.9, y['?'] += 0.6.  if q < 0.5: y['!'] -= 0.4, y['?'] -= 0.6.
All other entries 0. Ids 0,1,2 are 0.""",
          CTXQ, {"comma": COMMA_W, "period": PERIOD_W, "excl": EXCL_W}, [ecase(t, endpunct_eval, q=q, note=n) for t, q, n in ep])

    # ---- quote
    qt = [("said, \"", 1, "just opened"), ("say. \"", 1, "opened after period"), ('go!"', 0, "just closed"), ("ok, ", 1, "inside quote, prev space"),
          ("wow!", 1, "inside quote after !"), ("me,", 1, "inside quote after comma"), ("home.", 1, "inside quote after period"), ("home.", 0, "outside after period: no rule"),
          ("she said, ", 0, "said, space -> quote"), ("asked, ", 0, "asked, "), ("her mom, ", 0, "mom, "), ("day, ", 0, "day, -> base only"), ("home. ", 0, "after period space outside"),
          ("plain", 1, "letter inside quote: only eos"), ("plain", 0, "letter outside: zeros"), ('said, "hi', 1, "letter inside")]
    write("cl_quote", """Quotation dynamics. Inputs ctx and q (q = 1 iff currently inside a quotation, ALREADY updated for row 7: right after an opening quote token q=1). Let p = the token in row 7. y is the SUM of:
 E:  if q >= 0.5: y[<eos>] += -6.
 R1: q>=0.5 and p is the double quote (just opened): add data.qopen letters; y['"'] += -3; y[' '] += -3; y[','] += -2; y['.'] += -2.
 R2: q<0.5 and p is the double quote (just closed): y[' '] += 0.8, y[','] += 0.3, y['.'] += 0.4, y['"'] += -3, and -1.5 to all 26 letters.
 R3: q>=0.5 and p in , . ! ?: y['"'] += 1.0 / 0.6 / 1.6 / 1.6 respectively (closing quote likely).
 R4: q<0.5 and p is space and row 6 is ',': y['"'] += 0.4; plus 2.1 more if row 5 is the letter 'd' (said, asked, ...); plus 0.6 more if row 5 is the letter 'm' (mom, ...).
 R5: q<0.5 and p is space and row 6 is one of . ! ?: y['"'] += 0.8.
 R6: q>=0.5 and p is space: y['"'] += -2.5.
Everything else 0 (E and R1 both apply just after an opening quote). Ids 0,1 are 0. It is a small case analysis on rows 5-7 and q.""",
          CTXQ, {"qopen": QOPEN}, [ecase(t, quote_eval, q=q, note=n) for t, q, n in qt])

    # ---- apos
    ap = [("___didn'", "didn' -> t"), ("____i'", "i' -> m"), ("____it'", "it' -> s"), ("_you'", "you' -> r"), ("__tom'", "tom' -> possessive s"), ("_____a'", "single unlisted letter -> zeros"), ("didn", "stem didn -> apostrophe"),
          ("_don", "don"), ("_i", "i alone -> apostrophe +1.0"), ("xdidn", "run 'xdidn' not anchored (row 0 letter) -> zeros"), ("_can", "can -> +0.4"), ("_let", "let -> +1.2"),
          ("_______'", "apostrophe with nothing before -> zeros"), ("a big '", "space before apostrophe -> zeros"), ("_can'", "can' -> t"), ("here'", "here' -> s"), ("_hello", "letter, unlisted: zeros")]
    write("cl_apos", """Apostrophe / contraction evidence. Two mutually exclusive cases on row 7 (p):
 A) p is the apostrophe token (id 8): let w = run of letter rows ending at row 6 (must be anchored by a BOUNDARY row, length between 1 and 6; otherwise y = zeros).
    In this order: w == 'i': y[m]=2.5, y[l]=1.2, y[v]=0.8, y[d]=0.3.
    elif w in {it,that,what,let,he,she,there,here,who,where,how}: y[s]=2.4, y[l]=0.6.
    elif w in {you,we,they}: y[r]=2.0, y[l]=1.2, y[v]=1.0, y[d]=0.4.
    elif w ends with 'n' and len(w)>=3: y[t]=2.6, y[s]=-1.0.
    elif len(w)>=2: y[s]=2.0 (possessive).   else zeros.
 B) p is a letter: w = anchored word-so-far (length <= 7; must be anchored): if w is a key of data.stems then y[apostrophe id 8] = data.stems[w]; otherwise zeros.
 Any other p: zeros. Ids 0,1,2 are 0.""", CTXIN, {"stems": APOS_STEM}, [ecase(t, apos_eval, note=n) for t, n in ap])

    # ---- name write / read
    def ncase(text, state, fn, note, rnd=4):
        rows = ctx_from(text)
        y = fn(rows, state)
        return {"note": note + f" [ctx={text!r}]", "in": {"ctx": onehots(rows), "nm": [float(x) for x in state]}, "expect": [round(float(x), rnd) for x in y], "tol": 0.02}
    z = np.zeros(NM)
    cases = [ncase("a girl named ", z, nm_write, "trigger: 'named ' starts capture"),
             ncase("named l", nm_state("", 1), nm_write, "first letter written"),
             ncase("named li", nm_state("l", 1), nm_write, "second letter written"),
             ncase("named lil", nm_state("li", 1), nm_write, "third"),
             ncase("named lily", nm_state("lil", 1), nm_write, "fourth"),
             ncase("lily ", nm_state("lily", 1), nm_write, "space ends capture, buffer kept"),
             ncase("lily ", nm_state("lily", 0), nm_write, "not capturing: unchanged"),
             ncase("sophies", nm_state("sophie", 1), nm_write, "6 letters full: extra letter ignored, L stays 6"),
             ncase("e named ", nm_state("bob", 0), nm_write, "new trigger clears old name"),
             ncase("lily.", nm_state("lily", 1), nm_write, "punctuation ends capture"),
             ncase("go home", z, nm_write, "nothing happens")]
    write("cl_name_write", """Name-cache writer (state update). Inputs ctx [8,36] and nm [224] (previous cache state); output y [224] = new cache state. Layout of nm: elements 0..215 = buffer B as 6 rows x 36 (row-major:
element 36*i+t is 1 iff the i-th captured letter has token id t); element 216 = cap (1 while a name is being captured, else 0); elements 217..223 = one-hot of L (the number of letters captured, 0..6).
Rules, in priority order, evaluated on this step's ctx (row 7 = the token just emitted):
 1. TRIGGER: if rows 2..7 of ctx are exactly n,a,m,e,d,space (ids 23,10,22,14,13,3): output buffer all zeros, cap = 1, L = 0 (element 217 = 1, the rest of 217..223 = 0).
 2. elif cap >= 0.5 and row 7 is a letter: if L < 6: set B[L] to the one-hot of row 7 and advance L by one (one-hot shifts to L+1); if L == 6: nothing changes; cap stays 1.
 3. elif cap >= 0.5 (row 7 is not a letter): cap = 0, buffer and L unchanged.
 4. else: output = input unchanged.
The state inputs are always valid (0/1 values, L one-hot, buffer rows one-hot for i < L, zero rows beyond). Meaning: remembers the letters of the most recent 'named <word>' (protagonist names like 'lily'), up to 6 letters.""",
          {"ctx": {"shape": [8, 36]}, "nm": {"shape": [NM]}}, {}, cases)

    lily = nm_state("lily", 0)
    cases = [ncase("day. ", lily, nm_read, "word start after space: boost first letter l"),
             ncase('said, "', lily, nm_read, "word start after quote: boost l"),
             ncase("she l", lily, nm_read, "k=1 matches: boost i (1.5)"),
             ncase("she li", lily, nm_read, "k=2 matches: boost l (2.5)"),
             ncase("she lil", lily, nm_read, "k=3: boost y (3.0)"),
             ncase("she lily", lily, nm_read, "k=4 = L: word-end boost"),
             ncase("she lo", lily, nm_read, "mismatch: zeros"),
             ncase("xlil", lily, nm_read, "not anchored: zeros"),
             ncase("she li", nm_state("lily", 1), nm_read, "still capturing: zeros"),
             ncase("day. ", z, nm_read, "no name stored: zeros"),
             ncase("day. ", nm_state("a", 0), nm_read, "L=1 < 2: zeros"),
             ncase("she tim", nm_state("tim", 0), nm_read, "name tim complete -> end boost"),
             ncase("a mia", nm_state("mia", 0), nm_read, "mia complete"),
             ncase("day,", lily, nm_read, "prev comma: zeros"),
             ncase("_sophi", nm_state("sophie", 0), nm_read, "k=5 of 6: boost e (3.0)")]
    write("cl_name_read", """Name-cache reader. Inputs ctx [8,36] and nm [224] (layout exactly as in cl_name_write). Output y [36] evidence. Let cap = nm[216], L = argmax of nm[217..223], buffer B = nm[0..215] as 6x36,
name[i] = argmax of B[i] for i < L. If cap >= 0.5 or L < 2: y = zeros. Otherwise, with p = token in row 7:
 * p is space (3) or double quote (9) (a word is about to start): y[name[0]] = 1.2.
 * p is a letter: let w = anchored word-so-far (run of letters ending at row 7, anchored by a BOUNDARY row, k = len(w) <= L). If w equals name[0:k]: if k < L (and k <= 5): y[name[k]] = g_k with g_1 = 1.5, g_2 = 2.5, g_3 = 3.0, g_4 = 3.0, g_5 = 3.0;
   if k == L: y[' '] = 1.5, y[','] = 0.7, y['.'] = 0.7, y["'"] = 0.6, y['!'] = 0.3, y['?'] = 0.2. Otherwise (mismatch, k > L, or not anchored): zeros.
 * any other p: zeros.
Ids 0,1,2 are 0. Inputs are always valid states. Dot products of buffer rows with ctx rows give the letter-match tests (ops), plus a few small cells or scale ops.""",
          {"ctx": {"shape": [8, 36]}, "nm": {"shape": [NM]}}, {"g0": 1.2, "g": G, "end": END}, cases)


if __name__ == "__main__":
    main()
