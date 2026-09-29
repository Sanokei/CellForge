# r1-L1: a grammar-state mixture of experts (ladder/1)

**Result (observed):** final proposal = eval 005: dev bpc **0.6829**, top1 0.8083 (the uniform start is 5.0). I used 6 of the 25 dev-evals.
Eval history: 000 0.7257 -> 001 0.6841 -> 002 0.6844 -> 003 0.6838 -> 004 0.6837 -> 005 0.6829.

## What I built (scripts/build.py expands my typed lists into proposal.json)
- **Trunk (root clusters).** `cl_wordcls` holds step units that recognise the word that just finished at row 6 of ctx when prev is a space (and also ',' '.').
  It maps that word to one of 15 categories. `cl_update` then updates a 17-d recurrent state:
  slot (8 bits, where "none set" means DET), NP number, rn (not-subject-role flag), md (main verb done), au ("and" used),
  a pending-verb stack v (base-3 digits in one scalar: push (b+v)/3, pop 3v-top), stack depth d, and D (modifier depth of the current NP).
  `cl_stack` decodes top/d/D with step units. `cl_feat` makes 21 binary features, including length buckets taken from `step`.
- **Router:** an identity cell maps the slot bits to 6 scores (DET gets a 0.5 bias). k=1 dispatch.
- **Experts:** ex_det, ex_adjn (adjectives and nouns), ex_post (after a noun: prep/that/verb/adverb), ex_verb (after an RC verb when more verbs are pending), ex_adv, ex_func ("and", the space after ',', and ' ' vs eos after '.').
  Each expert has 4 clusters:
  1. prefix-match step units over ctx, one per (word, prefix length), plus end-char units;
  2. a situation one-hot from the features, then a lookup of log word-probabilities, then exp;
  3. mul, then a char map;
  4. `cl_logout`: a relative floor eps=1e-6, a piecewise-linear ln (74 knots, ratio sqrt2), and pad/bos set to -60.
- **Route counts (eval 005):** det 13141, adjn 22200, post 13593, verb 1905, adv 1273, func 2374.

## Key numbers (all typed by me)
- Dets: the .32, this .135, every .13, a .115, many .105, these .10, some .095. P(adj)=.31. P(sg | "the")=.6. P(IV | main verb)=.58.
- Adverb at clause end: .28. ", and": .19 (at most one per sentence).
- Modifier probability (prep/that) by depth: subject head .15/.125, object head .08/.16, depth-1 .10/.12, depth-2 .002/.002.
- eos after '.': .001 if length <140, .03 for 140-149, .99 for >=150.

## Knowledge sources
- **From reading dev_train by eye** (about 45 documents; I kept hand tallies, no code):
  - the closed lexicon (7 dets, 8 adjectives, 12 nouns with irregular plurals, 6 intransitive and 6 transitive verbs, 4 preps, 5 adverbs, that, and);
  - the grammar: NP = Det Adj? N with at most one PP or object-gap relative clause; at most one ", and" per sentence; "a old" is used, so there is no a/an rule;
  - that no depth-2 NP ever takes a modifier;
  - that documents stop at the first '.' once length is about 150 or more (from hand-counted lengths 144 continue / 154 stop);
  - all probabilities above, from my tallies of 24 documents.
- **General English knowledge:** subject-verb agreement (-s on singular verbs); relative clauses are center-embedded, so pending verbs behave like a stack; the noun/verb/adjective roles of the words.
## Observed vs hypotheses
- Observed: removing the uniform floor (eps 1e-3 -> 1e-6) plus the det/modifier re-estimates gave -0.042 bpc (000->001).
  The depth rule gave -0.0006 (003). Sharper eos cut after_period bpc from 0.044 to 0.007 (005). No dev position cost more than 6.3 bits, so on these 300 docs the grammar is never broken.
- Re-estimating the parameters from a second batch of documents did nothing (002). My guess is that the model now sits close to the entropy of the source grammar.
- Hypotheses I did not use: a per-document cache of nouns (hand-estimated lambda about .1, worth under 0.003 bpc) and a per-document adverb rate (weak overdispersion).
- Next steps: count more documents by hand to pin down the sentence-start det distribution (after_period_space is 2.04 bits against about 1.95 expected), and a split of depth-1 modifiers into RC subject vs PP object.
