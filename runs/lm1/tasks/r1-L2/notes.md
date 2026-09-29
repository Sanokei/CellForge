# r1-L2: hand-built grammar-parser mixture of experts (tinylm/1, ladder/1)

**Final:** proposal.json (the same as the eval-011 candidate): dev bpc **0.6790**, top1 0.8075, 0 failures (evals/010/report.json). 11 of 25 dev-evals used.

## What I built
- **Trunk cluster `cl_features`**: (a) 213 prefix detectors, step cells. Each one fires only when the current word is exactly a given prefix, anchored on a preceding space or <bos>; there are also detectors for <bos>, ' ', ',' and '.'. (b) 60 whole-word detectors map the word just completed to 14 word classes (det sg/pl/the, adj, noun sg/pl, prep, that, verb-with-object +s/-s, verb-without-object +s/-s, adv, and). (c) Gates for ',' and '.'.
- **Recurrent state (343 dims)** holds a parser: a one-hot phase (47) plus a one-hot stack (296). The phase transition is looked up from phase x word class x (stack top type, whether this is the clause after ", and"). The stack changes through 8 operations: push-subject, set-number s/p, push-relative-clause, pop, subject-to-object, clear, reset. The stack holds the main frame (subject with sg/pl, or object) plus up to 2 relative-clause frames with number. ',' and '.' override the state to COMMA or PERIOD. Clusters: `cl_fsm_in`, `cl_fsm_phase`, `cl_fsm_stack`, `cl_fsm_punct`.
- **Router**: phase x stack summary (top frame, frame below it, and-flag) goes through a lookup cell, scores the 60 experts, and dispatches with k=1.
- **Experts (60)**: each is one syntactic situation, for example "expect determiner", "after sg determiner in subject", "after subject head noun, pl", "inside relative-clause subject, sg", "after object noun in the clause after ', and'". Each expert holds a prefix -> next-char log-prob table. The PERIOD expert adds an EOS term based on document length: 6*tanh(0.6*(step-148.5)), applied only when prev='.'.
- scripts/build.py expands my typed lexicon, grammar rules and probabilities into all of these tables. It enumerates the parser states my rules reach, marginalises my word distribution over each prefix, and groups states with identical tables into one expert. It never reads the corpus. scripts/selftest.py scores only sentences I typed myself, to debug the parser.

## Knowledge sources
- **Read in dev_train by eye:** the closed vocabulary (7 dets, 8 adjs, 12 nouns with plurals, 4 preps, 6+6 verbs, 5 advs, "that", ", and"); the grammar (NP = det adj? noun (PP | object relative clause)?; clause = NP verb-without-object or NP verb NP, optional final adverb; at most one ", and"); sg/pl agreement; "a old" (no a/an rule). Hand tallies: adj about 0.28, main-clause modifier about 0.33 (RC share about 0.65), transitive about 0.6, adverb about 0.3, ", and" about 0.24, "the" about 2x as common as each other det (27 of 107 subjects). Hand-counted document lengths: the document continued at 131-146 chars and ended at 151-169, which gives T=148.5.
- **From dev-eval route counts (feedback):** experts for modified objects in the clause after ", and" and for nested/modified relative-clause subjects inside objects got 0 positions. From this I infer (hypothesis) a depth limit: NPs at depth >= 2 are never modified.
- **General English knowledge:** only spelling and morphology. No outside statistics were used.

## Eval history (observed)
evals/000 0.6969 base model -> 001 0.6950 (no modifiers on objects after ", and") -> 002 0.6961 (re-estimated modifier rates, worse) -> 003 0.6935 ('the' = 0.25) -> 004 0.6922 (object modifier 0.35) -> 005 0.6919 -> 006 **0.6819** (backoff mass 0.023 -> 0.0012; the biggest single gain) -> 007 0.6812 (sharper EOS: after_period 0.081 -> 0.031 bpc) -> 008 0.6793 (NP depth tracking) -> 009 0.6791 -> 010 0.6790.
- In evals/010, 69% of bits come from word starts (2.94 bpc). These are mostly unavoidable word choices, for example det about 2.0 bits and noun about 3.6 bits. In-word chars cost 0.209 bpc. The parser never entered its UNK state on dev, and no dev position needed the backoff: the worst position costs 5.3 bits.

## What failed / hypotheses / next
- Changing probabilities one at a time now moves bpc by only about 0.0005. The model sits near the entropy of my grammar.
- Hypothesis, untested: each document may have its own rate of modifiers and ", and" (my hand tallies look over-dispersed). A counter in the recurrent state could adapt to it; I estimate a gain of 0.002-0.005 bpc at most.
- A PP inside a relative-clause subject is still the worst event, at 5.3 bits. Its true rate is uncertain.
