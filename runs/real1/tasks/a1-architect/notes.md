# a1-architect Phase C notes
Final: dev bpc 2.9825 (eval 8 of 30 used; proposal.json = p_7.json built by scripts/integrate.py + scripts/cfg7.json). Bigram-only baseline 3.443.
Knowledge: design/weights typed by me from a by-eye read of ~20 dev stories + general English; no corpus statistics code. Weights are hand-picked scalars.

## Integration
scripts/integrate.py concatenates fragment ops verbatim, adds 7 expert bundles, router, step_root (state q + name cache, dim 225), dispatch k=1.
Runs valid under ladder-micro/1. ~2 min per dev-eval (3.6M node execs; dominated by cl_words_open ~2.6k ops and cl_wordstart ~480 ops, run per position).

## Fragments NOT usable (need revised contract versions; I did not edit them)
* Cell cap counts W1+b1 slots (profiles.py): cells with W1 [36,1] + b1 [36] = 72 > 40 -> validator rejects:
  cl_words_fn (all 156 cells), cl_trigram (41 cells), cl_endpunct (c_q, 72), cl_apos (c_place, 72), cl_colloc (c_t3_0, 41).
  Fix: keep cell slots incl. bias <= 40 (e.g. W1 [1,k] with b1 [1], or split 36-vector outputs into two cells / use param: nodes + mul ops instead of cells).
  My contract said "each cell/param holds at most 40 numbers"; it should have said W1+b1 (+W2+b2) <= 40. My fault as much as theirs.
* cl_name_write: op:mul in node Bn fails for batch > 1 (shapes (4,6,36) vs (4,1)); dev-eval uses a batch. Needs explicit [1,1] reshape of the gate. Because of this the name cache is disabled (state passes q only; nm slice passed through), and cl_name_read is not used.
Used and valid: cl_router, cl_qstate, cl_bigram, cl_words_open, cl_suffix, cl_wordstart, cl_quote.

## Eval history (dev bpc)
1 bigram only 3.443 (in_word 3.41, word_start 4.13). Worst: '!'/'?' after in-quote words at 12.6 bits (cl_endpunct unavailable).
2 +open .6/.7(long, bigram .7 there), suffix 1, wordstart .8, quote 1: 3.068
3 open .35/.4: 3.182 (tries help; weaker is worse)
4 open 1.0: 3.023
5 wordstart 1.5: 3.001
6 wordstart 2.5, suffix 1.6: 2.983 (suffix effect ~0.004)
7 wordstart 4, open 1.4: 3.041 (both overshoot)
8 wordstart 3.2 (final): 2.9825.
Contribution: words_open about -0.42 bpc (in_word 3.41 -> 2.91); wordstart about -0.39 on word_start (4.13 -> 3.75); after_period_space 3.65 -> 3.235; suffix ~0.00 (in_word 2.909 -> 2.905 at 1.6); quote small (after_comma 0.30 -> 0.33, slightly worse; after_period 0.65 -> 0.62). Nothing hurt materially except over-weighting.

## Remaining gaps (hypotheses, unmeasured)
Missing circuits (trigram, words_fn, endpunct, apos, colloc, name cache) are the main gap; words_fn (the/and/a/to/was...) covers the most tokens, so I expect it plus endpunct to give the largest gain. Unlisted words are penalised by tries (e.g. 'mr' 13.9 bits); bigram backbone (3.44) is weaker than a counted bigram (3.19). Predicted 2.75 in Phase A is not reached; with all circuits fixed I would still expect ~2.7-2.8.
