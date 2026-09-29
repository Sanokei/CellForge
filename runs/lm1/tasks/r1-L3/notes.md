# r1-L3 notes: hand-written grammar LM as a hierarchical MoE (ladder/1)
## What I built (proposal.json = scratch/p05.json, made by scripts/build.py from scripts/grammar.py)
- Trunk cluster `cl_lex`: 666 step units over the flattened ctx. Unit (w,k) fires when the current partial
  word equals the first k letters of lexicon word w, anchored on a preceding space/<bos>. Fixed tables turn
  this into C[w,:] (the next letter if the word is w) and uend[w] (the word is complete).
- Trunk cluster `cl_detect`: 74 step units that fire when word w has just finished followed by a space, mapped
  to 14 word classes, plus PERIOD/COMMA/BOS taken from prev and NONE = 1 - the rest.
- Trunk cluster `cl_trans` (recurrent state, 57 dims): a one-hot phase register (19: D/A/N with NP role and
  determiner number, PN, RV, V, ADV, AND), a one-hot stack register (36: number of the pending main verb and
  up to 2 pending relative-clause verb numbers), and a 2-way clause-2 flag. Each register is updated by a
  lookup table on outer products (register x class x small summary of the other register). The tables come
  from my rule function grammar.trans. I checked that the factorization matches the joint machine (0 conflicts).
- Router (`cl_route`): scores come from the phase register, plus +10 on prev in {'.', ','}. 4 experts, k=1:
  ex_np (after det/adj or where a det is expected), ex_post (after a noun or RC verb: verbs/preps/that/clause end),
  ex_tail (after an intransitive verb/adverb, or "and" expected), ex_punct (after '.' or ',').
- Lexical experts: P[w] = table(phase x stack key) + floor 2e-5, and terminator masses T[w,{' ',',','.'}] set from
  the successor state's clause-end probability. Char mass = P@C + sum_w uend[w] T[w], with the comma mass moved
  to '.' when clause-2=1. It goes through a shared piecewise-linear log cell (49 knots, ratio sqrt2, char floor 1e-6).
- ex_punct: after ',' -> ' '. After '.', the eos logit is a clipped ramp 6*(step-149.5), so docs end at the first
  period at or after character 150.
## Knowledge sources
- From reading dev_train by eye: the whole lexicon (7 dets, 8 adjs, 12 nouns sg/pl, 6 intransitive and 6 transitive
  verbs, 4 preps, that, and, 5 adverbs), the grammar (NP = det adj? noun (PP|object-RC)*, clause = NP VP adv?,
  sentence = clause (", and" clause)? "."), "some"/"these"/"many" always plural, "a old" is used (no "an").
  I estimated rates by hand-tallying small samples (docs ~400-425 and 1200-1209): adj ~0.36, "the" ~0.22,
  main-subject noun -> verb/prep/that ~.72/.14/.14, object noun -> end/adv/prep/that ~.58/.17/.12/.13, ", and" ~0.2
  per sentence. I found the doc-length rule by counting characters by hand in 4 docs: 149 -> continued, 150 -> stopped.
- General English knowledge: agreement (singular subject -> verb+s), relative-clause verbs agree with the RC subject,
  and a stack for nested RCs. No code read or counted the corpus.
## Dev-eval results (observed; 5 of 25 used)
- evals/000 0.7125 bpc / top1 .798: first build (word_start 3.0 bits = 67% of bits, fixed p(eos|'.')=.25 cost 0.815 bits).
- evals/001 0.6849: eos step ramp (after_period 0.815 -> 0.001 bpc), word floor 5e-4 -> 2e-5, denser log knots, rates recalibrated.
- evals/002 0.6836: clause-2 flag (no second ", and"). evals/003 0.6825: adj .30->.36, the .18->.22. evals/004 0.6819 bpc / top1 .806 (final): main-subject and object-noun follow-up rates recalibrated.
- Final: word_start 2.966 bpc (69% of bits), in_word 0.209, after_period_space 2.03. The worst positions are all
  grammatical-but-rare choices (a prep after an RC-subject noun, 5.5 bits). None are grammar violations.
- Route counts (final): ex_np 35341, ex_post 15647, ex_tail 2072, ex_punct 1426. Samples: 100% in-lexicon words.
## Hypotheses / next
- The remaining bits look like the generator's own choice entropy (which noun, which det). My estimate is that further
  rate calibration is worth under 0.003 bpc. Unverified: the rate of noun repeats within a sentence might be above 1/12,
  and modifier rates might depend on nesting depth.
