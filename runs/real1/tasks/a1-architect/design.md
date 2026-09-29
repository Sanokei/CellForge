# a1-architect design (delegation/1, ladder-micro/1, tinyreal/1) - Phase A

Knowledge used: I read ~20 stories from `data/tinyreal/dev_train.json` by eye (recurring shapes: "once upon a time, there was a ...",
"one day, a little girl named lily ...", repeated protagonist names, `said, "..."`, `didn't`, `-ing/-ed/-ly` words, many short function words),
plus general knowledge of English. All numbers in contracts (word weights, evidence values) were typed by me from that; nothing was counted
by code. `scripts/lists.py`, `scripts/lists2.py` hold my typed tables; `scripts/gen_contracts.py` only evaluates the formulas printed in each
contract's spec on those tables to produce expected vectors for hand-picked cases.

## Model in one paragraph

Log-linear "product of experts" character LM. Every circuit outputs a [36] vector of evidence (nats) added to the logits: `cl_bigram` gives
ln p(next|prev) (the backbone), everything else gives corrections that are 0 when the circuit has no opinion. The window is 8 tokens, so the
model effectively sees "the word so far" (<=7 letters, anchored by a left boundary), the previous word (<=6 letters), and exact 8-gram phrase
contexts. Recurrent state carries an inside-quote flag and a cache of the protagonist's name (letters after `named `). A hard router picks one of
7 experts by class of `prev` (word start, in-word short/long, after punctuation, after quote, after apostrophe, bos); the expert is a weighted sum
of its circuits. Weights are architect scalars (`op:scale`), tuned in Phase C.

## Circuits (14) and what they contribute

| circuit | role | used in |
|---|---|---|
| cl_router | 7 scores (exact class of prev + word length) | router |
| cl_qstate | q' = toggle on `"`, reset on bos | root trunk |
| cl_name_write | state update for name cache (`named ` trigger, up to 6 letters) | root trunk |
| cl_bigram | ln p(c\|prev) backbone | all experts |
| cl_trigram | ~40 letter-pair deltas (th, ng, ed, qu, ...) | in-word |
| cl_words_fn | trie for ~80 function words; y=min(0.7 ln(1+F),5) | in-word |
| cl_words_open | trie for ~170 content words/names | in-word |
| cl_suffix | word-end evidence from -ing/-ed/-ly/-ful/-s/-er/-y/-est | in-word long |
| cl_wordstart | first-letter evidence given previous word / sentence / comma | wordstart |
| cl_colloc | 8-gram phrase evidence (once upon a time, wanted to, ...) | wordstart, in-word |
| cl_endpunct | `,` `.` `!` `?` after words, quote-aware | in-word |
| cl_quote | open/close quote, eos suppression inside quotes | wordstart, punct, quote |
| cl_apos | contraction stems (didn -> ' -> t), possessive 's | in-word, apos |
| cl_name_read | boost letters of cached name at word start / mid-word / word end | wordstart, in-word, quote |

Circuits are independent: each takes only `ctx` (and `q` or `nm`) and is fully specified by its contract plus embedded `data` tables.

## Wiring I will do in Phase C

Root `step_root(ctx, prev, step, state)`:
1. `q0 = slice(state, axis0, 0..1)`, `nm0 = slice(state, 1..225)`.
2. `qn = cl_qstate(prev, q0)`, `nmn = cl_name_write(ctx, nm0)` (shared trunk clusters; `ctx[7] == prev`, so experts see flags already updated for prev).
3. `r = router(ctx)` -> `cl_router` scores [7]; `d = dispatch(scores=r.scores, ctx, qn, nmn)` with branches in the order
   `[ex_bos, ex_wordstart, ex_inword_short, ex_inword_long, ex_punct, ex_quote, ex_apos]`, k=1.
4. outputs: `logits = d.logits`, `state = concat(qn, nmn)` (state_dim 225).

Each expert (`role: expert`, inputs `ctx, q, nm`) = clusters wrapping cl_* instances, then
`logits = sum_i w_i * y_i + mask` where `mask` is a 36-vector param (arch-owned, <=40 elements): pad, bos = -20, everything else 0.
Bigram weight 1.0 (0.6 in `ex_inword_long`); initial evidence weights 0.5-0.8 for tries (they add up to 5 nats and would over-sharpen if 1.0),
1.0 for cl_quote / cl_apos / cl_suffix / cl_endpunct / cl_name_read / cl_colloc, 0.7 for cl_wordstart, 0.8 for cl_trigram. Wrapper clusters (`cl_<x>_w`) are thin
`role: cluster` bundles (a cluster may contain ops/params only if allowed; if the validator wants cells inside clusters I will add one 1-element identity cell as the
weight instead of a raw scale op). I never edit circuit internals.

Expected router usage: word-start ~17% of positions, in-word ~70% (split ~45/25), punctuation ~7%, quote ~2%, apos ~1%, bos once per story.

## Phase C plan (30 dev-evals)

Order: (1) bigram-only expert set to validate wiring; (2) add tries; (3) add wordstart/colloc/endpunct; (4) quote/apos/name/suffix; (5) tune the ~10 architect weights
one or two at a time, using `route_counts`, per-expert loss if reported, and the sampled text to find what fails. Circuit revisions are requested through the
orchestrator with a new contract version only if a component test shows a contract is wrong (for example a table entry that hurts).

## Predicted bpc (truthful)

Reference: uniform 5.17, unigram 4.21, bigram 3.19, trigram 2.30. A hand-written log-linear model has no data fit, so its calibration is weaker than a counted trigram
despite richer features. Reasoning: bigram backbone ~3.2; letter-pair deltas plus suffix rules bring roughly -0.2; the two tries cover the ~250 most frequent word types
(a large majority of tokens in these stories) and make in-word letters cheap, maybe -0.4; word-start conditioning (previous word, phrases, name cache) another -0.15;
punctuation/quote handling -0.1. Uncorrected double counting between bigram and tries, and unlisted words being penalised relative to listed words, will eat some of it.

**Predicted final dev bpc about 2.75 (plausible range 2.45 to 3.2).** I judge P(beating the 2.30 trigram) at under 10%; the design's real upside is a clear win over the 3.19 bigram and
a demonstration that delegation can scale hand-authoring. If Phase C weights tuning works well the low end (about 2.5) is reachable.

## Known risks

* Additive evidence is not normalised: tuning weights (Phase C) is essential; contracts fix only the evidence tables.
* Trie circuits are large (roughly 3-5k ops, hundreds of 36-element cells); circuit-makers must generate them with helper scripts from `data.words` (allowed) - if a maker judges this too large for one sitting, I will split `cl_words_open` in two by word initial letter.
* `cl_name_write`/`cl_name_read` are the most intricate circuits; if unreliable I will drop them (state stays 1-dim) at a cost of about 0.05 bpc.
* Ladder validator: raw ops in root/experts are allowed, cells only inside clusters; circuits are clusters, so all cells sit in clusters.
