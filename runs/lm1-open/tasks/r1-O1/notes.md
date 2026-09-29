# r1-O1 notes: a hand-written grammar compiled into a character LM

**Final result:** proposal.json (= proposal_v3.json) scores dev bpc **0.6806**, top1 0.8093, 0 failures (evals/002). I used 3 of the 25 dev-evals.

## What I built
- **Grammar:** a probabilistic grammar I wrote by hand in scripts/grammar.py. Clause = NP VP. An NP is Det Adj? N plus an optional modifier: a prepositional phrase (PP) or an object-gap relative clause ("that NP Vt"). A VP is Vi Adv? or Vt NP Adv?. A sentence may take ", and" plus a second clause. Nouns and verbs agree in number (sg/pl). Relative clauses nest like a stack.
- **Automaton:** scripts/build.py expands that grammar into a deterministic automaton. Each state is a belief over weighted parser stacks: 2292 states after minimisation. A token is a word plus its separator (177 tokens), which makes the token set prefix-free.
- **State vector (2480 values):** a one-hot grammar state g, a mask of tokens still "alive" (consistent with the characters so far), and a one-hot position inside the current token.
- **Step 1, consume the previous character:** narrow the alive mask using the spelling table and advance the position. When a token completes, look up the next g by feeding the pair (g, token class) into a cell. That cell outputs the next state as a 12-bit binary code, and a step-activation cell decodes it back to one-hot.
- **Step 2, emit:** for each next character c, v[c] = sum of P(token|g) over alive tokens whose next character is c. The logits are log(v), computed by a 52-knot relu cell that approximates log piecewise-linearly (floor 1e-6).
- **Recovery:** if no token stays alive (a word the grammar doesn't know), the model outputs near-uniform logits and resets at the next separator.
- **End of document:** the odds of <eos> after a period are multiplied by exp(clip(step-151, ±6)).
- **Size:** 1.31M unique parameters. Nothing is fitted: every value comes from the numbers below, expanded by the scripts.

## Numbers I chose
- Number: sg/pl 0.5 each.
- Determiners: sg {every, this, a, the}, pl {these, many, some, the}, uniform within each.
- P(adj) = 0.35.
- PP by NP depth: 0.14 / 0.08 / 0.04 / 0.02 / 0.
- Relative clause by NP depth: 0.17 / 0.10 / 0.05 / 0.025 / 0.
- P(Vi) = 0.45, P(adv) = 0.27, P(", and") = 0.21.
- Word choice is uniform within each class.
- <eos> gate centred at 151 characters.

## Where the knowledge came from
- **From reading dev text by eye:** the whole vocabulary and the word classes; the grammar rules, including the number agreement and the det-to-number mapping; the doc-length rule (by hand-counting a few docs I found one that continued at 148 chars and one that stopped at 154); all the rates above. The rates come from tallies I made by eye over 35 documents (docs 600-614 and 1200-1219), for example 26/123 sentences with "and" and 67/150 main verbs intransitive.
- **From general knowledge:** English morphology (plurals such as children and mice, the verb -s) and the idea of treating the text as a PCFG with subject-verb agreement.
- No code counted anything in the text.

## Observed results
| eval | model | dev bpc |
|---|---|---|
| evals/000 | v1: first grammar, constant P(eos)=0.25 | 0.7022 |
| evals/001 | v2: depth-dependent modifiers + length-gated <eos> | 0.6824 |
| evals/002 | v3: rates re-estimated from 35 docs | 0.6806 |

- The length gate cut the after-period cost from 0.815 to 0.033 bpc (evals 000 → 001).
- In evals/002, word starts carry 69% of all bits (2.96 bpc) and in-word characters 0.21 bpc.
- The worst positions are rare prepositions after a relative-clause subject, costing about 5.5 bits each. Those are low-probability events under the grammar, not parse failures.

## Hypotheses (untested) and next steps
- On samples drawn from itself, the model's own entropy is 0.694 bpc. That is higher than the dev cross-entropy, so the model may be slightly too broad somewhere. Suspects: modifier rates that differ between subjects and objects, or adjective rates that vary from document to document (the adjective counts looked over-dispersed in my tallies).
- A cache for repeated nouns: my tally found 19 repeats where about 16 would be expected by chance, so the gain would be small.
