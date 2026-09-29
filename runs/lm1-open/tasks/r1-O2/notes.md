# r1-O2 notes: a hand-built grammar-state character LM (open/1)

**Final: dev bpc 0.6827, top1 0.8053 (evals/012). proposal.json is byte-identical to cand12. 15 dev-evals used (one of them failed validation).**

## Architecture (all tables come from scripts/build.py, which expands my own typed lists and probabilities)
- `word_events` (a step-act cell): 222 detectors, one per (word, terminator ' ' / ',' / '.'). Each is a template over the flattened ctx that needs a boundary (space or bos) before the word. W2 maps the detectors to 42 events (14 classes x 3 terminators), plus a "none" event.
- `grammar` bundle: three one-hot recurrent registers, state dim 130.
  - phase (37): CS, DET/ADJ x NP-type {MS, OB, PPN, RS} x det number {s, p, u}, NOUN x type/DONE, PREP, THAT, VT, VI, ADV, COMMA, PERIOD.
  - agreement stack (91 configs, depth up to 4): pending main/relative verbs and their numbers. Ops: pushM, pushR, set_s/p, pop, clear.
  - clause index (2): whether ", and" is still allowed.
  - Each update is a lookup on an outer product (register x event); the phase update also takes the stack's top-kind.
- `speller` bundle: the stack is reduced to a 13-way summary; (phase x summary x ci) maps to one of 153 distribution classes. A prefix detector bank finds which of 212 prefixes the partial word is. Logits are a lookup on the outer product (class x prefix), split into two halves so each param stays under the pattern element limit. Each entry is log P(char) worked out from my word-level model, mixed 0.3% with any-word and 0.05% with uniform.
- `eos_timer`: after '.', the eos logit gets 0.5*(step-147.5).

## Knowledge used
- **From reading dev_train (by eye; no counting code):**
  - The closed vocabulary: 7 determiners, 8 adjectives, 12 nouns in singular/plural, 6 intransitive and 6 transitive verbs, 4 prepositions, 5 adverbs, "that", ", and".
  - The sentence structure: NP V [NP] [adv], with object-gap relative clauses; each NP has at most one modifier, and modifiers nest.
  - Documents stop once the length reaches somewhere in (142, 152]. I counted a dozen docs by hand.
  - Rates I tallied by eye over about 3 batches of text: 'the' is about 2x the other determiners; adjective about 0.3; intransitive share about 0.5; adverb about 0.3; ", and" about 0.22; modifier rates by NP type (main subject highest, relative-clause subject lowest).
- **From general English knowledge:** subject-verb agreement (every/this/a take singular, these/many/some take plural, 'the' takes either); the agreement stack for nested relative clauses; punctuation conventions.

## Observed (dev)
- 000: first full model 0.6939.
- 001: "one modifier per NP" + typed rates 0.6913.
- 003: better determiner weights / vi / adjective rate 0.6880.
- 004: RS prep 0.06 + smaller epsilons 0.6842.
- 006: eos ramp re-centred 0.6835 (after_period 0.077 -> 0.043).
- 008 / 010 / 012: adjective 0.34, adverb 0.27, main-subject modifiers (0.14, 0.17) -> 0.6827.
- Changes that did not help: vi 0.57 (007), adverb 0.34 (009), more main-subject modifiers (011), nesting-depth decay of modifiers (013, 0.6828), eos slope 1.0 (014, same total).
- The bits split roughly 69% word-start and 24% in-word. The worst positions are grammatical but rare events (a preposition after a relative-clause subject), about 5.6 bits each.

## Hypotheses / next
- My own grammar's self-entropy, sampled with scripts/selfent.py, is about 0.68 bpc, so the model looks close to the floor that this grammar allows. What is left is probably non-uniform word choice and small rate errors.
- I did not measure the lemma frequencies for nouns, verbs and adjectives. Per-lemma weights read off a larger batch would be the next thing to try. I would also pin down the exact stopping length.
