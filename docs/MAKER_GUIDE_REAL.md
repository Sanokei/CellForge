# Maker guide addendum: tinyreal/1 (`maker-guide-real/1`)

Same interface and rules as `docs/MAKER_GUIDE_LM.md` (read it and
`docs/MAKER_GUIDE.md` first) with these differences.

* Text: simple children's stories (TinyStories, MIT), normalized to lowercase
  with the 33 characters ` ,.!?'"a-z`. Vocabulary (36 tokens):
  `<pad>, <bos>, <eos>, ' ', ',', '.', '!', '?', "'", '"', a..z`. Each document
  is one whole story (150-600 chars).
* Files you may read: `data/tinyreal/dev_train.json` (2,088 stories) and
  `data/tinyreal/dev_eval.json` (165 stories, scored by `dev-eval`).
  Shapes: `ctx [8,36]`, `prev [36]`, `step [1]`, `state [S]`; outputs `logits [36]`, `state`.
* There is no closed grammar: the text is natural English with an open
  vocabulary of thousands of words. Reference levels on fresh text:
  uniform 5.17 bpc, unigram 4.21, bigram 3.19, trigram 2.30.
* **Statistics rule (unchanged, and it matters more here):** you may read
  text; you may not run code that counts, tallies or estimates anything from
  it. Anything data-derived must come from your own reading/knowledge and be
  typed by you (or produced by a helper script expanding lists/formulas you
  typed). State in notes.md which numbers came from where.
* Parameter cap: runs of profile `ladder-micro/1` cap every cell AND every
  `param:` tensor at 40 elements; total unique parameters up to 20M. A table
  bigger than 40 elements must be spread across many cells (e.g. one cell per
  output character: identity activation, W1 `[1, in]` gives one logit as a
  linear function of `in` inputs, 37 slots for a 36-way one-hot). Reuse
  definitions freely: many instances of one cell definition cost nothing.
