# Maker guide addendum: tinylm/1 and the cell → cluster → expert ladder (`maker-guide-lm/1`)

Read `docs/MAKER_GUIDE.md` first; all its hard rules apply. This addendum
replaces its task-specific parts (image input, OCR decoding).

## Task

Character-level language modeling. The model reads text one character at a
time and outputs logits for the next character. Text is English-like
sentences (lowercase letters, space, comma, period). Author-visible text:

* `data/tinylm/dev_train.json` — 2,000 documents (~356k chars) you may read.
* `data/tinylm/dev_eval.json` — documents that `dev-eval` scores.

Vocabulary (index order, 32 tokens): `<pad>, <bos>, <eos>, ' ', ',', '.',
a, b, …, z`. Every document is `<bos>` + text + `<eos>`; the model must
predict every character and the final `<eos>`.

### Additional hard rule for text

You may **read** the text files yourself and form your own estimates of
which characters/words follow which. You may **not** run code (python, grep
-c, wc, sort|uniq, etc.) that counts or otherwise computes statistics from
the text, and you may not build tables from text programmatically. Word
lists and probability tables you type yourself (possibly via a helper script
that expands your own typed lists/formulas into arrays) are fine.

## Step-bundle interface (no encode entry)

Entry `step` bundle inputs:
* `ctx` `[8, 32]` — one-hot rows of the last 8 tokens, oldest first, most
  recent in row 7. Rows before the document start are all zeros.
* `prev` `[32]` — one-hot of the most recent token (= `ctx[7]`).
* `step` `[1]` — position index.
* `state` `[S]` — your recurrent state (zeros at the start; `set_state_dim`).

Outputs: `logits` `[32]`, `state` `[S]`. Logits for `<pad>`/`<bos>` should
be very negative. Scoring is bits per character (bpc) under teacher forcing:
lower is better. Uniform = 5.0 bpc. `dev-eval` also samples free-running
text from your model and reports it.

## Ladder profile (runs with `profile: ladder/1`)

Validator-enforced structure:

| role | kind | may contain |
|---|---|---|
| cell | `"kind": "cell"` | – (primitive) |
| cluster | bundle, `"role": "cluster"` | ops, params, **cells** |
| expert | bundle, `"role": "expert"` | ops, params, **clusters** |
| router | bundle, `"role": "router"` | ops, params, clusters; must output `scores` |
| root | the `step` entry bundle | ops, params, clusters (shared trunk), router instances, dispatch |

The root must contain at least one `dispatch` node selecting among ≥2
experts, with `scores` taken from a router instance output (e.g.
`"scores": "r.scores"`). Raw cells may appear only inside clusters; experts
may not contain experts. Expert outputs are merged by the dispatch node;
every branch must produce the same output names/shapes.

`dispatch` args: `{"branches": [...], "k": 1}`. `k=1`: each position runs
only its top-scoring expert (lazy). `k>1`: top-k experts run and outputs are
mixed with softmax weights over their scores. `dev-eval` reports
`route_counts` (how many positions each expert served).

In `open/1` runs there is no role requirement (any nesting, routing optional).

## Example skeleton (ladder)

```json
{"op": "put_def", "name": "c_embed", "def": {"kind": "cell", "act": "identity",
   "params": {"W1": "embed.W", "b1": "embed.b"}}},
{"op": "put_def", "name": "cl_embed", "def": {"kind": "bundle", "role": "cluster",
   "inputs": ["x"], "nodes": [{"id": "c", "use": "c_embed", "in": {"x": "$x"}}],
   "outputs": {"y": "c"}}},
{"op": "put_def", "name": "router", "def": {"kind": "bundle", "role": "router",
   "inputs": ["prev"], "nodes": [{"id": "s", "use": "cl_route", "in": {"x": "$prev"}}],
   "outputs": {"scores": "s"}}},
{"op": "put_def", "name": "ex_wordstart", "def": {"kind": "bundle", "role": "expert",
   "inputs": ["ctx", "prev"], "nodes": [ ... clusters ... ], "outputs": {"logits": "..."}}},
{"op": "put_def", "name": "step_root", "def": {"kind": "bundle",
   "inputs": ["ctx", "prev", "step", "state"], "nodes": [
     {"id": "r", "use": "router", "in": {"prev": "$prev"}},
     {"id": "d", "use": "dispatch", "in": {"scores": "r.scores", "ctx": "$ctx", "prev": "$prev"},
      "args": {"branches": ["ex_wordstart", "ex_inword"], "k": 1}}],
   "outputs": {"logits": "d.logits", "state": "$state"}}}
```

Useful op idioms: `op:slice` rows of `ctx` to get older tokens (`axis 0,
start 6, stop 7` is the token before `prev`); `op:mul` of a transposed
`[32,1]` slice with a `[1,32]` slice gives a pair one-hot; `op:reshape` to
flatten. A cell with `act: identity` and only `W1/b1` is a lookup table when
fed a one-hot.

## Deliverables

Same as the base guide (`proposal.json`, `notes.md`, `observations.json`,
`scripts/`). In `notes.md`, also state which knowledge you used: what you read
in dev text vs. what you assumed from general knowledge of English.
