# cl_words_fn notes
Design: one cluster `cl_words_fn`. Ops on ctx rows give a boundary indicator per row
(B = 1 - sum(letter cols) - apostrophe col; all-zero rows count as boundary) and exact letter probes
(slice row, slice column). For each trie node (prefix of length 1..7 of a listed word; 156 nodes)
the activation = B[row 7-j] * product of the j letter probes at rows 8-j..7, so it is 1 only for the
anchored word-so-far. Each node drives one cell (identity, W1 [36,1], shared zero b1 [36]) that
emits its 36-vector of evidence; the 156 outputs are summed and reshaped to [36].
Evidence values min(0.7*ln(1+F),5) with F = sum of weights of longer words through w+c, and
the space entry from the word's own weight, are computed by scripts/gen.py from contract data.words
(architect-typed list; no corpus is read).
Params: 157 unique tensors (156 x 36 elements + one 36-zero bias), 5,652 numbers. Cells: 156 instances
of 156 definitions (no cell shares a param except the zero bias). Max tensor size 36 <= 40.
Component test: 16/16 cases passed (tol 0.02).
Concerns: op graph is large (about 1,500 nodes) but linear; runtime cost per step is small.
Words of length > 7 letters, unanchored runs, apostrophe-preceded runs yield exact zeros by construction
(no detector matches). Ids 0,1,2 always 0.
