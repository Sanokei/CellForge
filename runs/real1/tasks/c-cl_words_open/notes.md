# cl_words_open notes
Design: parameter-free matching with ops on ctx, one 36-vector param per trie node.
- Boundary indicator B_r (r=0..6) = 1 - (ctx[r,apostrophe] + sum ctx[r,10:36]); exact 0/1, zero rows count as boundary.
- Letter indicators i_r_c = slice of ctx row r, column c (shared across nodes).
- For each prefix w (length L, 1..7) of a listed word: match = B_{7-L} * prod_i ctx[8-L+i, w_i]; chains share partial products (trie per start row).
- Node output = match * V_w, V_w [36] holds min(0.7*ln(1+F),5) per next letter, and y[' '] if w is itself a word (len<=7).
- Node vectors are summed with an add chain, reshaped to [36]. Only one prefix matches at a time, so it is exact.
- Words from contract data.words (all ~170 included, none dropped); 'together'/'beautiful' contribute only via prefixes <=7.
Helper: scripts/gen.py expands contract data.words into proposal.json (does not read the corpus; no search loops).
Size: 420 prefix nodes = 420 unique params (36 elements each, cap 40); 0 cells; 2576 nodes inside the single cluster (ops + params only).
component-test: 16/16 passed.
Concerns: cluster holds no cells (contract/validator accepted it); large node count (~2.6k) may slow the graph slightly.
