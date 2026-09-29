# cl_wordstart notes
Design: parameter-light gated lookup, no cells. For each of the 58 contract keys (@sent, @comma, 56 words) one param
`cl_wordstart.t_<key>` [36] (values copied from contract data.table via scripts/gen.py) multiplied by a 0/1 gate.
Gate = (row7==space) * product of exact one-hot letter slices of rows 6-k+1..6 * boundary(row 6-k) (if 6-k>=0).
Boundary(r) = 1 - sum(ctx[r, 8:36]) (letters + apostrophe are non-boundary; all-zero rows and bos/space/punct/quote are).
@sent gate = sum of ctx[6, '.','!','?']; @comma gate = ctx[6,','].  Terms concatenated and summed -> y [36].
k=6 words anchor at row 0 (checked); 7+ letter runs have no key, so zeros.
Params: 58 tensors x 36 = 2088 elements (all <=40). Cells: 0 (ops/params only inside one cluster, 478 nodes).
Component test: 18/18 passed.
Concerns: no cells used (validator accepted); a key that is a suffix of a longer word is correctly rejected by the
anchor test. Values are exactly the contract tables, no independent knowledge used; no corpus code run.
