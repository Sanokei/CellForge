# cl_suffix notes
Design: one cluster, no cells, no matmul weights. Ops only, plus 8 typed delta params.
- For rows 3..7 of ctx: slice row, slice cols 10:36, sum -> letter indicator L_r (exact 0/1).
- Single-token indicators are column slices of a row. Each suffix gate is a product of token
  indicators and L_r (e.g. ING = g7*n6*i5*L4; FUL also needs L3, L4 for n>=5).
- S gate: L6 minus (s,u,i,a at row 6). Y gate: L6 minus (a,e,i,o,u,y,l at row 6), which also excludes 'ly'.
- Each gate is reshaped to [1] and multiplied by its delta vector (param d_<rule>, 36 elements),
  then all 8 terms are added, so overlaps sum as the contract says.
- Delta values are copied from the contract's rule text. Nothing was read or counted from the corpus.
Params: 8 tensors x 36 = 288 unique values (max tensor 36 <= 40). Cells: 0. Bundle nodes: see proposal (cluster only).
Component test: 19/19 passed.
Concerns: none known. The gates rely on a run of letters being contiguous, so the n>=k conditions
reduce to L_r products for the rows just before the suffix. n is capped at 8 by the input window.
