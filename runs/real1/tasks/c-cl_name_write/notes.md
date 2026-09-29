# c-cl_name_write notes
Design: one cluster `cl_name_write`, zero cells, parameter-free exact ops except one 7-element constant param `cl_name_write.e0` = [1,0,0,0,0,0,0] (the L=0 one-hot).
- Trigger: slice ctx rows 2..7, pick tokens n,a,m,e,d,space (23,10,22,14,13,3) per row, multiply the six scalars -> trig (0/1).
- Letter flag of row 7 = sum of ctx[7,10:36].
- w = cap * letter7 * (1-trig) is the write flag. Buffer: B' = (1-trig) * (B + outer(w*L[0:6], ctx[7])). Rows past L are zero by contract, so adding is equivalent to set; L==6 gives no write because L[0:6] is all zero.
- L': s = w*(1-L[6]); L' = (1-s)*L + s*shift(L,1); then (1-trig)*L' + trig*e0.
- cap' = max(trig, cap*letter7): trigger -> 1, letter while capturing -> 1, non-letter -> 0, idle -> 0.
- Output = concat(B'[216], cap'[1], L'[7]) = 224.
Params: 7 unique (one tensor). Cells: 0. Ops: about 75 nodes (slice/mul/add/etc.).
Result: component-test 11/11 passed.
Concerns: assumes cap and L are exactly 0/1 as the contract guarantees; with soft state values the outputs would be blended rather than snapped. The ops use small constants in args (scale -1, bias 1, slice indices), which are structural.
No corpus data was used. Helper: scripts/build.py (expands my typed wiring into proposal.json).
