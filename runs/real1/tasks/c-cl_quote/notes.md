# cl_quote notes
Design: no cells. One cluster of ops + 5 param vectors (each 36 elements): w3 (punct weights for R3), u9 (unit vector for the double quote), evec (eos -6), v1 (R1 qopen table + punct penalties), v2 (R2 closed-quote vector).
Row features (ctx[r][t]) come from slice/slice/reshape (exact, param-free). q gate = step(q-0.5).
E, R1, R2 = gate * fixed vector. R3, R4, R5, R6 all only affect y['"'], so their scalar amounts are summed into one scalar and multiplied by u9.
R3 amount = q1 * (w3 . row7); R4 = q0*sp7*(row6==',') * (0.4 + 2.1*[row5=d] + 0.6*[row5=m]); R5 = 0.8*q0*sp7*(row6 in .!?); R6 = -2.5*q1*sp7.
Params: 5 unique tensors, 180 numbers. Cells: 0. Only the bundle contains ops/params.
Component test: 16/16 passed.
Concerns: none; contract data (qopen) typed from contract into the script. Numbers all come from the contract.
