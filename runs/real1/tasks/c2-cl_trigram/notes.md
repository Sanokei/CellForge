# cl_trigram v2 notes
Revision for the ladder-micro/1 cap (v1 had 72-slot cells).
Changed: no cells for 36-wide vectors. 26 shared letter-selector cells (W1 [1,36] one-hot, b1 [1]; 37 slots each)
are applied to row 6 and row 7 (both sliced+reshaped to [36]). Per pair: gate = op:mul(sel_a(r6), sel_b(r7)),
reshaped to [1], then op:mul with a param:cl_trigram.d<k> ([36] delta vector, rest_letters folded in). 40 products summed.
Values come only from contract data.pairs / rest_letters (scripts/gen.py); no corpus access.
Params: 68 unique (1 shared zero bias, 26 selectors, 41 pair deltas); max tensor 36 elements.
Cells: 26 defs (one cluster cl_trigram). Nodes in cluster: 290.
component-test (enforced cap, roles, batch-2 identity): 13/13 pass.
