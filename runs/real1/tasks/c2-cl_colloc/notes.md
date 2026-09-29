# c2-cl_colloc notes (revision of v1)
Same design as v1: scripts/gen.py expands contract data.phrases into prefix-shared one-hot product chains on ctx rows,
boundary-gated for j<8, grouped by target char; strengths are exactly the contract's (I chose no other numbers).
Fix: each identity cell now has at most 39 strengths, so W1(39)+b1(1) = 40 slots (cap 40). Largest param is 39 elements.
Params: 26 unique, 25 cell instances, 1777 nodes. All strengths cells share b0=[0].
