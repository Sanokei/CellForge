# c-cl_colloc notes
Design: scripts/gen.py expands data.phrases from contract.json (typed by architect; script reads nothing else, no corpus).
For each phrase position j>=3, the exact match of the preceding m=min(j,8) chars is a product of one-hot ctx
entries (op:slice of row/char, op:mul chain; chains share prefixes from row 7 leftwards). For j<8 the fire signal
is multiplied by a boundary indicator of row 7-j (1 - letters_sum - apostrophe). Fire signals are grouped by target
char; one identity cell per <=40 fire signals holds the strengths (W1 [1,n]); rules with identical match+target are
merged (strengths summed by the script). 36 scalars are concatenated (zero for untouched ids incl. 0,1,2) and reshaped to [36].
Numbers: strengths are exactly the contract's; I chose no other values (shared b0=[0]).
Size: 26 params (364 elements incl. bias), 25 cell instances, ~1780 nodes, all tensors <= 40 elements.
Component test: 18/18 passed.
Concerns: node count (~1.8k ops) makes this a heavy-ish circuit at runtime per step; exact-boundary semantic follows spec (apostrophe not a boundary). No unsatisfiable cases.
