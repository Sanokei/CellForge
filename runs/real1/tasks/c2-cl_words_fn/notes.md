# cl_words_fn v2 notes
Change from v1: no cells at all. Each of the 156 trie nodes now has a `param:` vector [36] (evidence for next char),
multiplied by its anchored-prefix gate: gate [1,1] -> op:reshape [1] -> op:mul with the [36] param (broadcast, batch-safe).
The 156 products are summed and reshaped to [36]. Detector logic (boundary row * letter probes) is unchanged from v1.
Evidence values min(0.7*ln(1+F),5) come from scripts/gen.py expanding the contract's architect-typed word list; no corpus read.
Params: 156 unique tensors x 36 = 5,616 numbers; max tensor 36 <= 40. Cells: 0 (v1's zero bias removed). Bundle nodes: 1,208.
Component test: 16/16 cases passed, ok=true (cap, role rules, batch-2 identity enforced).
