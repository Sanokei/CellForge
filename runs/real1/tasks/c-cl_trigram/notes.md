# cl_trigram notes
Design: single flat cluster. Rows 6 and 7 of ctx are sliced ([1,36] each). For each of the 40 keys in
contract data.pairs: an identity cell with W1 [1,36] one-hot on the first letter (reads row 6), another on the
second letter (row 7), op:mul of the two scalars (exact 0/1 pair gate), then an identity cell with W1 [36,1]
holding that pair's delta vector (rest_letters folded into the 26 letter entries). 40 [36] outputs are summed with
op:add and reshaped to [36]. Non-letter/empty rows give zero one-hot hits, so y=0 automatically; ids 0,1,2 stay 0.
Values come only from the contract's authoritative table (expanded by scripts/gen.py); no corpus access.
Params: 125 unique set_param (2 shared zero-bias vectors + 40*3); max tensor 36 elements. Cells: 123 defs
(each used once) in one cluster; 40*4 cell instances.
Cases: 13/13 pass.
Concerns: none known; cluster has ~250 nodes (long add chain), fine for the runtime but slightly verbose. Could
share the 26 letter selector cells across pairs, but it costs nothing at the 40-element cap.
