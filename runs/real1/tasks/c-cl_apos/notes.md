# cl_apos notes
Design: pure exact logic over ctx (no learned fuzziness). ctx is flattened to [288]; per-row letter mask L_r, apostrophe A_r and
boundary B_r = 1-L_r-A_r come from slice/sum/neg/bias ops. Each word test (stems, and the case-A word lists) concatenates the k
letter slices plus the anchoring boundary slice, sums them, and applies step(sum-(k+0.5)); this is exact for one-hot inputs.
Case B: 24 stem indicators -> concat [24] -> cell c_stem (W1 [1,24] = the contract stem values) -> cell c_place (W1 [36,1], one-hot
on the apostrophe id) -> y. Case A (gated by A_7): exact-run-length indicators E2..E6 (anchored), word-group sums for 'i',
{it,that,...,how}, {you,we,they}; n-ending len>=3 = [row6=='n'] * (E3+..+E6); possessive = (E2+..+E6) - groups - n-ending
(groups are disjoint and none ends in n, so this is exact). Each group gates a 36-vector param (values typed from the contract).
Counts: 277 unique params (5x36 output vectors, 24 stem values + 1 bias, 36 place col, 36 place bias); 2 cell definitions/2 instances;
~420 ops/nodes inside the one cluster. All tensors/cells <= 36 elements.
Cases passed: 17/17 (component-test).
Concerns: op args carry structural constants (slice indices, step thresholds) which resource reports count; not weights.
Cluster holds ~420 nodes, which may be heavy for the parent if instanced many times. Helper: scripts/gen.py (only expands typed lists).
