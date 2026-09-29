# cl_apos v2 notes
Revision of v1 for the ladder-micro/1 cap. Only change: v1's c_place cell (W1 [36,1] + b1 [36] = 72 slots) removed.
Replaced by param cl_apos.place_v [36] (one-hot on apostrophe id 8) multiplied by the [1] stem value via op:mul (broadcast).
Remaining cell: c_stem, W1 [1,24] + b1 [1] = 25 slots (stem values typed from the contract). Largest param: 36.
Logic otherwise unchanged from v1: exact slice/sum/step word matchers over ctx rows; 5 typed 36-vectors gated by case A indicators.
component-test (cap + role + batch-2 enforced): 17/17 passed, ok true.
Unique params: 24 stem + 1 bias + 5*36 output vectors + 36 place_v = 241. 1 cell def (1 instance), 421 nodes in the one cluster.
Concern: cluster is large (421 nodes) if instanced many times. Op args hold structural constants (slice indices, thresholds), no weights.
Helper: scripts/gen.py (expands typed lists only; no corpus access).
