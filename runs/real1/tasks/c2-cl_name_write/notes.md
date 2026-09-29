# c2-cl_name_write (v2, revision of c-cl_name_write)
Fix: v1 multiplied Bsum (B,6,36) by ntrig (B,1), which only broadcast at B=1. Now ntrig is reshaped to per-example [1,1] before the multiply.
Other nodes re-checked: [1]-shaped scalars (trig, w, s, cap flags) only meet [1]/[6]/[7]-vectors (right-aligned, batch-safe);
Lwc [6,1] * last [1,36] has matching rank; param e0 is reshaped to [7] and broadcasts over the batch.
Design unchanged: one cluster, zero cells, exact slice/mul/add/shift ops, one 7-element constant param cl_name_write.e0=[1,0,0,0,0,0,0].
Result: component-test 11/11 passed (batch 1 and 2 identical). No corpus data used.
Helper: scripts/build.py writes proposal.json.
