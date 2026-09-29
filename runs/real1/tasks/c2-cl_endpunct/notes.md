# cl_endpunct v2 notes
Revision of v1: the q cell (72 slots) is gone; cap fix only, behavior unchanged.
- q term: two [36] param tensors (qv: 1.3 at '!', 1.2 at '?'; qb: -0.4, -0.6), q*qv+qb, gated by row-7 letter indicator.
- Rest as v1: exact anchored-word gating via slice/mul ops on ctx rows; one [36] param vector per table word.
- Words >7 letters ("surprise", "together") are skipped: contract says lookups apply only for w length <= 7,
  longer runs get only q terms, so this matches the contract as written.
- Numbers all come from the contract's data tables (expanded by scripts/gen.py); no corpus use.
- Params: 77 word vectors x 36 + 72 = 2844 numbers, 79 tensors, all <=40. Cells: 0. Nodes: 718 (parameter-free ops except param nodes).
- component-test: 14/14 pass, batch-2 identical.
