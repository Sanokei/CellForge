# Shared-policy development probe, 2026-10-02

Two charged writer calls through the existing maker route: one generic founder,
then one runtime repair. No manager calls, game-specific prompt cards, downloads,
API inference or evaluation-split access. Identical source/tensors, fresh memory,
seed 0 and 1500 actions per game. Neither run is a qualified ONE_STANDARD study.

First proposal (`runs/weights-arc3-shared-probe-20261002`) failed before any
action: `NameError: name 'str' is not defined`. All ablations failed as well;
no competence or weight dependence can be inferred. Fresh replay matches.

Repair (`runs/weights-arc3-shared-repair-20261002`) used the exact runtime error
and sandbox builtin list, without game feedback. It ran validly but solved no
levels. All zero, three random-tensor draws and blind runs were valid and also
solved no levels.

| Game | Levels / credited | Actions | No-op actions | Distinct frames |
|---|---:|---:|---:|---:|
| ar25 | 0 / 0 | 1500 | 1363 | 107 |
| lp85 | 0 / 0 | 1500 | 1500 | 1 |
| sc25 | 0 / 0 | 1500 | 353 | 689 |

This is an exploration failure on existing public development games, not an
estimate of new-game generalisation. In particular, one tested policy cannot
establish that shared adaptation is impossible. There is no measured basis to
expand the proposed 16-call population campaign yet. A further hypothesis must
address action-effect learning and no-op recovery, then be tested under the same
shared-genome protocol before spending on a larger population.

Shared meter: 61% before the founder, 62% after it; 62% before and after repair.
Desktop activity shares that meter, so the one-point change is not a per-call
cost estimate. Cap remains 85%; no lab campaigns are left running after replay.
Prompt, responses, literal weights, runtime records, bounded traces (first 400
actions) and manifests remain in the local/server run directories, not Git.

Validation:

- Shared runner safety/budget/replay checks: 3/3 pass.
- Pre-probe standard suites: 25/25 pass after legacy archival.
- Sandbox/archive compatibility: 38 tests, OK (one platform skip).
- Existing ARC goal independently recomputed on the server: 14/14 pass in
  304.723 seconds; this remains public-training evidence under the existing floor.
- Broader architecture gate: 78 tests, two failures; preserved in
  `owner-resumption-gates-20261002.log`.
- New founder qualification: 25 tests, two failures and three errors, including
  single-seed/unequal proposal budgets and missing semantic evaluator; preserved
  in `shared-probe-standard-qualification-20261002.log`. No tests were changed and
  the new probe was not hidden from enumeration to obtain a green suite.
