# Protected control-gate diagnostic, 2026-10-08

Base source: `02beaf18251fca5ed1f1eb55e92f00a3e3820e88`.

## Reproduced defect

The new additive `tests/test_control_gate_diagnostics.py` uses only in-memory
fixtures and mocked I/O. It does not read ARC data, load a historical evaluator,
score a real run, dispatch authors, or alter protected gate files.

- `tests/standard_checks.py:130–133` handles lists but turns a tensor dictionary
  into scalar `0.0`, losing names and shapes.
- `tests/arc_goal_checks.py:145` supplies the whole tensor dictionary to it.
- `tests/arc_goal_checks.py:56–60` catches evaluator exceptions and returns the
  same `False` used for a valid wrong prediction.
- The combined fixture proves `check_weights_matter` returns an empty issue
  list even when its only ablation execution rejects that malformed scalar.
  The identical fixture solves with shape-correct zero tensors, deliberately
  exposing that the current check can accept a tensor-independent evaluator.

These are expected-defect assertions. Their passing status confirms the defect
is reproducible; it is **not** scientific acceptance, a strengthened production
check, a historical replay, or evidence of learned capability. They should be
revisited when an explicitly authorized protected-gate correction lands.

## Verification

`python -m unittest tests.test_control_gate_diagnostics -v`

Result: 3 tests passed, 0.002 seconds. No model calls or historical scoring.

## Authority boundary

Current `AGENTS.md:31` states: "Do not skip, weaken or delete these tests;
changing them needs the owner."

`AGENTS.md:36–37` states: "Do not edit these tests, the floor, or route around
them (baseline cells, answer lookups, evaluators that ignore the weights)."

`LAB.md:45` states: "never edit `ARC_FLOOR.json`, `HARD_LANES.json`, the
standard/arc-goal tests."

These restrictions cover edits beyond merely weakening assertions. This change
adds independent diagnostics only. A prospective repair needs explicit owner
permission to preserve dictionary structure and distinguish invalid execution
from valid ablation failure. It should strengthen fail-closed behavior and
retain control errors, not bypass the existing gate or alter historical receipts.

Historical green ARC-goal status alone cannot establish valid zero ablation.
This diagnostic does not establish that every historical control was invalid;
that would require a separately scoped, correctly shaped, error-aware replay.

## Pinned sources

- [Zero helper](https://github.com/Sanokei/CellForge/blob/02beaf18251fca5ed1f1eb55e92f00a3e3820e88/tests/standard_checks.py#L130-L133)
- [Execution error handling](https://github.com/Sanokei/CellForge/blob/02beaf18251fca5ed1f1eb55e92f00a3e3820e88/tests/arc_goal_checks.py#L56-L60)
- [Ablation caller](https://github.com/Sanokei/CellForge/blob/02beaf18251fca5ed1f1eb55e92f00a3e3820e88/tests/arc_goal_checks.py#L134-L153)
- [Repository instructions](https://github.com/Sanokei/CellForge/blob/02beaf18251fca5ed1f1eb55e92f00a3e3820e88/AGENTS.md)
- [Current lab restrictions](https://github.com/Sanokei/CellForge/blob/02beaf18251fca5ed1f1eb55e92f00a3e3820e88/LAB.md)
