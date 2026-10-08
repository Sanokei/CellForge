# Bounded local ray adapter

`specialist_runtime.py` now admits the existing `ray_cast` template alongside
`tile_orient`, `assoc_induct` and `glyph_recolor`. Other templates still fail
closed. The dispatcher continues to call `skillcore.predict_bank` with the
original expert topology and exact saved or evaluator-supplied tensors.
No numeric tensor, template, frozen bank, oracle or live driver is changed.

## Admission proof and accounting

The inspected `skillcore.ray_cast` preserves its input shape. Its eight fixed
directions each have a nonzero unit row or column step. For an `h` by `w` grid:

- At most `h*w` pixels can be sources.
- Each can launch at most eight rays: `8*h*w` launch units.
- A ray has at most `max(h,w)-1` in-bounds loop iterations, so path steps are
  bounded by `8*h*w*(max(h,w)-1)`.

The bound ignores source masks, selected directions and early blocking, making
it conservative for every legal saved, zero, random or other override tensor.
Source selection reads the unchanged input array; newly painted cells cannot
spawn additional rays within the same call.
Launches are charged even for a 1x1 grid whose rays have zero in-bounds steps.
The input scan is included in the existing grid-cell visit count. These units
bound relevant traversal counts; they are not FLOPs, elapsed time or currency.

Shape preflight traverses every expert's leave-one-out scoring calls and every
possible routed query. At each chain it includes both intermediate prediction
and recursively staged demonstration **inputs and outputs**. Every ray leaf
in that traversal contributes its shape-specific launches and path steps.
All possible query experts are counted although the router selects only one.

The previous limits remain: eight experts, chain depth two, 30x30 at every
intermediate, 4,096 leaf calls and 1,000,000 grid-cell visits. In addition,
`grid_cell_visits + ray_launches + ray_path_steps` must not exceed 1,000,000.
The host budget reserves that combined amount before `predict_bank`, plus the
existing one-unit dispatch charge. A budget covering grid visits alone is
insufficient. Conservative tile bounds still assume fourfold growth per axis;
some safe particular tensors or large composition inputs remain unsupported.

Receipts use `worst-case-template-shapes/3` and include `ray_launches_bound`,
`ray_path_steps_bound` and `work_units_bound`. Source pins must be refreshed for
the reviewed adapter change. Source hashes assume trusted, unmodified loaded
modules as documented in the runtime; changed ray directions/implementation
require a new review of this proof. Deadlines remain cooperative checks around
synchronous execution, not a way to interrupt a running numerical function.

## Narrow operational scope

The selected bank is read explicitly from
`runs/weights-native-composition01-20261006/SELECTED_SKILL_BANK.json`, file SHA-256
`1415099c2d3b6f44de91d5a47dcdc9341f71a9aa871c395caded2923dc869253` and artifact hash
`40deb8db88c0bb0ba3fdd472c45a1e2093cec0e061f1001298c8d8fa370763fd`.
No default binding, auto-discovery, external tool executor or deployment is added.
The independent composition audit is separate; this change makes no claim to
repeat or validate its 24-task score or establish broader capability.

`tests/test_specialist_ray_adapter.py` checks the geometric bound, recursive
demonstration accounting, total-work rejection, old shape/unknown-template
rejections, exact-bank routing on two new 2x3 fixtures, unchanged zero/all-eight-
direction overrides, and budget rejection before prediction. It uses no saved
search/development tasks, no model calls, no new experiment or run directories.

Run with the existing NumPy-capable Python, `scripts` on `PYTHONPATH`, bytecode
disabled and one numerical thread:

```text
python -B -m unittest tests.test_specialist_ray_adapter -v
```

The full scientific suites are outside this bounded operational check. Existing
qualification failures and unmodified floors are unaffected.

Verification on October 6, 2026: **7 tests passed in 0.087 seconds**, under a
30-second child-process limit with one numerical thread and bytecode disabled.
Both separate 2x3 fixtures produced the expected column/horizontal output; zero
and all-direction overrides matched direct `predict_bank` calls. Grid-only
budget allocation was rejected before prediction. AST and whitespace checks
passed. Selected-bank and `skillcore.py` byte hashes stayed unchanged. No
composition-panel scoring, model calls, campaigns, deployment or commit occurred.
