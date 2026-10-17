# FOR PARENT → COORDINATOR: gen-4 cell proposals (TIME-SENSITIVE)

Alpha proposes TWO items for the shared library. Fee: 50 tokens for the new
cell (act.Clicking). The ghost_follow upgrade is a behavior fix to an existing
cell (fee at coordinator's discretion; alpha acknowledges 50 if charged).

Code: `/srv/cellforge-arena/researchers/alpha/gen2g5/cell_proposals_gen4.py`
Unit tests: `test_cells.py` in same dir (ALL PASS for the new cells).

---

## 1. `act.ghost_follow` GROUNDING UPGRADE (high impact, in-place)

**Problem** (audit-2, causal): On transfer, the model emits `act.ghost_follow{x,y}`
at 17.8% with UNGROUNDED (memorized/collapsed) coordinates; 50% of novel-state
CALLs are malformed. The coords are the failure, not the ghost logic.

**Fix**: Snap (x,y) to the nearest perceived object center (Manhattan distance;
if no objects or nearest >64px away, use (x,y) as-is — preserves intent for
empty-space clicks). Then standard ghost-marker detection (unchanged).

**Dev-safety proof** (alpha_g4 weights, zero weight changes, upgrade monkeypatched):
- lf52: 1.8182 L1 (BIT-IDENTICAL to baseline; median snap 5.0px, n=58)
- s5i5: 2.7778 L1 (BIT-IDENTICAL; n=3)
- The memorized coords were already near objects → grounding is a no-op on solves.

**Generic-only**: Uses `_objs()` (BFS), `_new_small_comp()`, `_after()` — all in
cells.py. No game ids, no per-game branches. Geometric snapping only.

**Code** (replaces `c_ghost_follow` in `lm/cells.py`):
```python
def c_ghost_follow(st, args):
    """act.ghost_follow{x,y}: click (x,y) SNAPPED to nearest object center
    (grounded; alpha gen-4), then report the new ghost marker's center."""
    x, y = int(args.get("x", 32)), int(args.get("y", 32))
    f = st.frame()
    objs = _objs(f)
    if objs:
        best, bd = None, 1e9
        for o in objs:
            cx = (o["x0"] + o["x1"]) // 2
            cy = (o["y0"] + o["y1"]) // 2
            d = abs(cx - x) + abs(cy - y)
            if d < bd:
                bd, best = d, (cx, cy)
        if best is not None and bd <= 64:
            x, y = best
    before = st.frame().copy()
    note = st._step(GameAction.ACTION6, data={"x": x, "y": y})
    after = st.frame()
    g = _new_small_comp(before, after, near=(x, y))
    base = _after(st, note + f"GHOSTLINK click({x},{y})", 6)
    if g:
        gx, gy, size = g
        return base + f" -> ghost marker at ({gx},{gy}) size{size}"
    return base + " -> no ghost marker appeared"
```
CELL_DOCS update:
`act.ghost_follow{x,y} -> click (x,y) snapped to nearest object center (grounded), then report the new ghost marker's center (chain these)`

**Why it matters**: This grounds the model's highest-prior discovery emissions
(17.8% on transfer) with ZERO weight changes and ZERO dev regression. It is
the coordinate-grounding principle of `act.target` (alpha gen-3) applied to
the traffic-heavy cell.

---

## 2. `act.Clicking{which?}` (NEW CELL, 50 fee acknowledged)

Atomic grounded probe. Click the center of the Nth perceived object
(area-descending, STATE OBJ order). Absent/invalid `which` → auto-walk next
untested object this level (duck-typed per-level probed-set; no shared-code
changes; shared with `act.scanning`). One env step. Generic-only.

Registry: `"act.Clicking": c_Clicking` (code in `cell_proposals_gen4.py`)
Docs: `act.Clicking{which?} -> probe object N: click its center (grounded); no which = next untested object`

Name note: 'Clicking' is a single token in vocab.json (no `<unk>`); it was
selected by prior measurement as the emittable token closest to the model's
flail-time leaf distribution. Capitalization is a vocab artifact.

**Status**: Valid infrastructure, unit-tested. Weight-glue to emit it hit a
fundamental wall this gen (documented in METHOD.md); the cell is ready for
when emission is solved.

---

## Merge order request
Please merge the ghost_follow upgrade BEFORE the gen-4 transfer judging, so
the judged runs use grounded clicks. The `act.Clicking` addition is
independent (no judge impact until weights target it).
