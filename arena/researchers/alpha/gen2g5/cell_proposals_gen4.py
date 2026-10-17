"""ALPHA cell proposals (gen2-gen4): act.Clicking + act.scanning.

Coordinate-grounded DISCOVERY cells. Generic infrastructure, no per-game
logic, no game names. Designed for the weights-only arena where the fixed
runner DISCARDS cell return strings -- the only feedback channel is the next
perceive.describe STATE text (chg/level fields). So these cells are designed
around ENV EFFECTS + EMITTABLE NAMES, not reports.

Problem they solve (audit-2 + alpha gen-3 findings):
- On novel states the model emits clicks with MEMORIZED/UNGROUNDED coordinates
  (50% malformed CALLs; ghost_follow with collapsed coords). Copying bbox
  arithmetic ((x0+x1)/2) through a 1.4M-param LM is unreliable.
- The model emits ~0 exploratory calls (sweep 0.000 in 360 dev decisions;
  0/540 transfer). Discovery must be the EASY path: one emittable leaf,
  minimal args, grounding done in code.

Cell 1: act.Clicking{which?} -- atomic grounded hypothesis test.
  Click the center of the Nth perceived object (area-descending, same order as
  the STATE OBJ list). If `which` is absent/invalid, the CELL picks the next
  object not yet probed this level (routing at the cell level via a per-level
  probed-set, duck-typed onto GameState -- no shared-code changes). One env
  step. The model sees the effect in the next STATE's chg field.

Cell 2: act.scanning{} -- batched grounded sweep.
  Click the center of EVERY perceived object once (area-descending, skipping
  already-probed), stop on LEVEL-UP / WIN / budget exhaustion. One decision =
  systematic click exploration. Shares the probed-set with act.Clicking, so the
  two compose (scan then probe finds nothing new until the next level).

Names: 'Clicking' and 'scanning' are SINGLE TOKENS in vocab.json (2209) --
they appear in the training THINK traces ('Clicking ...', 'scanning ...'),
so their head rows are trained, not random. Verified:
  encode('act.Clicking') -> ['act', '.', 'Clicking']  (no <unk>)
  encode('act.scanning') -> ['act', '.', 'scanning'] (no <unk>)
The capitalization is a vocab artifact, documented here. 'Clicking' was chosen
by measurement: at the cell-leaf position it has the smallest nat-gap to the
argmax of any emittable candidate (ramble 1.93 / unsolved-dev 1.31 nats;
next-best 'reveal' 3.28/2.81) -- i.e. the model is already closest to emitting
it when flailing, so a small C1-safe head boost flips flail contexts to
grounded discovery while the routine margin (2.76) keeps memorized solves.

Generic-only statement: both cells use only _objs() (BFS components, shared),
frame differencing, and GameState._step. No game ids, no level logic, no
per-game branches. They work on any game with clickable objects and degrade
to no-ops (no click issued) when no objects are visible.
"""

# ---------------------------------------------------------------------------
# Cell 1
def c_Clicking(st, args):
    """act.Clicking{which?}: probe what object N does (click center, grounded).

    which: int index into the STATE OBJ list (area-descending; 0 = largest).
    Absent/invalid which -> cell auto-walks the next untested object this level.
    One env step. Shares per-level probed-set with act.scanning.
    """
    f = st.frame()
    objs = _objs(f)  # area-descending, same order as perceive.describe
    if not objs:
        return "Clicking: no objects visible; no click issued."
    lvl = st.level
    if getattr(st, "_probe_lvl", None) != lvl:
        st._probe_lvl = lvl
        st._probed = set()
    try:
        which = int(args.get("which", -1))
    except Exception:
        which = -1
    if 0 <= which < len(objs):
        pick = which
    else:
        pick = None
        for i, o in enumerate(objs):
            sig = (o["c"], o["x0"], o["y0"], o["x1"], o["y1"])
            if (lvl, sig) not in st._probed:
                pick = i
                break
        if pick is None:
            # full pass done and model still probing: fresh pass (state changed)
            st._probed = set()
            pick = 0
    o = objs[pick]
    st._probed.add((lvl, (o["c"], o["x0"], o["y0"], o["x1"], o["y1"])))
    cx = (o["x0"] + o["x1"]) // 2
    cy = (o["y0"] + o["y1"]) // 2
    note = st._step(GameAction.ACTION6, data={"x": cx, "y": cy})
    base = _after(st, note + "CLICKPROBE obj%d c%d@(%d,%d)" % (pick, o["c"], cx, cy), 6)
    return base


# ---------------------------------------------------------------------------
# Cell 2
def c_scanning(st, args):
    """act.scanning{}: click every perceived object's center once (grounded).

    Area-descending, skips objects already probed this level (shared set with
    act.Clicking). Stops early on LEVEL-UP / WIN / budget exhaustion.
    """
    if st.steps_used >= st.budget:
        return "scanning skipped: budget exhausted."
    if getattr(st, "_probe_lvl", None) != st.level:
        st._probe_lvl = st.level
        st._probed = set()
    lvl0 = st.level
    out = []
    n = 0
    while st.steps_used < st.budget and n < 64:
        # re-fetch: clicks change the frame, new objects may appear
        objs = [o for o in _objs(st.frame())
                if (st.level, (o["c"], o["x0"], o["y0"], o["x1"], o["y1"]))
                not in st._probed]
        if not objs:
            break
        o = objs[0]  # largest untested
        st._probed.add((st.level, (o["c"], o["x0"], o["y0"], o["x1"], o["y1"])))
        cx = (o["x0"] + o["x1"]) // 2
        cy = (o["y0"] + o["y1"]) // 2
        note = st._step(GameAction.ACTION6, data={"x": cx, "y": cy})
        n += 1
        out.append("obj@(%d,%d)" % (cx, cy))
        if st.resp.state == GState.WIN or st.level != lvl0:
            out.append("LEVEL-UP!")
            break
    return ("SCANNING " + " ".join(out) +
            " steps%d/%d" % (st.steps_used, st.budget)) if out else \
           "SCANNING nothing new."


# ---------------------------------------------------------------------------
# REGISTRY additions (for coordinator merge into lm/cells.py):
#   "act.Clicking": c_Clicking,
#   "act.scanning": c_scanning,
# (Requires: _objs, _after, GameAction, GState -- all already in cells.py.)
#
# CELL_DOCS additions:
#   act.Clicking{which?} -> probe object N: click its center (grounded); no which = next untested object
#   act.scanning{} -> click every perceived object's center once (grounded); stops on LEVEL-UP
