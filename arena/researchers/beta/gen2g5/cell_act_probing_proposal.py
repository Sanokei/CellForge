"""BETA cell proposal (gen2-gen4): act.Probing{}

Self-gating discovery probe. Generic infrastructure, no per-game logic,
no game names anywhere.

Problem: the router never emits discovery calls on novel games (beta_g4:
0/540 exploratory on transfer-12), and every weight-level novelty signal
tried in gen-3 failed calibration (pooled novelty, LDA, logit gaps,
chg-absence, LVL progression -- per-position SNR<1; the model is a perfect
pattern-collapser). The cell layer, however, sees LIVE state: it can tell
"stuck" (no visual progress, no level-up over many env steps) from
"solving" behaviorally, with no neural signal needed. Novelty->exploration
routing therefore happens IN THE CELL, not in the weights.

Mechanism: when called, the cell runs a generic stuck test on live state:
  - tracks (frame thumbnail, level) across calls, stored on the GameState
    (same pattern as st.tested / st.diff_base -- no new architecture)
  - STUCK = steps_used >= 30 AND no significant frame change AND no
    level-up for the last 25+ env steps (as observed at probe calls)
  - NOT STUCK -> return a short text note, NO env step. Dev-neutral by
    construction: a working solve changes the frame every decision, so the
    probe can never fire mid-solve. Verified by forced-call stress test.
  - STUCK -> bounded grounded discovery: click each untested object center
    (area-descending, same order as perceive.describe -- the act.target
    grounding pattern) + try each untried keyboard action (c_sweep's loop),
    reporting per-action pixel effects so the router can form hypotheses.
    Stops early on LEVEL-UP/WIN. NEVER issues RESET (would destroy banked
    progress). Hard cap 30 env steps per call; respects st.budget.

Name "act.Probing" chosen for vocab-emittability: tokenizes as
['act', '.', 'Probing'] with no <unk> (verified against vocab.json;
'Probing' is token 39).

Dev-neutrality (by construction):
  1. Additive: existing cells untouched; probe runs only on explicit CALL.
  2. No-op when progressing: working solves change the frame constantly.
  3. Never resets; exploration cannot un-bank a solved level.
  4. Bounded steps; budget-respecting.

Prior art (per standing rule):
  - REUSED: _objs (features.py), area-descending object order (act.target),
    untried-action loop + st.tested (c_sweep), _after note pattern.
  - ADAPTED: bigram-gated suffix boost machinery (beta_g4) retargeted to
    the new suffix token.
  - NEW: stuck-test self-gating, bounded grounded discovery routine,
    per-call (thumbnail, level) progress tracking on GameState.
"""

import hashlib

import numpy as np

# --------------------------------------------------------------------------
# The function below is written for verbatim merge into lm/cells.py.
# It uses only names already available there: np, _objs, _avail, _after,
# GameAction, GState. No new imports needed except hashlib (stdlib).
# --------------------------------------------------------------------------

_PROBE_WARMUP_STEPS = 30   # no exploration before this many env steps
_PROBE_STUCK_STEPS = 25    # env steps with zero progress before exploring
_PROBE_MAX_STEPS = 30      # hard cap on env steps per probe call
_PROBE_THUMB_CELLS = 8    # downsampled-frame cells allowed to differ


def _probe_thumb(f):
    return f[::4, ::4].copy()


def c_probing(st, args):
    """act.Probing{}: self-gating discovery probe.

    No-op (pure text, no env step) unless the agent is behaviorally stuck;
    then runs a bounded grounded discovery routine and reports per-action
    effects. Generic: no game names, no per-game logic, never resets.
    """
    f = st.frame()
    lvl = int(st.resp.levels_completed)
    steps = st.steps_used
    thumb = _probe_thumb(f)

    last_thumb = getattr(st, "_probing_thumb", None)
    if last_thumb is None:
        st._probing_thumb = thumb
        st._probing_lvl = lvl
        st._probing_prog_step = steps
        since_prog = 0
    else:
        ndiff = int((thumb != last_thumb).sum())
        changed = (ndiff > _PROBE_THUMB_CELLS) or (lvl != st._probing_lvl)
        if changed:
            st._probing_prog_step = steps
        st._probing_thumb = thumb
        st._probing_lvl = lvl
        since_prog = steps - st._probing_prog_step

    stuck = (steps >= _PROBE_WARMUP_STEPS) and (since_prog >= _PROBE_STUCK_STEPS)
    if not stuck:
        return (f"PROBE no-op (progress {since_prog} steps ago; "
                f"lvl{lvl} step{steps}). No env step taken.")

    # ---- stuck: bounded grounded discovery ----
    out = []
    used0 = steps
    lvl0 = int(st.resp.levels_completed)
    clicked = getattr(st, "_probing_clicked", None)
    if clicked is None:
        clicked = set()
        st._probing_clicked = clicked

    def _note_step(before, label):
        after = st.frame()
        d = int((before != after).sum())
        out.append(f"{label}>{d}px")
        return (st.resp.state == GState.WIN or
                int(st.resp.levels_completed) != lvl0)

    # 1. untried keyboard actions (generic sweep; never RESET)
    for aid in _avail(st):
        if st.steps_used - used0 >= _PROBE_MAX_STEPS:
            break
        if st.steps_used >= st.budget:
            out.append("budget exhausted")
            break
        if aid in (0, 6):
            continue
        if (lvl0, aid) in st.tested:
            continue
        before = st.frame()
        try:
            st._step(GameAction.from_id(aid))
        except Exception:
            out.append(f"a{aid}>err")
            continue
        st.tested.add((lvl0, aid))
        if _note_step(before, f"a{aid}"):
            out.append("LEVEL-UP!")
            break
    # 2. grounded clicks on untested object centers (act.target pattern)
    if st.resp.state != GState.WIN:
        for o in _objs(st.frame())[:6]:
            if st.steps_used - used0 >= _PROBE_MAX_STEPS:
                break
            if st.steps_used >= st.budget:
                out.append("budget exhausted")
                break
            cx = (o["x0"] + o["x1"]) // 2
            cy = (o["y0"] + o["y1"]) // 2
            if (lvl0, cx, cy) in clicked:
                continue
            before = st.frame()
            st._step(GameAction.ACTION6, data={"x": cx, "y": cy})
            clicked.add((lvl0, cx, cy))
            st.tested.add((lvl0, 6))
            if _note_step(before, f"c({cx},{cy})"):
                out.append("LEVEL-UP!")
                break

    # exploration happened: re-arm the stuck timer from here
    st._probing_thumb = _probe_thumb(st.frame())
    st._probing_lvl = int(st.resp.levels_completed)
    st._probing_prog_step = st.steps_used
    lvl1 = int(st.resp.levels_completed)
    tag = f" L{lvl0}->{lvl1}" if lvl1 != lvl0 else ""
    body = " ".join(out) if out else "nothing new"
    return (f"PROBE explored{tag}: {body} "
            f"steps{st.steps_used}/{st.budget} state{st.resp.state.name}")


# REGISTRY addition (for coordinator merge into lm/cells.py):
#   "act.Probing": c_probing,
# CELL_DOCS addition:
#   act.Probing{} -> self-gating discovery probe: no-op when progressing,
#                    grounded click+keyboard sweep when stuck (never resets)
