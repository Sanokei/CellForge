"""Cell registry for the tiny-LM agentic router.

Glue protocol: the LM emits one line  CALL {"cell": "<name>", "args": {...}}
Each cell takes a GameState + args and returns a compact observation string.
Perception cells cost no env steps; action cells step the real env.
"""
import sys
import os
from collections import deque

import numpy as np

sys.path.insert(0, "/srv/arc3-claude/v6run/model")
from features import _components  # reuse BFS components

from arcengine import GameAction, GameState as GState


class GameState:
    def __init__(self, env, budget):
        self.env = env
        self.budget = budget
        self.resp = env.observation_space
        self.prev_frame = None
        self.steps_used = 0
        self.diff_base = None
        self.tested = set()   # (level, action_id) tried
        self.level = 0

    def frame(self):
        return np.asarray(self.resp.frame[-1], dtype=np.uint8)

    def _step(self, action, data=None):
        if self.resp.state == GState.GAME_OVER:
            self.resp = self.env.step(GameAction.RESET)
            self.steps_used += 1
            self.level = int(self.resp.levels_completed)
            return "auto-reset after game-over; "
        self.prev_frame = self.frame()
        r = self.env.step(action) if data is None else self.env.step(action, data=data)
        self.steps_used += 1
        if r is None:
            return "env returned None; "
        self.resp = r
        self.level = int(r.levels_completed)
        return ""


def _objs(frame):
    pres = frame > 0
    comps = _components(pres)
    out = []
    for (size, x0, y0, x1, y1) in comps:
        sub = frame[y0:y1+1, x0:x1+1]
        vals, cnts = np.unique(sub[sub > 0], return_counts=True)
        color = int(vals[np.argmax(cnts)]) if len(vals) else 0
        out.append({"c": color, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "area": size})
    out.sort(key=lambda o: -o["area"])
    return out


def _avail(st):
    return sorted(int(x) for x in st.resp.available_actions)


# ------------------------------------------------------------ perception
def c_describe(st, args):
    f = st.frame()
    objs = _objs(f)[:6]
    parts = []
    for o in objs:
        parts.append(f"c{o['c']}@{o['x0']},{o['y0']}-{o['x1']},{o['y1']}a{o['area']}")
    chg = ""
    if st.prev_frame is not None:
        d = f != st.prev_frame
        n = int(d.sum())
        if n:
            ys, xs = np.nonzero(d)
            chg = f" chg{n}@({xs.min()},{ys.min()},{xs.max()},{ys.max()})"
    return (f"LVL{st.level} STEP{st.steps_used} OBJ{len(objs)} "
            + " ".join(parts) + chg + f" AVAIL{_avail(st)}")


def c_objects(st, args):
    f = st.frame()
    objs = _objs(f)[:10]
    lines = [f"OBJ{i}: color {o['c']} bbox ({o['x0']},{o['y0']})-({o['x1']},{o['y1']}) area {o['area']}"
             for i, o in enumerate(objs)]
    return f"{len(objs)} objects. " + " | ".join(lines) if lines else "0 objects."


def c_diff(st, args):
    f = st.frame()
    base = st.diff_base if st.diff_base is not None else (st.prev_frame if st.prev_frame is not None else f)
    d = f != base
    n = int(d.sum())
    st.diff_base = f.copy()
    if n == 0:
        return "no change since last diff."
    ys, xs = np.nonzero(d)
    return (f"{n} px changed since last diff, bbox "
            f"({xs.min()},{ys.min()})-({xs.max()},{ys.max()}).")


def c_hud(st, args):
    f = st.frame()
    rows = []
    for i in range(3):
        vals, cnts = np.unique(f[i], return_counts=True)
        top = sorted(zip(vals.tolist(), cnts.tolist()), key=lambda t: -t[1])[:3]
        rows.append(f"r{i}:" + ",".join(f"{v}x{c}" for v, c in top))
    return "HUD " + " ".join(rows)


# ---------------------------------------------------------------- action
def _after(st, note, action_id=None):
    f = st.frame()
    chg = 0
    if st.prev_frame is not None:
        chg = int((f != st.prev_frame).sum())
    if action_id is not None:
        st.tested.add((st.level, action_id))
    return (f"{note} lvl{st.level} chg{chg}px steps{st.steps_used}/{st.budget} "
            f"state{st.resp.state.name}")


def c_try(st, args):
    aid = int(args.get("action", 1))
    note = st._step(GameAction.RESET) if aid == 0 else ""
    if aid == 0:
        return _after(st, note + "ACT reset")
    if aid == 6:
        x, y = int(args.get("x", 32)), int(args.get("y", 32))
        note += st._step(GameAction.ACTION6, data={"x": x, "y": y})
        return _after(st, note + f"ACT click({x},{y})", 6)
    try:
        note += st._step(GameAction.from_id(aid))
    except Exception as e:
        return f"ACT {aid} invalid: {e}"
    return _after(st, note + f"ACT {aid}", aid)


def c_sweep(st, args):
    if st.steps_used >= st.budget:
        return "sweep skipped: budget exhausted."
    out = []
    for aid in _avail(st):
        if st.steps_used >= st.budget:
            break
        if (st.level, aid) in st.tested:
            continue
        before = st.frame().copy()
        if aid == 6:
            continue  # clicks need coordinates; sweep keyboard only
        note = st._step(GameAction.from_id(aid))
        after = st.frame()
        d = int((before != after).sum())
        st.tested.add((st.level, aid))
        out.append(f"{aid}>{d}px")
        if st.resp.state == GState.WIN or int(st.resp.levels_completed) != st.level:
            out.append("LEVEL-UP!")
            st.level = int(st.resp.levels_completed)
            break
    return "SWEEP " + " ".join(out) + f" steps{st.steps_used}/{st.budget}" if out else "SWEEP nothing new."


def c_key_seq(st, args):
    """act.key_seq{seq:[ids]} -> run simple action ids as ONE decision.

    General keyboard macro: no game knowledge, just action ids 1-5,7 (0=reset).
    Stops early on LEVEL-UP / WIN / budget exhaustion. Reports per-action
    pixel change so the LM can learn action->effect mappings.
    """
    seq = args.get("seq", [])
    if not isinstance(seq, list):
        return "key_seq invalid: seq must be a list of action ids"
    seq = seq[:200]  # hard cap
    lvl0 = int(st.resp.levels_completed)
    out = []
    for raw in seq:
        if st.steps_used >= st.budget:
            out.append("budget exhausted")
            break
        try:
            aid = int(raw)
        except Exception:
            out.append(f"skip bad {raw}")
            continue
        if aid == 6:
            out.append("skip 6 (needs coords)")
            continue
        if aid == 0:
            st._step(GameAction.RESET)
            out.append("reset")
        else:
            try:
                st._step(GameAction.from_id(aid))
            except Exception:
                out.append(f"{aid} invalid")
                continue
            f = st.frame()
            chg = int((f != st.prev_frame).sum()) if st.prev_frame is not None else 0
            st.tested.add((st.level, aid))
            out.append(f"{aid}>{chg}px")
        if st.resp.state == GState.WIN or int(st.resp.levels_completed) > lvl0:
            out.append("LEVEL-UP!")
            break
    return ("KEYSEQ " + " ".join(out) +
            f" steps{st.steps_used}/{st.budget} state{st.resp.state.name}")


def c_click_at(st, args):
    x, y = int(args.get("x", 32)), int(args.get("y", 32))
    note = st._step(GameAction.ACTION6, data={"x": x, "y": y})
    return _after(st, note + f"CLICK({x},{y})", 6)


def c_pick_drop(st, args):
    x1, y1 = int(args.get("x1", 0)), int(args.get("y1", 0))
    x2, y2 = int(args.get("x2", 0)), int(args.get("y2", 0))
    note = st._step(GameAction.ACTION6, data={"x": x1, "y": y1})
    note += st._step(GameAction.ACTION6, data={"x": x2, "y": y2})
    return _after(st, note + f"PICKDROP({x1},{y1})->({x2},{y2})", 6)


def c_reset(st, args):
    if st.steps_used == 0:
        return "reset refused: no actions taken yet (would drop to level 0)."
    note = st._step(GameAction.RESET)
    st.tested = {(l, a) for (l, a) in st.tested if l != st.level}
    return _after(st, note + "RESET level")


def c_repeat_click(st, args):
    """Dial macro: click the same spot n times (tokens that are dials, not toggles)."""
    x, y = int(args.get("x", 32)), int(args.get("y", 32))
    n = max(1, min(int(args.get("n", 2)), 20))
    note = ""
    for _ in range(n):
        if st.steps_used >= st.budget:
            break
        note += st._step(GameAction.ACTION6, data={"x": x, "y": y})
    return _after(st, note + f"REPEATCLICK({x},{y})x{n}", 6)


def _new_small_comp(before, after, near=None, max_area=120):
    """Find same-colour components in `after` not present in `before`
    (matched by representative pixel + colour); return the smallest one's
    representative coords. Mirrors blitz11's ghost detection."""
    def comps(fr):
        h, w = fr.shape
        seen = np.zeros((h, w), bool)
        out = []
        for yy in range(h):
            for xx in range(w):
                if seen[yy, xx]:
                    continue
                c = int(fr[yy, xx])
                q = deque([(yy, xx)])
                seen[yy, xx] = True
                pix = []
                while q:
                    cy, cx = q.popleft()
                    pix.append((cy, cx))
                    if cy > 0 and not seen[cy-1, cx] and fr[cy-1, cx] == c:
                        seen[cy-1, cx] = True; q.append((cy-1, cx))
                    if cy + 1 < h and not seen[cy+1, cx] and fr[cy+1, cx] == c:
                        seen[cy+1, cx] = True; q.append((cy+1, cx))
                    if cx > 0 and not seen[cy, cx-1] and fr[cy, cx-1] == c:
                        seen[cy, cx-1] = True; q.append((cy, cx-1))
                    if cx + 1 < w and not seen[cy, cx+1] and fr[cy, cx+1] == c:
                        seen[cy, cx+1] = True; q.append((cy, cx+1))
                ys = [p[0] for p in pix]; xs = [p[1] for p in pix]
                my, mx = sum(ys) / len(ys), sum(xs) / len(xs)
                ry, rx = min(pix, key=lambda p: (p[0]-my)**2 + (p[1]-mx)**2)
                out.append((c, len(pix), rx, ry))
        return out
    before_set = {(rx, ry, c) for c, _, rx, ry in comps(before)}
    best = None
    for c, size, rx, ry in comps(after):
        if (rx, ry, c) in before_set or size > max_area or c == 0:
            continue
        # smallest new marker first (ghost-like), distance as tie-break
        score = (size, abs(rx - near[0]) + abs(ry - near[1]) if near else 0)
        if best is None or score < best[0]:
            best = (score, rx, ry, size)
    return best[1:] if best else None


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


def c_Clicking(st, args):
    """act.Clicking{which?}: probe what object N does (click center, grounded).
    which: int index into the STATE OBJ list (area-descending; 0 = largest).
    Absent/invalid which -> cell auto-walks the next untested object this level.
    One env step. Generic cell, alpha gen-4."""
    f = st.frame()
    objs = _objs(f)
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
            st._probed = set()
            pick = 0
    o = objs[pick]
    st._probed.add((lvl, (o["c"], o["x0"], o["y0"], o["x1"], o["y1"])))
    cx = (o["x0"] + o["x1"]) // 2
    cy = (o["y0"] + o["y1"]) // 2
    note = st._step(GameAction.ACTION6, data={"x": cx, "y": cy})
    base = _after(st, note + "CLICKPROBE obj%d c%d@(%d,%d)" % (pick, o["c"], cx, cy), 6)
    return base




def c_target(st, args):
    """act.target{which}: click center of Nth perceived object (area-descending,
    same order as STATE OBJ list; which=0 is largest). Generic cell, alpha gen-3."""
    f = st.frame()
    objs = _objs(f)  # area-descending, same order as perceive.describe
    if not objs:
        return "target: no objects visible; no click issued."
    try:
        which = int(args.get("which", 0))
    except Exception:
        return "target invalid: which must be an int"
    which = max(0, min(which, len(objs) - 1))
    o = objs[which]
    cx = (o["x0"] + o["x1"]) // 2
    cy = (o["y0"] + o["y1"]) // 2
    note = st._step(GameAction.ACTION6, data={"x": cx, "y": cy})
    base = _after(st, note + f"TARGET obj{which} c{o['c']}@({cx},{cy})", 6)
    return base


_PROBE_WARMUP_STEPS = 30
_PROBE_STUCK_STEPS = 25
_PROBE_MAX_STEPS = 30
_PROBE_THUMB_CELLS = 8


def _probe_thumb(f):
    return f[::4, ::4].copy()


def c_probing(st, args):
    """act.Probing{}: self-gating discovery probe.

    No-op (pure text, no env step) unless the agent is behaviorally stuck;
    then runs a bounded grounded discovery routine and reports per-action
    effects. Generic: no game names, no per-game logic, never resets.
    Beta gen-4."""
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

    st._probing_thumb = _probe_thumb(st.frame())
    st._probing_lvl = int(st.resp.levels_completed)
    st._probing_prog_step = st.steps_used
    lvl1 = int(st.resp.levels_completed)
    tag = f" L{lvl0}->{lvl1}" if lvl1 != lvl0 else ""
    body = " ".join(out) if out else "nothing new"
    return (f"PROBE explored{tag}: {body} "
            f"steps{st.steps_used}/{st.budget} state{st.resp.state.name}")

REGISTRY = {
    "perceive.describe": c_describe,
    "perceive.objects": c_objects,
    "perceive.diff": c_diff,
    "perceive.hud": c_hud,
    "act.try": c_try,
    "act.sweep": c_sweep,
    "act.key_seq": c_key_seq,
    "act.click_at": c_click_at,
    "act.target": c_target,
    "act.pick_drop": c_pick_drop,
    "act.repeat_click": c_repeat_click,
    "act.ghost_follow": c_ghost_follow,
    "act.Clicking": c_Clicking,
    "act.Probing": c_probing,
    "act.reset": c_reset,
}

CELL_DOCS = """perceive.describe{} -> compact state summary (free)
perceive.objects{} -> object list with bboxes (free)
perceive.diff{} -> changed pixels since last diff (free)
perceive.hud{} -> top-3-rows summary (free)
act.try{action,x?,y?} -> one env step (action 0=reset, 6=click needs x,y)
act.target{which} -> click center of Nth perceived object (area-descending; which=0 largest)
act.sweep{} -> try each untested keyboard action once, report effects
act.key_seq{seq:[ids]} -> run simple action ids 1-5,7 as ONE decision (keyboard macro)
act.click_at{x,y} -> click at 0-63 coords
act.pick_drop{x1,y1,x2,y2} -> two clicks (grab then drop)
act.repeat_click{x,y,n} -> click the same spot n times (dial macro)
act.ghost_follow{x,y} -> click (x,y) SNAPPED to nearest object center (grounded), then report the new ghost marker's center (chain these)
act.Clicking{which?} -> probe object N: click its center (grounded); no which = next untested object
act.Probing{} -> self-gating discovery probe: no-op when progressing, grounded click+keyboard sweep when stuck (never resets)
act.reset{} -> reset current level"""


def call_cell(st, name, args):
    fn = REGISTRY.get(name)
    if fn is None:
        return f"unknown cell '{name}'."
    try:
        return fn(st, args or {})
    except Exception as e:
        return f"cell {name} error: {e}"
