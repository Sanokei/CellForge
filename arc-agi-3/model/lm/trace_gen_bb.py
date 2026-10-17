"""Blackboard teacher traces: minimal-action play with INTAKE lines in the prompt.

Each decision row: STATE + INTAKE lines (from the REAL observer loop) +
THINK (authored from the actual top intake) + CALL (act.* only) + OBS.

Sources (dev games only; held_v6.jsonl sealed):
  s5i5 dial grammar seeds 0..5 (13 actions), lf52 ghost chain seeds 0..5
  (8 actions), ls20 blitz8 L0 translated (64 actions).
Efficiency weighting: rows duplicated round(64/episode_actions)x.
Writes lm/traces_bb.jsonl
"""
import json
import logging
import os
import random
import re
import sys
import importlib.util
sys.path.insert(0, "/srv/arc3-claude/v6run")          # v6run/cells for blitz_agent
sys.path.insert(0, "/srv/arc3-claude/v6run/blitz")    # blitz_agent, blitz_w2
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm/cells_intake")
_spec = importlib.util.spec_from_file_location(
    "lm_cells", "/srv/arc3-claude/v6run/model/lm/cells.py")
lm_cells = importlib.util.module_from_spec(_spec)
sys.modules["lm_cells"] = lm_cells
_spec.loader.exec_module(lm_cells)
GameState, call_cell = lm_cells.GameState, lm_cells.call_cell
# blackboard intake via spec (v6run/model/intake.py is the parallel track's)
_spec2 = importlib.util.spec_from_file_location(
    "bb_intake", "/srv/arc3-claude/v6run/model/lm/cells_intake/intake.py")
bb_intake = importlib.util.module_from_spec(_spec2)
sys.modules["bb_intake"] = bb_intake
_spec2.loader.exec_module(bb_intake)
new_ctx = bb_intake.new_ctx
collect_intake = bb_intake.collect_intake
format_intake = bb_intake.format_intake
format_state = bb_intake.format_state
from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
OUT = os.path.join(BASE, "v6run", "model", "lm", "traces_bb.jsonl")
GHOST_RE = re.compile(r"ghost marker at \((\d+),(\d+)\)")
DIALS = [((46, 21), 7), ((24, 45), 6)]

lg = logging.getLogger("q")
lg.addHandler(logging.NullHandler())
lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)


def think_from_intake(top, fallback, echo_xy=None):
    """Author THINK from the actual top intake. echo_xy: when the decision's
    coordinates come from the intake (ghost chain), echo them explicitly so
    the tiny model learns the copy chain INTAKE -> THINK -> CALL."""
    if not top:
        return "No strong intake. " + fallback
    n, t, b = top[0]
    if n == "novelty" and echo_xy is not None:
        gx, gy = echo_xy
        return (f"Intake: novelty cell reports {t}. The new marker is at "
                f"({gx},{gy}) — clicking ({gx},{gy}) in one macro call, no "
                f"wasted clicks.")
    if n == "novelty":
        return (f"Intake: novelty cell reports {t} — a new marker appeared. "
                f"New markers are targets. {fallback}")
    if n == "scene":
        return f"Intake: scene shows {t}. {fallback}"
    if n == "counter":
        return f"Intake: counts shifted ({t}). {fallback}"
    if n == "change":
        return f"Intake: {t} changed. {fallback}"
    return f"Intake {n}: {t}. {fallback}"


def decide_row(st, ctx, prev_frame, cell, args, think_tail, echo_xy=None):
    frame = st.frame()
    top = collect_intake(frame, prev_frame, ctx)
    state_txt = format_state(st.level, st.steps_used,
                             st.resp.available_actions).strip()
    think = think_from_intake(top, think_tail, echo_xy)
    obs = call_cell(st, cell, args)
    if cell in ("act.click_at", "act.ghost_follow", "act.repeat_click",
                "act.pick_drop") and "x" in args and "y" in args:
        ctx["last_click"] = (int(args["x"]), int(args["y"]))
    row = {"state": state_txt,
           "intake": format_intake(top).strip(),
           "think": think,
           "call": {"cell": cell, "args": args},
           "obs": obs[:200]}
    # return the PRE-action frame: next decision compares post-action vs
    # pre-action (frame==prev would blind change/novelty observers)
    return row, frame


def gen_s5i5(arc, seeds):
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == "s5i5")
    eps = []
    for seed in seeds:
        card = arc.create_scorecard()
        env = arc.make(info.game_id, seed=seed, scorecard_id=card)
        st = GameState(env, 3000)
        ctx = new_ctx()
        rows, prev = [], None
        r, prev = decide_row(st, ctx, prev, "act.repeat_click",
                             {"x": 46, "y": 21, "n": 7},
                             "Two dial tokens need turning. Efficient play: one "
                             "macro call per dial, no probing. Turning the first "
                             "dial with 7 repeated clicks.")
        rows.append(r)
        if st.level == 0:
            r, prev = decide_row(st, ctx, prev, "act.repeat_click",
                                 {"x": 24, "y": 45, "n": 6},
                                 "First dial done. Turning the second dial with "
                                 "6 repeated clicks.")
            rows.append(r)
        ok = st.level > 0
        n_act = st.steps_used
        arc.close_scorecard(card)
        print(f"s5i5 seed {seed}: {n_act} actions ok={ok}", flush=True)
        if ok:
            eps.append((rows, n_act))
    return eps


def token_candidates(st):
    import re as _re
    obs = call_cell(st, "perceive.objects", {})
    toks = []
    for m in _re.finditer(r"OBJ\d+: color (\d+) bbox \((\d+),(\d+)\)-\((\d+),(\d+)\)", obs):
        c, x0, y0, x1, y1 = map(int, m.groups())
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if area <= 64:
            toks.append((area, (x0 + x1) // 2, (y0 + y1) // 2))
    toks.sort(reverse=True)
    return toks


def gen_lf52(arc, seeds):
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == "lf52")
    eps = []
    for seed in seeds:
        card = arc.create_scorecard()
        env = arc.make(info.game_id, seed=seed, scorecard_id=card)
        st = GameState(env, 3000)
        toks = token_candidates(st)
        arc.close_scorecard(card)
        good = None
        for att in range(min(10, len(toks))):
            c2 = arc.create_scorecard()
            e2 = arc.make(info.game_id, seed=seed, scorecard_id=c2)
            s2 = GameState(e2, 3000)
            tx, ty = toks[att][1], toks[att][2]
            o2 = call_cell(s2, "act.ghost_follow", {"x": tx, "y": ty})
            m = GHOST_RE.search(o2)
            if m and not (int(m.group(1)) == 0 and int(m.group(2)) == 0):
                good = (tx, ty)
            arc.close_scorecard(c2)
            if good:
                break
        if not good:
            print(f"lf52 seed {seed}: no token", flush=True)
            continue
        card = arc.create_scorecard()
        env = arc.make(info.game_id, seed=seed, scorecard_id=card)
        st = GameState(env, 3000)
        ctx = new_ctx()
        rows, prev = [], None
        tx, ty = good
        r, prev = decide_row(st, ctx, prev, "act.ghost_follow",
                             {"x": tx, "y": ty},
                             f"Clicking the token at ({tx},{ty}) to reveal where "
                             "it wants to go.",
                             echo_xy=(tx, ty))
        rows.append(r)
        obs, links = r["obs"], 0
        while links < 12 and st.level == 0 and st.resp.state != GState.WIN:
            m = GHOST_RE.search(obs)
            if not m:
                break
            gx, gy = int(m.group(1)), int(m.group(2))
            r, prev = decide_row(st, ctx, prev, "act.ghost_follow",
                                 {"x": gx, "y": gy},
                                 "Following the marker chain — one macro call "
                                 "per link, no wasted clicks.",
                                 echo_xy=(gx, gy))
            rows.append(r)
            obs = r["obs"]
            links += 1
        ok = st.level > 0
        n_act = st.steps_used
        arc.close_scorecard(card)
        print(f"lf52 seed {seed}: {n_act} actions ok={ok}", flush=True)
        if ok:
            eps.append((rows, n_act))
    return eps


def gen_ls20(arc):
    """blitz8 L0 (painter-demotion, 64 actions) replayed through the blackboard."""
    from blitz_agent import AGENTS
    import blitz_w2  # noqa
    import numpy as np
    from arcengine import GameAction
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == "ls20")

    def aid_of(action):
        if action == GameAction.RESET:
            return 0
        if action == GameAction.ACTION6:
            return 6
        return int(action.value)

    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=0, scorecard_id=card)
    frames, aids = [], []
    cur = {"resp": env.observation_space}
    frames.append(np.asarray(cur["resp"].frame[-1], dtype=np.uint8).copy())
    orig_step = env.step

    def rec_step(action, data=None):
        r = orig_step(action) if data is None else orig_step(action, data=data)
        frames.append(np.asarray(r.frame[-1], dtype=np.uint8).copy())
        aids.append(aid_of(action))
        cur["resp"] = r
        return r

    env.step = rec_step
    ex = AGENTS["blitz8"](env, 3000, seed=0, log=lambda *a: None,
                          flags=("pr", "nm", "ck", "gv", "d6"), tags=info.tags)
    try:
        ex.run()
    except Exception as e:
        print("blitz8 err", e, flush=True)
    arc.close_scorecard(card)
    # replay L0 only through a fresh env + blackboard
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=0, scorecard_id=card)
    st = GameState(env, 3000)
    ctx = new_ctx()
    rows, prev = [], None
    for aid in aids:
        if st.level > 0:
            break
        if aid == 0:
            cell, args, tail = "act.reset", {}, "Resetting the level to try fresh."
        elif aid == 6:
            cell, args, tail = "act.click_at", {"x": 32, "y": 32}, "Clicking."
        else:
            cell, args = "act.try", {"action": aid}
            tail = (f"Action {aid} only deposits new pixels, never revisits "
                    "known states — painter-like. Demoting it, trying others first."
                    if aid == 2 else f"Trying keyboard action {aid}.")
        r, prev = decide_row(st, ctx, prev, cell, args, tail)
        rows.append(r)
        if st.resp.state == GState.WIN or st.level > 0:
            break
    n_act = st.steps_used
    arc.close_scorecard(card)
    print(f"ls20: {len(rows)} rows from {n_act} actions", flush=True)
    return [(rows, n_act)]


def gen_copydrills(n=400):
    """Synthetic copy drills: random marker positions force the LM to learn
    the GENERAL copy operation (intake -> think -> call), not memorize the
    8 deterministic lf52 positions. No env needed; obs is templated."""
    rng = random.Random(99)
    rows = []
    for i in range(n):
        x, y = rng.randrange(64), rng.randrange(64)
        c, size = rng.randrange(1, 16), 1
        step = rng.randrange(8)
        intake = (f"INTAKE novelty: new c{c}@({x},{y})a{size}\n"
                  f"INTAKE scene: 34obj c10@31,31a2844")
        think = (f"Intake: novelty cell reports new c{c}@({x},{y})a{size}. "
                 f"The new marker is at ({x},{y}) — clicking ({x},{y}) in one "
                 f"macro call, no wasted clicks.")
        rows.append({
            "state": f"STATE LVL0 STEP{step} AVAIL[1, 2, 3, 4, 6, 7]",
            "intake": intake, "think": think,
            "call": {"cell": "act.ghost_follow", "args": {"x": x, "y": y}},
            "obs": f"GHOSTLINK click({x},{y}) lvl0 chg1px steps{step+1}/3000 "
                   f"stateNOT_FINISHED -> ghost marker at ({x},{y}) size{size}"})
    return [(rows, 64)]  # n_act=64 -> dup=1; 400 drills is plenty


def main():
    arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                 recordings_dir="/tmp/bb_rec")
    seeds = [0, 1, 2, 3, 4, 5]
    # NOTE: no primer episode — primer rows on s5i5 states taught conflicting
    # actions for the same (state,intake) and poisoned the dial policy
    # (2026-10-16). Vocab coverage for unused cells comes from mandatory
    # tokens in train_bb.py instead.
    eps = gen_s5i5(arc, seeds) + gen_lf52(arc, seeds) + gen_ls20(arc) \
        + gen_copydrills(400)
    out_rows = []
    for rows, n_act in eps:
        dup = max(1, round(64 / max(1, n_act)))
        for _ in range(dup):
            out_rows.extend(rows)
    random.Random(0).shuffle(out_rows)
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"WROTE {len(out_rows)} rows from {len(eps)} episodes -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
