"""Trace generator v2: run blitz13 (all wave-2 fixes) on dev games, translate the
raw action stream into cell-call distillation traces with strategy-grounded
English reasoning.

Usage: python trace_gen_w2.py [seed] [budget]
Writes lm/traces_w2.jsonl. Excludes lp85 (lucky-solve game: don't teach luck).
"""
import json
import logging
import os
import random
import sys

import numpy as np

sys.path.insert(0, "/srv/arc3-claude/v6run/blitz")
sys.path.insert(0, "/srv/arc3-claude/v6run")
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
from blitz_agent import AGENTS
import blitz_w2  # noqa: registers blitz6..13
# load the LM cell registry under a distinct module name (v6run/cells.py
# must keep the plain "cells" name for explorer5/blitz imports)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "lm_cells", "/srv/arc3-claude/v6run/model/lm/cells.py")
lm_cells = importlib.util.module_from_spec(_spec)
sys.modules["lm_cells"] = lm_cells
_spec.loader.exec_module(lm_cells)
_objs = lm_cells._objs
_new_small_comp = lm_cells._new_small_comp
from arc_agi import Arcade
from arcengine import GameAction, GameState as GState

BASE = "/srv/arc3-claude"
OUT = os.path.join(BASE, "v6run", "model", "lm", "traces_w2.jsonl")
# train games minus lp85 (lucky 5-action solve — excluded from teaching)
GAMES = ["bp35", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
         "ls20", "re86", "sc25", "sk48"]


class Shim:
    """Minimal state for c_describe-style summaries from recorded frames."""
    def __init__(self, frame, prev, avail, level, steps):
        self._f = frame
        self.prev_frame = prev
        self.level = level
        self.steps_used = steps
        self.resp = type("R", (), {"available_actions": avail})()

    def frame(self):
        return self._f


def describe(frame, prev, avail, level, steps):
    f = np.asarray(frame, dtype=np.uint8)
    objs = _objs(f)[:6]
    parts = [f"c{o['c']}@{o['x0']},{o['y0']}-{o['x1']},{o['y1']}a{o['area']}"
             for o in objs]
    chg = ""
    if prev is not None:
        d = f != np.asarray(prev, dtype=np.uint8)
        n = int(d.sum())
        if n:
            ys, xs = np.nonzero(d)
            chg = f" chg{n}@({xs.min()},{ys.min()},{xs.max()},{ys.max()})"
    return (f"LVL{level} STEP{steps} OBJ{len(objs)} " + " ".join(parts) +
            chg + f" AVAIL{sorted(int(x) for x in avail)}")


def aid_of(action):
    if action == GameAction.RESET:
        return 0
    if action == GameAction.ACTION6:
        return 6
    return int(action.value)


def run_game(game, seed, budget):
    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                 recordings_dir=f"{BASE}/recordings")
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)

    rec = []  # (frame_before, aid, x, y, frame_after, level_after, avail_after)
    cur = {"resp": env.observation_space}
    orig_step = env.step

    def rec_step(action, data=None):
        fb = np.asarray(cur["resp"].frame[-1], dtype=np.uint8).copy()
        av = list(cur["resp"].available_actions)
        aid = aid_of(action)
        x = y = -1
        if aid == 6 and data:
            x, y = int(data.get("x", 0)), int(data.get("y", 0))
        r = orig_step(action) if data is None else orig_step(action, data=data)
        fa = np.asarray(r.frame[-1], dtype=np.uint8).copy()
        rec.append((fb, aid, x, y, fa, int(r.levels_completed),
                    list(r.available_actions)))
        cur["resp"] = r
        return r

    env.step = rec_step
    ex = AGENTS["blitz13"](env, budget, seed=seed, log=lambda *a: None,
                           flags=("pr", "nm", "ck", "gv", "d6"),
                           tags=info.tags)
    try:
        ex.run()
    except Exception as e:
        print(f"AGENT ERROR {game}: {e}", file=sys.stderr)
    sc = arc.close_scorecard(card)
    run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
    return rec, {"game": game, "levels": run.get("levels_completed"),
                 "score": run.get("score")}


def is_painter(fb, fa):
    """True if the transition only ADDS pixels (0->nonzero), never removes/changes."""
    added = (fb == 0) & (fa != 0)
    removed_or_changed = (fb != 0) & (fa != fb)
    return bool(added.any() and not removed_or_changed.any())


def translate(rec, rng):
    """Raw action stream -> list of trace dicts."""
    traces = []
    n = len(rec)
    i = 0
    step_no = 0

    def emit(fb, fa, avail_b, avail_a, lvl_b, lvl_a, think, cell, args, steps):
        s = describe(fb, None, avail_b, lvl_b, step_no)
        o = describe(fa, fb, avail_a, lvl_a, step_no + steps)
        traces.append({"state": s, "think": think,
                       "call": {"cell": cell, "args": args}, "obs": o})

    while i < n:
        fb, aid, x, y, fa, lvl_a, av_a = rec[i]
        avail_b = rec[i - 1][6] if i > 0 else av_a
        lvl_b = rec[i - 1][5] if i > 0 else 0
        if aid == 0:
            think = rng.choice([
                "No progress here. Resetting the level to try fresh.",
                "Stuck. Resetting and re-scanning."])
            emit(fb, fa, avail_b, av_a, lvl_b, lvl_a, think,
                 "act.reset", {}, 1)
            step_no += 1
            i += 1
        elif aid == 6:
            # gather click run
            j = i
            clicks = []
            while j < n and rec[j][1] == 6:
                clicks.append(rec[j])
                j += 1
            coords = [(c[2], c[3]) for c in clicks]
            if len(clicks) >= 2 and len(set(coords)) == 1:
                cx, cy = coords[0]
                think = (f"This spot is a dial, not a toggle — it needs repeated "
                         f"clicks. Repeating {len(clicks)}x at ({cx},{cy}).")
                emit(fb, clicks[-1][4], avail_b, clicks[-1][6], lvl_b,
                     clicks[-1][5], think, "act.repeat_click",
                     {"x": cx, "y": cy, "n": len(clicks)}, len(clicks))
            elif len(clicks) >= 2:
                # check ghost chaining: each click lands on previous click's ghost
                is_chain = True
                for k in range(1, len(clicks)):
                    g = _new_small_comp(clicks[k-1][0], clicks[k-1][4],
                                        near=(clicks[k][2], clicks[k][3]))
                    if g is None or abs(g[0]-clicks[k][2]) + abs(g[1]-clicks[k][3]) > 6:
                        is_chain = False
                        break
                if is_chain:
                    # one trace per link (chaining, not one-shot)
                    prev_g = None
                    for k, c in enumerate(clicks):
                        cfb, _, cx, cy, cfa, clvl, cav = c
                        if k == 0:
                            think = (f"Clicking the token at ({cx},{cy}) to reveal "
                                     f"where it wants to go.")
                        else:
                            think = (f"A ghost marker appeared at ({cx},{cy}). Ghosts mark "
                                     f"the next target — following it.")
                        emit(cfb, cfa, avail_b if k == 0 else clicks[k-1][6],
                             cav, lvl_b if k == 0 else clicks[k-1][5], clvl,
                             think, "act.ghost_follow", {"x": cx, "y": cy}, 1)
                else:
                    (x1, y1), (x2, y2) = coords[0], coords[1]
                    think = (f"Moving the piece from ({x1},{y1}) to ({x2},{y2}).")
                    emit(fb, clicks[1][4], avail_b, clicks[1][6], lvl_b,
                         clicks[1][5], think, "act.pick_drop",
                         {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, 2)
                    # remaining clicks as singles
                    for c in clicks[2:]:
                        cfb, _, cx, cy, cfa, clvl, cav = c
                        emit(cfb, cfa, cav, cav, clvl, clvl,
                             f"Probing click at ({cx},{cy}).",
                             "act.click_at", {"x": cx, "y": cy}, 1)
            else:
                think = f"Probing click at ({x},{y})."
                emit(fb, fa, avail_b, av_a, lvl_b, lvl_a, think,
                     "act.click_at", {"x": x, "y": y}, 1)
            step_no += len(clicks)
            i = j
        else:
            # keyboard run
            j = i
            while j < n and rec[j][1] not in (0, 6):
                j += 1
            run = rec[i:j]
            aids = [r[1] for r in run]
            if len(set(aids)) >= 3 and len(aids) <= 9:
                think = rng.choice([
                    "I don't know what the actions do here. Running a sweep.",
                    f"{len(set(aids))} actions to learn. Sweeping to map effects."])
                emit(fb, run[-1][4], avail_b, run[-1][6], lvl_b, run[-1][5],
                     think, "act.sweep", {}, len(run))
            else:
                # per-action, with painter detection
                for c in run:
                    cfb, caid, _, _, cfa, clvl, cav = c
                    if is_painter(cfb, cfa):
                        think = (f"Action {caid} only paints new pixels, never changes "
                                 f"existing structure — painter-like. Demoting it.")
                    else:
                        think = rng.choice([f"Trying action {caid}.",
                                            f"Probing action {caid}."])
                    emit(cfb, cfa, cav, cav, clvl, clvl, think,
                         "act.try", {"action": caid}, 1)
            step_no += len(run)
            i = j
    return traces


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    rng = random.Random(1234)
    n_all = 0
    with open(OUT, "w") as f:
        for g in GAMES:
            rec, info = run_game(g, seed, budget)
            tr = translate(rec, rng)
            kinds = {}
            for t in tr:
                kinds[t["call"]["cell"]] = kinds.get(t["call"]["cell"], 0) + 1
                f.write(json.dumps(t) + "\n")
            n_all += len(tr)
            print(json.dumps({"game": g, "traces": len(tr), "kinds": kinds,
                              "levels": info["levels"],
                              "score": info["score"]}), flush=True)
    print(f"WROTE {n_all} traces to {OUT}", flush=True)


if __name__ == "__main__":
    main()
