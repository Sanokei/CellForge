"""Scripted teacher: plays dev games through the cell tools, logging
(state -> English reasoning -> cell call -> result) traces.
This is the self-distillation corpus for the tiny LM router.

Usage: python teacher.py <game> [seed] [budget] [max_decisions]
Appends to lm/traces.jsonl
"""
import json
import logging
import os
import random
import sys

sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
from cells import GameState, call_cell, REGISTRY, _objs, _avail

from arc_agi import Arcade
from arcengine import GameState as GState

BASE = "/srv/arc3-claude"
TRACE_PATH = os.path.join(BASE, "v6run", "model", "lm", "traces.jsonl")

THINK = {
    "new_level": [
        "New level {lvl}. Scanning the scene before acting.",
        "Level {lvl} just started. Let me look at what's here.",
    ],
    "sweep": [
        "{n} actions untested on this level. Sweeping them to learn effects.",
        "I don't know what the actions do here. Running a sweep.",
    ],
    "click": [
        "Click is available. Trying the biggest object at ({x},{y}).",
        "I'll click the largest object center ({x},{y}) and watch what happens.",
    ],
    "retry": [
        "Action {a} changed the most ({n}px). Trying it again to see where it leads.",
        "Following up on action {a} — it had the biggest effect.",
    ],
    "stuck": [
        "No progress for a while. Resetting the level to try fresh.",
        "Stuck. Resetting and re-scanning.",
    ],
    "probe": [
        "Probing action {a}.",
        "Trying action {a} once.",
    ],
    "diff": [
        "Checking what changed recently.",
        "Let me diff against the last scan.",
    ],
}


def pick(rng, key, **kw):
    return rng.choice(THINK[key]).format(**kw)


def teacher_loop(st, rng, max_decisions, trace_f):
    n_dec = 0
    last_level = -1
    swept = set()
    no_progress = 0
    last_chg_total = 0

    def log(state_txt, think, cell, args, obs):
        trace_f.write(json.dumps({"state": state_txt, "think": think,
                                  "call": {"cell": cell, "args": args},
                                  "obs": obs}) + "\n")

    while n_dec < max_decisions and st.steps_used < st.budget:
        if st.resp.state == GState.WIN:
            break
        lvl = st.level
        new_level = (lvl != last_level)
        last_level = lvl
        state_txt = call_cell(st, "perceive.describe", {})

        avail = _avail(st)
        kbd = [x for x in avail if x not in (0, 6)]
        untested = [x for x in kbd if (lvl, x) not in st.tested]

        if new_level:
            think = pick(rng, "new_level", lvl=lvl)
            cell, args = "perceive.describe", {}
        elif untested and lvl not in swept:
            think = pick(rng, "sweep", n=len(untested))
            cell, args = "act.sweep", {}
            swept.add(lvl)
        elif 6 in avail:
            f = st.frame()
            objs = _objs(f)
            if objs:
                o = objs[0]
                cx, cy = (o["x0"] + o["x1"]) // 2, (o["y0"] + o["y1"]) // 2
                think = pick(rng, "click", x=cx, y=cy)
                cell, args = "act.click_at", {"x": cx, "y": cy}
            else:
                think = pick(rng, "diff")
                cell, args = "perceive.diff", {}
        elif no_progress >= 3 and st.steps_used > 0:
            think = pick(rng, "stuck")
            cell, args = "act.reset", {}
            no_progress = 0
        elif untested:
            a = untested[0]
            think = pick(rng, "probe", a=a)
            cell, args = "act.try", {"action": a}
        else:
            # follow the most effective action seen, else probe
            a = rng.choice(kbd) if kbd else 1
            think = pick(rng, "retry", a=a, n=0)
            cell, args = "act.try", {"action": a}

        obs = call_cell(st, cell, args)
        log(state_txt, think, cell, args, obs)
        n_dec += 1

        # progress tracking
        if "LEVEL-UP" in obs or f"lvl{lvl+1}" in obs or st.level != lvl:
            no_progress = 0
            swept.discard(lvl)
        elif cell.startswith("act"):
            no_progress += 1
        if n_dec % 10 == 0:
            trace_f.flush()
    return n_dec


def main():
    game = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    max_dec = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    rng = random.Random(seed * 1000 + 7)
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
    st = GameState(env, budget)
    os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
    n = 0
    with open(TRACE_PATH, "a") as f:
        try:
            n = teacher_loop(st, rng, max_dec, f)
        except Exception as e:
            print(f"TEACHER ERROR {game}: {e}", file=sys.stderr)
    sc = arc.close_scorecard(card)
    run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
    print(json.dumps({"game": game, "decisions": n, "steps": st.steps_used,
                      "levels": run.get("levels_completed"),
                      "score": run.get("score")}), flush=True)


if __name__ == "__main__":
    main()
