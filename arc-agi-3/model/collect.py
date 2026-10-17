"""Behavioral-cloning data collection: run the heuristic explorer on dev games,
recording (frame, available_actions, action_id, click_x, click_y, levels_completed)
at every step. Saves model/data/<game>.npz.

Usage: python collect.py <game> [seed] [budget]
"""
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, "/srv/arc3-claude/v6run")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_agi import Arcade
from arcengine import GameAction, GameState
from explorer5 import Explorer

BASE = "/srv/arc3-claude"
OUT = os.path.join(BASE, "v6run", "model", "data")
os.makedirs(OUT, exist_ok=True)


def action_id(a):
    if a == GameAction.RESET:
        return 0
    if a == GameAction.ACTION6:
        return 6
    return int(a.value) if hasattr(a, "value") else int(a)


def main():
    game = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
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

    rec = {"frames": [], "avails": [], "acts": [], "cxs": [], "cys": [],
           "levels": []}
    cur = {"obs": env.observation_space}
    orig_step = env.step

    def rec_step(action, data=None):
        obs = cur["obs"]
        fr = np.asarray(obs.frame[-1], dtype=np.uint8)
        rec["frames"].append(fr)
        rec["avails"].append(list(obs.available_actions))
        aid = action_id(action)
        rec["acts"].append(aid)
        if aid == 6 and data:
            rec["cxs"].append(int(data.get("x", 0)))
            rec["cys"].append(int(data.get("y", 0)))
        else:
            rec["cxs"].append(-1)
            rec["cys"].append(-1)
        r = orig_step(action) if data is None else orig_step(action, data=data)
        rec["levels"].append(int(r.levels_completed))
        cur["obs"] = r
        return r

    env.step = rec_step
    ex = Explorer(env, budget, seed=seed, log=lambda *a: None,
                  flags=("pr", "nm", "ck", "gv"))
    try:
        ex.run()
    except Exception as e:
        print(f"EXPLORER ERROR {game}: {e}", file=sys.stderr)

    n = len(rec["acts"])
    av = rec["avails"]
    maxa = max((len(v) for v in av), default=0)
    av_arr = np.full((n, maxa), -1, dtype=np.int64)
    for i, v in enumerate(av):
        av_arr[i, :len(v)] = v
    np.savez_compressed(
        os.path.join(OUT, f"{game}.npz"),
        frames=np.stack(rec["frames"]) if n else np.zeros((0, 64, 64), np.uint8),
        avails=av_arr,
        acts=np.array(rec["acts"], dtype=np.int64),
        cxs=np.array(rec["cxs"], dtype=np.int64),
        cys=np.array(rec["cys"], dtype=np.int64),
        levels=np.array(rec["levels"], dtype=np.int64),
    )
    sc = arc.close_scorecard(card)
    run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
    print(json.dumps({"game": game, "steps": n,
                      "levels": run.get("levels_completed"),
                      "score": run.get("score")}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
