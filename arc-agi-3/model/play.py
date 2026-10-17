"""End-to-end eval: play dev games with the tiny neural model through the
official ARC-AGI framework + scorecard.

Usage: python play.py <model.npz> <game1> [game2 ...] [--seed S] [--budget B]
Prints RESULT JSON per game + summary table.
"""
import argparse
import json
import logging
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, N_ACTIONS
from mlp import load, predict
from arc_agi import Arcade
from arcengine import GameAction, GameState

BASE = "/srv/arc3-claude"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("games", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=3000)
    a = ap.parse_args()
    ps = load(a.model)
    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)

    results = []
    for game in a.games:
        arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                     recordings_dir=f"{BASE}/recordings")
        info = next(e for e in arc.get_environments()
                    if e.game_id.split("-")[0] == game)
        card = arc.create_scorecard()
        env = arc.make(info.game_id, seed=a.seed, scorecard_id=card)
        t0 = time.time()
        resp = env.observation_space
        prev = None
        used = 0
        while used < a.budget and resp.state != GameState.WIN:
            if resp.state == GameState.GAME_OVER:
                resp = env.step(GameAction.RESET)
                used += 1
                prev = None
                continue
            frame = np.asarray(resp.frame[-1], dtype=np.uint8)
            feats = extract_features(frame, prev, resp.available_actions)
            pa, px, py = predict(ps, feats[None])
            pa = pa[0].copy()
            avail = set(int(x) for x in resp.available_actions)
            mask = np.zeros(N_ACTIONS)
            for i in avail:
                if 0 <= i < N_ACTIONS:
                    mask[i] = 1.0
            pa = pa * mask
            act = int(np.argmax(pa)) if mask.sum() else 0
            if act == 6:
                x, y = int(np.argmax(px[0])), int(np.argmax(py[0]))
                resp = env.step(GameAction.ACTION6, data={"x": x, "y": y})
            elif act == 0:
                resp = env.step(GameAction.RESET)
            else:
                try:
                    resp = env.step(GameAction.from_id(act))
                except Exception:
                    resp = env.step(GameAction.RESET)
            prev = frame
            used += 1
            if resp is None:
                break
        dt = time.time() - t0
        sc = arc.close_scorecard(card)
        run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
        out = {"agent": f"tinynet:{os.path.basename(a.model)}", "game": game,
               "seed": a.seed, "budget": a.budget, "used": used,
               "levels": run.get("levels_completed"),
               "n_levels": len(info.baseline_actions),
               "score": run.get("score"), "actions": run.get("actions"),
               "state": str(run.get("state")), "secs": round(dt, 1),
               "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "tags": info.tags}
        results.append(out)
        print("RESULT " + json.dumps(out), flush=True)

    tot_l = sum(r["levels"] or 0 for r in results)
    mean_s = float(np.mean([r["score"] or 0 for r in results]))
    print(f"SUMMARY games={len(results)} levels={tot_l} mean_score={mean_s:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
