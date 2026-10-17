"""probeB_cursor.py: for each keyboard action, report cursor displacement.
Cursor color given; prints centroid before/after and delta, plus changed count.
Usage: python probeB_cursor.py <game> <cursor_color> [--seed 0]
"""
import argparse, logging, os, sys
sys.path.insert(0, "/srv/arc3-claude/v6run")
from arc_agi import Arcade
from arcengine import GameAction, GameState
import numpy as np

BASE = "/srv/arc3-claude"
lg = logging.getLogger("quiet"); lg.addHandler(logging.NullHandler()); lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)
arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files", recordings_dir=f"{BASE}/recordings")

ap = argparse.ArgumentParser()
ap.add_argument("game"); ap.add_argument("color", type=int); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)

def centroid(f):
    ys, xs = np.nonzero(f == a.color)
    return (round(xs.mean(), 2), round(ys.mean(), 2), len(xs))

for aid in list(arc.make(info.game_id, seed=a.seed, scorecard_id=arc.create_scorecard()).observation_space.available_actions):
    if aid in (0, 6):
        continue
    env = arc.make(info.game_id, seed=a.seed, scorecard_id=arc.create_scorecard())
    f0 = np.asarray(env.observation_space.frame[-1], dtype=np.uint8)
    c0 = centroid(f0)
    r = env.step(GameAction.from_id(aid))
    f1 = np.asarray(r.frame[-1], dtype=np.uint8)
    c1 = centroid(f1)
    ch = int((f0 != f1).sum())
    print(f"key {aid}: cursor {c0} -> {c1} delta=({round(c1[0]-c0[0],1)},{round(c1[1]-c0[1],1)}) changed={ch} level={r.levels_completed}")
