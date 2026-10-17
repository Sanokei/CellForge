"""probeB_cursor2.py: exact cursor pixel coords before/after a plan.
Usage: python probeB_cursor2.py <game> <color> "<plan>" [--seed 0]
Prints sorted pixel coords of <color> before and after.
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
ap.add_argument("game"); ap.add_argument("color", type=int); ap.add_argument("plan")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)
env = arc.make(info.game_id, seed=a.seed, scorecard_id=arc.create_scorecard())
f0 = np.asarray(env.observation_space.frame[-1], dtype=np.uint8)
def pix(f):
    ys, xs = np.nonzero(f == a.color)
    return sorted(zip(xs.tolist(), ys.tolist()))
print("before", pix(f0))
r = env.observation_space
for tok in a.plan.split(","):
    t = tok.strip().split()
    if t[0] == "C":
        r = env.step(GameAction.ACTION6, data={"x": int(t[1]), "y": int(t[2])})
    elif t[0] == "K":
        r = env.step(GameAction.from_id(int(t[1])))
    f1 = np.asarray(r.frame[-1], dtype=np.uint8)
    print("after", " ".join(t), pix(f1), "level", r.levels_completed, "changed", int((f0 != f1).sum()))
    f0 = f1
