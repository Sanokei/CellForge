"""probeB_diffmap.py: print a 64x64 diff map after a plan: '.' unchanged, new colour char where changed.
Usage: python probeB_diffmap.py <game> "<plan>" [--seed N]
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
CHARS = " .:-=+*#%@abcdefgh"

ap = argparse.ArgumentParser()
ap.add_argument("game"); ap.add_argument("plan"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)
card = arc.create_scorecard()
env = arc.make(info.game_id, seed=a.seed, scorecard_id=card)
r = env.observation_space
f0 = np.asarray(r.frame[-1], dtype=np.uint8)
for tok in a.plan.split(","):
    t = tok.strip().split()
    if t[0] == "C":
        r = env.step(GameAction.ACTION6, data={"x": int(t[1]), "y": int(t[2])})
    elif t[0] == "K":
        r = env.step(GameAction.from_id(int(t[1])))
    elif t[0] == "R":
        r = env.step(GameAction.RESET)
f1 = np.asarray(r.frame[-1], dtype=np.uint8)
print("level", r.levels_completed, "state", r.state)
d = f0 != f1
for y in range(64):
    print("".join("." if not d[y, x] else CHARS[min(int(f1[y, x]), 15)] for x in range(64)))
