"""probeB_clear.py: click the center of every component of given color(s), report level.
Usage: python probeB_clear.py <game> <colors> [--seed 0]   (colors like 14 or 14,11)
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
sys.path.insert(0, "/srv/arc3-claude/v6run")
from cells import components

ap = argparse.ArgumentParser()
ap.add_argument("game"); ap.add_argument("colors"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
want = set(map(int, a.colors.split(",")))
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)
env = arc.make(info.game_id, seed=a.seed, scorecard_id=arc.create_scorecard())
r = env.observation_space
f = np.asarray(r.frame[-1], dtype=np.uint8)
tgts = [c for c in components(f) if c["color"] in want]
print(f"{len(tgts)} targets")
for c in tgts:
    r = env.step(GameAction.ACTION6, data={"x": c["x"], "y": c["y"]})
    print(f"click ({c['x']},{c['y']}) color={c['color']} size={c['size']} -> level {r.levels_completed} state {r.state}")
    if r.levels_completed > 0 or r.state == GameState.WIN:
        print("LEVEL COMPLETE / WIN")
        break
