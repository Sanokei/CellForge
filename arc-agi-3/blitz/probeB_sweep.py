"""probeB_sweep.py: click each point of a grid on a FRESH env; report effect hash + changed count.

Usage: python probeB_sweep.py <game> x0,y0,x1,y1,step [--seed N]
For each click: prints x,y,changed,level_up,frame-hash of the block region.
Groups identical outcomes.
"""
import argparse, hashlib, logging, os, sys
sys.path.insert(0, "/srv/arc3-claude/v6run")
from arc_agi import Arcade
from arcengine import GameAction, GameState
import numpy as np

BASE = "/srv/arc3-claude"
lg = logging.getLogger("quiet"); lg.addHandler(logging.NullHandler()); lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)
arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files", recordings_dir=f"{BASE}/recordings")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game"); ap.add_argument("region")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    x0, y0, x1, y1, st = map(int, a.region.split(","))
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)
    groups = {}
    for y in range(y0, y1, st):
        for x in range(x0, x1, st):
            card = arc.create_scorecard()
            env = arc.make(info.game_id, seed=a.seed, scorecard_id=card)
            f0 = np.asarray(env.observation_space.frame[-1], dtype=np.uint8)
            r = env.step(GameAction.ACTION6, data={"x": x, "y": y})
            f1 = np.asarray(r.frame[-1], dtype=np.uint8)
            ch = int((f0 != f1).sum())
            h = hashlib.blake2b(f1.tobytes(), digest_size=6).hexdigest()
            groups.setdefault((ch, r.levels_completed, h), []).append((x, y))
    for (ch, lv, h), pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"n={len(pts)} changed={ch} level_up={lv} hash={h} pts={pts[:12]}{'...' if len(pts)>12 else ''}")

main()
