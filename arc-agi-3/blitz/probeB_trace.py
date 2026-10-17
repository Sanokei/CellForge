"""probeB_trace.py: trace an action plan on one env, show ascii before/after + diff regions.

Usage:
  python probeB_trace.py <game> "<plan>" [x0,y0,x1,y1] [--seed N]
Plan tokens comma separated: C x y (click), K id (key), R (reset), W (no-op / observe).
Crop shown as ascii. Prints: level, state, n changed, changed pixel bboxes per colour delta.
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
def ascii_frame(f):
    return "\n".join("".join(CHARS[min(int(v), 15)] for v in row) for row in f)

def diff_summary(f0, f1):
    d = f0 != f1
    n = int(d.sum())
    if n == 0:
        return "no pixel change"
    ys, xs = np.nonzero(d)
    out = f"{n} px changed, bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]"
    ch = {}
    for y, x in zip(ys, xs):
        ch[(int(f0[y, x]), int(f1[y, x]))] = ch.get((int(f0[y, x]), int(f1[y, x])), 0) + 1
    out += " transitions: " + ", ".join(f"{a}->{b}:{c}" for (a, b), c in sorted(ch.items()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game"); ap.add_argument("plan")
    ap.add_argument("--crop", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ascii", action="store_true")
    a = ap.parse_args()
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=a.seed, scorecard_id=card)
    r = env.observation_space
    crop = tuple(map(int, a.crop.split(","))) if a.crop else None
    def show(f, tag):
        if a.ascii:
            g = f[crop[1]:crop[3], crop[0]:crop[2]] if crop else f
            print(f"--- {tag} ---"); print(ascii_frame(g))
    f0 = np.asarray(r.frame[-1], dtype=np.uint8)
    show(f0, "initial")
    print(f"avail {list(r.available_actions)} level {r.levels_completed}")
    for i, tok in enumerate(a.plan.split(",")):
        t = tok.strip().split()
        prev = f0
        if t[0] == "C":
            r = env.step(GameAction.ACTION6, data={"x": int(t[1]), "y": int(t[2])})
        elif t[0] == "K":
            r = env.step(GameAction.from_id(int(t[1])))
        elif t[0] == "R":
            r = env.step(GameAction.RESET)
        f0 = np.asarray(r.frame[-1], dtype=np.uint8)
        print(f"[{i}] {' '.join(t)}: level {r.levels_completed} state {r.state} avail {list(r.available_actions)}")
        print("    " + diff_summary(prev, f0))
        show(f0, tok.strip())
    print("FINAL level", r.levels_completed, "state", r.state)

main()
