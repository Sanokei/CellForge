"""Dev-only probe: print tags, actions, frame and per-action pixel change for one game."""
import sys, logging
sys.path.insert(0, "/srv/arc3-claude/agent")
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

g = sys.argv[1]
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
a = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files", recordings_dir="/srv/arc3-claude/recordings")
info = next(e for e in a.get_environments() if e.game_id.startswith(g))
c = a.create_scorecard(); env = a.make(info.game_id, scorecard_id=c)
r = env.observation_space
print(info.tags, info.baseline_actions, r.available_actions, len(r.frame))
f = np.asarray(r.frame[-1])
print("colors", dict(zip(*np.unique(f, return_counts=True))))


def show(f):
    ch = ".123456789abcdef"
    for row in f:
        print("".join(ch[v] for v in row))


if len(sys.argv) > 2:
    show(f)
for act in r.available_actions:
    if act in (0, 6):
        continue
    r2 = env.step(GameAction.from_id(act)); f2 = np.asarray(r2.frame[-1]); d = f != f2
    ys, xs = np.nonzero(d)
    print("act", act, "nframes", len(r2.frame), "changed", d.sum(),
          (ys.min(), xs.min(), ys.max(), xs.max()) if d.any() else None, r2.state.name, r2.levels_completed)
    f = f2
    if len(sys.argv) > 2:
        show(f)
