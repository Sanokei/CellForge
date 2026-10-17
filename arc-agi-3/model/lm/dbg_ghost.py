import sys
sys.path.insert(0, "/srv/arc3-claude/v6run")
from cells import components, salience_order, last_frame
import numpy as np, logging
from arc_agi import Arcade
from arcengine import GameAction
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "lf52")
env = arc.make(info.game_id, seed=0, scorecard_id=arc.create_scorecard())
resp = env.observation_space
f = last_frame(resp)
vals, cnts = np.unique(f, return_counts=True); bg = int(vals[np.argmax(cnts)])
comps = components(f)
picks = [c for c in salience_order(comps, bg)
         if c["color"] != bg and c["size"] <= 64][:6]
for p in picks:
    fb = last_frame(resp).copy()
    before = {(c["x"], c["y"], c["color"]) for c in components(fb)}
    r = env.step(GameAction.ACTION6, data={"x": p["x"], "y": p["y"]})
    f1 = last_frame(r)
    d = int((fb != f1).sum())
    ghosts = [c for c in components(f1)
              if (c["x"], c["y"], c["color"]) not in before
              and c["size"] <= 32 and c["color"] != bg]
    gi = [(g["x"], g["y"], g["color"], g["size"]) for g in ghosts]
    print("click (%d,%d) col%d sz%d: %dpx ghosts=%s lvl=%s"
          % (p["x"], p["y"], p["color"], p["size"], d, gi,
             r.levels_completed))
    resp = r
