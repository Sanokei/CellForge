"""Dev debug: run v6 on one game and dump director beliefs at the end."""
import sys, os, logging
sys.path.insert(0, "/srv/arc3-claude/v6run")
from arc_agi import Arcade
from explorer5 import Explorer
g, budget = sys.argv[1], int(sys.argv[2])
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files", recordings_dir="/srv/arc3-claude/recordings")
info = next(e for e in arc.get_environments() if e.game_id.startswith(g))
env = arc.make(info.game_id, scorecard_id=arc.create_scorecard())
ex = Explorer(env, budget, log=lambda *a: print(*a), flags=["pr", "nm", "ck", "gv", "d6"])
ex.run()
d = ex.director
print("avatar", d.fx.avatar())
print("trials", dict(d.fx.trials), "noops", dict(d.fx.noops))
print("walls free", sorted(d.walls.free), "blocked", sorted(d.walls.blocked), "hazard", sorted(d.walls.hazard))
print("goals win", dict(d.goals.win), "vanish", dict(d.goals.vanish), "inert", dict(d.goals.inert), "solid", dict(d.goals.solid))
print("stats", d.stats, "levels", ex.level)
print(getattr(d, "win_log", None))
