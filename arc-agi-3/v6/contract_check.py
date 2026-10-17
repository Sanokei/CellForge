"""Check the is_done()/choose_action() adapter reproduces the direct run (levels cleared after N actions).
usage: contract_check.py GAME N   (run on the server)"""
import logging, sys
sys.path.insert(0, "/srv/arc3-claude/v6run")
sys.path.insert(0, "/srv/arc3-claude/v6run/v6")
from arc_agi import Arcade
from arcengine import GameAction
from agent import Arc3AgentV6

g, n = sys.argv[1], int(sys.argv[2])
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files", recordings_dir="/srv/arc3-claude/recordings")
info = next(e for e in arc.get_environments() if e.game_id.startswith(g))
env = arc.make(info.game_id, scorecard_id=arc.create_scorecard())
agent = Arc3AgentV6(budget=n)
frames, latest = [], env.observation_space
while not agent.is_done(frames, latest):
    a = agent.choose_action(frames, latest)
    data = getattr(a, "action_data", None)
    d = getattr(data, "model_dump", lambda: None)() if data is not None else None
    d = {k: v for k, v in (d or {}).items() if k in ("x", "y")} or None
    latest = env.step(a, data=d) if d else env.step(a)
    frames.append(latest)
print("CONTRACT", g, "actions", agent.actions, "levels", latest.levels_completed)
agent.close()
