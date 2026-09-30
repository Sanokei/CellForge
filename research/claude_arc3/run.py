"""Run one agent on one public ARC-AGI-3 game and print a JSON result line.

usage: python run.py --agent random|explorer|explorer2 --game ls20 --budget 3000 [--seed 0]
Runs locally (NORMAL mode: game source fetched once by arc_agi, stepped in-process).
Never logs the anonymous API key (arc_agi's own logger is silenced).
"""
import argparse, json, logging, os, random, resource, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_agi import Arcade
from arcengine import GameAction, GameState


def random_agent(env, budget, seed):
    rng = random.Random(seed)
    used = 0
    resp = env.observation_space
    while used < budget and resp.state != GameState.WIN:
        if resp.state == GameState.GAME_OVER:
            resp = env.step(GameAction.RESET)
        else:
            a = rng.choice([i for i in resp.available_actions if i != 0])
            if a == 6:
                resp = env.step(GameAction.ACTION6, data={"x": rng.randrange(64), "y": rng.randrange(64)})
            else:
                resp = env.step(GameAction.from_id(a))
        used += 1
        if resp is None:
            break
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="explorer")
    ap.add_argument("--game", required=True)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    lg = logging.getLogger("quiet"); lg.addHandler(logging.NullHandler()); lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    base = os.environ.get("ARC3_DIR", "/srv/arc3-claude")
    arc = Arcade(logger=lg, environments_dir=f"{base}/environment_files", recordings_dir=f"{base}/recordings")
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == a.game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=a.seed, scorecard_id=card)
    t0 = time.time()
    events = []
    if a.agent == "random":
        used = random_agent(env, a.budget, a.seed)
    else:
        if a.agent.startswith("v5"):
            from explorer5 import Explorer as E5
            flags = a.agent.split(":")[1].split(",") if ":" in a.agent else []
        else:
            from explorer import Explorer
        logf = (lambda *x: print(*x, file=sys.stderr)) if a.verbose else (lambda *x: events.append(" ".join(map(str, x))))
        if a.agent.startswith("v5"):
            ex = E5(env, a.budget, seed=a.seed, log=logf, flags=flags)
        else:
            ex = Explorer(env, a.budget, seed=a.seed, log=logf, use_action_stats=(a.agent == "explorer2"))
        used = ex.run()
    dt = time.time() - t0
    sc = arc.close_scorecard(card)
    run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
    out = dict(agent=a.agent, game=a.game, seed=a.seed, budget=a.budget, used=used,
               levels=run.get("levels_completed"), n_levels=len(info.baseline_actions),
               score=run.get("score"), actions=run.get("actions"), state=str(run.get("state")),
               level_actions=run.get("level_actions"), secs=round(dt, 1),
               peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               tags=info.tags, events=events[-12:],
               level_used=getattr(ex, 'level_paths', None) if a.agent != 'random' else None)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
