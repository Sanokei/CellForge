"""Run one blitz agent on one public ARC-AGI-3 game and print a JSON result line.

usage: python run_blitz.py --agent blitz1[:flags] --game ls20 --budget 3000 [--seed 0]
Mirrors v6run/run.py but dispatches blitz agents (with game tags passed in).
"""
import argparse, json, logging, os, random, resource, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
V6RUN = os.path.dirname(HERE)
sys.path.insert(0, V6RUN)
sys.path.insert(0, HERE)
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
    ap.add_argument("--agent", default="blitz1")
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
    logf = (lambda *x: print(*x, file=sys.stderr)) if a.verbose else (lambda *x: events.append(" ".join(map(str, x))))
    if a.agent.startswith("blitz"):
        from blitz_agent import AGENTS
        try:
            import blitz_w2  # noqa: F401  (registers blitz6/7/8)
        except ImportError:
            pass
        name = a.agent.split(":")[0]
        flags = a.agent.split(":")[1].split(",") if ":" in a.agent else ["pr", "nm", "ck", "gv", "d6"]
        ex = AGENTS[name](env, a.budget, seed=a.seed, log=logf, flags=flags, tags=info.tags)
    elif a.agent == "random":
        ex = None
        used = random_agent(env, a.budget, a.seed)
    else:
        raise SystemExit(f"unknown blitz agent {a.agent}")
    if ex is not None:
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
               director=(ex.director.stats if getattr(ex, "director", None) else None),
               level_used=getattr(ex, "level_paths", None) if ex is not None else None)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
