"""Reproducible synthetic-only benchmark against an explicit generator checkout."""
import argparse
from collections import defaultdict
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .agent import GeneAgent


def load_generator(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / "research" / "claude_arc3"))
    return importlib.import_module("gan.games")


def rollout(game, games, budget, seed, disabled=(), random_agent=False):
    from arcengine import GameAction
    import random
    rng = random.Random(seed)
    env = games.SynthEnv(game)
    agent = GeneAgent(budget, seed, disabled)
    latest, history = env.observation_space, []
    while not agent.is_done(history, latest):
        if random_agent:
            aid = 0 if latest.state.name == "GAME_OVER" else rng.choice(latest.available_actions)
            act = GameAction.from_id(aid)
            if aid == 6:
                act.set_data({"x": rng.randrange(64), "y": rng.randrange(64)})
            agent.actions += 1
        else:
            act = agent.choose_action(history, latest)
        latest = env.step(act)
        history.append(latest)
    return dict(win=latest.state.name == "WIN", levels=latest.levels_completed,
                used=env.used, deaths=env.game_overs, score=env.score(), events=dict(agent.events))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator-root", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ablations", action="store_true")
    args = ap.parse_args()
    games = load_generator(args.generator_root)
    commit = subprocess.check_output(["git", "-C", args.generator_root, "rev-parse", "HEAD"], text=True).strip()
    rows = []
    t0 = time.time()
    variants = ["genes", "random"] + (["no_" + x for x in ("effects", "terrain", "goals", "keydoor", "push", "hazard_motion")] if args.ablations else [])
    for family in games.FAMILIES:
        for seed in range(args.seeds):
            genome = dict(family=family, n=5 + seed % 5, density=0.05 * (seed % 4),
                          extra=seed % 5, n_levels=2, perm=(seed * 7) % 24, palette=1009 * seed + 73,
                          noop=seed % 2)
            game = games.make_game(genome, seed + 43000)
            assert games.verify_oracle(game), "generator oracle replay failed"
            for variant in variants:
                disabled = (variant[3:],) if variant.startswith("no_") else ()
                result = rollout(game, games, args.budget, seed, disabled, variant == "random")
                rows.append(dict(family=family, seed=seed, genome=genome, variant=variant,
                                 opt=game.opt, **result))
        print(f"finished {family}", flush=True)
    summary = defaultdict(lambda: dict(episodes=0, wins=0, levels=0, used=0, deaths=0, score=0.0))
    for row in rows:
        s = summary[row["family"] + "/" + row["variant"]]
        s["episodes"] += 1
        for k in ("levels", "used", "deaths", "score"):
            s[k] += row[k]
        s["wins"] += row["win"]
    for s in summary.values():
        s["mean_score"] = s.pop("score") / s["episodes"]
    sources = list(Path(__file__).parent.glob("*.py"))
    sources += list((Path(args.generator_root) / "research/claude_arc3/gan").glob("*.py"))
    output = dict(generator_commit=commit, generator_root=str(Path(args.generator_root).resolve()),
                  source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  budget=args.budget, seeds=args.seeds, seconds=round(time.time() - t0, 2),
                  summary=dict(summary), episodes=rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
