"""TRACK B eval driver: true paper-spec RHAE for decide() agents.

Self-contained (does not depend on other tracks' files): drives the agent
through agent_api, reads the framework scorecard run, scores with the pure
rhae_scorer.environment_score (paper spec: S=min(1,h/a)^2, level-weighted).

Protocol parity with the baselines: --budget 3000 --maxdec 300
--token-budget 500, one run per game per seed, no retries.

Usage: python trackb_eval.py <agent.py> <g1,g2,...> [--seeds 0]
       [--budget 3000] [--maxdec 300] [--token-budget 500] [--out path]
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/srv/arc3-claude/v6run/model")
import agent_api
from rhae_scorer import environment_score
from arc_agi import Arcade

lg = logging.getLogger("trackb")
lg.addHandler(logging.NullHandler())
lg.propagate = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("games")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--maxdec", type=int, default=300)
    ap.add_argument("--token-budget", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    games = [g.strip() for g in a.games.split(",") if g.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    arc = Arcade(logger=lg,
                 environments_dir="/srv/arc3-claude/environment_files",
                 recordings_dir="/srv/arc3-claude/recordings")
    mod = agent_api.load_agent(a.agent)
    results = []
    for game in games:
        info = next(e for e in arc.get_environments()
                    if e.game_id.split("-")[0] == game)
        baselines = list(info.baseline_actions or [])
        for seed in seeds:
            card = arc.create_scorecard()
            env = arc.make(info.game_id, seed=seed, scorecard_id=card)
            t0 = time.time()
            res, _trace = agent_api.run_agent_game(
                mod, game, seed, a.budget, a.maxdec, arc, info, card, env,
                token_budget=a.token_budget)
            sc = arc.close_scorecard(card)
            run = {}
            if sc and sc.environments and sc.environments[0].runs:
                run = sc.environments[0].runs[-1].model_dump()
            la = run.get("level_actions") or []
            lc = run.get("levels_completed") or 0
            per_lvl = {l + 1: la[l] for l in range(min(lc, len(la)))}
            e, scores = environment_score(baselines, per_lvl)
            results.append({
                "agent": getattr(mod, "NAME", a.agent),
                "game": game, "seed": seed,
                "rhae": round(e * 100, 4),
                "levels_completed": lc,
                "n_levels": len(baselines),
                "level_scores": [scores.get(l + 1, 0.0)
                                 for l in range(len(baselines))],
                "level_actions": la,
                "level_baselines": baselines,
                "env_actions": run.get("actions"),
                "decisions": res["decisions"],
                "parse_fails": res["parse_fails"],
                "think_tokens": res["think_tokens"],
                "capped": res["capped"],
                "secs": round(time.time() - t0, 1),
            })
            print("RESULT " + json.dumps(results[-1]), flush=True)
    rhaes = [r["rhae"] or 0 for r in results]
    mean_rhae = sum(rhaes) / max(len(rhaes), 1)
    tot_lv = sum(r["levels_completed"] or 0 for r in results)
    print(f"SUMMARY runs={len(results)} mean_RHAE={mean_rhae:.2f} "
          f"levels={tot_lv}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"results": results, "mean_rhae": mean_rhae,
                       "levels": tot_lv}, f, indent=1)
        print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
