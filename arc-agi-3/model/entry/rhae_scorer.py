"""True RHAE scorer for the CellForge evolution loop.

Uses the arc_agi framework's OWN canonical scoring (EnvironmentScoreCalculator):
  per completed level: ((baseline_actions / actions_taken)^2) * 100, cap 115
  incomplete levels: 0
  environment score: level-index-weighted mean of level scores
                     (later levels weigh more), capped by completed weight
  RHAE total: mean of environment scores (0..~115 scale; leaderboard reports %)

The framework tracks actions_by_level itself via the scorecard, so action
accounting is authoritative (every env.step counts). We just play the game
through the framework with a scorecard and read the closed run.

Usage:
  python rhae_scorer.py --agent <agent.py> --games lf52,s5i5 --seeds 0,1 [--budget 3000] [--maxdec 300] [--token-budget 500]
  python rhae_scorer.py --lm <model.pt> --games lf52,s5i5 --seeds 0 [--budget 3000] [--maxdec 60]
Prints RESULT JSON lines + SUMMARY. Writes JSON to --out if given.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/srv/arc3-claude/v6run/model")
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")

ARC3 = "/srv/arc3-claude"
REC = "/srv/arc3-claude/recordings"

lg = logging.getLogger("rhae")
lg.addHandler(logging.NullHandler())
lg.propagate = False
logging.getLogger("arc_agi").setLevel(logging.WARNING)


def get_arc():
    from arc_agi import Arcade
    return Arcade(logger=lg, environments_dir=f"{ARC3}/environment_files",
                  recordings_dir=REC)


def close_run(arc, card):
    sc = arc.close_scorecard(card)
    if sc and sc.environments:
        runs = sc.environments[0].runs
        if runs:
            return runs[-1].model_dump()
    return {}


def paper_rhae(level_actions, levels_completed, baselines):
    """Exact ARC-AGI-3 tech-report spec (arXiv 2603.24621, sec 4.1).

    Per completed level l (1-based): S_l = min(1.0, h_l / a_l)^2.
    Incomplete levels: 0. Environment: E = sum(l * S_l) / (n(n+1)/2).
    Returns (E in 0..1, per-level scores list).
    Note: uses the framework's shipped baseline_actions as h_l
    (the paper defines it as second-best human; we take shipped as given).
    """
    n = len(baselines or [])
    if n == 0:
        return 0.0, []
    lc = min(levels_completed or 0, n)
    la = level_actions or []
    scores = []
    for l in range(1, n + 1):
        if l <= lc:
            a = la[l - 1] if l - 1 < len(la) else 0
            h = baselines[l - 1]
            s = min(1.0, h / a) ** 2 if a and a > 0 else 0.0
        else:
            s = 0.0
        scores.append(round(s, 6))
    env = sum(l * s for l, s in enumerate(scores, 1)) / (n * (n + 1) / 2)
    return round(env, 6), scores


def score_decide_agent(agent_path, games, seeds, budget=3000, maxdec=300,
                       token_budget=None):
    import agent_api
    from arcengine import GameState as GState  # noqa: F401 (api parity)
    arc = get_arc()
    mod = agent_api.load_agent(agent_path)
    results = []
    for game in games:
        info = next(e for e in arc.get_environments()
                    if e.game_id.split("-")[0] == game)
        for seed in seeds:
            card = arc.create_scorecard()
            env = arc.make(info.game_id, seed=seed, scorecard_id=card)
            t0 = time.time()
            res, _trace = agent_api.run_agent_game(
                mod, game, seed, budget, maxdec, arc, info, card, env,
                token_budget=token_budget)
            run = close_run(arc, card)
            dt = time.time() - t0
            baselines = list(info.baseline_actions or [])
            env_rhae, lvl_scores = paper_rhae(
                run.get("level_actions"), run.get("levels_completed"), baselines)
            results.append({
                "agent": getattr(mod, "NAME", agent_path),
                "game": game, "seed": seed,
                "rhae": round(env_rhae * 100, 4),  # paper-spec RHAE %
                "framework_score": run.get("score", 0.0),
                "levels_completed": run.get("levels_completed", 0),
                "n_levels": len(baselines),
                "level_scores": lvl_scores,
                "level_actions": run.get("level_actions"),
                "level_baselines": baselines,
                "env_actions": run.get("actions"),
                "decisions": res["decisions"], "parse_fails": res["parse_fails"],
                "think_tokens": res["think_tokens"], "capped": res["capped"],
                "secs": round(dt, 1),
            })
            print("RESULT " + json.dumps(results[-1]), flush=True)
    return results


def score_lm(model_pt, games, seeds, budget=3000, maxdec=60):
    import torch
    from arcengine import GameState as GState
    sys.path.insert(0, os.path.join(ARC3, "v6run", "model", "lm"))
    from cells import GameState, call_cell, REGISTRY
    from tok import load_vocab, encode, decode
    from gpt import GPT
    import eval_lm as ev

    torch.manual_seed(0)
    vocab, stoi = load_vocab(os.path.join(ARC3, "v6run", "model", "lm", "vocab.json"))
    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
    model = GPT(**ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    n_params = model.param_count()
    arc = get_arc()
    results = []
    for game in games:
        info = next(e for e in arc.get_environments()
                    if e.game_id.split("-")[0] == game)
        for seed in seeds:
            card = arc.create_scorecard()
            env = arc.make(info.game_id, seed=seed, scorecard_id=card)
            st = GameState(env, budget)
            t0 = time.time()
            hist, ndec, parse_fails, gen_tokens = [], 0, 0, 0
            while ndec < maxdec and st.steps_used < budget:
                if st.resp.state == GState.WIN:
                    break
                state_txt = call_cell(st, "perceive.describe", {})
                blocks = hist[-2:]
                prompt = "".join(b + "\n" for b in blocks)
                prompt += f"STATE {state_txt}\nTHINK "
                idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
                if idx.shape[1] > 160:
                    idx = idx[:, -160:]
                with torch.no_grad():
                    out = model.generate(idx, max_new_tokens=64,
                                         temperature=0.7, top_k=40)
                gen = decode(out[0].tolist()[len(idx[0]):], vocab)
                gen_tokens += len(out[0]) - len(idx[0])
                name, args = ev.parse_call(gen)
                if name is None:
                    parse_fails += 1
                    name, args = "perceive.describe", {}
                obs = call_cell(st, name, args)
                think_txt = gen.split("CALL")[0].strip()
                hist.append(f"STATE {state_txt}\nTHINK {think_txt}\n"
                            f"CALL {json.dumps({'cell': name, 'args': args})}\n"
                            f"OBS {obs[:160]}")
                ndec += 1
            run = close_run(arc, card)
            dt = time.time() - t0
            baselines = list(info.baseline_actions or [])
            env_rhae, lvl_scores = paper_rhae(
                run.get("level_actions"), run.get("levels_completed"), baselines)
            results.append({
                "agent": f"lmrouter:{os.path.basename(model_pt)}",
                "params": n_params,
                "game": game, "seed": seed,
                "rhae": round(env_rhae * 100, 4),  # paper-spec RHAE %
                "framework_score": run.get("score", 0.0),
                "levels_completed": run.get("levels_completed", 0),
                "n_levels": len(baselines),
                "level_scores": lvl_scores,
                "level_actions": run.get("level_actions"),
                "level_baselines": baselines,
                "env_actions": run.get("actions"),
                "decisions": ndec, "parse_fails": parse_fails,
                "gen_tokens": gen_tokens,
                "secs": round(dt, 1),
            })
            print("RESULT " + json.dumps(results[-1]), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=None)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--games", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--maxdec", type=int, default=300)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    games = [g.strip() for g in a.games.split(",") if g.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    if a.agent:
        results = score_decide_agent(a.agent, games, seeds, a.budget,
                                     a.maxdec, a.token_budget)
    else:
        results = score_lm(a.lm, games, seeds, a.budget, a.maxdec)
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
