"""COMPETE with staking + token budgets (CellForge loop).

Flow per generation:
  1. Researchers declare (stake, token_budget) -> bank.declare_stake
  2. bank.close_staking(gen, seeds)  -- seeds drawn AFTER close (structural)
  3. python compete.py <gen_dir> --gen gen0   (this script)
     - asserts staking closed, uses ledger seeds + token budgets
     - harness meters think-tokens, cuts off at cap
     - bills compute + think tokens, settles payouts, checks eliminations

Usage: python compete.py <gen_dir> --gen gen0 [--games g1,g2] [--budget B] [--maxdec D]
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, "/srv/arc3-claude/v6run/model")
from agent_api import load_agent, run_agent_game, purity_check
import bank

from arc_agi import Arcade

BASE = "/srv/arc3-claude"
BOX_DOLLAR_PER_HR = 0.0089  # Hetzner cx23-class box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("--gen", default="gen0")
    ap.add_argument("--games", default="lf52,s5i5,ls20,g50t,tu93,bp35")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--maxdec", type=int, default=50)
    a = ap.parse_args()
    games = a.games.split(",")
    gen = a.gen

    d = bank.load()
    g = d["generations"].get(gen)
    assert g and g["closed"], f"staking not closed for {gen}; cannot run eval"
    seeds = g["eval_seeds"]
    stakes = g["stakes"]
    print(f"gen {gen}: staking closed, eval seeds {seeds}", flush=True)

    lg = logging.getLogger("qc")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)

    agents = {}
    for name in sorted(os.listdir(a.gen_dir)):
        p = os.path.join(a.gen_dir, name, "agent.py")
        if os.path.exists(p):
            try:
                agents[name] = load_agent(p)
                print(f"loaded agent {name}", flush=True)
            except Exception as e:
                print(f"FAILED to load {name}: {e}", flush=True)

    out_path = os.path.join(a.gen_dir, "compete.jsonl")
    results = []
    think_totals = {}
    with open(out_path, "w") as f:
        for aname, mod in agents.items():
            tbudget = stakes.get(aname, {}).get("token_budget")  # None = house
            think_totals[aname] = 0
            for game in games:
                for seed in seeds:
                    arc = Arcade(logger=lg,
                                 environments_dir=f"{BASE}/environment_files",
                                 recordings_dir=f"{BASE}/recordings")
                    info = next(e for e in arc.get_environments()
                                if e.game_id.split("-")[0] == game)
                    card = arc.create_scorecard()
                    env = arc.make(info.game_id, seed=seed,
                                   scorecard_id=card)
                    try:
                        res, trace = run_agent_game(
                            mod, game, seed, a.budget, a.maxdec,
                            arc, info, card, env, token_budget=tbudget)
                    except Exception as e:
                        res, trace = {"error": str(e)[:200],
                                      "think_tokens": 0}, []
                    sc = arc.close_scorecard(card)
                    run = (sc.environments[0].runs[-1].model_dump()
                           if sc and sc.environments else {})
                    think_totals[aname] += res.get("think_tokens", 0)
                    nrolls = len(games) * len(seeds)
                    row = {
                        "agent": aname, "game": game, "seed": seed,
                        "levels": run.get("levels_completed"),
                        "n_levels": len(info.baseline_actions),
                        "score": run.get("score"),
                        "actions": run.get("actions"),
                        "env_steps": res.get("steps"),
                        "decisions": res.get("decisions"),
                        "think_tokens": res.get("think_tokens"),
                        "token_budget": tbudget,
                        "capped": res.get("capped"),
                        "secs": res.get("secs"),
                        "dollar_per_task": round(
                            (res.get("secs") or 0) / 3600 * BOX_DOLLAR_PER_HR, 6),
                        "parse_fails": res.get("parse_fails"),
                        "error": res.get("error"),
                        "tags": info.tags,
                    }
                    score = row["score"] or 0
                    tt = row["think_tokens"] or 1
                    row["score_per_kthink"] = round(score / tt * 1000, 6)
                    results.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"RESULT {aname} {game} s{seed}: "
                          f"lv={row['levels']} score={row['score']} "
                          f"think={row['think_tokens']}/{tbudget} "
                          f"${row['dollar_per_task']}", flush=True)

    # bill think tokens (researchers only)
    for aname, tt in think_totals.items():
        if aname in stakes:
            bill = bank.bill_think_tokens(d, aname, gen, tt)
            print(f"billed {aname}: {tt} think tokens -> {bill} bankroll tokens",
                  flush=True)

    # leaderboard + settle
    print("\n=== LEADERBOARD (mean score over games x seeds) ===")
    board = []
    for aname in agents:
        rs = [r for r in results if r["agent"] == aname
              and r["score"] is not None]
        mean_s = float(np.mean([r["score"] for r in rs])) if rs else 0.0
        mean_d = float(np.mean([r["dollar_per_task"] for r in rs])) if rs else 0.0
        mean_spk = float(np.mean([r["score_per_kthink"] for r in rs])) if rs else 0.0
        lv = sum(r["levels"] or 0 for r in rs)
        board.append((aname, mean_s, lv, mean_d, mean_spk))
    board.sort(key=lambda t: -t[1])
    for aname, ms, lv, md, mspk in board:
        print(f"{aname:12s} mean_score={ms:.4f} levels={lv} "
              f"$/task={md:.6f} score/kthink={mspk:.4f}")

    ranked = [(aname, ms) for aname, ms, _, _, _ in board if aname in stakes]
    if ranked:
        report = bank.settle(d, gen, ranked)
        print("\n=== PAYOUTS ===")
        for r in report:
            print(f"{r['researcher']:12s} rank={r['rank']} score={r['score']} "
                  f"stake={r['stake']} returned={r['returned']} "
                  f"balance={r['balance']} [{r['note']}]")
    # elimination check (next gen eval cost assumed same as this gen)
    eval_cost_next = len(games) * len(seeds)
    elim = bank.elimination_check(d, gen, eval_cost_next)
    if elim:
        print("\n=== ELIMINATED ===")
        for name, reason in elim:
            print(f"{name}: {reason}")
    bank.save(d)
    print(f"\nledger saved: {bank.LEDGER}", flush=True)

    print("\n=== PURITY ===")
    purity_results = {}
    for aname in agents:
        p = os.path.join(a.gen_dir, aname, "agent.py")
        try:
            pur = purity_check(p)
            dead = [k for k, v in pur.items() if v is not True]
            purity_results[aname] = {"load_bearing": all(v is True for v in pur.values()),
                                     "dead": dead, "detail": pur}
            print(f"{aname}: {'all load-bearing' if not dead else 'DEAD: '+str(dead)}")
        except Exception as e:
            purity_results[aname] = {"load_bearing": False, "dead": ["check-failed"],
                                     "detail": str(e)}
            print(f"{aname}: purity check failed: {e}")

    # ---- publish public leaderboard + champion ----
    import shutil
    import time as _time
    champ = board[0][0] if board else None
    champ_pub = None
    if champ:
        cdir = "/srv/arc3-claude/v6run/model/champions"
        os.makedirs(cdir, exist_ok=True)
        champ_pub = os.path.join(cdir, f"{gen}_{champ}.py")
        shutil.copy(os.path.join(a.gen_dir, champ, "agent.py"), champ_pub)
    lb = {
        "gen": gen,
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "eval": {"games": games, "seeds": seeds},
        "researchers": {},
        "agents": [{"agent": an, "mean_score": round(ms, 4), "levels": lv,
                    "dollar_per_task": md, "score_per_kthink": round(mspk, 4)}
                   for an, ms, lv, md, mspk in board],
        "champion": {"researcher": champ, "public_agent": champ_pub},
        "purity": purity_results,
    }
    rank_of = {an: i + 1 for i, (an, _, _, _, _) in enumerate(board)}
    for aname in agents:
        r = d["researchers"].get(aname, {})
        lb["researchers"][aname] = {
            "bankroll": r.get("balance"),
            "stake": stakes.get(aname, {}).get("stake"),
            "token_budget": stakes.get(aname, {}).get("token_budget"),
            "agent_score": next((round(ms, 4) for an2, ms, _, _, _ in board
                                 if an2 == aname), None),
            "think_tokens_burned": think_totals.get(aname),
            "rank": rank_of.get(aname),
            "status": "active" if r.get("alive") else "eliminated",
        }
    lb_path = "/srv/cellforge-arena/ledger/leaderboard.json"
    mirror_path = "/srv/cellforge-arena/common/leaderboard.json"
    for p in (lb_path, mirror_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(lb, f, indent=1)
    print(f"\nleaderboard published: {lb_path} (mirrored to common/)")
    # lineage record
    lin_dir = os.path.join(a.gen_dir, "lineage")
    os.makedirs(lin_dir, exist_ok=True)
    lineage = {"gen": gen, "champion": champ,
               "parents": {an: None for an in agents},  # gen0 founders
               "notes": "gen0 founders: authored from scratch + wave-2 findings"}
    with open(os.path.join(lin_dir, f"{gen}.json"), "w") as f:
        json.dump(lineage, f, indent=1)
    print(f"champion {champ} exported to {champ_pub}")


if __name__ == "__main__":
    main()
