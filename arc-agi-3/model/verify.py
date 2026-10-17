"""Metered self-verification for researchers: every test rollout costs
1 bankroll token. Use this for ALL agent testing — the ledger is audited.

Usage: python verify.py <researcher> <agent.py> <game> [seed] [budget] [maxdec]
"""
import logging
import os
import sys

sys.path.insert(0, "/srv/arc3-claude/v6run/model")
from agent_api import load_agent, run_agent_game
import bank

from arc_agi import Arcade

BASE = "/srv/arc3-claude"


def main():
    researcher, agent_path, game = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    budget = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    maxdec = int(sys.argv[6]) if len(sys.argv) > 6 else 50
    d = bank.load()
    r = d["researchers"].get(researcher)
    assert r and r["alive"], f"{researcher} unknown or eliminated"
    assert bank.spend(d, researcher, 1, "verify",
                      f"self-test rollout {game} s{seed}"), "insufficient funds"
    bank.save(d)
    mod = load_agent(agent_path)
    lg = logging.getLogger("qv")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    arc = Arcade(logger=lg, environments_dir=f"{BASE}/environment_files",
                 recordings_dir=f"{BASE}/recordings")
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    res, _ = run_agent_game(mod, game, seed, budget, maxdec, arc, info,
                            card, env)
    sc = arc.close_scorecard(card)
    run = sc.environments[0].runs[-1].model_dump() if sc and sc.environments else {}
    d = bank.load()
    bal = d["researchers"][researcher]["balance"]
    print(f"VERIFY {researcher} {game} s{seed}: levels={run.get('levels_completed')} "
          f"score={run.get('score')} steps={res.get('steps')} "
          f"think={res.get('think_tokens')} balance={bal}", flush=True)


if __name__ == "__main__":
    main()
