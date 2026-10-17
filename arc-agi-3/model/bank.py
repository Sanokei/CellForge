"""CellForge staking ledger: researcher bankrolls, compute spending, stakes,
payouts, eliminations. The mechanism-design ledger for the paper.

Ledger: /srv/arc3-claude/v6run/model/../../cellforge-arena/ledger/bankroll.json
(kept on the arena ledger path per spec).

Rules:
- Start: 1000 tokens per researcher. Tokens buy compute (1/game-seed rollout)
  and are staked. Every balance change is logged.
- Staking: before a generation's eval, each researcher declares stake s
  (0 < s <= bankroll - eval_cost) + per-game agent token budget.
- Staking CLOSES before eval seeds are drawn/announced (structural ordering).
- Payouts by that generation's population rank:
    rank #1        -> 3x stake back
    top 25%        -> 1.5x stake back
    above mean     -> 1.1x stake back
    below mean     -> lose 50% of stake
    bottom 25%     -> lose entire stake
- Elimination: bankrupt (can't afford a full eval) -> eliminated, replaced
  with 1000. Bottom-25% twice running -> eliminated even if solvent.
- Agent think-tokens: priced at 1 bankroll token per 100 think tokens (ceil),
  deducted after eval, metered harness-side.
"""
import json
import math
import os
import time

LEDGER = "/srv/cellforge-arena/ledger/bankroll.json"
START_BANKROLL = 1000
THINK_TOKENS_PER_BANKROLL_TOKEN = 100


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load():
    if not os.path.exists(LEDGER):
        return {"researchers": {}, "log": [], "generations": {}}
    with open(LEDGER) as f:
        return json.load(f)


def save(d):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    tmp = LEDGER + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, LEDGER)


def _log(d, rtype, researcher, amount, gen, note=""):
    r = d["researchers"][researcher]
    r["balance"] += amount
    d["log"].append({"ts": _now(), "gen": gen, "researcher": researcher,
                     "type": rtype, "amount": amount,
                     "balance_after": r["balance"], "note": note})


def init_researchers(d, names, gen="gen0"):
    for n in names:
        if n not in d["researchers"]:
            d["researchers"][n] = {"balance": START_BANKROLL, "alive": True,
                                   "consecutive_bottom25": 0,
                                   "eliminated_gen": None}
            d["log"].append({"ts": _now(), "gen": gen, "researcher": n,
                             "type": "init", "amount": START_BANKROLL,
                             "balance_after": START_BANKROLL,
                             "note": "starting bankroll"})


def spend(d, researcher, amount, gen, reason):
    """Deduct `amount` tokens for compute. Returns True if affordable."""
    r = d["researchers"][researcher]
    if not r["alive"] or r["balance"] < amount:
        return False
    _log(d, "compute_spend", researcher, -amount, gen, reason)
    return True


def declare_stake(d, researcher, gen, stake, token_budget, eval_cost):
    """Record stake + agent token budget. Stakes lock at close time."""
    r = d["researchers"][researcher]
    assert r["alive"], f"{researcher} not alive"
    g = d["generations"].setdefault(gen, {"stakes": {}, "closed": False})
    assert not g["closed"], f"staking already closed for {gen}"
    assert 0 < stake <= r["balance"] - eval_cost, \
        f"stake {stake} invalid: balance {r['balance']}, eval_cost {eval_cost}"
    assert token_budget > 0, "token budget must be positive"
    g["stakes"][researcher] = {"stake": stake, "token_budget": token_budget,
                               "eval_cost": eval_cost}
    d["log"].append({"ts": _now(), "gen": gen, "researcher": researcher,
                     "type": "stake_declared", "amount": 0,
                     "balance_after": r["balance"],
                     "note": f"stake={stake} token_budget={token_budget}"})


def close_staking(d, gen, seeds):
    """Close staking, THEN draw/announce eval seeds. Structural ordering."""
    g = d["generations"][gen]
    assert not g["closed"], "already closed"
    # lock stakes + deduct eval costs
    for researcher, s in g["stakes"].items():
        r = d["researchers"][researcher]
        _log(d, "stake_locked", researcher, -s["stake"], gen,
             f"stake locked for {gen}")
        _log(d, "compute_spend", researcher, -s["eval_cost"], gen,
             f"official eval compute for {gen}")
    g["closed"] = True
    g["closed_ts"] = _now()
    g["eval_seeds"] = seeds  # drawn only after close
    d["log"].append({"ts": _now(), "gen": gen, "researcher": "-",
                     "type": "staking_closed", "amount": 0,
                     "balance_after": 0,
                     "note": f"seeds announced: {seeds}"})


def settle(d, gen, ranking):
    """ranking: list of (researcher, mean_score) sorted desc. Applies payouts,
    think-token bills must already be logged. Returns payout report."""
    g = d["generations"][gen]
    assert g["closed"], "staking not closed"
    n = len(ranking)
    scores = [s for _, s in ranking]
    mean_s = sum(scores) / n if n else 0
    top25_n = max(1, math.ceil(0.25 * n))
    report = []
    for i, (researcher, score) in enumerate(ranking):
        rank = i + 1
        s = g["stakes"][researcher]["stake"]
        if rank == 1:
            back, note = 3 * s, "rank #1 -> 3x"
        elif rank <= top25_n:
            back, note = int(1.5 * s), "top 25% -> 1.5x"
        elif rank > n - top25_n:
            back, note = 0, "bottom 25% -> lose stake"
        elif score > mean_s:
            back, note = int(1.1 * s), "above mean -> 1.1x"
        else:
            back, note = s // 2, "below mean -> lose 50%"
        _log(d, "payout", researcher, back, gen,
             f"{note} (stake was {s}, score {score:.4f})")
        r = d["researchers"][researcher]
        is_bottom25 = rank > n - top25_n
        r["consecutive_bottom25"] = (r["consecutive_bottom25"] + 1
                                     if is_bottom25 else 0)
        report.append({"researcher": researcher, "rank": rank,
                       "score": round(score, 4), "stake": s,
                       "returned": back, "balance": r["balance"],
                       "note": note})
    g["settled"] = True
    g["ranking"] = [{"researcher": r, "score": round(s, 4)}
                    for r, s in ranking]
    return report


def elimination_check(d, gen, eval_cost_next):
    """Returns list of eliminated researcher names. Bankruptcy or
    bottom-25% twice running."""
    out = []
    for name, r in d["researchers"].items():
        if not r["alive"]:
            continue
        if r["balance"] < eval_cost_next + 1:  # can't stake anything either
            r["alive"] = False
            r["eliminated_gen"] = gen
            out.append((name, "bankrupt"))
        elif r["consecutive_bottom25"] >= 2:
            r["alive"] = False
            r["eliminated_gen"] = gen
            out.append((name, "bottom-25% twice running"))
    for name, reason in out:
        d["log"].append({"ts": _now(), "gen": gen, "researcher": name,
                         "type": "eliminated", "amount": 0,
                         "balance_after": d["researchers"][name]["balance"],
                         "note": reason})
    return out


def bill_think_tokens(d, researcher, gen, think_tokens):
    bill = math.ceil(think_tokens / THINK_TOKENS_PER_BANKROLL_TOKEN)
    if bill > 0:
        _log(d, "think_tokens", researcher, -bill, gen,
             f"{think_tokens} think tokens @ 100:1")
    return bill
