"""Cell Wars, tournament edition: head to head, zero-sum, with stakes. Pure functions (no ARC data, no maker calls), recomputable by anyone.

Why this exists: season 1 scored every house independently against random luck, so three correct houses all won and a house that was behind saw the
leader's tensors. That is cooperation. Here two houses meet in a duel on a niche and cannot see each other's tensors; the better cell takes the
niche, only the winner's cells enter the shared bank, and a correct loser is discarded. A single-elimination bracket decides the cycle.

Gauntlet: hidden referee tasks at the standard tier, plus a hard tier (fewer demonstrations) that counts double. A house must beat the best of
LUCK_DRAWS random entries on the standard tier to win at all. Ties on the gauntlet go to the sparser tensor (fewer nonzero entries).
Merge duels: the two finalists each write ONE cell for a group of earlier same-template cells; the merged cell that solves the whole group (and,
beyond that, other patterns of the family) replaces its parents when ARC selection does not suffer.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_core as A  # noqa: E402
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402

HARD_DEMOS = 2
DUEL_WIN, DUEL_DRAW, BOUT_WIN, MERGE_WIN = 3.0, 1.0, 5.0, 4.0
PLACING = (10.0, 6.0, 3.0, 3.0)  # champion, runner-up, semifinal losers


def hard_seed(i):
    return A.SEED * 1000 + 800 + i


def make_tasks(niche, patterns, seed_fn, ndemos=None):
    out = {}
    for i, p in enumerate(patterns):
        try:
            out[p["id"]] = niche.task(p, seed_fn(i), ndemos=ndemos)
        except RuntimeError:  # a pattern that cannot yield a verifiable task is dropped, never faked
            pass
    return out


def build(niche, limit, pair):
    """Patterns every duelist and both referee tiers can use; per-house training tasks; hidden referee (standard and hard)."""
    patterns = niche.patterns(limit, A.SEED)
    referee = make_tasks(niche, patterns, A.referee_seed)
    hard = make_tasks(niche, patterns, hard_seed, HARD_DEMOS)
    train = {h: make_tasks(niche, patterns, lambda i, h=h: A.train_seed(h, i)) for h in pair}
    keep = [p for p in patterns if p["id"] in referee and all(p["id"] in train[h] for h in pair)]
    ids = [p["id"] for p in keep]
    return {"patterns": keep, "referee": {i: referee[i] for i in ids}, "hard": {i: hard[i] for i in ids if i in hard},
            "train": {h: {i: train[h][i] for i in ids} for h in pair}}


def nonzero(entry):
    """Number of nonzero tensor entries: the sparser of two equally good cells wins the tie (clean weights, nothing unused)."""
    return int(sum(int((np.abs(np.asarray(v, float)) > 1e-9).sum()) for c in entry.values() for v in c["params"].values()))


def score_row(niche, house, gen, entry, arena, prompt_sha, rid, opponent, season, declined=False, flags=0):
    base = A.score_row(niche, house, gen, entry, arena["referee"], prompt_sha, declined=declined, scouted=False, season=season)
    solved_hard = A.entry_solved(niche, entry, arena["hard"])
    base.update(round_id=rid, opponent=opponent, solved_hard=solved_hard, hard_total=len(arena["hard"]), referee_ids=sorted(arena["referee"]),
                hard_ids=sorted(arena["hard"]), nonzero=nonzero(entry), score=base["solved"] + 2 * solved_hard, flags=flags)
    return base


def standing_key(row):
    return (row["score"], -row["nonzero"])


def best_by_round(rows):
    """One row per (season, round, niche, house): the best generation by (gauntlet score, sparsity)."""
    best = {}
    for r in rows:
        if r.get("kind") != "match" or not r.get("round_id"):
            continue
        k = (r["season"], r["round_id"], r["niche"], r["house"])
        if k not in best or standing_key(r) > standing_key(best[k]):
            best[k] = r
    return best


def duel(a, b):
    """Zero-sum niche duel. Returns (winning house or None, reason). Beating luck on the standard tier is required to win anything."""
    live = [r for r in (a, b) if r["solved"] > r["luck"]]
    if not live:
        return None, "no contest"
    if len(live) == 1:
        return live[0]["house"], "forfeit"
    ka, kb = standing_key(a), standing_key(b)
    if ka == kb:
        return None, "draw"
    return (a if ka > kb else b)["house"], "score" if ka[0] != kb[0] else "parsimony"


def bout_winner(a, b, duels):
    """Winner of a head-to-head match over several niches. duels: [{winner, rows: {house: row}}]. Ties: gauntlet score, sparsity, higher seed (a)."""
    wins = {a: 0, b: 0}
    for d in duels:
        if d["winner"] in wins:
            wins[d["winner"]] += 1
    if wins[a] != wins[b]:
        return (a if wins[a] > wins[b] else b), "niche wins"
    score = {h: sum(d["rows"][h]["score"] for d in duels if h in d["rows"]) for h in (a, b)}
    if score[a] != score[b]:
        return (a if score[a] > score[b] else b), "gauntlet score"
    nz = {h: sum(d["rows"][h]["nonzero"] for d in duels if h in d["rows"]) for h in (a, b)}
    if nz[a] != nz[b]:
        return (a if nz[a] < nz[b] else b), "parsimony"
    return a, "higher seed"


def pairings(alive):
    """Single elimination round: the top seed takes the bye when the field is odd. Returns (pairs of (higher seed, lower seed), bye or None)."""
    if len(alive) % 2:
        bye, rest = alive[0], alive[1:]
    else:
        bye, rest = None, alive
    return [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)], bye


def placing_points(place):
    return PLACING[min(place, len(PLACING)) - 1]


# ---------------------------------------------------------------- merge duels
def merge_row(niche, house, rid, gid, pids, ood_ids, cell, referee, prompt_sha, opponent, season):
    entry = {"*": cell} if cell else {}
    solved_group = sum(1 for p in pids if cell and p in referee and S.passes(cell, referee[p]))
    ood = sum(1 for p in ood_ids if cell and p in referee and S.passes(cell, referee[p]))
    return {"kind": "merge", "season": season, "round_id": rid, "niche": niche.name, "house": house, "opponent": opponent, "group": gid, "pids": list(pids),
            "ood_ids": list(ood_ids), "solved_group": solved_group, "ood_solved": ood, "nonzero": nonzero(entry) if cell else 0,
            "entry_sha": A.sha(A.canon_entry(entry)), "prompt_sha256": prompt_sha}


def merge_winner(a, b):
    """The merged cell must solve its whole group to count; then more other patterns solved wins, then the sparser tensor."""
    live = [r for r in (a, b) if r["solved_group"] == len(r["pids"]) and r["nonzero"] > 0]
    if not live:
        return None, "no contest"
    if len(live) == 1:
        return live[0]["house"], "forfeit"
    ka, kb = (a["ood_solved"], -a["nonzero"]), (b["ood_solved"], -b["nonzero"])
    if ka == kb:
        return None, "draw"
    return (a if ka > kb else b)["house"], "generalisation" if ka[0] != kb[0] else "parsimony"


# ---------------------------------------------------------------- standings
def table(rows, houses=A.HOUSES):
    t = {h: {"house": h, "points": 0.0, "duel_wins": 0, "duel_draws": 0, "duel_losses": 0, "bouts_won": 0, "titles": 0, "merge_wins": 0,
             "kill_points": 0.0} for h in houses}
    for r in rows:
        k = r.get("kind")
        if k == "duel":
            for h in (r["a"], r["b"]):
                if h not in t:
                    continue
                if r["winner"] == h:
                    t[h]["duel_wins"] += 1
                    t[h]["points"] += DUEL_WIN
                elif r["winner"] is None and r["reason"] == "draw":
                    t[h]["duel_draws"] += 1
                    t[h]["points"] += DUEL_DRAW
                elif r["winner"] is not None:
                    t[h]["duel_losses"] += 1
        elif k == "bout" and r["winner"] in t:
            t[r["winner"]]["bouts_won"] += 1
            t[r["winner"]]["points"] += BOUT_WIN
        elif k == "placing" and r["house"] in t:
            t[r["house"]]["points"] += r["points"]
            t[r["house"]]["titles"] += int(r["place"] == 1)
        elif k == "merge_duel" and r["winner"] in t:
            t[r["winner"]]["merge_wins"] += 1
            t[r["winner"]]["points"] += MERGE_WIN
        elif k == "kills" and r["house"] in t and r.get("tournament"):
            t[r["house"]]["kill_points"] += A.KILL_POINTS * r["tasks"]
            t[r["house"]]["points"] += A.KILL_POINTS * r["tasks"]
    return sorted(t.values(), key=lambda x: (-x["points"], x["house"]))


# ---------------------------------------------------------------- verification
def verify(rows, entries, limits=None):
    """Recompute every tournament row from the saved tensors: referee tiers regenerated from the niche seeds, luck re-drawn, duels, merges and
    bouts re-derived. `entries` maps (niche, house, gen, round_id) -> canonical entry, merges (niche, house, group, round_id) -> canonical entry."""
    problems, cache = [], {}

    def tasks(name):
        if name not in cache:
            niche = N.NICHES[name]
            patterns = niche.patterns((limits or N.LIMITS)[name], A.SEED)
            cache[name] = (make_tasks(niche, patterns, A.referee_seed), make_tasks(niche, patterns, hard_seed, HARD_DEMOS))
        return cache[name]

    matches = {}
    for r in rows:
        if r.get("kind") != "match" or not r.get("round_id"):
            continue
        key = (r["niche"], r["house"], r["gen"], r["round_id"])
        canon = entries.get(key)
        if canon is None:
            problems.append("%s: tensors missing" % (key,))
            continue
        if A.sha(canon) != r["entry_sha"]:
            problems.append("%s: tensors do not match the ledger hash" % (key,))
            continue
        niche, entry = N.NICHES[r["niche"]], A.load_entry(canon)
        ref_all, hard_all = tasks(r["niche"])
        referee, hard = {i: ref_all[i] for i in r["referee_ids"] if i in ref_all}, {i: hard_all[i] for i in r["hard_ids"] if i in hard_all}
        solved, hsolved = A.entry_solved(niche, entry, referee), A.entry_solved(niche, entry, hard)
        luck = A.luck_champion(niche, entry, referee, seed=A.luck_seed(r["season"], r["gen"], r["house"]))
        got = (solved, hsolved, luck, len(referee), len(hard), nonzero(entry), solved + 2 * hsolved, A.entry_charged(entry))
        want = (r["solved"], r["solved_hard"], r["luck"], r["total"], r["hard_total"], r["nonzero"], r["score"], r["charged"])
        if got != want:
            problems.append("%s: recomputed %s != ledger %s (solved, hard solved, luck, total, hard total, nonzero, score, charged)" % (key, got, want))
        matches[key] = r
    best = best_by_round([r for r in matches.values()])
    for r in rows:
        if r.get("kind") == "duel":
            a, b = (best.get((r["season"], r["round_id"], r["niche"], h)) for h in (r["a"], r["b"]))
            if a is None or b is None:
                problems.append("duel %s %s: match rows missing" % (r["round_id"], r["niche"]))
                continue
            if duel(a, b) != (r["winner"], r["reason"]):
                problems.append("duel %s %s: recomputed %s != ledger %s" % (r["round_id"], r["niche"], duel(a, b), (r["winner"], r["reason"])))
        elif r.get("kind") == "merge":
            key = (r["niche"], r["house"], r["group"], r["round_id"])
            canon = entries.get(key)
            if canon is None or A.sha(canon) != r["entry_sha"]:
                problems.append("merge %s: tensors missing or altered" % (key,))
                continue
            ref_all, _ = tasks(r["niche"])
            cell = A.load_entry(canon).get("*")
            again = merge_row(N.NICHES[r["niche"]], r["house"], r["round_id"], r["group"], r["pids"], r["ood_ids"], cell, ref_all, r["prompt_sha256"],
                              r["opponent"], r["season"])
            for field in ("solved_group", "ood_solved", "nonzero"):
                if again[field] != r[field]:
                    problems.append("merge %s: %s recomputed %s != ledger %s" % (key, field, again[field], r[field]))
    duels_by_round = {}
    for r in rows:
        if r.get("kind") == "duel":
            a, b = (best.get((r["season"], r["round_id"], r["niche"], h)) for h in (r["a"], r["b"]))
            if a is not None and b is not None:
                duels_by_round.setdefault(r["round_id"], []).append({"winner": r["winner"], "rows": {r["a"]: a, r["b"]: b}, "a": r["a"], "b": r["b"]})
    for r in rows:
        if r.get("kind") == "bout":
            ds = duels_by_round.get(r["round_id"], [])
            if ds and bout_winner(r["a"], r["b"], ds) != (r["winner"], r["reason"]):
                problems.append("bout %s: recomputed %s != ledger %s" % (r["round_id"], bout_winner(r["a"], r["b"], ds), (r["winner"], r["reason"])))
    merge_rows = {(r["round_id"], r["niche"], r["group"], r["house"]): r for r in rows if r.get("kind") == "merge"}
    for r in rows:
        if r.get("kind") == "merge_duel":
            a, b = (merge_rows.get((r["round_id"], r["niche"], r["group"], h)) for h in (r["a"], r["b"]))
            if a is None or b is None:
                problems.append("merge duel %s %s %s: merge rows missing" % (r["round_id"], r["niche"], r["group"]))
            elif merge_winner(a, b) != (r["winner"], r["reason"]):
                problems.append("merge duel %s %s %s: recomputed %s != ledger %s" % (r["round_id"], r["niche"], r["group"], merge_winner(a, b), (r["winner"], r["reason"])))
    return problems
