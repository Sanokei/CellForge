"""Cell Wars scoring core: pure functions, no ARC data and no maker calls, so every number can be recomputed by anyone.

Several independent makers ("houses") write literal weight tensors for the SAME niche. Each house is shown different generated tasks;
all are judged on hidden referee tasks (a third seed family nobody sees) against the best of LUCK_DRAWS random-tensor entries with the
same number of cells and positions. Points are the margin over luck, so a niche that random numbers already satisfy pays nothing.
Nothing here trusts a reported number: `verify_ledger` regenerates the referee tasks and re-scores the saved tensors.
"""
import hashlib
import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402

SEED = 11
HOUSES = ("Ember", "Tide", "Moss", "Flint")
LUCK_DRAWS = 24
NICHE_POINTS = 10.0   # a house that solves every referee task and luck solves none earns this much for the niche
KILL_POINTS = 5.0     # per ARC selection-split task newly solved by that house's cells (ablation, recomputed)
RIGGED = 0.25          # a niche where the best random entry reaches this fraction pays almost nothing and is listed as rigged
DECLINE_PHRASES = ("no valid", "no satisfying", "no exact", "no revision", "no tensors", "cannot", "no such")
# structure authors: the workshop that wrote the template (everything else is the head workshop)
CODEX_NICHES = ("wallpruned", "corridormarker", "anchorassign", "anchorspin", "anchorclip", "framecorner", "frameinfer", "referenceobject",
                "referencepaint")


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def house_index(house):
    return HOUSES.index(house) + 1


def train_seed(house, i):
    return SEED * 1000 + 100 * house_index(house) + i


def referee_seed(i):
    return SEED * 1000 + 700 + i


def rand_cell(template, rng):
    return S.make_cell(template, {k: np.array([rng.uniform(-9, 9) for _ in range(int(np.prod(s)))]).reshape(s).tolist()
                                  for k, s in S.TEMPLATES[template]["shapes"].items()})


def make_tasks(niche, patterns, seed_fn):
    out = {}
    for i, p in enumerate(patterns):
        try:
            out[p["id"]] = niche.task(p, seed_fn(i))
        except RuntimeError:  # a pattern that cannot yield a verifiable task is dropped, never faked
            pass
    return out


def build_arena(niche, limit, houses):
    """Patterns usable by every house and the referee, the per-house training tasks and the hidden referee tasks."""
    patterns = niche.patterns(limit, SEED)
    referee = make_tasks(niche, patterns, referee_seed)
    train = {h: make_tasks(niche, patterns, lambda i, h=h: train_seed(h, i)) for h in houses}
    keep = [p for p in patterns if p["id"] in referee and all(p["id"] in train[h] for h in houses)]
    return {"patterns": keep,
            "referee": {p["id"]: referee[p["id"]] for p in keep},
            "train": {h: {p["id"]: train[h][p["id"]] for p in keep} for h in houses}}


def entry_solved(niche, entry, referee):
    """Hidden referee tasks solved. Instance niche: the cell written for pattern p on referee task p. General niche: one cell on every task."""
    if niche.general:
        cell = entry.get("*")
        return sum(S.passes(cell, t) for t in referee.values()) if cell else 0
    return sum(1 for pid, cell in entry.items() if pid in referee and S.passes(cell, referee[pid]))


def entry_total(niche, entry, referee):
    return len(referee)


def luck_champion(niche, entry, referee, draws=LUCK_DRAWS, seed=0):
    """Best of `draws` random entries that submit the same number of cells (instance niches: for the same patterns) as the house did."""
    best = 0
    pids = sorted(entry) if not niche.general else ["*"]
    for d in range(draws):
        rng = random.Random(seed * 100003 + d)
        if niche.general:
            cell = rand_cell(niche.template, rng)
            solved = sum(S.passes(cell, t) for t in referee.values())
        else:
            solved = sum(1 for pid in pids if pid in referee and S.passes(rand_cell(niche.template, rng), referee[pid]))
        best = max(best, solved)
    return best


def luck_seed(season, gen, house):
    """Season 1 drew a different luck champion per house, which made identical results score differently (noise). From season 2 the draw is shared."""
    return gen * 10 + (house_index(house) if season <= 1 else 0)


def points(solved, luck, total, per=NICHE_POINTS):
    return round(per * max(0, solved - luck) / max(total, 1), 3)


def canon_entry(entry):
    """The scored artifact: template and literal tensors only (metadata never enters a hash)."""
    return {k: {"template": c["template"], "params": c["params"]} for k, c in sorted(entry.items())}


def load_entry(canon):
    return {k: S.make_cell(v["template"], v["params"]) for k, v in canon.items()}


def entry_charged(entry):
    return int(sum(S.positions(c) for c in entry.values()))


def is_decline(raw_cells, rationales):
    return not raw_cells and any(str(r or "").lower().startswith(DECLINE_PHRASES) for r in rationales)


def score_row(niche, house, gen, entry, referee, prompt_sha, declined=False, scouted=False, season=1):
    solved = entry_solved(niche, entry, referee)
    luck = luck_champion(niche, entry, referee, seed=luck_seed(season, gen, house))
    total = len(referee)
    return {"kind": "match", "season": season, "niche": niche.name, "house": house, "gen": gen, "solved": solved, "luck": luck, "total": total,
            "points": points(solved, luck, total), "charged": entry_charged(entry), "entry_sha": sha(canon_entry(entry)), "prompt_sha256": prompt_sha,
            "declined": declined, "scouted": scouted}


def winners(rows):
    """Houses sharing the best (points, fewest charged positions) among rows of ONE niche; nobody wins a niche that pays nothing."""
    ranked = sorted(((-r["points"], r["charged"], r["house"]) for r in rows if r["points"] > 0))
    if not ranked:
        return []
    top = ranked[0][:2]
    return [h for p, c, h in ranked if (p, c) == top]


def best_rows(rows):
    """One row per (niche, house): the best generation (every house gets the same number of generations)."""
    best = {}
    for r in rows:
        if r.get("kind") != "match":
            continue
        k = (r["season"], r["niche"], r["house"])
        if k not in best or (r["points"], -r["charged"]) > (best[k]["points"], -best[k]["charged"]):
            best[k] = r
    return best


def leaderboard(rows, kills=None, houses=HOUSES):
    """Standings: match points (best generation per niche and house) plus ARC kill points, and the mechanical titles."""
    kills = kills or {}
    table = {h: {"house": h, "points": 0.0, "match_points": 0.0, "kill_points": 0.0, "niches_won": 0, "titles": []} for h in houses}
    by_niche = {}
    for (season, name, house), r in best_rows(rows).items():
        if house not in table:
            continue
        by_niche.setdefault((season, name), []).append(r)
        table[house]["match_points"] += r["points"]
        if r["declined"]:
            table[house]["titles"].append("Honest Null: %s" % name)
    for (season, name), rs in sorted(by_niche.items()):
        won = winners(rs)
        for h in won:
            table[h]["niches_won"] += 1
        paid = [r["house"] for r in rs if r["points"] > 0]
        if len(paid) == 1:
            table[paid[0]]["titles"].append("Sole Survivor: %s" % name)
        paying = [r for r in rs if r["points"] > 0 and not (r["total"] and r["luck"] / r["total"] >= RIGGED)]  # a rigged niche draws no first blood
        if paying:
            key = lambda r: (r["gen"], -r["points"], r["charged"])  # noqa: E731  earliest generation, then margin, then fewest positions; exact ties share it
            top = min(key(r) for r in paying)
            for r in paying:
                if key(r) == top:
                    table[r["house"]]["titles"].append("First Blood: %s" % name)
    for h, k in kills.items():
        if h in table:
            table[h]["kill_points"] += KILL_POINTS * k
    for h in table:
        table[h]["points"] = round(table[h]["match_points"] + table[h]["kill_points"], 3)
    return sorted(table.values(), key=lambda t: (-t["points"], t["house"]))


def near_duplicate(a, b, tol=0.5):
    if a["template"] != b["template"] or set(a["params"]) != set(b["params"]):
        return False
    return all(np.asarray(a["params"][k], float).shape == np.asarray(b["params"][k], float).shape
               and float(np.max(np.abs(np.asarray(a["params"][k], float) - np.asarray(b["params"][k], float)), initial=0.0)) < tol
               for k in a["params"])


def verify_ledger(rows, entries, niche_limits=None):
    """Recompute every match row from the saved tensors: referee tasks regenerated from the niche seeds, luck re-drawn, points re-derived.
    `entries` maps (niche, house, gen) -> canonical entry (as saved). Returns a list of mismatches (empty = ledger honest)."""
    problems, cache = [], {}
    for r in rows:
        if r.get("kind") != "match":
            continue
        key = (r["niche"], r["house"], r["gen"])
        canon = entries.get(key)
        if canon is None:
            problems.append("%s: tensors missing" % (key,))
            continue
        if sha(canon) != r["entry_sha"]:
            problems.append("%s: tensors do not match the ledger hash" % (key,))
            continue
        entry = load_entry(canon)
        niche = N.NICHES[r["niche"]]
        if niche.name not in cache:
            limit = (niche_limits or N.LIMITS)[niche.name]
            cache[niche.name] = make_tasks(niche, niche.patterns(limit, SEED), referee_seed)
        referee = cache[niche.name]
        solved = entry_solved(niche, entry, referee)
        luck = luck_champion(niche, entry, referee, seed=luck_seed(r["season"], r["gen"], r["house"]))
        if (solved, luck, len(referee), points(solved, luck, len(referee))) != (r["solved"], r["luck"], r["total"], r["points"]):
            problems.append("%s: recomputed (%d solved, luck %d of %d, %.3f pts) != ledger (%d, %d of %d, %.3f)" % (
                key, solved, luck, len(referee), points(solved, luck, len(referee)), r["solved"], r["luck"], r["total"], r["points"]))
        if r["charged"] != entry_charged(entry):
            problems.append("%s: charged positions %d != recomputed %d" % (key, r["charged"], entry_charged(entry)))
    return problems


def census_category(task):
    """Coarse family of an ARC task from its demonstrations (used only to size bounties, never to select or prompt)."""
    pairs = [(np.asarray(d["input"]), np.asarray(d["output"])) for d in task["train"]]
    if any(a.shape != b.shape for a, b in pairs):
        if all(b.size < a.size for a, b in pairs):
            return "smaller output"
        if all(b.size > a.size for a, b in pairs):
            return "larger output"
        return "mixed size"
    adds = moves = recolors = False
    other = False
    for a, b in pairs:
        if (a == b).all():
            continue
        changed = a != b
        if (a[changed] == 0).all() and (b[changed] != 0).all():
            adds = True
        elif (a[changed] != 0).all() and (b[changed] != 0).all():
            recolors = True
        elif sorted(a.ravel().tolist()) == sorted(b.ravel().tolist()):
            moves = True
        else:
            other = True
    kinds = [adds, moves, recolors, other]
    if sum(kinds) != 1:
        return "same shape, mixed"
    return ["same shape, additive", "same shape, movement", "same shape, recolor only", "same shape, mixed"][kinds.index(True)]
