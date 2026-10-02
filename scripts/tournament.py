"""CELL WARS tournament driver: head-to-head, zero-sum, single elimination, with stakes.

Each cycle seeds the houses (rotating), plays a bracket of bouts, and every bout is a set of niche duels. In a duel two houses write literal weight
tensors for the same niche, cannot see each other, and are judged on a hidden two-tier gauntlet against random-tensor luck. The duel winner challenges the
INCUMBENT cell of each slot (niche, pattern) in the shared bank: only a strictly better cell takes the slot, the dethroned cell is evicted, and a correct
loser is discarded. After the bracket the two finalists fight merge duels (one cell for a group of earlier same-template cells), then the bank breeds.

  python tournament.py runs/weights-head-tourney1 --houses 4 --season 4 --cycles 2 --seed-from runs/weights-head-arena3
  python tournament.py verify runs/weights-head-tourney1
"""
import argparse
import collections
import json
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena as AR  # noqa: E402
import arena_core as A  # noqa: E402
import head_arc_cells as H  # noqa: E402
import head_driver as D  # noqa: E402
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402
import tournament_core as T  # noqa: E402

POOL = ("wallpruned", "corridormarker", "anchorassign", "frameinfer", "referencepaint", "anchorspin", "framecorner", "referenceobject", "anchorclip",
        "tooluse", "analogy", "association", "glyph", "fillenc", "context2", "objmove3", "blockred", "outline")
PER_BOUT = 6


def duel_header(house, opponent, season, rid, standings):
    table = "; ".join("%s %.0f" % (t["house"], t["points"]) for t in standings)
    return ("CELL WARS TOURNAMENT, season %d, %s. You are House %s, head to head with House %s on this family. You each write tensors independently and you "
            "cannot see theirs. Hidden gauntlet: referee tasks at the standard tier, then a harder tier with fewer demonstrations that counts double; you must also "
            "beat the best of %d random-tensor entries to win at all. The winner takes the niche: only the winner's cells can enter the shared bank, and only by "
            "beating the cell already holding the slot; a loser's cells are discarded even if they are correct. Equal gauntlet scores go to the sparser tensor, so "
            "write clean tensors and put 0 in every entry you do not need. Standings: %s. Always give your best complete attempt; if you believe no tensors of "
            "this template can express the rule, ALSO set \"cannot_express\": true on that entry (it costs nothing and never replaces the attempt). Do not use "
            "tools or files.\n" % (season, rid, house, opponent, A.LUCK_DRAWS, table))


def merge_prompt(niche, items):
    t = S.TEMPLATES[niche.template]
    return ("You merge literal weight tensors of grid cells (no code, no tools, no files). Template %s, tensors %s. Semantics: %s\nFor EACH group below, two earlier "
            "cells of this template each solve one pattern (their tensors and sample demonstration sets are given). Write ONE complete tensor set of the same "
            "template and shapes (finite numbers in [-9, 9], every entry present, 0 for unused) that solves BOTH patterns and, if you can see the underlying rule, "
            "other patterns of the same family too. Return ONLY a JSON array: [{\"task_id\": group id, \"params\": {tensor_name: nested lists}, \"rationale\": str}]."
            "\nGroups:\n%s" % (niche.template, {k: list(v) for k, v in t["shapes"].items()}, t["doc"], json.dumps(items)))


def covers(cell, pid):
    """A cell holds the slot of its own pattern, and a merged cell holds the slots of every pattern it was merged from."""
    pattern = cell.get("pattern") or ""
    return pattern == pid or (pattern.startswith("merged:") and pid in pattern[len("merged:"):].split("+"))


class Tournament(AR.Arena):
    def __init__(self, root, houses, season, fake=False, workers=2):
        super().__init__(root, houses, season, fake, workers)
        import threading
        self.lock = threading.Lock()
        self.state.setdefault("cycle", 0)

    def append(self, row):
        with self.lock:
            super().append(row)

    def round_entry_path(self, name, rid, stem):
        path = self.root / "ARENA" / "entries" / name / rid
        path.mkdir(parents=True, exist_ok=True)
        return path / (stem + ".json")

    # -- one niche duel: both houses write, nothing is shared
    def play_niche(self, name, pair, rid, gens):
        niche = N.NICHES[name]
        arena = T.build(niche, N.LIMITS[name], pair)
        pats, train = arena["patterns"], arena["train"]
        if len(pats) < 2:
            self.log({"stage": "tourney:%s:%s:skip" % (rid, name), "utc": D.now(), "note": "fewer than two verifiable patterns"})
            return None
        by_id = {p["id"]: p for p in pats}
        st = {h: {"entry": {}, "pass": {}, "attempts": {}, "failing": None, "submitted": 0, "prompts": [], "rate": 0.0} for h in pair}
        out = {"name": name, "niche": niche, "arena": arena, "st": st, "rows": [], "entries": {}}
        for gen in range(1, gens + 1):
            standings = T.table(self.rows, self.houses)
            jobs = []
            for h in pair:
                opp = pair[1] if h == pair[0] else pair[0]
                head = duel_header(h, opp, self.season, rid, standings)
                if niche.general:
                    if gen > 1 and st[h]["rate"] >= 0.9:
                        continue  # already passes its own training tasks: keeps its best generation and sits this one out
                    tasks = list(train[h].values())
                    shown = random.Random(A.house_index(h) * 31 + gen).sample(tasks, min(3, len(tasks)))
                    jobs.append((h, head + D.prompt_general(niche, shown, st[h]["failing"]), "t%d-%s-%s-%s-g%d" % (self.season, rid, name, h, gen), [{"id": "general"}],
                                 lambda: [{"task_id": "general", "params": pats[0]["truth"], "rationale": "oracle"}]))
                else:
                    todo = [pid for pid in by_id if not st[h]["pass"].get(pid)]
                    for b, i in enumerate(range(0, len(todo), 10)):
                        ids = todo[i:i + 10]
                        items = []
                        for pid in ids:
                            item = {"task_id": pid, "demonstrations": train[h][pid]["train"]}
                            if st[h]["attempts"].get(pid):
                                cell = st[h]["attempts"][pid][-1]
                                item["previous_failed_attempt"] = dict((D.counterexamples(cell, [train[h][pid]], 1) or [{"note": "passed"}])[0], params=cell["params"])
                            items.append(item)
                        jobs.append((h, head + D.prompt_instance(niche, items), "t%d-%s-%s-%s-g%d-b%d" % (self.season, rid, name, h, gen, b),
                                     [{"id": pid} for pid in ids], lambda ids=ids: [{"task_id": pid, "params": by_id[pid]["truth"], "rationale": "oracle"} for pid in ids]))
            if not jobs:
                break

            def play(job):
                h, text, tag, chosen, oracle = job
                raw, error = self.try_ask(text, tag, chosen, oracle)
                return h, text, raw, error

            with ThreadPoolExecutor(len(pair)) as pool:
                results = list(pool.map(play, jobs))
            raws = collections.defaultdict(list)
            for h, text, raw, error in results:
                st[h]["prompts"].append(A.sha(text))
                raws[h].append((raw, error))
            for h in pair:
                if not raws[h] and gen > 1:
                    continue
                opp = pair[1] if h == pair[0] else pair[0]
                cells, rationales, errors, flags = [], [], 0, 0
                for raw, error in raws[h]:
                    errors += error is not None
                    entries = [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []
                    parsed = D.parse_cells(raw, niche.template) if isinstance(raw, list) else []
                    rationales += [e.get("rationale") for e in entries]
                    flags += sum(1 for e in entries if e.get("cannot_express") is True)
                    cells += [c for c in parsed if niche.general or c.get("task_id") in by_id]
                st[h]["submitted"] += sum(S.positions(c) for c in cells)
                if niche.general:
                    if cells:
                        cell = cells[0]
                        st[h]["entry"] = {"*": cell}
                        st[h]["rate"] = float(sum(S.passes(cell, t) for t in train[h].values()) / max(len(train[h]), 1))
                        st[h]["failing"] = {"params": cell["params"], "failing_tasks": D.counterexamples(cell, list(train[h].values()), 2)}
                else:
                    for c in cells:
                        pid = c["task_id"]
                        st[h]["attempts"].setdefault(pid, []).append(c)
                        ok = S.passes(c, train[h][pid])
                        if ok or not st[h]["pass"].get(pid):
                            st[h]["entry"][pid] = c
                            st[h]["pass"][pid] = ok
                entry = dict(st[h]["entry"])
                declined = A.is_decline(cells, rationales) and errors == 0
                row = T.score_row(niche, h, gen, entry, arena, A.sha(st[h]["prompts"]), rid, opp, self.season, declined=declined, flags=flags)
                row.update(charged_submitted=st[h]["submitted"], maker_errors=errors, utc=D.now())
                out["rows"].append(row)
                out["entries"][(h, gen)] = entry
            if niche.general and all(st[h]["rate"] >= 0.9 for h in pair):
                break
        return out

    # -- slot admission: a challenger must strictly beat the incumbent of the slot
    def cell_key(self, niche, cell, pid, arena):
        if niche.general:
            std = sum(S.passes(cell, t) for t in arena["referee"].values())
            hard = sum(S.passes(cell, t) for t in arena["hard"].values())
        else:
            std = int(pid in arena["referee"] and S.passes(cell, arena["referee"][pid]))
            hard = int(pid in arena["hard"] and S.passes(cell, arena["hard"][pid]))
        return (std + 2 * hard, -T.nonzero({"x": cell}))

    def admit_cells(self, out, house, row, rid):
        niche, arena, st, name = out["niche"], out["arena"], out["st"], out["name"]
        entry = out["entries"][(house, row["gen"])]
        incoming = []
        if niche.general:
            cell = entry.get("*")
            if cell is not None and st[house]["rate"] >= 0.9 and row["solved"] / max(row["total"], 1) >= 0.75:
                incoming.append(("*", cell))
        else:
            for pid, cell in entry.items():
                if S.passes(cell, arena["train"][house][pid]) and S.passes(cell, arena["referee"][pid]):
                    incoming.append((pid, cell))
        adds, evict = [], []
        for pid, cell in incoming:
            incumbent = next((c for c in self.registry if c.get("kind") == "cell" and c.get("niche") == name and covers(c, pid)), None)
            if incumbent is not None and incumbent.get("merged_from"):
                continue  # a merged cell holds the slot together with its sibling pattern: it is protected, not dethroned by a single-pattern cell
            new = dict(cell)
            new.update(niche=name, pattern=pid, house=house, general=bool(niche.general), generation=row["gen"], duel_round=rid,
                       origin="maker gpt-6.1-sol low, cell wars tournament season %d %s house %s" % (self.season, rid, house))
            if incumbent is None:
                adds.append(new)
            elif incumbent["id"] != new["id"] and not A.near_duplicate(incumbent, new) and self.cell_key(niche, new, pid, arena) > self.cell_key(niche, incumbent, pid, arena):
                adds.append(new)
                evict.append(incumbent)
        if not adds:
            return {"added": 0, "replaced": 0, "kills": 0}
        before = self.cells()
        base = self.sel(before)

        def trial_of(adds_, evict_):
            ids = {c["id"] for c in evict_}
            return self.sel([c for c in before if c["id"] not in ids] + adds_)

        trial = trial_of(adds, evict)
        lost = [t for t, v in base["outcomes"].items() if v and not trial["outcomes"][t]]
        if lost and evict:  # ARC veto: dethroning an incumbent must not lose a selection task; keep it and take only empty slots
            taken = {e["pattern"] for e in evict}
            adds = [c for c in adds if c["pattern"] not in taken]
            evict = []
            trial = trial_of(adds, evict) if adds else base
        gain = trial["solved"] - base["solved"]
        ids = {c["id"] for c in evict}
        self.registry = [c for c in self.registry if c["id"] not in ids] + adds
        if gain > 0:
            self.state["kills"][house] = self.state["kills"].get(house, 0) + gain
            self.append({"kind": "kills", "season": self.season, "niche": name, "house": house, "tasks": gain, "utc": D.now(), "tournament": True})
        return {"added": len(adds), "replaced": len(evict), "kills": max(0, gain)}

    def resolve(self, out, pair, rid):
        name = out["name"]
        for r in out["rows"]:
            self.append(r)
            H.write(self.round_entry_path(name, rid, "%s-g%d" % (r["house"], r["gen"])), A.canon_entry(out["entries"][(r["house"], r["gen"])]))
        best = T.best_by_round(out["rows"])
        rows = {h: best.get((self.season, rid, name, h)) for h in pair}
        if rows[pair[0]] is None or rows[pair[1]] is None:
            return None
        winner, reason = T.duel(rows[pair[0]], rows[pair[1]])
        self.append({"kind": "duel", "season": self.season, "round_id": rid, "niche": name, "a": pair[0], "b": pair[1], "winner": winner, "reason": reason,
                     "utc": D.now()})
        taker = winner or (pair[0] if reason == "draw" else None)  # a draw goes to the higher seed; no contest admits nobody
        got = self.admit_cells(out, taker, rows[taker], rid) if taker else {"added": 0, "replaced": 0, "kills": 0}
        self.log({"stage": "tourney:%s:%s" % (rid, name), "utc": D.now(), "winner": winner, "reason": reason, "scores": {h: rows[h]["score"] for h in pair},
                  "luck": {h: rows[h]["luck"] for h in pair}, "nonzero": {h: rows[h]["nonzero"] for h in pair}, **got})
        return {"winner": winner, "reason": reason, "rows": rows, "admit": got, "niche": name}

    def bout(self, a, b, card, rid, gens):
        self.status(stage="tourney:" + rid, phase="bout", season=self.season)
        self.log({"stage": "tourney:%s:start" % rid, "utc": D.now(), "a": a, "b": b, "card": card})
        def safe_play(n):
            try:
                return self.play_niche(n, (a, b), rid, gens)
            except Exception as error:  # one broken niche must not end the tournament: it is logged and the duel does not happen
                self.log({"stage": "tourney:%s:%s:error" % (rid, n), "utc": D.now(), "error": "%s: %s" % (type(error).__name__, str(error)[:200])})
                return None

        with ThreadPoolExecutor(2) as pool:
            plays = list(pool.map(safe_play, card))
        duels =[d for d in (self.resolve(p, (a, b), rid) for p in plays if p) if d]
        winner, reason = T.bout_winner(a, b, duels)
        self.append({"kind": "bout", "season": self.season, "round_id": rid, "a": a, "b": b, "winner": winner, "reason": reason, "niches": len(duels), "utc": D.now()})
        self.save()
        self.measure("tourney:" + rid, bout="%s vs %s" % (a, b), winner=winner, reason=reason, duels=len(duels),
                     added=sum(d["admit"]["added"] for d in duels), replaced=sum(d["admit"]["replaced"] for d in duels))
        return winner, sum(d["admit"]["added"] for d in duels), sum(d["admit"]["replaced"] for d in duels)

    # -- merge duels: the finalists combine earlier cells into larger ones
    def merge_phase(self, finalists, rid, cycle, max_niches=3, max_groups=6):
        a, b = finalists
        by_niche = collections.defaultdict(list)
        for c in self.registry:
            nz = N.NICHES.get(c.get("niche"))
            if c.get("kind") == "cell" and nz and not nz.general and c.get("pattern") not in (None, "*") and not c.get("merged_from"):
                by_niche[c["niche"]].append(c)
        names = [n for n in sorted(by_niche) if len({c["pattern"] for c in by_niche[n]}) >= 2]
        if not names:
            return 0
        shift = (cycle * max_niches) % len(names)
        names = (names[shift:] + names[:shift])[:max_niches]
        applied = 0
        for name in names:
            niche = N.NICHES[name]
            arena = T.build(niche, N.LIMITS[name], (a, b))
            seen, uniq = set(), []
            for c in sorted(by_niche[name], key=lambda c: c["pattern"]):
                if c["pattern"] not in seen and c["pattern"] in arena["referee"]:
                    seen.add(c["pattern"])
                    uniq.append(c)
            groups = [(uniq[i], uniq[i + 1]) for i in range(0, len(uniq) - 1, 2)][:max_groups]
            if not groups:
                continue
            gid = {i: "g%d" % i for i in range(len(groups))}
            group_pids = {gid[i]: [g[0]["pattern"], g[1]["pattern"]] for i, g in enumerate(groups)}
            ood = {gid[i]: [p for p in arena["referee"] if p not in group_pids[gid[i]]] for i in range(len(groups))}

            def play(h, name=name, niche=niche, arena=arena, groups=groups):
                items = [{"task_id": "g%d" % i, "parents": [{"pattern": c["pattern"], "params": c["params"],
                                                              "sample_demonstrations": arena["train"][h][c["pattern"]]["train"][:2]} for c in g]}
                         for i, g in enumerate(groups)]
                opp = b if h == a else a
                text = duel_header(h, opp, self.season, rid + "-merge-" + name, T.table(self.rows, self.houses)).split("Hidden gauntlet")[0] + (
                    "MERGE DUEL. Only the winner's merged cell is considered for the bank. The judge counts how many of the group's patterns it solves (it must solve "
                    "both), then how many OTHER patterns of the family it solves, then prefers the sparser tensor. Do not use tools or files.\n") + merge_prompt(niche, items)
                raw, error = self.try_ask(text, "t%d-%s-merge-%s-%s" % (self.season, rid, name, h), [{"id": it["task_id"]} for it in items], lambda: [])
                return h, text, raw, error

            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(play, (a, b)))
            merged = {}
            for h, text, raw, error in results:
                by_task = {e.get("task_id"): e for e in raw if isinstance(e, dict)} if isinstance(raw, list) else {}
                for i in range(len(groups)):
                    entry = by_task.get("g%d" % i)
                    cell = S.make_cell(niche.template, entry["params"]) if entry and S.valid_params(niche.template, entry.get("params")) else None
                    row = T.merge_row(niche, h, rid, "g%d" % i, group_pids["g%d" % i], ood["g%d" % i], cell, arena["referee"], A.sha(text), b if h == a else a, self.season)
                    row["utc"] = D.now()
                    self.append(row)
                    H.write(self.round_entry_path(name, rid, "%s-g%d" % (h, i)), A.canon_entry({"*": cell} if cell else {}))
                    merged[(h, i)] = (cell, row)
            for i, parents in enumerate(groups):
                ra, rb = merged[(a, i)][1], merged[(b, i)][1]
                winner, reason = T.merge_winner(ra, rb)
                self.append({"kind": "merge_duel", "season": self.season, "round_id": rid, "niche": name, "group": "g%d" % i, "a": a, "b": b, "winner": winner,
                             "reason": reason, "utc": D.now()})
                if not winner:
                    continue
                cell = merged[(winner, i)][0]
                ids = {p["id"] for p in parents}
                before = self.cells()
                base = self.sel(before)
                new = dict(cell)
                new.update(niche=name, pattern="merged:" + "+".join(group_pids["g%d" % i]), house=winner, merged_from=sorted(ids), generation=cycle,
                           origin="maker gpt-6.1-sol low, cell wars tournament season %d merge duel %s house %s" % (self.season, rid, winner))
                replaced = self.sel([c for c in before if c["id"] not in ids] + [new])
                lost = [t for t, v in base["outcomes"].items() if v and not replaced["outcomes"][t]]
                if not lost and replaced["solved"] >= base["solved"]:  # a larger cell replaces its parents only if ARC selection does not suffer
                    self.registry = [c for c in self.registry if c["id"] not in ids] + [new]
                    trial, how = replaced, "replaced"
                else:
                    self.registry = self.registry + [new]
                    trial, how = self.sel(before + [new]), "added"
                applied += 1
                gain = trial["solved"] - base["solved"]
                if gain > 0:
                    self.state["kills"][winner] = self.state["kills"].get(winner, 0) + gain
                    self.append({"kind": "kills", "season": self.season, "niche": "merge-" + name, "house": winner, "tasks": gain, "utc": D.now(), "tournament": True})
                self.log({"stage": "tourney:%s:merge:%s:g%d" % (rid, name, i), "utc": D.now(), "winner": winner, "reason": reason, "how": how,
                          "ood_solved": merged[(winner, i)][1]["ood_solved"], "ood_total": len(ood["g%d" % i]), "gain": gain})
        self.save()
        return applied

    # -- cycle = bracket + merge duels + breeding
    def cycle(self, n, pool, gens):
        order = self.houses[n % len(self.houses):] + self.houses[:n % len(self.houses)]  # seeding rotates every cycle
        alive, eliminated, rnd, taken, added, replaced = list(order), [], 0, 0, 0, 0
        while len(alive) > 1:
            if (self.root / "STOP").exists():
                return None
            rnd += 1
            pairs, bye = T.pairings(alive)
            nxt, losers = ([bye] if bye else []), []
            for i, (a, b) in enumerate(pairs):
                start = (n * 3 * PER_BOUT + taken * PER_BOUT) % len(pool)
                card = [pool[(start + j) % len(pool)] for j in range(PER_BOUT)]
                taken += 1
                winner, add, rep = self.bout(a, b, card, "t%d-c%d-r%d-m%d" % (self.season, n, rnd, i), gens)
                added, replaced = added + add, replaced + rep
                nxt.append(winner)
                losers.append(b if winner == a else a)
            alive = sorted(nxt, key=order.index)
            eliminated.append(losers)
        champion = alive[0]
        runner_up = eliminated[-1][0]
        places = {champion: 1, runner_up: 2}
        for h in order:
            places.setdefault(h, 3)
        for h, place in places.items():
            self.append({"kind": "placing", "season": self.season, "cycle": n, "house": h, "place": place, "points": T.placing_points(place), "utc": D.now()})
        merges = self.merge_phase((champion, runner_up), "t%d-c%d-merge" % (self.season, n), n) if len(self.houses) > 1 else 0
        children = self.breed()["children_admitted"]
        self.save()
        self.measure("tourney:c%d" % n, champion=champion, runner_up=runner_up, cells_added=added, cells_replaced=replaced, merges_applied=merges, children=children)
        self.write_board()
        return {"champion": champion, "runner_up": runner_up, "added": added, "replaced": replaced, "merges": merges, "children": children}

    def write_board(self):
        cells = self.cells()
        cur = self.sel(cells)
        census = collections.Counter(A.census_category(r["task"]) for r in self.rows_sel if not cur["outcomes"][r["id"]])
        table = T.table(self.rows, self.houses)
        tag = "cell wars tournament season %d" % self.season
        mine = [c for c in self.registry if tag in c.get("origin", "")]
        lines = ["# CELL WARS TOURNAMENT, season %d" % self.season, "",
                 "Updated %s. Head to head, zero-sum: the winner of a duel challenges the incumbent of each slot and only a strictly better cell takes it. "
                 "Every row recomputes from saved tensors: `python scripts/tournament.py verify <this run>`." % D.now(), "",
                 "## Standings", "", "| rank | house | points | duels W-D-L | bouts won | titles | merge wins | ARC kill pts |", "|---|---|---|---|---|---|---|---|"]
        for i, t in enumerate(table, 1):
            lines.append("| %d | %s | %.0f | %d-%d-%d | %d | %d | %d | %.0f |" % (i, t["house"], t["points"], t["duel_wins"], t["duel_draws"], t["duel_losses"],
                                                                         t["bouts_won"], t["titles"], t["merge_wins"], t["kill_points"]))
        bouts = [r for r in self.rows if r.get("kind") == "bout" and r["season"] == self.season]
        lines += ["", "## Bouts", "", "| round | a | b | winner | decided by | niches |", "|---|---|---|---|---|---|"]
        lines += ["| %s | %s | %s | %s | %s | %d |" % (r["round_id"], r["a"], r["b"], r["winner"], r["reason"], r["niches"]) for r in bouts]
        duels = [r for r in self.rows if r.get("kind") == "duel" and r["season"] == self.season]
        reasons = collections.Counter(r["reason"] for r in duels)
        lines += ["", "## Duels", "", "%d duels: %s" % (len(duels), ", ".join("%d %s" % (n, k) for k, n in reasons.most_common()) or "none"), ""]
        merges = [r for r in self.rows if r.get("kind") == "merge_duel" and r["season"] == self.season]
        lines += ["## Merge duels", "", "%d groups fought: %s" % (len(merges), ", ".join("%d %s" % (n, k) for k, n in collections.Counter(
            "%s (%s)" % (r["winner"] or "none", r["reason"]) for r in merges).most_common()) or "none"), "",
                  "## Bank", "", "cells %d; added or replaced by this tournament: %d; merged cells: %d; ARC selection solved %d of %d" % (
                      len(self.registry), len(mine), sum(1 for c in self.registry if c.get("merged_from")), cur["solved"], len(self.rows_sel)), "",
                  "## Bounty board (unsolved ARC selection tasks)", "", "| family | unsolved |", "|---|---|"]
        lines += ["| %s | %d |" % (k, v) for k, v in census.most_common()]
        (self.root / "TOURNAMENT_BOARD.md").write_text("\n".join(lines) + "\n")
        H.write(self.root / "TOURNAMENT_BOARD.json", {"utc": D.now(), "season": self.season, "houses": table, "bounties": dict(census), "arc_selection_solved": cur["solved"],
                                                       "bank_cells": len(self.registry)})

    def tournament_run(self, pool, cycles, gens):
        self.baseline()
        idle = 0
        for _ in range(cycles):
            n = self.state["cycle"] + 1
            self.state["cycle"] = n
            before = self.arc()["outcomes"]
            res = self.cycle(n, pool, gens)
            if res is None:
                self.status(state="user_stop", season=self.season)
                return
            after = self.arc()["outcomes"]
            moved = sum(1 for k, v in after.items() if v and not before[k]) + sum(1 for k, v in before.items() if v and not after[k])
            idle = 0 if (res["added"] or res["replaced"] or res["merges"] or res["children"] or moved) else idle + 1
            self.save()
            if idle >= 2:
                self.status(state="plateau", season=self.season, note="two cycles with no admitted cell, replacement, merge, child or ARC change")
                return
        self.status(state="season_complete", season=self.season)


def verify(root):
    rows = [json.loads(line) for line in (root / "POINTS.jsonl").read_text().splitlines() if line.strip()]
    entries = {}
    for r in rows:
        if r.get("kind") == "match" and r.get("round_id"):
            path = root / "ARENA" / "entries" / r["niche"] / r["round_id"] / ("%s-g%d.json" % (r["house"], r["gen"]))
            if path.exists():
                entries[(r["niche"], r["house"], r["gen"], r["round_id"])] = json.loads(path.read_text())
        elif r.get("kind") == "merge":
            path = root / "ARENA" / "entries" / r["niche"] / r["round_id"] / ("%s-%s.json" % (r["house"], r["group"]))
            if path.exists():
                entries[(r["niche"], r["house"], r["group"], r["round_id"])] = json.loads(path.read_text())
    problems = T.verify(rows, entries)
    print("rows %d, problems %d" % (len(rows), len(problems)))
    for p in problems:
        print(" ", p)
    return 1 if problems else 0


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "verify":
        sys.exit(verify(pathlib.Path(sys.argv[2])))
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--houses", type=int, default=4)
    ap.add_argument("--season", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--pool", default=",".join(POOL))
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--seed-from", type=pathlib.Path)
    a = ap.parse_args()
    t = Tournament(a.root, A.HOUSES[:a.houses], a.season, a.fake)
    if a.seed_from and not t.registry:
        old = H.load(a.seed_from / "REGISTRY.json")
        t.registry = [c for c in old if (c["kind"] == "cell" and c.get("origin", "").startswith("maker")) or c.get("arc_selection_gain")]
        t.state["done"] = sorted({c["niche"] for c in t.registry if c.get("niche") in N.NICHES})
        t.save()
    t.tournament_run([n for n in a.pool.split(",") if n], a.cycles, a.gens)


if __name__ == "__main__":
    main()
