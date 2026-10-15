"""CELL WARS driver: independent makers ("houses") write literal weight tensors for the same niche, are scored on hidden referee tasks against
random-tensor luck, and the winners' cells enter the shared bank that is scored on real ARC. Safe to stop (STOP file) and resume.

Rewards are real but mechanical: points are recomputed from the saved tensors (`python arena.py verify RUN`), the standings go into the next
prompt, winners' cells are admitted to the bank and credited when they newly solve ARC selection tasks, and a house that is behind sees the
leading house's tensors in the next generation. No gradient, no reward model: selection and information only.

  python arena.py runs/weights-head-arena1 --houses 3 --gens 2 --card wallpruned,tooluse,... --seed-from runs/weights-head-skillbank-v12
  python arena.py verify runs/weights-head-arena1
"""
import argparse
import collections
import json
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_core as A  # noqa: E402
import head_arc_cells as H  # noqa: E402
import head_driver as D  # noqa: E402
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402

CARD = ("wallpruned", "corridormarker", "anchorassign", "frameinfer", "referencepaint", "tooluse", "analogy", "context2", "objmove3", "blockred", "outline")


def header(house, season, standings):
    table = "; ".join("%s %.1f" % (t["house"], t["points"]) for t in standings)
    return ("CELL WARS, season %d. You are House %s. Independent makers write tensors for this same family; each house is shown DIFFERENT example tasks "
            "and all are scored on hidden referee tasks nobody sees, against the best of %d random-tensor entries with the same number of cells and "
            "positions. Points are the margin over that luck baseline, recomputed from your tensors by code, so only tensors that genuinely generalise "
            "pay, and a family that random numbers already satisfy pays nothing. Standings: %s. If previous_attempt carries rival_champion, that is the "
            "leading house's tensors and referee fraction from the last match; you may build on it or ignore it. If no tensors of this template can "
            "express the rule, say so: begin the rationale with 'no valid' and return no cell; an honest decline costs nothing and posts a structure "
            "bounty. Do not use tools or files.\n" % (season, house, A.LUCK_DRAWS, table))


def boss_header(house, season, standings):
    table = "; ".join("%s %.1f" % (t["house"], t["points"]) for t in standings)
    return ("CELL WARS boss fight, season %d. You are House %s. Every house receives the SAME items and must return ONE entry per item with your best "
            "complete revised tensors, even when you doubt they will land. A revision counts only if, added to the shared bank, it strictly raises the score "
            "on real selection tasks without losing any task already solved, and it is compared against a matched random edit with the same number of "
            "changed positions; the first house whose revision lands takes the kill (5 points per task). Standings: %s. If you believe no tensors of the "
            "template can express the rule, ALSO set \"cannot_express\": true on that entry: it posts a structure bounty and costs nothing, but it never "
            "replaces the attempt. Do not use tools or files.\n" % (season, house, table))


def uses_niche(cell, names):
    if cell.get("niche") in names:
        return True
    return any(uses_niche(p, names) for p in cell.get("parts", []) if isinstance(p, dict))


class Arena(D.Driver):
    def __init__(self, root, houses, season, fake=False, workers=3):
        super().__init__(root, fake, workers)
        self.houses, self.season = list(houses), season
        self.ledger = root / "POINTS.jsonl"
        self.rows = [json.loads(line) for line in self.ledger.read_text().splitlines() if line.strip()] if self.ledger.exists() else []
        self.state.setdefault("arena_done", [])
        self.state.setdefault("kills", {})
        for h in self.houses:
            self.state["kills"].setdefault(h, 0)

    # -- persistence
    def append(self, row):
        self.rows.append(row)
        with open(self.ledger, "a") as f:
            f.write(json.dumps(row) + "\n")

    def save_entry(self, name, house, gen, entry):
        path = self.root / "ARENA" / "entries" / name
        path.mkdir(parents=True, exist_ok=True)
        H.write(path / ("%s-g%d.json" % (house, gen)), A.canon_entry(entry))

    def try_ask(self, text, tag, chosen, oracle):
        try:
            return self.ask(text, tag, chosen, oracle), None
        except Exception as error:  # a failed or tool-using maker call is an invalid generation: logged, never credited, never faked
            self.log({"stage": "arena:maker_error", "utc": D.now(), "tag": tag, "error": "%s: %s" % (type(error).__name__, str(error)[:200])})
            return None, error

    # -- one niche contest
    def contest(self, name, gens):
        niche = N.NICHES[name]
        arena = A.build_arena(niche, N.LIMITS[name], self.houses)
        pats, referee, train = arena["patterns"], arena["referee"], arena["train"]
        if len(pats) < 2:
            self.log({"stage": "arena:%s:skip" % name, "utc": D.now(), "note": "fewer than two verifiable patterns"})
            return
        by_id = {p["id"]: p for p in pats}
        st = {h: {"entry": {}, "pass": {}, "attempts": {}, "failing": None, "submitted": 0, "prompts": []} for h in self.houses}
        gen_rows = {}
        entries = {}
        for gen in range(1, gens + 1):
            standings = A.leaderboard(self.rows, self.state["kills"], self.houses)
            jobs = []
            for h in self.houses:
                head = header(h, self.season, standings)
                if niche.general:
                    if gen > 1 and st[h].get("rate", 0) >= 0.9:
                        continue  # already passes its own training tasks: it keeps its best generation and sits this one out
                    tasks = list(train[h].values())
                    shown = random.Random(A.house_index(h) * 31 + gen).sample(tasks, min(3, len(tasks)))
                    failing = st[h]["failing"]
                    if failing and gen > 1:
                        top = max((r for r in gen_rows.get(gen - 1, []) if r["house"] != h), key=lambda r: r["solved"], default=None)
                        mine = next((r for r in gen_rows.get(gen - 1, []) if r["house"] == h), None)
                        if top and mine and top["solved"] > mine["solved"] and entries.get((top["house"], gen - 1), {}).get("*"):
                            failing = dict(failing, rival_champion={"params": entries[(top["house"], gen - 1)]["*"]["params"],
                                                                    "referee_fraction": round(top["solved"] / top["total"], 3)})
                    jobs.append((h, None, head + D.prompt_general(niche, shown, failing), "arena%d-%s-%s-g%d" % (self.season, name, h, gen), [{"id": "general"}],
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
                        jobs.append((h, ids, head + D.prompt_instance(niche, items), "arena%d-%s-%s-g%d-b%d" % (self.season, name, h, gen, b),
                                     [{"id": pid} for pid in ids], lambda ids=ids: [{"task_id": pid, "params": by_id[pid]["truth"], "rationale": "oracle"} for pid in ids]))
            if not jobs:
                break

            def play(job):
                h, ids, text, tag, chosen, oracle = job
                raw, error = self.try_ask(text, tag, chosen, oracle)
                return h, ids, text, raw, error

            with ThreadPoolExecutor(self.workers) as pool:
                results = list(pool.map(play, jobs))
            raws = collections.defaultdict(list)
            for h, ids, text, raw, error in results:
                st[h]["prompts"].append(A.sha(text))
                raws[h].append((ids, raw, error))
            gen_rows[gen] = []
            for h in self.houses:
                if not raws[h] and gen > 1:
                    continue  # sat out (general niche) or nothing left to fix (instance niche)
                cells, rationales, errors = [], [], 0
                for ids, raw, error in raws[h]:
                    errors += error is not None
                    parsed = D.parse_cells(raw, niche.template) if isinstance(raw, list) else []
                    rationales += [e.get("rationale") for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []
                    cells += [c for c in parsed if niche.general or c.get("task_id") in by_id]
                st[h]["submitted"] += sum(S.positions(c) for c in cells)
                if niche.general:
                    if cells:
                        cell = cells[0]
                        rate = float(np.mean([S.passes(cell, t) for t in train[h].values()]))
                        st[h]["entry"] = {"*": cell}
                        st[h]["failing"] = {"params": cell["params"], "failing_tasks": D.counterexamples(cell, list(train[h].values()), 2)}
                        st[h]["rate"] = rate
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
                row = A.score_row(niche, h, gen, entry, referee, A.sha(st[h]["prompts"]), declined=declined, scouted=bool(gen > 1 and niche.general),
                                  season=self.season)
                row["charged_submitted"] = st[h]["submitted"]
                row["maker_errors"] = errors
                row["utc"] = D.now()
                self.append(row)
                self.save_entry(name, h, gen, entry)
                entries[(h, gen)] = entry
                gen_rows[gen].append(row)
                self.log({"stage": "arena:%s:%s:g%d" % (name, h, gen), "utc": row["utc"], "solved": row["solved"], "luck": row["luck"], "total": row["total"],
                          "points": row["points"], "declined": declined, "maker_errors": errors})
            if all(st[h].get("rate", 0) >= 0.9 for h in self.houses) and niche.general:
                break
        self.admit(name, niche, st, referee, train, entries, gens)

    # -- bank admission and ARC kill credit (marginal, recomputed)
    def admit(self, name, niche, st, referee, train, entries, gens):
        best = A.best_rows([r for r in self.rows if r["niche"] == name and r["season"] == self.season])
        order = sorted(self.houses, key=lambda h: -best[(self.season, name, h)]["points"] if (self.season, name, h) in best else 0)
        before = self.cells()
        base = self.sel(before)["solved"]
        admitted = []
        for h in order:
            row = best.get((self.season, name, h))
            if not row or row["points"] <= 0:
                continue
            entry = entries[(h, row["gen"])]
            new = []
            if niche.general:
                cell = entry.get("*")
                if cell and st[h].get("rate", 0) >= 0.9 and row["solved"] / row["total"] >= 0.75:
                    new.append((None, cell))
            else:
                for pid, cell in entry.items():
                    if S.passes(cell, train[h][pid]) and S.passes(cell, referee[pid]) and not any(c.get("pattern") == pid for c in admitted):
                        new.append((pid, cell))
            kept = []
            for pid, cell in new:
                if any(c.get("niche") == name and A.near_duplicate(c, cell) for c in self.registry + admitted):
                    continue
                cell = dict(cell)
                cell.update(niche=name, pattern=pid or "*", house=h, general=bool(niche.general), generation=row["gen"],
                            origin="maker gpt-6.1-sol low, cell wars season %d house %s" % (self.season, h))
                kept.append(cell)
            if not kept:
                continue
            trial = self.sel(before + admitted + kept)["solved"]
            gain = trial - (self.sel(before + admitted)["solved"] if admitted else base)
            for cell in kept:
                cell["arc_kill_credit_house"] = gain
            self.state["kills"][h] = self.state["kills"].get(h, 0) + max(0, gain)
            self.append({"kind": "kills", "season": self.season, "niche": name, "house": h, "tasks": gain, "utc": D.now()})
            admitted += kept
        self.registry.extend(admitted)
        self.state["arena_done"].append(name)
        self.save()
        self.measure("arena:" + name, niche_cells=len(admitted), houses=len(self.houses))
        self.write_board()

    # -- boards
    def write_board(self):
        cells = self.cells()
        cur = self.sel(cells)
        unsolved = [r for r in self.rows_sel if not cur["outcomes"][r["id"]]]
        census = collections.Counter(A.census_category(r["task"]) for r in unsolved)
        codex_names = set(A.CODEX_NICHES)
        no_codex = [c for c in cells if not uses_niche(c, codex_names)]
        head_names = {c.get("niche") for c in cells if c.get("niche") and c.get("niche") not in codex_names and c.get("niche") != "legacy"}
        no_head = [c for c in cells if c.get("niche") == "legacy" or not uses_niche(c, head_names)]
        full = self.arc(cells)
        report_codex = self.split(full["outcomes"])["report"] - self.split(self.arc(no_codex)["outcomes"])["report"]
        tag = "cell wars season %d" % self.season
        mine = [c for c in cells if tag in c.get("origin", "")]
        lose = lambda group: cur["solved"] - self.sel([c for c in cells if c not in group])["solved"]  # noqa: E731
        workshop = {"codex": cur["solved"] - self.sel(no_codex)["solved"], "head": cur["solved"] - self.sel(no_head)["solved"],
                    "codex_report": report_codex,
                    "codex_season": lose([c for c in mine if c.get("niche") in codex_names]),
                    "head_season": lose([c for c in mine if c.get("niche") not in codex_names])}
        table = A.leaderboard(self.rows, self.state["kills"], self.houses)
        best = A.best_rows(self.rows)
        worst = {}
        for (s, n, h), r in best.items():
            if r["total"] and r["luck"] / r["total"] >= 0.25 and (n not in worst or r["luck"] / r["total"] > worst[n][0] / worst[n][1]):
                worst[n] = (r["luck"], r["total"])
        rigged = sorted((n, l, t) for n, (l, t) in worst.items())
        declines = sorted({(n, h) for (s, n, h), r in best.items() if r["declined"]})
        ids = collections.defaultdict(list)
        for r in unsolved:
            ids[A.census_category(r["task"])].append(r["id"])  # selection split only (dev + validation): the report split is never listed
        board = {"utc": D.now(), "season": self.season, "houses": table, "workshops": workshop, "arc_selection_solved": cur["solved"],
                 "arc_selection_total": len(self.rows_sel), "bounties": dict(census), "bounty_task_ids": {k: sorted(v) for k, v in ids.items()},
                 "rigged_tables": rigged, "structure_bounties": declines}
        H.write(self.root / "ARENA_BOARD.json", board)
        lines = ["# CELL WARS, season %d" % self.season, "",
                 "Updated %s. Every number is recomputed from saved tensors: `python scripts/arena.py verify <this run>`." % board["utc"], "",
                 "## Houses", "", "| rank | house | points | match | ARC kills | niches won | titles |", "|---|---|---|---|---|---|---|"]
        for i, t in enumerate(table, 1):
            lines.append("| %d | %s | %.1f | %.1f | %.1f | %d | %s |" % (i, t["house"], t["points"], t["match_points"], t["kill_points"], t["niches_won"],
                                                                    "; ".join(t["titles"][:6]) or "-"))
        lines += ["", "## Workshop duel (ARC selection tasks that would be lost if the workshop's cells were removed)", "",
                  "all time: head workshop %d | codex workshop %d (report split, never optimised: %d)" % (workshop["head"], workshop["codex"], workshop["codex_report"]),
                  "this season (cells admitted by Cell Wars season %d): head %d | codex %d" % (self.season, workshop["head_season"], workshop["codex_season"]), "",
                  "## Bounty board (unsolved ARC selection tasks: %d of %d; %.0f points each to the house whose cells newly solve one)" % (
                      len(unsolved), len(self.rows_sel), A.KILL_POINTS), "", "| family | unsolved | pot |", "|---|---|---|"]
        for fam, n in census.most_common():
            lines.append("| %s | %d | %.0f |" % (fam, n, n * A.KILL_POINTS))
        bosses = [r for r in self.rows if r.get("kind") == "boss" and r["season"] == self.season]
        if bosses:
            lines += ["", "## Boss fights (every house revises the same real ARC near-misses; first strict selection gain is the kill)", "",
                      "| round | house | tried | no entry | cannot_express flags | admitted | kills | matched random edits that would gain |", "|---|---|---|---|---|---|---|---|"]
            lines += ["| %d | %s | %d | %d | %d | %d | %d | %d |" % (b["round"], b["house"], b["tried"], b["declined"], b.get("cannot_express", 0), b["admitted"], b["kills"],
                                                                   b["random_edits_that_would_gain"]) for b in bosses]
        lines += ["", "## Rigged tables (random numbers already score 25% or more: these pay nothing until hardened)", ""]
        lines += ["- %s: best random entry solved %d of %d (worst house draw)" % r for r in rigged] or ["- none"]
        lines += ["", "## Structure bounties (a house said no tensors of the template can express the rule)", ""]
        flags = collections.Counter(f["template"] for b in bosses for f in b.get("flagged", []))
        lines += ["- %s (House %s)" % d for d in declines]
        lines += ["- boss fights: template %s flagged %d times as unable to express the rule" % kv for kv in flags.most_common()]
        if not declines and not flags:
            lines += ["- none"]
        (self.root / "ARENA_BOARD.md").write_text("\n".join(lines) + "\n")

    # -- boss fights: every house revises the SAME real ARC near-misses; the first admitted strict gain is the kill
    def boss_round(self, n_targets=10):
        rnd = self.state.get("boss_round", 0) + 1
        cells = self.cells()
        cur = self.sel(cells)
        targets = self.mutation_targets(cells, cur, n_targets)
        self.state["boss_round"] = rnd
        if not targets:
            self.log({"stage": "arena:boss:round%d" % rnd, "utc": D.now(), "targets": 0, "note": "no near-miss cells on the selection tasks"})
            self.save()
            return 0
        items, text = self.mutation_prompt(targets)
        standings = A.leaderboard(self.rows, self.state["kills"], self.houses)

        def play(h):
            raw, error = self.try_ask(boss_header(h, self.season, standings) + text, "arena%d-boss%d-%s" % (self.season, rnd, h), [{"id": r["id"]} for _, r, _ in targets], lambda: [])
            return h, raw, error

        with ThreadPoolExecutor(self.workers) as pool:
            results = {h: (raw, error) for h, raw, error in pool.map(play, self.houses)}
        turn = (rnd - 1) % len(self.houses)  # who is judged first rotates, so the leader has no standing advantage
        order = self.houses[turn:] + self.houses[:turn]
        admitted, current, killed = [], cur, set()
        rng = random.Random(8000 + rnd)
        for h in order:
            raw, error = results[h]
            by_task = {e.get("task_id"): e for e in raw if isinstance(e, dict)} if isinstance(raw, list) else {}
            tally = {"tried": 0, "declined": 0, "admitted": 0, "kills": 0, "charged": 0, "random_edits_that_would_gain": 0, "cannot_express": 0}
            flagged = []
            for e, row, cell in targets:
                entry = by_task.get(row["id"])
                if entry and entry.get("cannot_express") is True:
                    tally["cannot_express"] += 1
                    flagged.append({"task": row["id"], "template": cell["template"]})
                if row["id"] in killed:
                    continue
                if not entry or not S.valid_params(cell["template"], entry.get("params")):
                    tally["declined"] += 1
                    continue
                new = np.concatenate([np.asarray(v, float).ravel() for v in entry["params"].values()])
                old = np.concatenate([np.asarray(cell["params"][k], float).ravel() for k in entry["params"]])
                if np.allclose(new, old) or str(entry.get("rationale", "")).lower().startswith(A.DECLINE_PHRASES):
                    tally["declined"] += 1
                    continue
                tally["tried"] += 1
                child = S.make_cell(cell["template"], entry["params"], niche=cell["niche"], pattern=cell.get("pattern"), parents=[cell["id"]], house=h,
                                    origin="maker boss round %d house %s, cell wars season %d" % (rnd, h, self.season), generation=rnd)
                tally["charged"] += S.positions(child)
                flat = old.copy()  # matched random edit: same number of changed positions, random values
                for k in rng.sample(range(len(flat)), min(max(1, int((old != new).sum())), len(flat))):
                    flat[k] = rng.uniform(-9, 9)
                rparams, at = {}, 0
                for k, v in cell["params"].items():
                    size = int(np.asarray(v).size)
                    rparams[k] = flat[at:at + size].reshape(np.asarray(v).shape).tolist()
                    at += size
                tally["random_edits_that_would_gain"] += self.sel(cells + admitted + [S.make_cell(cell["template"], rparams, niche=cell["niche"])])["solved"] > current["solved"]
                trial = self.sel(cells + admitted + [child])
                lost = [t for t, v in current["outcomes"].items() if v and not trial["outcomes"][t]]
                if trial["solved"] > current["solved"] and not lost:
                    gain = trial["solved"] - current["solved"]
                    child["arc_selection_gain"] = gain
                    admitted.append(child)
                    current = trial
                    killed.add(row["id"])
                    tally["admitted"] += 1
                    tally["kills"] += gain
                    self.state["kills"][h] = self.state["kills"].get(h, 0) + gain
                    self.append({"kind": "kills", "season": self.season, "niche": "boss%d" % rnd, "house": h, "tasks": gain, "utc": D.now()})
            self.append(dict(kind="boss", season=self.season, round=rnd, house=h, targets=len(targets), maker_errors=int(results[h][1] is not None),
                             prompt_sha256=A.sha(text), utc=D.now(), flagged=flagged, **tally))
            self.log(dict(stage="arena:boss:round%d:%s" % (rnd, h), utc=D.now(), **tally))
        self.registry.extend(admitted)
        self.save()
        self.measure("arena:boss:round%d" % rnd, targets=len(targets), houses=len(self.houses), admitted=len(admitted))
        self.write_board()
        return len(admitted)

    # -- season
    def season_run(self, card, gens, rounds):
        self.baseline()
        for name in card:
            if (self.root / "STOP").exists():
                self.status(state="user_stop")
                return
            if name in self.state["arena_done"]:
                continue
            self.status(stage="arena:" + name, phase="contest", season=self.season)
            self.contest(name, gens)
        idle = 0
        for _ in range(rounds):  # tournament day: winners' cells cross over with the whole bank, near misses are fought over
            if (self.root / "STOP").exists():
                self.status(state="user_stop", season=self.season)
                return
            children = self.breed()["children_admitted"]
            got = self.boss_round() if len(self.houses) > 1 else (self.mutate().get("mutations_admitted") or 0)
            idle = 0 if (children or got) else idle + 1
            if idle >= 2:  # identical near-misses and identical prompts would only re-roll the same fight
                self.write_board()
                self.status(state="plateau", season=self.season, note="two tournament days with no admitted child or boss kill")
                return
        self.write_board()
        self.status(state="season_complete", season=self.season)


def verify(root):
    rows = [json.loads(line) for line in (root / "POINTS.jsonl").read_text().splitlines() if line.strip()]
    entries = {}
    for r in rows:
        if r.get("kind") == "match":
            path = root / "ARENA" / "entries" / r["niche"] / ("%s-g%d.json" % (r["house"], r["gen"]))
            if path.exists():
                entries[(r["niche"], r["house"], r["gen"])] = json.loads(path.read_text())
    problems = A.verify_ledger(rows, entries)
    print("rows %d, problems %d" % (len(rows), len(problems)))
    for p in problems:
        print(" ", p)
    return 1 if problems else 0


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "verify":
        sys.exit(verify(pathlib.Path(sys.argv[2])))
    if len(sys.argv) > 2 and sys.argv[1] == "board":  # rewrite the board of a finished run with the current scoring code (no maker calls)
        root = pathlib.Path(sys.argv[2])
        houses = sorted({json.loads(l)["house"] for l in (root / "POINTS.jsonl").read_text().splitlines() if l.strip() and "house" in l}, key=A.HOUSES.index)
        Arena(root, houses, int(sys.argv[3]) if len(sys.argv) > 3 else 1).write_board()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--houses", type=int, default=3)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--season", type=int, default=1)
    ap.add_argument("--card", default=",".join(CARD))
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed-from", type=pathlib.Path)
    a = ap.parse_args()
    arena = Arena(a.root, A.HOUSES[:a.houses], a.season, a.fake, a.workers)
    if a.seed_from and not arena.registry:
        old = H.load(a.seed_from / "REGISTRY.json")
        arena.registry = [c for c in old if (c["kind"] == "cell" and c.get("origin", "").startswith("maker")) or c.get("arc_selection_gain")]
        arena.state["done"] = sorted({c["niche"] for c in arena.registry if c.get("niche") in N.NICHES})
        arena.save()
    arena.season_run([n for n in a.card.split(",") if n], a.gens, a.rounds)


if __name__ == "__main__":
    main()
