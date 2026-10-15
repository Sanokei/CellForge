"""LLM-managed weight writing for ARC-AGI-3 (server). Orchestrator regime 2026-10-02.

Roles (all LLM roles are the owner-configured maker route through scripts/maker.py, metered against the shared weekly limit):
  manager  one call per cycle. Reads a digest of every game (attempts, best levels, credited levels, last failure) and the budget meter, and orders the
           next writer slots: which game, which house, and a focused revision instruction. It writes no weights.
  writers  houses Ember/Tide/Moss/Flint. One call = one policy cell for one game: restricted Python `act(params, mem, frame, avail, info)` plus literal tensors.
  verifier plain code. Plays the cell, runs matched controls (zeroed tensors, 3 random draws, BLIND frame), and credits levels only when the cell beats all of
           them. `verify` replays every saved best cell in a fresh process and recomputes the levels from the stored weights; nothing self-reported counts.

What the manager is accountable for: **no cycle ends without weights being submitted.** If the manager call fails, is empty, or names invalid games, the
slots are filled deterministically (fewest attempts, fewest credited levels first) and the cycle still runs. An honest decline is not an exit: a writer that
says "cannot express" must still submit its best attempt, which is played and charged like any other.

  python lab_manager.py run runs/weights-arc3-managed1 --cycles 6 --slots 3 --games ls20,vc33,tu93
  python lab_manager.py status runs/weights-arc3-managed1
  python lab_manager.py verify runs/weights-arc3-managed1
  python lab_manager.py transfer runs/weights-arc3-managed1
"""
import argparse
import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arc3_env as E  # noqa: E402
import arc3cell as C  # noqa: E402
import maker as M  # noqa: E402

HOUSES = ("Ember", "Tide", "Moss", "Flint")
BUDGET_STEPS = 1500          # game actions per play (matched for the cell, its controls and the random agent)
MAX_ATTEMPTS_PER_GAME = 6    # writer calls per game over the whole run; a game that already has a credited level may use twice as many (push for the next level)
SOURCE_HINT = 12000


def attempt_cap(game_state):
    return MAX_ATTEMPTS_PER_GAME * (2 if game_state["credited"] > 0 else 1)


def now():
    return M.now()


def jload(path, default=None):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else default


_io_lock = threading.RLock()  # writers run in threads; STATE.json and the ledgers are shared


def jsave(path, obj):
    p = pathlib.Path(path)
    with _io_lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp%d" % threading.get_ident())
        tmp.write_text(json.dumps(obj, indent=1))
        tmp.replace(p)


def append(path, row):
    with _io_lock, open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ prompts
CONTRACT = """You write ONE policy cell for an ARC-AGI-3 game: restricted Python plus literal weight tensors (no tools, no files, no network, no imports).

def act(params, mem, frame, avail, info):
    # params: dict tensor_name -> float numpy array, exactly the shapes you declare. Every task-specific number lives here: which colour is the agent / a wall /
    #         a target, step sizes, the priority order of actions, thresholds, click targets, how many repeats before switching.
    # mem:    a dict that persists across the steps of ONE game run. Use it to learn online: previous frame, what each action did, visited state keys, plans.
    # frame:  2-D numpy int array 64x64 with colours 0..15 (the settled frame after the last action).
    # avail:  list of legal action ids. 1..5 are simple actions; 6 is a click at (x, y) with x = column, y = row in 0..63.
    # info:   dict(levels=int completed so far, resets=int, step=int actions used, key=str digest of this frame: equal keys = identical frames).
    # return: (action_id, x, y); x and y only matter for action 6. The harness issues RESET itself after GAME_OVER (it costs one action).

Rules checked by code:
- Only `def` at top level (helper functions are fine), `np` is available. No imports, try/with/class/global/f-strings/decorators, no names or attributes that
  start with an underscore, no open/exec/eval/getattr/print/hash. Numpy only through named functions (array, zeros, where, argwhere, nonzero, unique, bincount,
  argmax, abs, sum, any, all, roll, flip, concatenate, ... and the usual list/dict/set methods incl. setdefault).
- At most 256 tensor entries in total, finite numbers within [-1000, 1000]. The tensors are the genome: keep them small and meaningful.
- Your cell is judged by levels completed within the step budget, against (a) the SAME code with zeroed tensors, (b) the same code with three random tensor draws,
  and (c) the same cell run BLIND (the frame replaced by zeros and info["key"] held constant). It earns credit only if it beats all of them, so a stored/replayed action sequence earns nothing:
  the policy must read the frame, and the tensors must matter.
- Games differ and their rules are not given. Learn them online inside the policy: apply an action, compare the frame with the previous one (mem), find which
  pixels/objects moved, infer goals from what changes when a level is completed (info['levels'] increases), avoid repeating states (info['key']), explore
  systematically (BFS/DFS over keys), and use clicks on distinct objects when 6 is available. A level counts the moment info['levels'] goes up.
Return ONLY one JSON array with ONE object, no markdown:
[{"game": str, "source_lines": [one string per line of Python], "shapes": {"tensor_name": [dims]}, "params": {"tensor_name": nested lists}, "rationale": str,
  "cannot_express": false}]
If you believe no policy of this kind can win, still submit your best attempt (it is played and charged like any other) and set "cannot_express": true.
"""


def game_card(probe):
    lines = ["game %s | tags %s | actions available at start %s | levels %d | human baseline actions per level %s" % (
        probe["game"], probe["tags"], probe["available"], probe["n_levels"], probe["human_baseline_actions"])]
    lines.append("colour pixel counts in the first frame: %s" % probe["colours"])
    for a, steps in probe["actions"].items():
        lines.append("action %s applied 3 times from the first frame -> per-step change: %s" % (a, json.dumps(steps)))
    if probe["clicks"]:
        lines.append("click probes (largest components of the first frame, one fresh game each): %s" % json.dumps(probe["clicks"]))
    lines.append("first frame (64 rows, hex colours, '.' = 0):\n%s" % probe["first_frame"])
    return "\n".join(lines)


def result_card(rec):
    if rec.get("rejected"):
        return "REJECTED by the static whitelist before it was ever played: %s. Fix exactly that and resubmit." % rec["rejected"]
    r = rec["result"]
    controls = rec.get("controls")
    lines = ["levels %d of %d, level completed at actions %s, actions used %d, resets %d, no-op actions %d, distinct frames visited %d, final state %s" % (
        r["levels"], r["n_levels"], r["level_steps"], r["used"], r["resets"], r["noops"], r["distinct"], r["state"]), "action histogram %s" % r["action_hist"]]
    if r.get("error"):
        lines.append("run ended with error: %s" % r["error"])
    if controls:
        lines.append("controls at the same positions: zero %d levels, random draws %s, blind %d (credited only if you beat all of them; yours %d)" % (
            controls["zero"], controls["random"], controls["blind"], r["levels"]))
    if r.get("final_frame"):
        lines.append("final frame:\n%s" % r["final_frame"])
    return "\n".join(lines)


def writer_prompt(house, probe, prior, instruction, second=None, donor=None):
    parts = ["CELL WARS ARC-3. You are House %s, a weight writer. Other houses write independent cells for the same game and cannot see yours." % house, CONTRACT,
             "GAME FACTS (from a scripted probe; no level solutions are included):\n" + game_card(probe)]
    if prior:
        parts.append("YOUR PREVIOUS ATTEMPT (best so far for this game) and what happened. Write a genuinely different, better revision; do not resubmit it unchanged.\n"
                     "source:\n%s\nshapes: %s\nparams: %s\nresult:\n%s" % ("\n".join(prior["source_lines"]), json.dumps(prior["shapes"]), json.dumps(prior["params"]),
                                                                              result_card(prior)))
    if second:
        parts.append("CROSSOVER: a SECOND cell from another house for the same game is shown below. Write a CHILD that keeps what works in each (compare their results), "
                     "not a copy of either.\nsecond source:\n%s\nshapes: %s\nparams: %s\nresult:\n%s" % ("\n".join(second["source_lines"]), json.dumps(second["shapes"]),
                                                                                                      json.dumps(second["params"]), result_card(second)))
    if donor:
        dgame, dcell = donor
        parts.append("DONOR CELL (cross-game transfer): this cell completed levels in a DIFFERENT game (%s), beating its zero/random/blind controls there. Games differ, but its "
                     "online-learning machinery may carry over. Adapt it to THIS game: keep the mechanism that worked, change what this game's probes and your previous attempt "
                     "show does not fit. Do not copy it blindly.\ndonor source:\n%s\nshapes: %s\nparams: %s\nresult in %s:\n%s" % (
                         dgame, "\n".join(dcell["source_lines"]), json.dumps(dcell["shapes"]), json.dumps(dcell["params"]), dgame, result_card(dcell)))
    if instruction:
        parts.append("MANAGER INSTRUCTION for this attempt: " + instruction)
    return "\n\n".join(parts)


def manager_prompt(state, slots, meter, second_cells=None, donors=None):
    second_cells = second_cells or {}
    donors = donors or {}
    rows = []
    for g, s in sorted(state["games"].items()):
        last = s["history"][-1] if s["history"] else None
        rows.append({"game": g, "tags": s["tags"], "levels_total": s["n_levels"], "attempts": s["attempts"], "best_levels": s["best_levels"], "credited": s["credited"],
                     "second_cell": bool(second_cells.get(g)),
                     "last": None if not last else {"house": last["house"], "levels": last["levels"], "error": last.get("error"), "noop_rate": last.get("noop_rate"),
                                                    "distinct": last.get("distinct"), "invalid": last.get("invalid"), "note": last.get("note")}})
    return ("You are the MANAGER of the weight-writing houses (%s) in an ARC-AGI-3 lab. You never write weights; you decide which writer slots run next and what each\n"
            "writer should do differently. The lab's only goal this cycle: submit weights that complete more levels than their zero/random/blind controls. Idle slots,\n"
            "declines and neutral rewrites are your failures. Weekly maker budget used: %s%% (stop at %s%%). Slots available: %d.\n"
            "State per game (credited = levels that beat all controls; attempts are capped at %d per game, doubled once a game has a credited level; second_cell = a played "
            "cell from a different house exists, so mode crossover is possible). Donor games (credited cells that can be ADAPTED into another game): %s\n%s\n"
            "Return ONLY JSON: {\"orders\": [{\"game\": str, \"house\": one of the houses, \"instruction\": str of at most 400 characters, concrete (what to probe/learn "
            "online, which failure to fix, e.g. 'stuck in a loop: add visited-key avoidance' or 'click games: try the other components'), \"mode\": \"revise\" or "
            "\"crossover\" (only where second_cell is true: merge two houses' cells) or \"adapt\" (also set \"donor\": a donor game id; the writer adapts that game's credited "
            "cell to this game: cross-game transfer, worth trying on games where plain revisions keep scoring 0)}], \"summary\": str}. "
            "At most %d orders, distinct games, prefer games where a revision can add a level or where nothing was tried." % (
                ", ".join(HOUSES), meter, M.cap(), slots, MAX_ATTEMPTS_PER_GAME, json.dumps(donors), json.dumps(rows), slots))



# ------------------------------------------------------------------ state
class Lab:
    def __init__(self, root, games):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("PROBES", "CELLS"):
            (self.root / sub).mkdir(exist_ok=True)
        self.state = jload(self.root / "STATE.json", {"games": {}, "cycle": 0, "created": now()})
        for g in games:
            if g not in self.state["games"]:
                info = E.info_of(g)
                self.state["games"][g] = {"tags": info.tags, "n_levels": len(info.baseline_actions), "attempts": 0, "best_levels": 0, "credited": 0, "best_cell": None,
                                          "history": []}
        self.save()

    def save(self):
        jsave(self.root / "STATE.json", self.state)

    def probe(self, game):
        path = self.root / "PROBES" / (game + ".json")
        if not path.exists():
            jsave(path, E.probe(game))
        return jload(path)

    def baselines(self):
        path = self.root / "RANDOM_BASELINE.json"
        data = jload(path, {})
        for g in self.state["games"]:
            if g not in data:
                data[g] = E.random_agent(g, BUDGET_STEPS)
                jsave(path, data)
        return data

    # one writer attempt: maker call -> validate -> play -> controls
    def second_cell(self, game):
        """Best played cell of this game from a different house than the current best (levels > 0), or None. Crossover needs two."""
        s = self.state["games"][game]
        best = jload(s["best_cell"]) if s["best_cell"] else None
        pool = []
        for path in sorted((self.root / "CELLS").glob(game + "-*.json")):
            cell = jload(path)
            if cell and not cell.get("rejected") and "result" in cell and (best is None or cell["house"] != best["house"]) and cell["result"]["levels"] > 0:
                pool.append((cell["result"]["levels"], cell))
        return max(pool, key=lambda x: x[0])[1] if pool else None

    def donor_cells(self):
        """Credited, played best cells by game: the sources for cross-game adaptation."""
        out = {}
        for g, s in self.state["games"].items():
            if s["credited"] > 0 and s["best_cell"]:
                cell = jload(s["best_cell"])
                if cell and not cell.get("rejected") and "result" in cell:
                    out[g] = cell
        return out

    def attempt(self, game, house, instruction, mode="revise", donor=None):
        s = self.state["games"][game]
        n = s["attempts"] + 1
        s["attempts"] = n
        self.save()
        work = self.root / ("AUTHOR-%s-%s-n%d" % (game, house, n))
        prior = jload(s["best_cell"]) if s["best_cell"] else None
        second = self.second_cell(game) if mode == "crossover" and prior else None
        donors = self.donor_cells()
        donor_pair = (donor, donors[donor]) if mode == "adapt" and donor in donors and donor != game else None
        probe = self.probe(game)
        rec = {"utc": now(), "cycle": self.state["cycle"], "game": game, "house": house, "n": n, "instruction": instruction, "mode": "adapt" if donor_pair else ("crossover" if second else "revise"),
               "donor": donor_pair[0] if donor_pair else None, "levels": 0, "credited": 0, "invalid": None, "positions": 0}
        text = writer_prompt(house, probe, prior, instruction, second, donor_pair)
        try:
            raw = M.ask(text, work, role="writer", track="arc3")
        except M.BudgetExhausted:
            s["attempts"] = n - 1
            raise
        except Exception as error:  # noqa: BLE001 a failed or tool-using maker call is an invalid, charged attempt
            rec["invalid"] = "maker: %s: %s" % (type(error).__name__, str(error)[:200])
            return self.finish(game, rec, None)
        entry = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], dict) else {}
        source = "\n".join(str(x) for x in entry["source_lines"]) if isinstance(entry.get("source_lines"), list) else str(entry.get("source") or "")
        shapes, params = entry.get("shapes"), entry.get("params")
        ok, why = C.validate(source)
        if ok:
            ok, why = C.check_shapes(shapes or {}, params or {})
        rec["cannot_express"] = entry.get("cannot_express") is True
        cell = {"game": game, "house": house, "n": n, "source_lines": source.split("\n"), "shapes": shapes or {}, "params": params or {},
                "rationale": str(entry.get("rationale", ""))[:400]}
        if not ok:
            rec["invalid"] = why
            rec["note"] = "rejected: " + why
            cell["rejected"] = why
            return self.finish(game, rec, cell if source.strip() else None)
        rec["positions"] = C.positions(shapes)
        t0 = time.time()
        result = E.play(source, params, game, BUDGET_STEPS)
        cell["result"] = result
        rec.update({"levels": result["levels"], "error": result.get("error"), "noop_rate": round(result["noops"] / max(1, result["used"]), 3), "distinct": result["distinct"]})
        if result["levels"] > 0:
            ctl = E.controls(source, params, game, BUDGET_STEPS)
            cell["controls"] = {"zero": ctl["zero"]["levels"], "random": [c["levels"] for c in ctl["random"]], "blind": ctl["blind"]["levels"]}
            beaten = max([cell["controls"]["zero"], cell["controls"]["blind"]] + cell["controls"]["random"])
            rec["credited"] = result["levels"] if result["levels"] > beaten else 0
            rec["controls"] = cell["controls"]
        rec["play_seconds"] = round(time.time() - t0, 1)
        return self.finish(game, rec, cell)

    def finish(self, game, rec, cell):
        s = self.state["games"][game]
        s["history"].append(rec)
        append(self.root / "ATTEMPTS.jsonl", rec)
        if cell is not None:
            jsave(self.root / "CELLS" / ("%s-%s-n%d.json" % (game, rec["house"], rec["n"])), cell)
            played = 0 if cell.get("rejected") else 1
            best = jload(s["best_cell"]) if s["best_cell"] else None
            best_rank = (s["credited"], s["best_levels"], 0 if (best is None or best.get("rejected")) else 1)
            if best is None or (rec["credited"], rec["levels"], played) > best_rank:
                s["credited"], s["best_levels"] = max(s["credited"], rec["credited"]), max(s["best_levels"], rec["levels"])
                s["best_cell"] = str(self.root / "CELLS" / ("%s-%s-n%d.json" % (game, rec["house"], rec["n"])))
        self.save()
        return rec

    # one cycle: the manager orders the slots, writers run in parallel
    def orders(self, slots):
        meter = M.last_used()[0]
        open_games = [g for g, s in self.state["games"].items() if s["attempts"] < attempt_cap(s) and s["best_levels"] < s["n_levels"]]  # a won game needs no more writers
        orders = []
        try:
            reply = M.ask(manager_prompt(self.state, slots, meter, {g: self.second_cell(g) for g in open_games},
                                                {g: c["result"]["levels"] for g, c in self.donor_cells().items()}), self.root / ("MANAGER-c%d" % (self.state["cycle"] + 1)), role="manager", track="arc3")
            raw = reply if isinstance(reply, dict) else (reply[0] if isinstance(reply, list) and reply else {})
            for o in raw.get("orders", [])[:slots]:
                if isinstance(o, dict) and o.get("game") in open_games and o["game"] not in [x["game"] for x in orders]:
                    orders.append({"game": o["game"], "house": o["house"] if o.get("house") in HOUSES else HOUSES[len(orders) % len(HOUSES)],
                                   "instruction": str(o.get("instruction", ""))[:400], "mode": o["mode"] if o.get("mode") in ("crossover", "adapt") else "revise",
                                   "donor": o.get("donor") if isinstance(o.get("donor"), str) else None, "by": "manager"})
            self.state.setdefault("manager_summaries", []).append({"cycle": self.state["cycle"] + 1, "summary": str(raw.get("summary", ""))[:600]})
        except M.BudgetExhausted:
            raise
        except Exception as error:  # noqa: BLE001 the manager failing must not stop the weights being written
            self.state.setdefault("manager_summaries", []).append({"cycle": self.state["cycle"] + 1, "summary": "manager failed: %s" % type(error).__name__})
        if len(orders) < slots:   # the guarantee: slots are never left idle
            taken = {o["game"] for o in orders}
            pool = sorted((g for g in open_games if g not in taken), key=lambda g: (self.state["games"][g]["credited"] > 0, self.state["games"][g]["attempts"], g))
            for g in pool[:slots - len(orders)]:
                orders.append({"game": g, "house": HOUSES[(len(orders) + self.state["cycle"]) % len(HOUSES)], "instruction": "", "mode": "revise", "by": "fallback"})
        return orders

    def cycle(self, slots):
        orders = self.orders(slots)
        if not orders:
            return []
        return self.run_orders(orders)

    def ab_round(self, slots, rng):
        """Randomised A/B round (no manager call): games with no credited level get a plain revision (arm revise) or a revision plus a donor cell from another
        game (arm adapt), assigned by a seeded coin, house and donor also drawn at random. The only difference between the arms is the donor."""
        pool = [g for g, s in self.state["games"].items() if s["credited"] == 0 and s["attempts"] < attempt_cap(s) and s["best_levels"] < s["n_levels"]]
        donors = sorted(self.donor_cells())
        rng.shuffle(pool)
        orders = []
        for g in sorted(pool[:slots]):
            others = [d for d in donors if d != g]
            arm = rng.choice(["revise", "adapt"]) if others else "revise"
            orders.append({"game": g, "house": HOUSES[rng.randrange(len(HOUSES))], "instruction": "", "mode": arm, "donor": rng.choice(others) if arm == "adapt" else None, "by": "ab"})
        return self.run_orders(orders) if orders else []

    def run_orders(self, orders):
        self.state["cycle"] += 1
        self.save()
        append(self.root / "ORDERS.jsonl", {"utc": now(), "cycle": self.state["cycle"], "orders": orders})
        with ThreadPoolExecutor(len(orders)) as pool:
            futures = [pool.submit(self.attempt, o["game"], o["house"], o["instruction"], o.get("mode", "revise"), o.get("donor")) for o in orders]
            out = []
            for f in futures:
                try:
                    out.append(f.result())
                except M.BudgetExhausted as error:
                    out.append({"budget": str(error)})
                except Exception as error:  # noqa: BLE001 one crashed writer must not stop the other slots or the loop
                    append(self.root / "ERRORS.jsonl", {"utc": now(), "cycle": self.state["cycle"], "error": "%s: %s" % (type(error).__name__, str(error)[:300])})
                    out.append({"crash": "%s: %s" % (type(error).__name__, str(error)[:120])})
        self.write_status()
        return out

    def write_status(self):
        total = sum(s["credited"] for s in self.state["games"].values())
        best = sum(s["best_levels"] for s in self.state["games"].values())
        rb = jload(self.root / "RANDOM_BASELINE.json", {})
        rlev = sum(v["levels"] for g, v in rb.items() if g in self.state["games"])
        meter = M.last_used()[0]
        lines = ["# ARC-3 managed weight writing: status", "", "updated %s | cycle %d | weekly maker meter %s%% (cap %s%%)" % (now(), self.state["cycle"], meter, M.cap()), "",
                 "credited levels (beat zero, random and blind controls): **%d** | raw best levels over games: %d | random agent at the same budget (%d actions): %d" % (
                     total, best, BUDGET_STEPS, rlev), "", "| game | levels | attempts | best | credited | last note |", "|---|---|---|---|---|---|"]
        for g, s in sorted(self.state["games"].items()):
            last = s["history"][-1] if s["history"] else {}
            lines.append("| %s | %d | %d | %d | %d | %s |" % (g, s["n_levels"], s["attempts"], s["best_levels"], s["credited"], (last.get("note") or last.get("error") or "")[:60]))
        (self.root / "STATUS.md").write_text("\n".join(lines) + "\n")


def verify(root):
    """Replay every game's best cell from its saved weights (fresh policy worker, fresh game) and recompute levels and controls."""
    lab = jload(pathlib.Path(root) / "STATE.json")
    report = []
    for g, s in sorted(lab["games"].items()):
        if not s["best_cell"]:
            continue
        cell = jload(s["best_cell"])
        source = "\n".join(cell["source_lines"])
        result = E.play(source, cell["params"], g, BUDGET_STEPS)
        ctl = E.controls(source, cell["params"], g, BUDGET_STEPS) if result["levels"] else None
        beaten = max([ctl["zero"]["levels"], ctl["blind"]["levels"]] + [c["levels"] for c in ctl["random"]]) if ctl else 0
        report.append({"game": g, "claimed_levels": cell["result"]["levels"], "recomputed_levels": result["levels"], "controls_best": beaten,
                       "credited": result["levels"] if result["levels"] > beaten else 0, "matches_claim": result["levels"] == cell["result"]["levels"]})
        print(json.dumps(report[-1]), flush=True)
    jsave(pathlib.Path(root) / "VERIFY.json", {"utc": now(), "rows": report, "credited_total": sum(r["credited"] for r in report)})
    print("recomputed credited levels:", sum(r["credited"] for r in report))


def transfer(root, only=None, steps=BUDGET_STEPS):
    """Cross-game transfer (CPU only, no maker calls): each credited cell is run, unchanged, on every other game. A transfer counts when the cell completes more
    levels there than its own zeroed tensors, three random draws and its blind run. One model across games is exactly this question."""
    lab = jload(pathlib.Path(root) / "STATE.json")
    cells = [(g, jload(s["best_cell"])) for g, s in sorted(lab["games"].items()) if s["credited"] > 0 and s["best_cell"] and (not only or g in only)]
    rows = []
    for home, cell in cells:
        source = "\n".join(cell["source_lines"])
        for g in sorted(lab["games"]):
            if g == home:
                continue
            res = E.play(source, cell["params"], g, steps, wall=90.0)
            if res["levels"] == 0:
                continue
            ctl = E.controls(source, cell["params"], g, steps)
            beaten = max([ctl["zero"]["levels"], ctl["blind"]["levels"]] + [c["levels"] for c in ctl["random"]])
            rows.append({"cell_from": home, "played_on": g, "levels": res["levels"], "controls_best": beaten, "transfers": res["levels"] > beaten})
            print(json.dumps(rows[-1]), flush=True)
    jsave(pathlib.Path(root) / "TRANSFER.json", {"utc": now(), "cells": [g for g, _ in cells], "rows": rows, "transfers": sum(r["transfers"] for r in rows)})
    print("cross-game transfers:", sum(r["transfers"] for r in rows), "from", len(cells), "credited cells")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "status", "verify", "transfer", "ab"))
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--games", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=BUDGET_STEPS)
    a = ap.parse_args()
    if a.command == "verify":
        return verify(a.root)
    if a.command == "transfer":
        return transfer(a.root, set(a.games.split(",")) if a.games else None, a.steps)
    games = a.games.split(",") if a.games else E.games()
    lab = Lab(a.root, games)
    if a.command == "status":
        lab.write_status()
        print((a.root / "STATUS.md").read_text())
        return
    lab.baselines()
    if a.command == "ab":
        import random
        rng = random.Random(a.seed)
        for _ in range(a.cycles):
            if (a.root / "STOP").exists():
                break
            try:
                out = lab.ab_round(a.slots, rng)
            except M.BudgetExhausted as error:
                print("budget exhausted:", error, flush=True)
                break
            if not out:
                print("no zero-credit games left", flush=True)
                break
            for rec in out:
                print(json.dumps({k: rec.get(k) for k in ("game", "house", "mode", "donor", "levels", "credited", "invalid", "budget", "crash")}), flush=True)
        lab.write_status()
        return
    for _ in range(a.cycles):
        if (a.root / "STOP").exists():
            print("STOP file present", flush=True)
            break
        try:
            out = lab.cycle(a.slots)
        except M.BudgetExhausted as error:
            print("budget exhausted:", error, flush=True)
            break
        if not out:
            print("no open games", flush=True)
            break
        for rec in out:
            print(json.dumps({k: rec.get(k) for k in ("game", "house", "levels", "credited", "invalid", "error", "budget")}), flush=True)
    lab.write_status()
    print((a.root / "STATUS.md").read_text())


if __name__ == "__main__":
    main()
