"""Continuous head-center driver: niche stages (maker writes literal-weight cells) -> breeding rounds (chains of earned cells) -> ARC scoring.

State lives in ROOT (REGISTRY.json, STATE.json, HISTORY.jsonl, LEADERBOARD.json, STATUS.json, AUTHOR-*/). Safe to stop (STOP file) and resume.
Maker: server `codex exec -m gpt-6.1-sol -c model_reasoning_effort=low` (owner-configured ChatGPT-subscription route; read-only; no API key).
Scoreboard: 100 DEV tasks and 579 held-out public-training tasks (audit population and evaluation split excluded); the maker sees only generated
niche tasks. Controls: matched random cells (same templates, same number of literal positions) and a random-chain control for breeding.
"""
import argparse
import hashlib
import os
import json
import pathlib
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import head_arc_cells as H  # noqa: E402
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402

RUN_SRC = pathlib.Path(H.RUN_SRC)
DATA = pathlib.Path(H.DATA)
TILE_RUN = pathlib.Path("/srv/cellforge-inbox/head-20261001/runs/weights-head-niche-tile")
QUEUE = [(n, N.LIMITS[n]) for n in ("tiling", "association", "sequence", "glyph", "neighbor", "object", "crop", "ray",
                                    "recall", "induction", "seqops", "selcopy", "grammar", "successor", "agree",
                                    "gravity", "fillenc", "symmetry", "scale", "halfbool", "denoise",
                                    "objmove", "floodm", "raycross", "context2", "objmove2", "floodthru", "snapline", "ocrdigit", "ocrseq", "legend", "grammar2", "analogy", "tooluse", "lineconn", "cropcolor", "blockpick", "seqcrop", "countout", "outline", "objmove3", "blockred", "selfkron")]
SEED = 11
QUEUE += [(name, N.LIMITS[name]) for name in ('wallpruned', 'corridormarker', 'anchorassign', 'frameinfer', 'referencepaint')]


def now():
    return H.now()


def rand_cell(template, rng):
    return S.make_cell(template, {k: np.array([rng.uniform(-9, 9) for _ in range(int(np.prod(s)))]).reshape(s).tolist()
                                  for k, s in S.TEMPLATES[template]["shapes"].items()})


def parse_cells(raw, template):
    cells = []
    for entry in raw if isinstance(raw, list) else []:
        params = entry.get("params")
        if S.valid_params(template, params):
            cells.append(S.make_cell(template, params, task_id=entry.get("task_id"), rationale=str(entry.get("rationale", ""))[:300]))
    return cells


def json_task(task):
    return json.dumps(task)


# ------------------------------------------------------------------ prompts
def prompt_instance(niche, items):
    t = S.TEMPLATES[niche.template]
    return ("You write literal weight tensors for a grid cell (no code, no tools, no files). Template %s, tensors %s. Semantics: %s\n"
            "For EACH task below, infer from the demonstrations how the output is built from the input and write complete tensors (finite numbers in "
            "[-9, 9], every entry present; use 0 for unused) so the cell reproduces EVERY demonstration and the same rule on a new input. If "
            "previous_failed_attempt is present, read its predicted vs expected and write a genuinely different revision. Return ONLY a JSON array, "
            "one object per task: [{\"task_id\": str, \"params\": {tensor_name: nested lists}, \"rationale\": str naming concrete demonstration evidence}]."
            "\nTasks:\n%s" % (niche.template, {k: list(v) for k, v in t["shapes"].items()}, t["doc"], json.dumps(items)))


def prompt_general(niche, tasks, failing):
    t = S.TEMPLATES[niche.template]
    body = {"example_tasks": [{"demonstrations": x["train"]} for x in tasks]}
    if failing:
        body["previous_attempt"] = failing
    return ("You write the literal weight tensors of ONE grid cell (no code, no tools, no files) that must solve EVERY task of a family. Template %s, "
            "tensors %s. Semantics: %s\nThe example tasks below are samples of the family: each has its own demonstration pairs, and the same cell must "
            "solve all of them (and unseen ones of the family) given only its own demonstrations. If previous_attempt is present it lists your last "
            "tensors and failing tasks with predicted vs expected: write a genuinely different, better revision. Return ONLY a JSON array with ONE "
            "object: [{\"task_id\": \"general\", \"params\": {tensor_name: nested lists}, \"rationale\": str naming concrete evidence}]. Finite numbers "
            "in [-9, 9], every entry present.\n%s" % (niche.template, {k: list(v) for k, v in t["shapes"].items()}, t["doc"], json.dumps(body)))


def counterexamples(cell, tasks, limit=2):
    out = []
    for task in tasks:
        if len(out) >= limit:
            break
        if S.passes(cell, task):
            continue
        t = task["test"][0]
        try:
            pred = S.run(cell, t["input"], task["train"])
        except Exception as error:
            pred = "error:" + type(error).__name__
        out.append({"demonstrations": task["train"], "query": t["input"], "expected": t["output"], "predicted": pred})
    return out


# ------------------------------------------------------------------ driver
class Driver:
    def __init__(self, root, fake=False, workers=3):
        self.root, self.fake, self.workers = root, fake, workers
        root.mkdir(parents=True, exist_ok=True)
        for name in ("pop1_core.py", "parent.json"):
            if not (root / name).exists():
                (root / name).write_bytes((RUN_SRC / name).read_bytes())
        for name in ("skillcore.py", "wall_skill.py", "wall_avoid_skill.py", "wall_pruned_skill.py", "anchor_skill.py", "frame_skill.py", "reference_skill.py", "anchor_clip_skill.py", "anchor_assign_skill.py", "frame_infer_skill.py", "reference_paint_skill.py", "corridor_skill.py", "structcode.py", "structlib.py"):
            (root / name).write_bytes((pathlib.Path(__file__).resolve().parent / name).read_bytes())
        (root / "arc_eval.py").write_text("import importlib.util\nimport sys\nfrom pathlib import Path\n_h = Path(__file__).parent\nsys.path.insert(0, str(_h))\n_s = importlib.util.spec_from_file_location('skillcore', _h / 'skillcore.py')\n"
                                          "_m = importlib.util.module_from_spec(_s)\n_s.loader.exec_module(_m)\n_m.load_legacy(_h / 'pop1_core.py')\n\ndef predict(weights, task):\n    return _m.predict(weights, task)\n")
        S.load_legacy(root / "pop1_core.py")
        parent = H.load(root / "parent.json")["tensors"]
        self.legacy, _ = S.from_tensors(parent)
        for c in self.legacy:
            c["niche"] = "legacy"
        self.legacy_router = (parent["route.W"], parent["route.b"])
        dev_ids = {x["id"] for x in H.load(RUN_SRC / "TASKS.json")["development"]}
        audit = lambda t: any(int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5 == 0 for s in (t, t + ".json"))  # noqa: E731
        dev_rows = [{"id": i, "task": H.load(DATA / (i + ".json"))} for i in sorted(dev_ids)]
        held_rows = [{"id": p.stem, "task": H.load(p)} for p in sorted(DATA.glob("*.json")) if p.stem not in dev_ids and not audit(p.stem)]
        in_val = lambda r: int(hashlib.sha256(("val:" + r["id"]).encode()).hexdigest(), 16) % 2 == 0  # noqa: E731
        self.rows_dev, self.rows_val = dev_rows, [r for r in held_rows if in_val(r)]
        self.rows_rep = [r for r in held_rows if not in_val(r)]  # REPORT split: never used to select anything
        self.rows_sel = self.rows_dev + self.rows_val
        self.all = S.Scorer(dev_rows + held_rows)
        self.cache_path = root.resolve().parent.parent / "ERRCACHE.pkl"  # shared by every run in this directory: restarts are warm
        self.all.load_cache(self.cache_path)
        self.registry = H.load(root / "REGISTRY.json") if (root / "REGISTRY.json").exists() else []
        self.state = H.load(root / "STATE.json") if (root / "STATE.json").exists() else {"done": [], "tried_pairs": [], "breed_round": 0, "makers_calls": 0}
        self.base = None

    # -- persistence
    def save(self):
        H.write(self.root / "REGISTRY.json", self.registry)
        H.write(self.root / "STATE.json", self.state)
        self.all.save_cache(self.cache_path)

    def status(self, **kw):
        H.write(self.root / "STATUS.json", dict(kw, utc=now(), cells=len(self.registry), pid=os.getpid()))

    def log(self, rec):
        with open(self.root / "HISTORY.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, (list, dict)) or k in ("held_out_gain_ids",)}), flush=True)

    # -- scoring against real ARC
    def cells(self):
        return self.legacy + self.registry

    def split(self, outcomes):
        count = lambda rows: sum(outcomes[r["id"]] for r in rows)  # noqa: E731
        return {"dev": count(self.rows_dev), "val": count(self.rows_val), "report": count(self.rows_rep)}

    def arc(self, cells=None):
        cells = cells if cells is not None else self.cells()
        router = S.default_router(cells, legacy=self.legacy_router)
        return self.all.score(cells, router, rows=self.rows_dev + self.rows_val + self.rows_rep)

    def sel(self, cells):
        """Selection score: dev + validation tasks only."""
        router = S.default_router(cells, legacy=self.legacy_router)
        return self.all.score(cells, router, rows=self.rows_sel)

    def baseline(self):
        if self.base is None:
            self.base = self.arc(self.legacy)
        return self.base

    def measure(self, stage, **extra):
        base = self.baseline()
        res = self.arc()
        now_split, base_split = self.split(res["outcomes"]), self.split(base["outcomes"])
        ids = sorted(k for k, v in res["outcomes"].items() if v and k not in {r["id"] for r in self.rows_dev})
        regress = [t for t, v in base["outcomes"].items() if v and not res["outcomes"][t]]
        rec = dict(stage=stage, utc=now(), cells=len(self.registry), depth2=sum(1 for c in self.registry if S.depth(c) > 1),
                   arc_dev=now_split["dev"], arc_val=now_split["val"], arc_report=now_split["report"], arc_held_out=now_split["val"] + now_split["report"],
                   baseline_dev=base_split["dev"], baseline_val=base_split["val"], baseline_report=base_split["report"],
                   baseline_held_out=base_split["val"] + base_split["report"], baseline_regressions=regress, held_out_solved_ids=ids, **extra)
        self.log(rec)
        board = H.load(self.root / "LEADERBOARD.json") if (self.root / "LEADERBOARD.json").exists() else []
        board.append({k: rec[k] for k in ("stage", "utc", "cells", "depth2", "arc_dev", "arc_val", "arc_report", "arc_held_out")})
        H.write(self.root / "LEADERBOARD.json", board)
        tensors = S.to_tensors(self.cells(), S.default_router(self.cells(), legacy=self.legacy_router))
        H.write(self.root / "FINAL_WEIGHTS.json", {"tensors": tensors})
        return rec

    # -- maker
    def ask(self, text, tag, chosen, oracle=None):
        self.state["makers_calls"] += 1
        if self.fake:  # fake = ORACLE: the true tensors, to test plumbing and measure the ceiling of the niches (never a maker result)
            (self.root / ("AUTHOR-" + tag)).mkdir(parents=True, exist_ok=True)
            return oracle()
        return H.ask_maker(text, self.root / ("AUTHOR-" + tag), False, None, chosen)

    # -- import of the finished tiling bank (earlier run, same provenance)
    def import_tiling(self):
        tensors = H.load(TILE_RUN / "FINAL_WEIGHTS.json")["tensors"]
        cells, _ = S.from_tensors(tensors)
        niche, patterns = N.NICHES["tiling"], N.NICHES["tiling"].patterns(60, SEED)
        tasks = {p["id"]: niche.task(p, SEED * 1000 + i) for i, p in enumerate(patterns)}
        imported = 0
        for c in cells:
            if c["kind"] == "cell" and c["template"] == "tile_orient":
                for p in patterns:
                    if S.passes(c, tasks[p["id"]]):
                        c.update(niche="tiling", pattern=p["id"], origin="maker gpt-6.1-sol low, tiling run 2026-10-01 (imported)", generation=1)
                        self.registry.append(c)
                        imported += 1
                        break
        self.state["done"].append("tiling")
        return imported

    # -- niche stages
    def stage(self, name, limit):
        niche = N.NICHES[name]
        self.status(stage=name, phase="authoring")
        if name == "tiling" and (TILE_RUN / "FINAL_WEIGHTS.json").exists() and not self.fake:
            n = self.import_tiling()
            self.log({"stage": "niche:tiling", "utc": now(), "imported_cells": n, "note": "imported finished bank; no new maker calls"})
            return self.measure("niche:tiling", niche_cells=n)
        (self.instance_stage if not niche.general else self.general_stage)(niche, limit)
        self.state["done"].append(name)
        self.save()

    def instance_stage(self, niche, limit):
        patterns, tasks, fresh = [], {}, []
        for i, p in enumerate(niche.patterns(limit, SEED)):
            try:
                tasks[p["id"]] = niche.task(p, SEED * 1000 + i)
                fresh.append({"id": p["id"], "task": niche.task(p, SEED * 1000 + 500 + i)})
                patterns.append(p)
            except RuntimeError:  # a pattern that cannot yield a verifiable task is dropped, never faked
                self.log({"stage": "niche:%s:skip" % niche.name, "utc": now(), "skipped_pattern": p["id"]})
        by_id = {p["id"]: p for p in patterns}
        attempts, bank, charged_total, rand_total = {}, {}, 0, [0, 0, 0]
        for gen in range(1, 4):
            todo = [pid for pid in by_id if pid not in bank]
            if not todo:
                break
            batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]

            def work(args):
                b, ids = args
                items = []
                for pid in ids:
                    item = {"task_id": pid, "demonstrations": tasks[pid]["train"]}
                    if attempts.get(pid):
                        cell = attempts[pid][-1]
                        item["previous_failed_attempt"] = (counterexamples(cell, [tasks[pid]], 1) or [{"note": "passed"}])[0]
                        item["previous_failed_attempt"]["params"] = cell["params"]
                    items.append(item)
                return self.ask(prompt_instance(niche, items), "%s-%d-%d" % (niche.name, gen, b), [{"id": i} for i in ids],
                                lambda: [{"task_id": pid, "params": by_id[pid]["truth"], "rationale": "oracle"} for pid in ids])

            with ThreadPoolExecutor(self.workers) as pool:
                raws = list(pool.map(work, list(enumerate(batches))))
            submitted = [c for raw in raws for c in parse_cells(raw, niche.template) if c.get("task_id") in by_id]
            charged = sum(S.positions(c) for c in submitted)
            charged_total += charged
            for c in submitted:
                attempts.setdefault(c["task_id"], []).append(c)
                if S.passes(c, tasks[c["task_id"]]):
                    c.update(niche=niche.name, pattern=c["task_id"], origin="maker gpt-6.1-sol low", generation=gen)
                    bank[c["task_id"]] = c
            for k, seed in enumerate((501, 502, 503)):
                r = random.Random(seed * 100 + gen)
                rand_total[k] += sum(S.passes(rand_cell(niche.template, r), tasks[c["task_id"]]) for c in submitted)
            fresh_hit = S.Scorer(fresh).score(list(bank.values()))["solved"] if bank else 0
            self.log({"stage": "niche:%s:gen%d" % (niche.name, gen), "utc": now(), "submitted": len(submitted), "eligible_total": len(bank),
                      "patterns": len(patterns), "fresh_transfer": fresh_hit, "charged_positions_per_arm": charged,
                      "random_eligible_total": list(rand_total)})
        self.registry.extend(bank.values())
        self.measure("niche:" + niche.name, niche_cells=len(bank), niche_patterns=len(patterns), charged_positions_per_arm=charged_total,
                     random_eligible=list(rand_total))

    def general_stage(self, niche, limit):
        patterns = niche.patterns(limit, SEED)
        tasks = [niche.task(p, SEED * 1000 + i) for i, p in enumerate(patterns)]
        fresh = [niche.task(p, SEED * 1000 + 500 + i) for i, p in enumerate(patterns)]
        best, failing, charged_total, rand_rates = None, None, 0, []
        for gen in range(1, 4):
            shown = tasks[:3] if gen == 1 else tasks[gen: gen + 3]
            raw = self.ask(prompt_general(niche, shown, failing), "%s-%d" % (niche.name, gen), [{"id": "general"}],
                           lambda: [{"task_id": "general", "params": patterns[0]["truth"], "rationale": "oracle"}])
            cells = parse_cells(raw, niche.template)
            charged = sum(S.positions(c) for c in cells)
            charged_total += charged
            if not cells:
                self.log({"stage": "niche:%s:gen%d" % (niche.name, gen), "utc": now(), "submitted": 0})
                continue
            cell = cells[0]
            rate = float(np.mean([S.passes(cell, t) for t in tasks]))
            fresh_rate = float(np.mean([S.passes(cell, t) for t in fresh]))
            r = random.Random(900 + gen)
            rand_rates.append([float(np.mean([S.passes(rand_cell(niche.template, r), t) for t in tasks])) for _ in range(3)])
            self.log({"stage": "niche:%s:gen%d" % (niche.name, gen), "utc": now(), "submitted": 1, "pass_rate": rate, "fresh_pass_rate": fresh_rate,
                      "charged_positions_per_arm": charged, "random_pass_rate": rand_rates[-1]})
            if best is None or rate > best[0]:
                best = (rate, cell)
            if rate >= 0.9:
                break
            failing = {"params": cell["params"], "failing_tasks": counterexamples(cell, tasks, 2)}
        added = 0
        if best and best[0] >= 0.9:
            best[1].update(niche=niche.name, pattern="*", origin="maker gpt-6.1-sol low", generation=gen, general=True)
            self.registry.append(best[1])
            added = 1
        self.measure("niche:" + niche.name, niche_cells=added, best_pass_rate=best[0] if best else 0.0, charged_positions_per_arm=charged_total)

    # -- breeding
    def pattern_of(self, cell, rng):
        niche = N.NICHES[cell["niche"]]
        patterns = niche.patterns(N.LIMITS[cell["niche"]], SEED)
        if cell.get("pattern") not in (None, "*"):
            return next(p for p in patterns if p["id"] == cell["pattern"])
        return rng.choice(patterns)

    def sample_input(self, pattern, rng):
        if pattern["niche"] == "association":
            colors = sorted(pattern["map"]) + pattern["ident"][:3]
            h, w = rng.randint(3, 6), rng.randint(3, 6)
            return [[rng.choice(colors) for _ in range(w)] for _ in range(h)]
        if pattern["niche"] == "glyph":
            return N.Glyph._pair(rng)["input"]
        return pattern["sample_input"](rng)

    def composite_tasks(self, a, b, rng, n=3):
        pa, pb = self.pattern_of(a, rng), self.pattern_of(b, rng)
        tasks = []
        for _ in range(n):
            pairs = []
            for _ in range(4):
                try:
                    x = self.sample_input(pa, rng)
                    mid = pa["apply"](x)
                    y = pb["apply"](mid)
                except (ValueError, IndexError, TypeError, KeyError):  # e.g. a transposing tile transform meeting a non-square grid: this composition does not exist
                    return None
                if max(len(y), len(y[0])) > 30 or y == mid or y == x:
                    return None
                pairs.append({"input": x, "output": y})
            tasks.append({"train": pairs[:3], "test": pairs[3:]})
        return tasks

    def breed(self, n_pairs=300, cap=40):
        rng = random.Random(1000 + self.state["breed_round"])
        parents = [c for c in self.registry if c["kind"] == "cell" and c.get("niche") in N.NICHES]
        tried = {tuple(p) for p in self.state["tried_pairs"]}
        pairs = [(a, b) for a in parents for b in parents if a["niche"] != b["niche"] and (a["id"], b["id"]) not in tried]
        rng.shuffle(pairs)
        admitted, evaluated, solved, ctrl, rejected_useless = [], 0, 0, 0, 0
        base_cells = self.cells()
        current = self.sel(base_cells)
        for a, b in pairs[:n_pairs]:
            self.state["tried_pairs"].append([a["id"], b["id"]])
            tasks = self.composite_tasks(a, b, rng)
            if tasks is None:
                continue
            evaluated += 1
            child = S.chain_cell(a, b, niche="chain", origin="crossover chain(%s,%s)" % (a["niche"], b["niche"]), round=self.state["breed_round"] + 1)
            ok = all(S.passes(child, t) for t in tasks) and sum(S.passes(a, t) or S.passes(b, t) for t in tasks) <= 1
            r = random.Random(rng.random())
            ctrl += all(S.passes(S.chain_cell(rand_cell(a["template"], r), rand_cell(b["template"], r)), t) for t in tasks)
            if not ok:
                continue
            solved += 1
            if len(admitted) >= cap:
                continue
            cur_cells = base_cells + admitted
            router = S.default_router(cur_cells, legacy=self.legacy_router)
            unsolved = [r for r in self.rows_sel if not current["outcomes"][r["id"]]]
            if not self.all.contends(child, cur_cells, router, unsolved):  # cannot be routed to on any unsolved task: no gain possible, skip the full pass
                rejected_useless += 1
                continue
            trial = self.sel(base_cells + admitted + [child])  # SELECTION on real ARC dev+validation: must strictly gain, never regress
            lost = [t for t, v in current["outcomes"].items() if v and not trial["outcomes"][t]]
            if trial["solved"] > current["solved"] and not lost:
                child.update(composite_fitness=1.0, a=a["niche"] + ":" + str(a.get("pattern")), b=b["niche"] + ":" + str(b.get("pattern")),
                             arc_selection_gain=trial["solved"] - current["solved"])
                admitted.append(child)
                current = trial
            else:
                rejected_useless += 1
        self.registry.extend(admitted)
        self.state["breed_round"] += 1
        self.save()
        combos = {}
        for c in admitted:
            combos[c["origin"]] = combos.get(c["origin"], 0) + 1
        return self.measure("breed:round%d" % self.state["breed_round"], pairs_evaluated=evaluated, children_valid=solved,
                            children_admitted=len(admitted), children_rejected_no_arc_gain=rejected_useless, random_chain_valid=ctrl,
                            admitted_by_pair=combos)

    # -- maker mutation of near-miss cells, selected on real ARC dev+validation, with a matched random-edit control
    def mutation_targets(self, cells, cur, n_targets):
        seen, targets = set(), []
        for row in self.rows_sel:
            if cur["outcomes"][row["id"]]:
                continue
            errs = [(self.all.demo_error(c, row), i) for i, c in enumerate(cells) if c["kind"] == "cell" and c.get("niche") != "legacy"]
            if not errs:
                continue
            e, i = min(errs)
            ident = float(np.mean([1.0 if np.asarray(d["input"]).shape != np.asarray(d["output"]).shape
                                   else float(np.mean(np.asarray(d["input"]) != np.asarray(d["output"]))) for d in row["task"]["train"]]))
            # a REAL near-miss explains at least half of the change the task needs; a small pixel error on a task that barely changes is just the identity
            if 0 < e <= 0.5 * ident and ident > 0 and cells[i]["id"] not in seen:
                seen.add(cells[i]["id"])
                targets.append((e, row, cells[i]))
        return sorted(targets, key=lambda t: t[0])[:n_targets]

    def mutation_prompt(self, targets):
        items = []
        for e, row, cell in targets:
            task = row["task"]
            d = task["train"][0]
            try:
                pred = S.run(cell, d["input"], task["train"][1:])
            except Exception as error:
                pred = "error:" + type(error).__name__
            items.append({"task_id": row["id"], "template": cell["template"], "current_params": cell["params"], "demonstrations": task["train"],
                          "counterexample": {"input": d["input"], "expected": d["output"], "predicted": pred}})
        template_docs = {i["template"]: S.TEMPLATES[i["template"]]["doc"] for i in items}
        text = ("You revise literal weight tensors of grid cells (no code, no tools, no files). For EACH item, the current cell nearly fits the task but fails. "
                "Read the counterexample (expected vs predicted), and write revised complete tensors of the SAME template and shapes (finite numbers in [-9, 9]) "
                "that reproduce every demonstration and generalise. Template semantics: %s\nReturn ONLY a JSON array: [{\"task_id\": str, \"params\": "
                "{tensor_name: nested lists}, \"rationale\": str}].\nItems:\n%s" % (json.dumps(template_docs), json.dumps(items)))
        return items, text

    def mutate(self, n_targets=6):
        cells = self.cells()
        cur = self.sel(cells)
        targets = self.mutation_targets(cells, cur, n_targets)
        if not targets:
            return self.measure("mutate:round%d" % (self.state.get("mutate_round", 0) + 1), targets=0, note="no near-miss cells on the selection tasks")
        items, text = self.mutation_prompt(targets)
        rnd = self.state.get("mutate_round", 0) + 1
        raw = self.ask(text, "mutate-%d" % rnd, [{"id": r["id"]} for _, r, _ in targets], lambda: [])
        by_task = {e.get("task_id"): e for e in raw if isinstance(raw, list)} if isinstance(raw, list) else {}
        admitted, tried, rand_admitted, charged, declined = [], 0, 0, 0, 0
        current = cur
        rng = random.Random(7000 + rnd)
        for e, row, cell in targets:
            entry = by_task.get(row["id"])
            if not entry or not S.valid_params(cell["template"], entry.get("params")):
                declined += 1
                continue
            same = np.allclose(np.concatenate([np.asarray(v, float).ravel() for v in entry["params"].values()]),
                               np.concatenate([np.asarray(cell["params"][k], float).ravel() for k in entry["params"]]))
            if same or str(entry.get("rationale", "")).lower().startswith(("no valid", "no satisfying", "no exact", "no revision", "no tensors", "cannot", "no such")):
                declined += 1  # the maker states that no tensors of this template can express the rule: a structural limit, not a weight problem
                continue
            tried += 1
            child = S.make_cell(cell["template"], entry["params"], niche=cell["niche"], pattern=cell.get("pattern"), parents=[cell["id"]],
                                origin="maker mutation round %d" % rnd, generation=rnd)
            charged += S.positions(child)
            old = np.concatenate([np.asarray(v, float).ravel() for v in cell["params"].values()])
            new = np.concatenate([np.asarray(v, float).ravel() for v in child["params"].values()])
            changed = max(1, int((old != new).sum()))
            flat = old.copy()
            for k in rng.sample(range(len(flat)), min(changed, len(flat))):
                flat[k] = rng.uniform(-9, 9)
            rparams, at = {}, 0
            for k, v in cell["params"].items():
                size = int(np.asarray(v).size)
                rparams[k] = flat[at:at + size].reshape(np.asarray(v).shape).tolist()
                at += size
            rchild = S.make_cell(cell["template"], rparams, niche=cell["niche"])
            rtrial = self.sel(cells + admitted + [rchild])
            rand_admitted += rtrial["solved"] > current["solved"]
            trial = self.sel(cells + admitted + [child])
            lost = [t for t, v in current["outcomes"].items() if v and not trial["outcomes"][t]]
            if trial["solved"] > current["solved"] and not lost:
                child["arc_selection_gain"] = trial["solved"] - current["solved"]
                admitted.append(child)
                current = trial
        self.registry.extend(admitted)
        self.state["mutate_round"] = rnd
        self.save()
        return self.measure("mutate:round%d" % rnd, targets=len(targets), mutations_tried=tried, mutations_declined_structural=declined, mutations_admitted=len(admitted),
                            random_edits_that_would_gain=rand_admitted, charged_positions_per_arm=charged)

    # -- main loop
    def run(self, max_rounds):
        self.baseline()
        self.log(dict(stage="baseline", utc=now(), **{"baseline_" + k: v for k, v in self.split(self.base["outcomes"]).items()}))
        for name, limit in QUEUE:
            if (self.root / "STOP").exists():
                break
            if name not in self.state["done"]:
                self.stage(name, limit)
                self.save()
        idle = 0
        last = None
        for _ in range(max_rounds):
            if (self.root / "STOP").exists():
                self.status(state="user_stop")
                return
            self.status(stage="breeding", phase="running", round=self.state["breed_round"] + 1)
            rec = self.breed()
            self.status(stage="mutation", phase="running", round=self.state.get("mutate_round", 0) + 1)
            rec2 = self.mutate()
            progressed = rec["children_admitted"] or rec2.get("mutations_admitted") or rec2["arc_held_out"] != last
            idle = 0 if progressed else idle + 1
            last = rec2["arc_held_out"]
            if idle >= 3:
                self.status(state="plateau", note="3 rounds of breeding and mutation with no admission and no held-out change; needs new niches")
                return
        self.status(state="rounds_complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed-from", type=pathlib.Path, help="earlier run dir: keep its maker-written cells, drop its chain children")
    a = ap.parse_args()
    d = Driver(a.root, a.fake, a.workers)
    if a.seed_from and not d.registry:
        old = H.load(a.seed_from / "REGISTRY.json")
        d.registry = [c for c in old if (c["kind"] == "cell" and c.get("origin", "").startswith("maker")) or c.get("arc_selection_gain")]
        d.state["done"] = sorted({c["niche"] for c in d.registry if c.get("niche") in N.NICHES})
        d.save()
    d.run(a.rounds)


if __name__ == "__main__":
    main()
