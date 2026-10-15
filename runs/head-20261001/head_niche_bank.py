"""Head-center skill-niche loop: tile_orient. gpt-6.1-sol (low) writes an INSTANCE BANK of literal-weight cells from generated niche tasks.

Niche = tiling-with-orientation (the largest single skill in the hand-authored ARC baseline). Tasks come from a catalog of generic dihedral
tiling patterns over random grids; the maker sees ONLY demonstration pairs (plus its previous failed attempt as predicted-vs-expected) and
writes one cell (template tile_orient + literal tensors) per task. Eligibility = reproduces every demonstration. Niche transfer = the bank
(routed by the demonstration-error router) solves a FRESH task of each pattern. Scoreboard = real ARC: 100 DEV tasks and the 579 held-out
training tasks (never shown to the maker; audit population and evaluation split excluded) versus the unchanged 5-cell baseline.
Matched random arm: same templates, same number of literal positions per submitted cell, uniform[-9,9] tensors, three lineages.
No gradients, no API key, no evaluation split.
"""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import head_arc_cells as H  # noqa: E402
import head_core as C  # noqa: E402

RUN_SRC = pathlib.Path(H.RUN_SRC)
DATA = pathlib.Path(H.DATA)
DIMS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3), (1, 4), (4, 1), (4, 4), (2, 4)]
FAMILIES = {
    "same": lambda i, j: 0,
    "flip_odd_cols": lambda i, j: 1 if j % 2 else 0,
    "flip_odd_rows": lambda i, j: 2 if i % 2 else 0,
    "flip_lr_odd_rows": lambda i, j: 1 if i % 2 else 0,
    "flip_ud_odd_cols": lambda i, j: 2 if j % 2 else 0,
    "mirror4": lambda i, j: (i % 2) * 2 + (j % 2),
    "mirror4_rev": lambda i, j: 3 - ((i % 2) * 2 + (j % 2)),
    "checker_rot180": lambda i, j: 3 if (i + j) % 2 else 0,
    "rot_cycle": lambda i, j: [0, 5, 3, 6][(i + j) % 4],
    "rot_cycle_rows": lambda i, j: [0, 5, 3, 6][i % 4],
    "diag_transpose": lambda i, j: 4 if i == j else 0,
}
SQUARE_ONLY = {"rot_cycle", "rot_cycle_rows", "diag_transpose"}


def catalog(limit, seed):
    seen, out = set(), []
    for (pr, pc) in DIMS:
        for name, f in FAMILIES.items():
            ks = tuple(tuple(f(i, j) if i < pr and j < pc else 0 for j in range(C.MAXT)) for i in range(C.MAXT))
            key = (pr, pc, ks)
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": "tile-%d%dx-%s" % (pr, pc, name), "pr": pr, "pc": pc, "ks": ks, "square": name in SQUARE_ONLY or any(
                ks[i][j] >= 4 for i in range(pr) for j in range(pc))})
    random.Random(seed).shuffle(out)
    return out[:limit]


def true_params(p):
    per, pec = [0.0] * C.MAXT, [0.0] * C.MAXT
    per[p["pr"] - 1] = 1.0
    pec[p["pc"] - 1] = 1.0
    orient = [[[1.0 if k == p["ks"][i][j] else 0.0 for k in range(8)] for j in range(C.MAXT)] for i in range(C.MAXT)]
    return {"per": per, "pec": pec, "orient": orient}


def sample_grid(rng, square):
    for _ in range(500):
        h = rng.choice([2, 3]) if not square else None
        n = rng.choice([2, 3])
        shape = (n, n) if square else (rng.choice([2, 3]), rng.choice([2, 3]))
        a = np.array([[rng.choice(range(10)) for _ in range(shape[1])] for _ in range(shape[0])])
        forms = [C.OPS[k](a) for k in range(8 if shape[0] == shape[1] else 4)]
        if len({f.tobytes() + bytes(f.shape) for f in forms}) == len(forms):
            return a.tolist()
    raise RuntimeError("no asymmetric grid")


def task_for(pattern, seed, ndemos=3):
    rng = random.Random(seed)
    params = true_params(pattern)
    pairs = []
    for _ in range(ndemos + 1):
        g = sample_grid(rng, pattern["square"])
        pairs.append({"input": g, "output": C.tile_orient(params, g)[0]})
    return {"train": pairs[:ndemos], "test": [pairs[ndemos]]}


def prompt_text(core, items):
    return ("You write literal weight tensors for a grid cell (no code, no tools, no files). Template tile_orient, tensors per (4), pec (4), "
            "orient (4 x 4 x 8). Semantics: " + C.DOCS["tile_orient"] + "\nFor EACH task below, infer from the demonstrations how the output is "
            "built from the input, and write complete tensors (finite numbers in [-9, 9], all 4x4x8 orient entries present; use 0 for unused) so the "
            "cell reproduces EVERY demonstration exactly and would apply to a new input of the same pattern. If previous_failed_attempt is present, "
            "read its predicted vs expected and write a genuinely different revision. Return ONLY a JSON array, one object per task: "
            "[{\"task_id\": str, \"template\": \"tile_orient\", \"params\": {\"per\": [..4..], \"pec\": [..4..], \"orient\": [[[..8..] x4] x4]}, "
            "\"rationale\": str naming the concrete demonstration evidence}].\nTasks:\n" + json.dumps(items))


def attempt_note(core, cell, task):
    for i, pair in enumerate(task["train"]):
        try:
            pred = core.PREDICTORS["tile_orient"](cell["params"], pair["input"], False)[0]
        except Exception as error:
            return {"params": cell["params"], "demo": i, "error": type(error).__name__}
        if pred != pair["output"]:
            return {"params": cell["params"], "demo": i, "input": pair["input"], "expected": pair["output"], "predicted": pred}
    return {"params": cell["params"], "note": "reproduced all demonstrations"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--patterns", type=int, default=60)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="inject the TRUE tensors for every pattern: measures the catalog's ceiling, not the maker")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    from cellforge.construction.arc_weight_moe import predict as route_predict, score
    from cellforge.construction.joint_weight_population import digest, experts
    root = a.root
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pop1_core.py", "parent.json"):
        if not (root / name).exists():
            (root / name).write_bytes((RUN_SRC / name).read_bytes())
    for name in ("head_core.py",):
        (root / name).write_bytes((pathlib.Path(__file__).resolve().parent / name).read_bytes())
    (root / "arc_eval.py").write_text(
        "from pathlib import Path\nimport importlib.util\n_h = Path(__file__).parent\n_s = importlib.util.spec_from_file_location('head_core', _h / 'head_core.py')\n"
        "_m = importlib.util.module_from_spec(_s)\n_s.loader.exec_module(_m)\nCORE = _m.make_head_core(_h / 'pop1_core.py')\n"
        "from cellforge.construction.arc_weight_moe import predict as _p\n\ndef predict(weights, task):\n    return _p(weights, task, CORE)\n")
    core = C.make_head_core(root / "pop1_core.py")
    base = H.load(root / "parent.json")["tensors"]
    patterns = catalog(a.patterns, a.seed)
    by_id = {p["id"]: p for p in patterns}
    demo_tasks = {p["id"]: task_for(p, a.seed * 1000 + i) for i, p in enumerate(patterns)}
    fresh_tasks = {p["id"]: task_for(p, a.seed * 1000 + 500 + i) for i, p in enumerate(patterns)}
    dev_ids = {x["id"] for x in H.load(RUN_SRC / "TASKS.json")["development"]}
    audit = lambda t: any(int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5 == 0 for s in (t, t + ".json"))  # noqa: E731
    dev_rows = [{"id": i, "task": H.load(DATA / (i + ".json"))} for i in sorted(dev_ids)]
    held_rows = [{"id": p.stem, "task": H.load(p)} for p in sorted(DATA.glob("*.json")) if p.stem not in dev_ids and not audit(p.stem)]
    base_dev, base_held = score(base, dev_rows, core), score(base, held_rows, core)
    print(json.dumps({"patterns": len(patterns), "dev": len(dev_rows), "held_out": len(held_rows), "baseline_dev": base_dev["solved"],
                      "baseline_held_out": base_held["solved"]}), flush=True)

    def bank_tensors(cells):
        t = H.with_cells(base, cells)
        order = experts(t, core)
        t["route.b"] = [1e-6 if prefix in {"0", "1", "2", "3", "4"} else 0.0 for prefix, _, _ in order]  # ties keep the original cells
        return t

    def niche_solved(cells):
        if not cells:
            return {}
        t = bank_tensors(cells)
        out = {}
        for pid, task in fresh_tasks.items():
            try:
                out[pid] = route_predict(t, task, core)[0][0] == task["test"][0]["output"]
            except Exception:
                out[pid] = False
        return out

    attempts, bank, rbank, report = {}, {}, {s: {} for s in (501, 502, 503)}, []
    prior_fresh = 0
    for n in range(1, a.generations + 1):
        todo = [pid for pid in by_id if pid not in bank]
        if not todo:
            break
        batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
        started = time.monotonic()

        def work(args):
            b, ids = args
            items = []
            for pid in ids:
                item = {"task_id": pid, "demonstrations": [{"input": p["input"], "output": p["output"]} for p in demo_tasks[pid]["train"]]}
                if attempts.get(pid):
                    item["previous_failed_attempt"] = attempt_note(core, attempts[pid][-1], demo_tasks[pid])
                items.append(item)
            text = prompt_text(core, items)
            return ids, text, H.ask_maker(text, root / ("AUTHOR-%d-%d" % (n, b)), a.fake, core, [{"id": i} for i in ids])

        if a.oracle:
            results = [([], "", [{"task_id": pid, "template": "tile_orient", "params": true_params(by_id[pid]), "rationale": "oracle"} for pid in todo])]
        else:
            with ThreadPoolExecutor(a.workers) as pool:
                results = list(pool.map(work, list(enumerate(batches))))
        charged, notes, submitted = 0, [], []
        for ids, text, raw in results:
            for entry in raw if isinstance(raw, list) else []:
                tid = entry.get("task_id")
                cell = {"template": entry.get("template"), "params": entry.get("params"), "task_id": tid}
                ok = tid in by_id and cell["template"] == "tile_orient" and H.valid_cell(core, cell)
                charged += H.positions(cell) if ok else 1
                if tid in by_id:
                    attempts.setdefault(tid, []).append(cell)
                if ok:
                    submitted.append(cell)
                else:
                    notes.append({"task_id": tid, "reason": "invalid"})
        new_cells = [c for c in submitted if H.fits(core, c, demo_tasks[c["task_id"]])]
        for c in new_cells:
            bank[c["task_id"]] = c
        for seed in rbank:
            r = random.Random(seed * 100 + n)
            for c in submitted:
                rc = dict(H.random_like(core, c, r), task_id=c["task_id"])
                if H.fits(core, rc, demo_tasks[c["task_id"]]):
                    rbank[seed][c["task_id"]] = rc
        cells = [bank[k] for k in sorted(bank)]
        fresh = niche_solved(cells)
        rfresh = [sum(niche_solved([v for k, v in sorted(rbank[s].items())]).values()) for s in sorted(rbank)]
        tensors = bank_tensors(cells) if cells else base
        dev_s, held_s = score(tensors, dev_rows, core), score(tensors, held_rows, core)
        regress = [t for t, v in base_held["outcomes"].items() if v and not held_s["outcomes"][t]]
        rec = {"generation": n, "submitted": len(submitted), "charged_positions_per_arm": charged, "eligible_total": len(bank),
               "eligible_new": len(new_cells), "random_eligible": [len(rbank[s]) for s in sorted(rbank)],
               "niche_fresh_solved": sum(fresh.values()), "niche_patterns": len(patterns), "random_niche_fresh_solved": rfresh,
               "arc_dev_solved": dev_s["solved"], "arc_held_out_solved": held_s["solved"], "baseline_held_out": base_held["solved"],
               "held_out_solved_ids": sorted(k for k, v in held_s["outcomes"].items() if v), "baseline_regressions": regress,
               "dev_solved_ids": sorted(k for k, v in dev_s["outcomes"].items() if v), "seconds": time.monotonic() - started, "notes": notes[:8]}
        report.append(rec)
        H.write(root / "NICHE_REPORT.json", {"niche": "tile_orient", "generations": report, "scope": "development niche + public-training held-out; "
                 "no evaluation split; router written by head-center (identity + tie bias), not by the maker"})
        H.write(root / "FINAL_WEIGHTS.json", {"tensors": tensors})
        print(json.dumps({k: v for k, v in rec.items() if k not in ("held_out_solved_ids", "dev_solved_ids", "notes")}), flush=True)
    print("held-out solved ids:", report[-1]["held_out_solved_ids"], flush=True)


if __name__ == "__main__":
    main()
