"""Write ARC_SCORE.json (contract: tests/arc_goal_checks.py) for a head driver run. Runs on the server, where the ARC data lives.

Scored set: the REPORT split only (302 public-training tasks that were never used to select, breed, mutate or admit anything; the 100 DEV tasks, the audit
population and the evaluation split are excluded). Per-task correctness is recomputed by the run's own evaluator on the run's own FINAL_WEIGHTS.json with the
test outputs stripped, exactly as the goal test does. Controls (zero weights, three random-weight seeds) go through the same evaluator on the same tasks.
History: one entry per saved earlier weights file, recomputed now on the same tasks (nothing is quoted from old logs).

  python make_arc_score.py runs/weights-head-tourney1 --history runs/weights-head-skillbank-v2 ... runs/weights-head-arena3
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/srv/cellforge-inbox/head-20261001")
sys.path.insert(0, "/srv/cellforge-inbox/two-loop-20261001/runs/weights-arc-moe-smallskill-v2/src")
import head_arc_cells as H  # noqa: E402
import head_driver as D  # noqa: E402
import skillcore as S  # noqa: E402

ARC = pathlib.Path("/srv/cellforge-private/arcagi2/data/training")
_G = {}


def strip(task):
    t = copy.deepcopy(task)
    for ex in t["test"]:
        ex.pop("output", None)
    return t


def load_eval(run):
    path = pathlib.Path(run) / "arc_eval.py"
    spec = importlib.util.spec_from_file_location("arc_eval_" + hashlib.md5(str(path).encode()).hexdigest()[:8], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def solved(ev, weights, task):
    try:
        preds = ev.predict(weights, strip(task))
    except Exception:
        return False
    if len(preds) != len(task["test"]):
        return False
    return all(any(a == ex["output"] for a in list(attempts)[:2]) for attempts, ex in zip(preds, task["test"]))


def init(run, weights):
    try:
        import structlib
        structlib.reset()  # forked workers must not share the parent's sandbox pipe
    except ImportError:
        pass
    _G["ev"], _G["w"] = load_eval(run), weights


def work(tid):
    return tid, solved(_G["ev"], _G["w"], json.loads((ARC / (tid + ".json")).read_text()))


def evaluate(run, weights, ids, procs, label):
    t0 = time.time()
    with mp.get_context("fork").Pool(procs, initializer=init, initargs=(run, weights)) as pool:
        per = dict(pool.map(work, ids, chunksize=4))
    print("%-14s solved %3d of %d (%.0fs)" % (label, sum(per.values()), len(ids), time.time() - t0), flush=True)
    return per


def zeros(v):
    return [zeros(x) for x in v] if isinstance(v, list) else 0.0


def randomised(w, seed):
    rng = np.random.default_rng(seed)
    return {k: rng.uniform(-9, 9, np.asarray(v, float).shape).tolist() for k, v in w.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--history", nargs="*", type=pathlib.Path, default=[])
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    d = D.Driver(pathlib.Path("/srv/cellforge-inbox/head-20261001/scratch/arcscore"), fake=True)
    import structlib
    for run in a.history + [a.run]:  # maker-proposed structures of every scored run must be registered before its tensors are unpacked
        structlib.load(S, run)
    rows = d.rows_rep
    ids = sorted(r["id"] for r in rows)
    weights_path = a.run / "FINAL_WEIGHTS.json"
    weights = json.loads(weights_path.read_text())["tensors"]
    per = evaluate(a.run, weights, ids, a.procs, "final")
    control_zero = evaluate(a.run, {k: zeros(v) for k, v in weights.items()}, ids, a.procs, "zero weights")
    seeds = [101, 102, 103]
    control_random = [evaluate(a.run, randomised(weights, s), ids, a.procs, "random %d" % s) for s in seeds]

    # history: baseline (the legacy 5-cell block) then every saved earlier weights file, recomputed now on the same tasks with the scorer
    def scored(tensors):
        cells, router = S.from_tensors(tensors)
        return int(d.all.score(cells, router, rows=rows)["solved"])

    history, seen = [], set()
    baseline = H.load(d.root / "parent.json")["tensors"]
    entries = [("baseline: the legacy 5-cell block (parent.json)", baseline, hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest())]
    for run in a.history + [a.run]:
        raw = (run / "FINAL_WEIGHTS.json").read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        if h not in seen:
            seen.add(h)
            entries.append((str(run.name), json.loads(raw)["tensors"], h))
    for i, (label, tensors, h) in enumerate(entries):
        history.append({"iteration": i, "solved": scored(tensors), "weights_hash": h, "weights": label})
        print("iteration %2d %-50s solved %d" % (i, label, history[-1]["solved"]), flush=True)
    final_scorer = scored(weights)
    claimed = sum(per.values())
    if final_scorer != claimed:
        print("NOTE: scorer says %d, evaluator says %d for the final weights" % (final_scorer, claimed), flush=True)
    score = {"split": "training", "task_ids": ids, "weights_file": "FINAL_WEIGHTS.json", "evaluator": "arc_eval.py",
             "claimed": {"solved": claimed, "total": len(ids), "per_task": per},
             "controls": {"zero_weights": {"solved": sum(control_zero.values())},
                          "random_weights": {"solved": max(sum(c.values()) for c in control_random), "seeds": seeds,
                                             "per_seed_solved": [sum(c.values()) for c in control_random]}},
             "history": history, "arc1_arc2_merged_by_task_id": True,
             "notes": "Scored set = the report split of the head driver (%d public-training tasks never used to select, breed, mutate or admit anything). "
                      "Data: ARC-AGI-2 public training directory, which already merges ARC-1 and ARC-2 tasks by id. The evaluation split and the audit population "
                      "were never read. Cells are code templates (structure) plus literal tensors written by gpt-6.1-sol (ChatGPT subscription route); no gradients."
                      % len(ids)}
    (a.run / "ARC_SCORE.json").write_text(json.dumps(score))
    print("wrote %s: claimed %d of %d (%.1f%%), zero %d, random %s" % (a.run / "ARC_SCORE.json", claimed, len(ids), 100 * claimed / len(ids),
                                                                      score["controls"]["zero_weights"]["solved"], score["controls"]["random_weights"]["per_seed_solved"]), flush=True)


if __name__ == "__main__":
    main()
