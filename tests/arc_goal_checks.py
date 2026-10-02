"""Checks that ARC-AGI task accuracy of the literal weights is the goal. Owner-controlled: do not weaken, skip or delete.

Contract: runs/weights-*/ARC_SCORE.json
{
  "split": "training|evaluation", "task_ids": [str], "weights_file": rel path (JSON/NPZ tensors of the FINAL model),
  "evaluator": rel path of a module exposing predict(weights, task) -> [[grid, grid?] per test input] (<=2 attempts),
  "claimed": {"solved": int, "total": int, "per_task": {task_id: bool}},
  "controls": {"zero_weights": {"solved": int}, "random_weights": {"solved": int, "seeds": [int]}},
  "history": [{"iteration": int, "solved": int, "weights_hash": str}],      # one entry per loop iteration
  "arc1_arc2_merged_by_task_id": true
}
The checks RECOMPUTE per-task correctness by running the evaluator on the run's weights against the official ARC task
files with test outputs STRIPPED, so the evaluator cannot read answers. ARC data root: env ARC_ROOT, else data/arc.
"""
import copy
import hashlib
import importlib.util
import os
import random
import re

from tests import standard_checks as S

MIN_TASKS = 100
MIN_ITERATIONS = 3
VERIFY_SAMPLE = 60
FORBIDDEN = re.compile(r"research[./\\]claude_arc|results_(training|evaluation)|solve_cache")


def arc_root():
    r = os.environ.get("ARC_ROOT")
    if r:
        return r
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "arc")


def task_file(root, split, tid):
    return os.path.join(root, split, tid + ".json")


def strip_outputs(task):
    t = copy.deepcopy(task)
    for ex in t["test"]:
        ex.pop("output", None)
    return t


def load_eval(run, rel):
    path = os.path.join(run, rel)
    spec = importlib.util.spec_from_file_location("arc_eval_" + hashlib.md5(path.encode()).hexdigest()[:8], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def solved(ev, weights, task):
    try:
        preds = ev.predict(weights, strip_outputs(task))
    except Exception:
        return False
    if len(preds) != len(task["test"]):
        return False
    for attempts, ex in zip(preds, task["test"]):
        if not any(a == ex["output"] for a in list(attempts)[:2]):
            return False
    return True


def _sample(ids, per, k=VERIFY_SAMPLE):
    rng = random.Random(0xA2C)
    claimed = sorted(i for i in ids if per.get(i))
    rest = sorted(i for i in ids if not per.get(i))
    rng.shuffle(claimed)
    rng.shuffle(rest)
    return claimed[:k] + rest[:k]


def load_score(run):
    p = os.path.join(run, "ARC_SCORE.json")
    return S.load(p) if os.path.exists(p) else None


def check_exists_and_scale(run):
    sc = load_score(run)
    if sc is None:
        return ["no ARC_SCORE.json: no ARC-AGI task accuracy was ever measured for these weights"]
    out = []
    if len(sc.get("task_ids", [])) < MIN_TASKS:
        out.append("fewer than %d ARC tasks scored" % MIN_TASKS)
    if sc.get("claimed", {}).get("total") != len(sc.get("task_ids", [])):
        out.append("claimed total != number of task_ids")
    if not sc.get("arc1_arc2_merged_by_task_id"):
        out.append("ARC-1/2 not merged by task id")
    return out


def check_arc_recomputed(run, root=None):
    sc = load_score(run)
    if sc is None:
        return ["no ARC_SCORE.json"]
    root = root or arc_root()
    if not os.path.isdir(os.path.join(root, sc["split"])):
        return ["ARC data not found at %s/%s (set ARC_ROOT); cannot recompute, so this FAILS" % (root, sc["split"])]
    weights = S.weights_of(run, sc["weights_file"])
    ev = load_eval(run, sc["evaluator"])
    per = sc["claimed"]["per_task"]
    out = []
    for tid in _sample(sc["task_ids"], per):
        got = solved(ev, weights, S.load(task_file(root, sc["split"], tid)))
        if got != bool(per.get(tid)):
            out.append("task %s: claimed %s, recomputed %s" % (tid, bool(per.get(tid)), got))
    if sum(bool(v) for v in per.values()) != sc["claimed"]["solved"]:
        out.append("claimed solved != sum of per_task")
    return out


def check_score_positive_and_beats_controls(run):
    sc = load_score(run)
    if sc is None:
        return ["no ARC_SCORE.json"]
    n, c = sc["claimed"]["solved"], sc.get("controls", {})
    out = []
    if n <= 0:
        out.append("model solves 0 ARC tasks")
    for k in ("zero_weights", "random_weights"):
        if k not in c:
            out.append("missing control " + k)
        elif n <= c[k]["solved"]:
            out.append("model (%d) does not beat %s control (%d)" % (n, k, c[k]["solved"]))
    if len(c.get("random_weights", {}).get("seeds", [])) < 3:
        out.append("random control needs >=3 seeds")
    return out


def check_weights_matter(run, root=None):
    """Zeroing the weights must lose the solved tasks, so the score comes from the tensors and not from evaluator code."""
    sc = load_score(run)
    if sc is None:
        return ["no ARC_SCORE.json"]
    root = root or arc_root()
    if not os.path.isdir(os.path.join(root, sc["split"])):
        return ["ARC data not found; cannot recompute"]
    weights = S.weights_of(run, sc["weights_file"])
    ev = load_eval(run, sc["evaluator"])
    zero = S._zeros_like(weights)
    per = sc["claimed"]["per_task"]
    ids = [i for i in _sample(sc["task_ids"], per) if per.get(i)]
    if not ids:
        return ["no claimed solved task to ablate"]
    still = [i for i in ids if solved(ev, zero, S.load(task_file(root, sc["split"], i)))]
    if len(still) > 0.2 * len(ids):
        return ["%d/%d solved tasks survive zeroed weights: score is not coming from the weights" % (len(still), len(ids))]
    return []


def check_no_answer_or_baseline_shortcuts(run):
    sc = load_score(run)
    if sc is None:
        return ["no ARC_SCORE.json"]
    src = open(os.path.join(run, sc["evaluator"]), encoding="utf-8").read()
    return ["evaluator references baseline/answer artifacts"] if FORBIDDEN.search(src) else []


def check_iterating_and_not_regressing(run):
    sc = load_score(run)
    if sc is None:
        return ["no ARC_SCORE.json"]
    h = sc.get("history", [])
    if len(h) < MIN_ITERATIONS:
        return ["only %d loop iterations with a measured ARC score (need >=%d)" % (len(h), MIN_ITERATIONS)]
    out = []
    its = [e["iteration"] for e in h]
    if its != sorted(set(its)):
        out.append("history iterations not strictly increasing")
    if h[-1]["solved"] != sc["claimed"]["solved"]:
        out.append("last history entry != claimed solved")
    if h[-1]["solved"] < max(e["solved"] for e in h[:-1]):
        out.append("final ARC score regressed below an earlier iteration")
    if h[-1]["solved"] <= h[0]["solved"]:
        out.append("ARC score did not improve from iteration %d to %d: evolution is not producing ARC gains" % (its[0], its[-1]))
    return out


def check_floor(run, floor_path):
    """ARC_FLOOR.json is owner-controlled: {"min_solved_fraction": x}. The loop may not finish below it."""
    sc = load_score(run)
    if sc is None:
        return ["no ARC_SCORE.json"]
    if not os.path.exists(floor_path):
        return ["ARC_FLOOR.json missing"]
    fl = S.load(floor_path)["min_solved_fraction"]
    frac = sc["claimed"]["solved"] / max(1, sc["claimed"]["total"])
    return [] if frac >= fl else ["ARC solved fraction %.4f below owner floor %.4f" % (frac, fl)]


WITH_ROOT = (check_arc_recomputed, check_weights_matter)
ALL = [check_exists_and_scale, check_arc_recomputed, check_score_positive_and_beats_controls, check_weights_matter,
       check_no_answer_or_baseline_shortcuts, check_iterating_and_not_regressing]
