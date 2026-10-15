"""The goal is ARC-AGI accuracy of the literal weights. Owner-controlled: do not weaken, skip, delete or work around.

Run: python -m unittest tests.test_arc_goal -v
Part 1 (MetaTests): a synthetic ARC split proves every check passes an honest run and catches each cheat.
Part 2 (RealRuns): at least one real runs/weights-*/ARC_SCORE.json must pass every check, recomputed from the weights.
These real-run tests FAIL (never skip) until that evidence exists. Finishing the loop means this file is green.
"""
import glob
import json
import os
import shutil
import tempfile
import unittest

from tests import arc_goal_checks as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SRC = '''
def predict(weights, task):
    k = int(round(weights["flip"][0]))
    return [[[r[::-1] for r in ex["input"]] if k else ex["input"]] for ex in task["test"]]
'''
CHEAT_ANSWER_SRC = '''
def predict(weights, task):
    return [[ex.get("output", ex["input"])] for ex in task["test"]]
'''
NO_WEIGHTS_SRC = '''
def predict(w, t):
    return [[[r[::-1] for r in ex["input"]]] for ex in t["test"]]
'''


def make(tmp, n=120, evaluator=EVAL_SRC, history=None, flip=1.0):
    run = os.path.join(tmp, "runs", "weights-x")
    arc = os.path.join(tmp, "arc", "training")
    os.makedirs(run)
    os.makedirs(arc)
    ids, per = [], {}
    for i in range(n):
        tid = "t%03d" % i
        ids.append(tid)
        g = [[(i + r + c) % 3 for c in range(3)] for r in range(3)]
        out = [r[::-1] for r in g]
        with open(os.path.join(arc, tid + ".json"), "w") as f:
            json.dump({"train": [{"input": g, "output": out}], "test": [{"input": g, "output": out}]}, f)
        per[tid] = bool(flip)
    with open(os.path.join(run, "w.json"), "w") as f:
        json.dump({"tensors": {"flip": [flip]}}, f)
    with open(os.path.join(run, "ev.py"), "w") as f:
        f.write(evaluator)
    ok = sum(per.values())
    sc = {"split": "training", "task_ids": ids, "weights_file": "w.json", "evaluator": "ev.py",
          "claimed": {"solved": ok, "total": n, "per_task": per},
          "controls": {"zero_weights": {"solved": 0}, "random_weights": {"solved": 3, "seeds": [1, 2, 3]}},
          "history": history or [{"iteration": 1, "solved": 5, "weights_hash": "a"},
                                 {"iteration": 2, "solved": 40, "weights_hash": "b"},
                                 {"iteration": 3, "solved": ok, "weights_hash": "c"}],
          "arc1_arc2_merged_by_task_id": True}
    save(run, sc)
    return run, os.path.join(tmp, "arc")


def save(run, sc):
    with open(os.path.join(run, "ARC_SCORE.json"), "w") as f:
        json.dump(sc, f)


def claim_everything(run):
    p = os.path.join(run, "ARC_SCORE.json")
    sc = json.load(open(p))
    for k in sc["claimed"]["per_task"]:
        sc["claimed"]["per_task"][k] = True
    sc["claimed"]["solved"] = sc["claimed"]["total"]
    save(run, sc)


def problems(run, arc):
    out = []
    for f in A.ALL:
        out += f(run, arc) if f in A.WITH_ROOT else f(run)
    return out


class MetaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_honest_run_passes(self):
        run, arc = make(self.tmp)
        self.assertEqual(problems(run, arc), [])

    def test_no_score_file_fails(self):
        run, arc = make(self.tmp)
        os.remove(os.path.join(run, "ARC_SCORE.json"))
        self.assertTrue(A.check_exists_and_scale(run))

    def test_inflated_claim_caught(self):
        run, arc = make(self.tmp, flip=0.0)
        claim_everything(run)
        self.assertTrue(A.check_arc_recomputed(run, arc))

    def test_answer_reading_evaluator_gets_nothing(self):
        run, arc = make(self.tmp, evaluator=CHEAT_ANSWER_SRC, flip=0.0)
        claim_everything(run)
        self.assertTrue(A.check_arc_recomputed(run, arc))

    def test_score_not_from_weights_caught(self):
        run, arc = make(self.tmp, evaluator=NO_WEIGHTS_SRC)
        self.assertTrue(A.check_weights_matter(run, arc))

    def test_baseline_import_caught(self):
        run, arc = make(self.tmp, evaluator=EVAL_SRC + "\nfrom research.claude_arc import cells\n")
        self.assertTrue(A.check_no_answer_or_baseline_shortcuts(run))

    def test_stalled_history_caught(self):
        h = [{"iteration": i, "solved": 120, "weights_hash": str(i)} for i in (1, 2, 3)]
        run, arc = make(self.tmp, history=h)
        self.assertTrue(A.check_iterating_and_not_regressing(run))

    def test_too_few_iterations_caught(self):
        run, arc = make(self.tmp, history=[{"iteration": 1, "solved": 120, "weights_hash": "a"}])
        self.assertTrue(A.check_iterating_and_not_regressing(run))

    def test_too_few_tasks_caught(self):
        run, arc = make(self.tmp, n=10)
        self.assertTrue(A.check_exists_and_scale(run))

    def test_not_beating_control_caught(self):
        run, arc = make(self.tmp)
        p = os.path.join(run, "ARC_SCORE.json")
        sc = json.load(open(p))
        sc["controls"]["random_weights"]["solved"] = 120
        save(run, sc)
        self.assertTrue(A.check_score_positive_and_beats_controls(run))

    def test_owner_floor(self):
        run, arc = make(self.tmp, flip=0.0)
        fl = os.path.join(self.tmp, "f.json")
        json.dump({"min_solved_fraction": 0.01}, open(fl, "w"))
        self.assertTrue(A.check_floor(run, fl))


class RealRuns(unittest.TestCase):
    def runs(self):
        return [os.path.dirname(p) for p in sorted(glob.glob(os.path.join(ROOT, "runs", "weights-*", "ARC_SCORE.json")))]

    def test_an_arc_scored_run_exists(self):
        self.assertTrue(self.runs(), "no runs/weights-*/ARC_SCORE.json: the loop has never measured ARC-AGI accuracy of its weights")

    def test_best_real_run_passes_every_arc_check(self):
        rs = self.runs()
        self.assertTrue(rs, "no ARC_SCORE.json")
        res = {r: problems(r, A.arc_root()) for r in rs}
        self.assertTrue(any(not v for v in res.values()), json.dumps(res, indent=1))

    def test_owner_floor_met(self):
        rs = self.runs()
        self.assertTrue(rs, "no ARC_SCORE.json")
        fp = os.path.join(ROOT, "ARC_FLOOR.json")
        self.assertTrue(any(not A.check_floor(r, fp) for r in rs), "no run meets ARC_FLOOR.json")


if __name__ == "__main__":
    unittest.main()
