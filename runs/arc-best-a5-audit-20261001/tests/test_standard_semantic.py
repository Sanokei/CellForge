"""Semantic anti-cheat tests for docs/ONE_STANDARD.md. Owner-controlled: do not weaken, skip or delete.

Run: python -m unittest tests.test_standard_semantic -v

Part 1 (MetaTests): builds a clean synthetic run and several deliberately cheating runs; each check in tests/standard_checks.py
must pass the clean run and catch its cheat. This proves the checks work independently of any real run.
Part 2 (RealRuns): applies every check to every real runs/weights-*/ directory and requires that at least one exists.
"""
import copy
import glob
import hashlib
import json
import os
import random
import shutil
import tempfile
import unittest

from tests import standard_checks as C

IDENT = [[1.0 if i == j else 0.0 for c_ in range(1)] for i in range(4) for j in range(4)]


def Wmap(mapping):
    """4x4 W where color c maps to mapping.get(c, c)."""
    W = [[0.0] * 4 for _ in range(4)]
    for c in range(4):
        W[mapping.get(c, c)][c] = 1.0
    return W


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def build_run(root, cheat=None):
    run = os.path.join(root, "weights-fake")
    os.makedirs(run, exist_ok=True)
    rng = random.Random(0)
    niche = "recolor12"
    genomes, proposals, founders = [], [], []
    ledger = []
    t0 = 1000.0
    seen_inputs = [[[9, 9, 9]] * 3]  # never equals a generated eval input

    def add_weights(name, tensors):
        write_json(os.path.join(run, "weights", name + ".json"), {"tensors": tensors})
        return "weights/%s.json" % name

    def prov(pid, kind, arm, seed, actor, tensors, shown, **kw):
        rec = dict(id=pid, kind=kind, arm=arm, seed=seed, niche=niche, actor=actor, ts_outbox=t0, ts_inbox=t0 + 1, ts_eval=t0 + 2,
                   shown={"counterexample_inputs": shown}, response={"tensors": tensors}, rationale="because", rng_seed=kw.get("rng_seed"))
        write_json(os.path.join(run, "provenance", pid + ".json"), rec)

    # founders (LLM-written): rough maps, 1->0 wrong, 1->2 right
    fmaps = [{1: 2}, {1: 0}, {1: 3}, {}]
    for i, m in enumerate(fmaps):
        T = {"W": Wmap(m)}
        h = C.weights_hash(T)
        wf = add_weights("founder%d" % i, T)
        pid = "f%d" % i
        prov(pid, "founder", "llm", 0, "claude-sonnet-5.5", T, [])
        founders.append(dict(hash=h, niche=niche, author="llm", actor="claude-sonnet-5.5", rationale="because", weights_file=wf,
                             accuracy=C.accuracy(__import__("tests.fake_eval", fromlist=["x"]), T, 1234, 200), provenance=pid, generation=0))
    fh = [f["hash"] for f in founders]

    def child(pid, arm, origin, seed, T, parent, gen, shown):
        h = C.weights_hash(T)
        wf = add_weights(pid, T)
        acc = C.accuracy(__import__("tests.fake_eval", fromlist=["x"]), T, 1234, 200)
        genomes.append(dict(hash=h, niche=niche, origin=origin, parents=[parent], weights_file=wf, accuracy=acc, generation=gen,
                            arm=arm, provenance=pid, seed=seed))
        proposals.append(dict(arm=arm, seed=seed, charged=True, outcome="accepted"))
        ledger.append({"type": "proposal", "id": pid})
        return h

    seeds = [1, 2, 3]
    budget = 3
    for seed in seeds:
        for k in range(budget):
            # LLM: proposes the right map with tiny perturbations of unused cells (distinct in float, meaningful in behaviour)
            T = {"W": Wmap({1: 2})}
            T["W"][3][3] = 1.0 + 0.01 * (seed * 10 + k)
            pid = "l%d_%d" % (seed, k)
            prov(pid, "llm_mutation", "llm", seed, "claude-sonnet-5.5", T, seen_inputs)
            child(pid, "llm", "llm_mutation", seed, T, fh[k % 4], 1 + k, seen_inputs)
            # random: random map from the same vocab
            T2 = {"W": Wmap({1: rng.choice([0, 1, 3]) if rng.random() < 0.8 else 2})}
            T2["W"][3][3] = 1.0 + 0.01 * (seed * 10 + k) + 0.5
            pid2 = "r%d_%d" % (seed, k)
            prov(pid2, "random_mutation", "random", seed, "rng", T2, [], rng_seed=seed * 100 + k)
            child(pid2, "random", "random_mutation", seed, T2, fh[k % 4], 1 + k, [])
    std = dict(arms={"llm": {}, "random": {}}, seeds=seeds, seeds_declared_before_run=True, budget_per_arm=budget, niches=[niche],
               founders=founders, genomes=genomes, proposals=proposals, evaluator="tests.fake_eval", eval_seeds=[11, 12, 13],
               maker_seen_seeds=[1234], random_vocab={"tensors": ["W"], "low": 0.0, "high": 2.5}, ledger_file="ledger.jsonl")

    # ---- cheats ----
    if cheat == "inflated_acc":
        for g in std["genomes"]:
            if g["arm"] == "llm":
                g["accuracy"] = 1.0
        for g in std["genomes"]:
            if g["arm"] == "random":
                g["accuracy"] = 1.0
        std["genomes"][0]["accuracy"] = 1.0
        # make a bad genome claim perfection
        Tbad = {"W": Wmap({1: 0})}
        hbad = C.weights_hash(Tbad)
        wf = add_weights("bad", Tbad)
        prov("bad", "llm_mutation", "llm", 1, "claude", Tbad, seen_inputs)
        std["genomes"].append(dict(hash=hbad, niche=niche, origin="llm_mutation", parents=[fh[0]], weights_file=wf, accuracy=1.0,
                                   generation=1, arm="llm", provenance="bad", seed=1))
        std["proposals"].append(dict(arm="llm", seed=1, charged=True, outcome="accepted"))
        ledger.append({"type": "proposal", "id": "bad"})
    if cheat == "near_dup":
        base = std["genomes"][0]
        T = C.weights_of(run, base["weights_file"])
        for i in range(40):
            T2 = copy.deepcopy(T)
            T2["W"][3][3] += 1e-9 * (i + 1)
            h = C.weights_hash(T2)
            wf = add_weights("dup%d" % i, T2)
            pid = "dup%d" % i
            prov(pid, "llm_mutation", "llm", 1, "claude", T2, seen_inputs)
            std["genomes"].append(dict(hash=h, niche=niche, origin="llm_mutation", parents=[base["hash"]], weights_file=wf,
                                       accuracy=base["accuracy"], generation=2, arm="llm", provenance=pid, seed=1))
            std["proposals"].append(dict(arm="llm", seed=1, charged=True, outcome="neutral"))
            ledger.append({"type": "proposal", "id": pid})
    if cheat == "seed_leak":
        std["maker_seen_seeds"] = [11]
    if cheat == "input_leak":
        x = C.evaluator(std).make_tasks(11, 1)[0][0]
        pf = os.path.join(run, "provenance", "l1_0.json")
        p = C.load(pf)
        p["shown"]["counterexample_inputs"].append(x)
        write_json(pf, p)
    if cheat == "fake_llm":
        pf = os.path.join(run, "provenance", "l1_0.json")
        p = C.load(pf)
        p["response"]["tensors"]["W"][0][0] = 0.123
        write_json(pf, p)
    if cheat == "no_counterexample":
        pf = os.path.join(run, "provenance", "l1_0.json")
        p = C.load(pf)
        p["shown"]["counterexample_inputs"] = []
        write_json(pf, p)
    if cheat == "random_saw_counterexamples":
        pf = os.path.join(run, "provenance", "r1_0.json")
        p = C.load(pf)
        p["shown"]["counterexample_inputs"] = seen_inputs
        write_json(pf, p)
    if cheat == "unfair_random":
        std["random_vocab"] = {"tensors": ["W"], "low": 0.0, "high": 0.5}
    if cheat == "hash_mismatch":
        std["genomes"][0]["hash"] = "0" * 64
    if cheat == "unequal_budget":
        for p in std["proposals"]:
            pass
        std["budget_per_arm"] = 2
    if cheat == "trivial_task":
        # make every genome equal zero weights: evaluator cannot separate anything
        for f in std["founders"] + std["genomes"]:
            T = {"W": [[0.0] * 4 for _ in range(4)]}
            write_json(os.path.join(run, f["weights_file"]), {"tensors": T})
    if cheat == "fewer_proposals_than_ledger":
        std["proposals"] = std["proposals"][:-2]

    write_json(os.path.join(run, "STANDARD.json"), std)
    # ledger chain
    prev = "GENESIS"
    lines = []
    for ev in ledger:
        line = json.dumps({"prev": prev, "event": ev}, sort_keys=True)
        lines.append(line)
        prev = hashlib.sha256(line.encode()).hexdigest()
    if cheat == "broken_chain":
        lines[3], lines[4] = lines[4], lines[3]
    if cheat == "truncated_ledger":
        lines = lines[:-3]
    with open(os.path.join(run, "ledger.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return run


CHEAT_TO_CHECK = {
    "inflated_acc": C.check_accuracy_recomputed,
    "near_dup": C.check_near_duplicates,
    "seed_leak": C.check_seed_separation,
    "input_leak": C.check_seed_separation,
    "fake_llm": C.check_provenance,
    "no_counterexample": C.check_provenance,
    "random_saw_counterexamples": C.check_provenance,
    "unfair_random": C.check_random_control_fair,
    "hash_mismatch": C.check_hash_integrity,
    "trivial_task": C.check_task_not_trivial,
    "broken_chain": C.check_ledger_chain,
    "truncated_ledger": C.check_budget_reconciled,
}


class MetaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_run_passes_every_check(self):
        run = build_run(self.tmp)
        for chk in C.ALL:
            self.assertEqual(chk(run), [], "%s flagged a clean run: %s" % (chk.__name__, chk(run)))

    def test_clean_run_llm_beats_random_recomputed(self):
        run = build_run(self.tmp)
        self.assertEqual(C.check_llm_vs_random_recomputed(run), [])

    def test_each_cheat_is_caught_by_its_check(self):
        for cheat, chk in CHEAT_TO_CHECK.items():
            with self.subTest(cheat=cheat):
                d = tempfile.mkdtemp(dir=self.tmp)
                run = build_run(d, cheat)
                problems = chk(run)
                self.assertTrue(problems, "check %s did not catch cheat %s" % (chk.__name__, cheat))

    def test_unequal_budget_cheat_is_caught(self):
        d = tempfile.mkdtemp(dir=self.tmp)
        run = build_run(d, "unequal_budget")
        self.assertTrue(C.check_budget_reconciled(run), "proposals above budget not caught")

    def test_random_arm_cannot_win_by_inflated_claims(self):
        """If random claims higher accuracy than its recomputed accuracy, the recompute check must flag it."""
        d = tempfile.mkdtemp(dir=self.tmp)
        run = build_run(d)
        s = C.load(os.path.join(run, "STANDARD.json"))
        for g in s["genomes"]:
            if g["arm"] == "random":
                g["accuracy"] = 1.0
        write_json(os.path.join(run, "STANDARD.json"), s)
        self.assertTrue(C.check_accuracy_recomputed(run))


class RouterMeta(unittest.TestCase):
    """Mini two-niche evaluator to prove the router checks catch a useless router."""

    def _mk(self, useful):
        import types
        ev = types.ModuleType("tmp_router_eval")
        def make_multi_tasks(seed, n):
            r = random.Random(seed)
            out = []
            for _ in range(n):
                k = r.randrange(2)
                x = (k, r.randrange(1, 4))
                out.append((x, x[1] + 1 if k == 0 else x[1] * 0))
            return out
        def predict_routed(cells, router, x):
            k, v = x
            sc = router["R"][k]
            pick = max(range(2), key=lambda j: sc[j])
            return v + 1 if pick == 0 else 0
        def uniform_router(router):
            return {"R": [[0.0, 0.0], [0.0, 0.0]]}
        ev.make_multi_tasks, ev.predict_routed, ev.uniform_router = make_multi_tasks, predict_routed, uniform_router
        import sys
        sys.modules["tmp_router_eval"] = ev
        d = tempfile.mkdtemp()
        run = os.path.join(d, "weights-r")
        R = [[1.0, 0.0], [0.0, 1.0]] if useful else [[0.0, 0.0], [0.0, 0.0]]
        write_json(os.path.join(run, "w.json"), {"tensors": {"C": [1.0]}})
        write_json(os.path.join(run, "r.json"), {"tensors": {"R": R}})
        std = dict(founders=[], genomes=[dict(hash="h", niche="n", origin="llm_mutation", parents=[], weights_file="w.json",
                                              router_weights_file="r.json", arm="llm", edit_scope="joint")],
                   evaluator="tmp_router_eval", eval_seeds=[1])
        write_json(os.path.join(run, "STANDARD.json"), std)
        return d, run

    def test_useful_router_passes_and_useless_router_is_caught(self):
        d, run = self._mk(True)
        self.assertEqual(C.check_router_present_and_matters(run), [])
        shutil.rmtree(d, ignore_errors=True)
        d, run = self._mk(False)
        self.assertTrue(C.check_router_present_and_matters(run), "uniform/useless router was not caught")
        shutil.rmtree(d, ignore_errors=True)

    def test_missing_router_weights_is_caught(self):
        d, run = self._mk(True)
        s = C.load(os.path.join(run, "STANDARD.json"))
        s["genomes"][0].pop("router_weights_file")
        write_json(os.path.join(run, "STANDARD.json"), s)
        self.assertTrue(C.check_router_present_and_matters(run))
        shutil.rmtree(d, ignore_errors=True)


class RealRuns(unittest.TestCase):
    def runs(self):
        rs = sorted(glob.glob(os.path.join(C.ROOT, "runs", "weights-*", "STANDARD.json")))
        self.assertTrue(rs, "no runs/weights-*/STANDARD.json exists: the LLM-writes-weights loop has not run under the standard")
        return [os.path.dirname(p) for p in rs]

    def test_every_semantic_check_passes_on_every_real_run(self):
        for run in self.runs():
            for chk in C.ALL:
                self.assertEqual(chk(run), [], "%s on %s" % (chk.__name__, run))

    def test_router_exists_matters_and_was_llm_mutated_on_a_real_run(self):
        for run in self.runs():
            self.assertEqual(C.check_router_present_and_matters(run), [], run)
            self.assertEqual(C.check_router_was_mutated_by_llm(run), [], run)

    def test_recomputed_llm_beats_random_on_a_real_run(self):
        ok = []
        for run in self.runs():
            ok.append(C.check_llm_vs_random_recomputed(run) == [])
        self.assertTrue(any(ok), "no real run shows LLM-written weights beating the random control when recomputed")


if __name__ == "__main__":
    unittest.main()
