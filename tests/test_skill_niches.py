"""Skill-cell core and niche generators (head-center). Synthetic only: no ARC data, no maker.
Run: python -m unittest tests.test_skill_niches -v
"""
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402


def solves(cell, task):
    return S.passes(cell, task)


def rand_params(template, rng):
    return {k: np.array([rng.uniform(-9, 9) for _ in range(int(np.prod(s)))]).reshape(s).tolist() for k, s in S.TEMPLATES[template]["shapes"].items()}


class TruthCellsSolveTheirOwnTasks(unittest.TestCase):
    def test_instance_niches(self):
        for name in ("tiling", "sequence", "neighbor", "object", "ray", "crop", "recall", "induction", "seqops", "selcopy", "grammar", "successor", "agree",
                     "gravity", "symmetry", "scale", "halfbool", "denoise", "objmove", "floodm", "raycross", "context2", "objmove2", "floodthru", "snapline", "legend", "grammar2", "lineconn", "cropcolor", "blockpick", "seqcrop", "countout", "outline", "objmove3", "blockred", "selfkron"):
            niche = N.NICHES[name]
            for pattern in niche.patterns(14, 3):
                cell = niche.truth_cell(pattern)
                for seed in (1, 2, 3):
                    self.assertTrue(solves(cell, niche.task(pattern, seed)), (name, pattern["id"], seed))

    def test_general_niches_one_cell_solves_every_pattern(self):
        for name in ("association", "glyph", "fillenc", "ocrdigit", "ocrseq", "analogy", "tooluse", "wallpruned", "corridormarker", "anchorspin", "framecorner", "referenceobject", "anchorclip", "anchorassign", "frameinfer", "referencepaint"):
            niche = N.NICHES[name]
            patterns = niche.patterns(20, 4)
            cell = niche.truth_cell(patterns[0])
            for pattern in patterns:
                for seed in (1, 2):
                    self.assertTrue(solves(cell, niche.task(pattern, seed)), (name, pattern["id"], seed))

    def test_instance_cell_does_not_solve_other_patterns(self):
        niche = N.NICHES["tiling"]
        patterns = niche.patterns(12, 3)
        cell = niche.truth_cell(patterns[0])
        wrong = sum(solves(cell, niche.task(p, 9)) for p in patterns[1:])
        self.assertLess(wrong, 3)

    def test_random_tensors_do_not_solve(self):
        rng = random.Random(0)
        for name, niche in N.NICHES.items():
            pattern = niche.patterns(3, 5)[0]
            hits = sum(solves(S.make_cell(niche.template, rand_params(niche.template, rng)), niche.task(pattern, s)) for s in range(30))
            # low-entropy selectors (mask colour x axis x period) are right by chance about 2% of the time: that is the control rate
            # in-context tables (neighbour, object) are partly learnable by several selector settings, so their random-control rate is higher
            # control rate = chance that random selectors land on a setting that solves the task
            self.assertLessEqual(hits, {"neighbor": 12, "object": 12, "induction": 10, "successor": 22, "denoise": 8, "scale": 6, "floodm": 8, "context2": 12, "objmove2": 8, "floodthru": 8, "legend": 8, "tooluse": 22, "countout": 8, "lineconn": 8}.get(name, 4), name)

    def test_association_uses_only_other_demonstrations(self):
        """Leave-one-out: an unseen mapping cannot be solved by peeking at the pair being scored."""
        niche = N.NICHES["association"]
        pattern = niche.patterns(1, 1)[0]
        cell = niche.truth_cell(pattern)
        task = niche.task(pattern, 3)
        lone = {"train": task["train"][:1], "test": task["test"]}
        full = S.run(cell, task["train"][0]["input"], task["train"][:1])
        self.assertEqual(full, task["train"][0]["output"])  # with itself as context it would trivially succeed ...
        loo = S.run(cell, task["train"][0]["input"], [])
        self.assertEqual(loo, task["train"][0]["input"])  # ... but with no other demonstration nothing is learned
        self.assertTrue(S.uses_context(cell))
        self.assertFalse(S.uses_context(N.NICHES["tiling"].truth_cell(N.NICHES["tiling"].patterns(1, 1)[0])))


class ChainsAreTheBreedingProduct(unittest.TestCase):
    def test_chain_of_tiling_then_association_solves_a_composite_task(self):
        tiling, assoc = N.NICHES["tiling"], N.NICHES["association"]
        tp, ap = tiling.patterns(5, 3)[0], assoc.patterns(3, 2)[1]
        a, b = tiling.truth_cell(tp), assoc.truth_cell(ap)
        child = S.chain_cell(a, b)
        rng = random.Random(8)
        pairs = []
        for _ in range(4):
            x = tp["sample_input"](rng)
            pairs.append({"input": x, "output": ap["apply"](tp["apply"](x))})
        task = {"train": pairs[:3], "test": pairs[3:]}
        self.assertTrue(solves(child, task))
        self.assertFalse(solves(a, task))
        self.assertFalse(solves(b, task))
        self.assertEqual(S.positions(child), 0)
        self.assertEqual(S.depth(child), 2)

    def test_tensor_roundtrip_preserves_chains_and_predictions(self):
        tiling, seq = N.NICHES["tiling"], N.NICHES["sequence"]
        a, b = tiling.truth_cell(tiling.patterns(3, 1)[0]), seq.truth_cell(seq.patterns(3, 1)[0])
        cells = [a, S.chain_cell(a, b), b]
        tensors = S.to_tensors(cells)
        back, router = S.from_tensors(tensors)
        self.assertEqual([c["id"] for c in back], [c["id"] for c in cells])
        pattern = tiling.patterns(3, 1)[0]
        task = tiling.task(pattern, 4)
        self.assertEqual(S.predict(tensors, task), [[task["test"][0]["output"]]])


class RouterAndScorer(unittest.TestCase):
    def test_router_picks_the_cell_that_reproduces_the_demonstrations(self):
        niche = N.NICHES["tiling"]
        patterns = niche.patterns(8, 2)
        cells = [niche.truth_cell(p) for p in patterns]
        rows = [{"id": p["id"], "task": niche.task(p, 11)} for p in patterns]
        result = S.Scorer(rows).score(cells)
        self.assertEqual(result["solved"], len(rows))
        self.assertEqual([result["picks"][p["id"]] for p in patterns], [c["id"] for c in cells])

    def test_exact_ties_go_to_the_older_cell(self):
        niche = N.NICHES["tiling"]
        pattern = niche.patterns(1, 2)[0]
        a = niche.truth_cell(pattern)
        twin = S.make_cell("tile_orient", S.make_cell("tile_orient", a["params"])["params"], note="twin")
        twin["id"] = "z" + a["id"]
        rows = [{"id": "t", "task": niche.task(pattern, 3)}]
        self.assertEqual(S.Scorer(rows).score([a, twin])["picks"]["t"], a["id"])

    def test_cache_makes_rescoring_free_and_equal(self):
        niche = N.NICHES["sequence"]
        patterns = niche.patterns(6, 1)
        cells = [niche.truth_cell(p) for p in patterns]
        rows = [{"id": p["id"], "task": niche.task(p, 2)} for p in patterns]
        scorer = S.Scorer(rows)
        first = scorer.score(cells)
        size = len(scorer.err)
        self.assertEqual(scorer.score(cells)["outcomes"], first["outcomes"])
        self.assertEqual(len(scorer.err), size)

    def test_zero_tensors_solve_nothing(self):
        niche = N.NICHES["glyph"]
        pattern = niche.patterns(1, 1)[0]
        zero = S.make_cell("glyph_recolor", {k: np.zeros_like(np.asarray(v)).tolist() for k, v in niche.truth_cell(pattern)["params"].items()})
        self.assertFalse(solves(zero, niche.task(pattern, 3)))


class SpeedPaths(unittest.TestCase):
    def test_persistent_cache_roundtrip_makes_a_restart_warm(self):
        import tempfile
        niche = N.NICHES["tiling"]
        patterns = niche.patterns(6, 2)
        cells = [niche.truth_cell(p) for p in patterns]
        rows = [{"id": p["id"], "task": niche.task(p, 5)} for p in patterns]
        first = S.Scorer(rows)
        result = first.score(cells)
        path = os.path.join(tempfile.mkdtemp(), "cache.pkl")
        first.save_cache(path)
        second = S.Scorer(rows)
        second.load_cache(path)
        self.assertEqual(len(second.err), len(first.err))
        self.assertEqual(second.score(cells)["outcomes"], result["outcomes"])

    def test_key_cache_does_not_change_context_table_predictions(self):
        for name in ("neighbor", "context2"):
            niche = N.NICHES[name]
            for pattern in niche.patterns(6, 4):
                task = niche.task(pattern, 3)
                cell = niche.truth_cell(pattern)
                S._KEYS.clear()
                cold = [S.run(cell, d["input"], task["train"]) for d in task["train"]]
                warm = [S.run(cell, d["input"], task["train"]) for d in task["train"]]
                self.assertEqual(cold, warm)
                self.assertTrue(S.passes(cell, task))

    def test_contender_prefilter_never_skips_a_cell_that_scoring_would_route_to(self):
        niche = N.NICHES["tiling"]
        patterns = niche.patterns(8, 2)
        cells = [niche.truth_cell(p) for p in patterns]
        rows = [{"id": p["id"], "task": niche.task(p, 7)} for p in patterns]
        scorer = S.Scorer(rows)
        for i in range(1, len(cells)):
            base, extra = cells[:i], cells[i]
            router = S.default_router(base)
            routed_to_extra = {rid for rid, pick in scorer.score(base + [extra])["picks"].items() if pick == extra["id"]}
            winners = set(scorer.contends(extra, base, router, rows))
            self.assertTrue(routed_to_extra <= winners, (i, routed_to_extra, winners))


if __name__ == "__main__":
    unittest.main()
