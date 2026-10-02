"""Guardrail tests for docs/ONE_STANDARD.md. Owner-controlled: do not weaken, skip or delete.

Run: python -m unittest tests.test_one_standard -v

Contract: every directory runs/weights-*/ must contain STANDARD.json:
{
  "arms": {"llm": {...}, "random": {...}, optional "breeding": {...}},
  "seeds": [ints], "seeds_declared_before_run": true,
  "budget_per_arm": int,                      # charged proposals, equal in every arm
  "niches": [str, ...],
  "founders": [{"hash", "niche", "author": "llm", "actor", "rationale", "weights_file"}],
  "genomes": [{"hash", "niche", "origin": "llm_founder|llm_mutation|random_mutation|crossover",
               "parents": [hash], "weights_file", "accuracy": float, "generation": int, "arm": "llm|random|breeding"}],
  "proposals": [{"arm", "seed", "hash_or_null", "charged": true, "outcome": "accepted|rejected|invalid|neutral"}]
}
Each weights_file is JSON or NPZ holding numeric tensors. Milestone tests define the target; they fail until it is met.
"""
import glob
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
MIN_FOUNDERS_PER_NICHE = 3
MIN_NICHES = 4
MIN_DISTINCT_GENOMES = 100
MIN_SEEDS = 3
ORIGINS = {"llm_founder", "llm_mutation", "random_mutation", "crossover"}


def standard_runs():
    return sorted(glob.glob(os.path.join(RUNS, "weights-*", "STANDARD.json")))


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tensors(path):
    if path.endswith(".npz"):
        import numpy as np
        z = np.load(path)
        return {k: z[k] for k in z.files}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("tensors", d)


def numeric_leaf_count(x):
    if isinstance(x, bool):
        return 0
    if isinstance(x, (int, float)):
        return 1
    if isinstance(x, (list, tuple)):
        return sum(numeric_leaf_count(i) for i in x)
    if isinstance(x, dict):
        return sum(numeric_leaf_count(i) for i in x.values())
    if hasattr(x, "size"):
        return int(x.size)
    return 0


class RepoGuards(unittest.TestCase):
    def test_agents_md_exists_and_points_to_standard(self):
        p = os.path.join(ROOT, "AGENTS.md")
        self.assertTrue(os.path.exists(p), "AGENTS.md missing")
        t = open(p, encoding="utf-8").read()
        self.assertIn("ONE_STANDARD", t)
        self.assertIn("literal weight", t.lower())

    def test_standard_doc_exists(self):
        self.assertTrue(os.path.exists(os.path.join(ROOT, "docs", "ONE_STANDARD.md")))

    def test_no_new_gene_list_runs_after_cutoff(self):
        """Runs created after the standard (mtime >= cutoff) must not be gene-list/selection ledgers unless under runs/baseline-*."""
        cutoff = os.path.getmtime(os.path.join(ROOT, "docs", "ONE_STANDARD.md")) if os.path.exists(
            os.path.join(ROOT, "docs", "ONE_STANDARD.md")) else 0
        offenders = []
        for led in glob.glob(os.path.join(RUNS, "*", "ledger.jsonl")):
            run = os.path.basename(os.path.dirname(led))
            if run.startswith(("weights-", "baseline-")) or os.path.getctime(led) < cutoff:
                continue
            with open(led, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if '"alleles"' in line and '"genome"' in line:
                        offenders.append(run)
                        break
        self.assertFalse(offenders, "new gene-list runs after the standard (rename to baseline-* or redo as weights-*): %s" % offenders)


class ContractOfEveryWeightsRun(unittest.TestCase):
    def runs(self):
        return [(p, load(p)) for p in standard_runs()]

    def test_required_fields(self):
        for p, s in self.runs():
            for k in ("arms", "seeds", "seeds_declared_before_run", "budget_per_arm", "niches", "founders", "genomes", "proposals"):
                self.assertIn(k, s, "%s missing %s" % (p, k))

    def test_llm_arm_and_random_control_present(self):
        for p, s in self.runs():
            self.assertIn("llm", s["arms"], p)
            self.assertIn("random", s["arms"], p)

    def test_seeds_predeclared_and_enough(self):
        for p, s in self.runs():
            self.assertTrue(s["seeds_declared_before_run"], p)
            self.assertGreaterEqual(len(set(s["seeds"])), MIN_SEEDS, p)

    def test_equal_charged_budget_every_arm_and_seed(self):
        for p, s in self.runs():
            for arm in s["arms"]:
                for seed in s["seeds"]:
                    n = sum(1 for x in s["proposals"] if x["arm"] == arm and x["seed"] == seed)
                    self.assertEqual(n, s["budget_per_arm"], "%s arm=%s seed=%s charged %d" % (p, arm, seed, n))
            self.assertTrue(all(x.get("charged") is True for x in s["proposals"]), "uncharged proposal in %s" % p)

    def test_founders_are_llm_written_literal_weights(self):
        for p, s in self.runs():
            base = os.path.dirname(p)
            for f in s["founders"]:
                self.assertEqual(f["author"], "llm", "founder not LLM-authored in %s" % p)
                self.assertTrue(f.get("actor") and f.get("rationale"), "founder lacks actor/rationale in %s" % p)
                w = os.path.join(base, f["weights_file"])
                self.assertTrue(os.path.exists(w), w)
                self.assertGreater(numeric_leaf_count(tensors(w)), 0, "founder has no numeric weights: %s" % w)

    def test_genomes_are_weights_with_valid_lineage(self):
        for p, s in self.runs():
            base = os.path.dirname(p)
            hashes = {g["hash"] for g in s["genomes"]} | {f["hash"] for f in s["founders"]}
            for g in s["genomes"]:
                self.assertIn(g["origin"], ORIGINS, p)
                self.assertTrue(os.path.exists(os.path.join(base, g["weights_file"])), g["hash"])
                if g["origin"] != "llm_founder":
                    self.assertTrue(g["parents"], "no parent for %s" % g["hash"])
                    self.assertTrue(all(q in hashes for q in g["parents"]), "dangling parent %s" % g["hash"])
                if g["origin"] == "crossover":
                    self.assertGreaterEqual(len(set(g["parents"])), 2)

    def test_mutation_origin_matches_arm(self):
        for p, s in self.runs():
            for g in s["genomes"]:
                if g["origin"] == "llm_mutation":
                    self.assertEqual(g["arm"], "llm", p)
                if g["origin"] == "random_mutation":
                    self.assertEqual(g["arm"], "random", p)


class Milestone(unittest.TestCase):
    """The target. These fail until a real weights run exists. Do not edit to make them pass."""

    def best(self):
        rs = [(p, load(p)) for p in standard_runs()]
        self.assertTrue(rs, "no runs/weights-*/STANDARD.json exists yet: the LLM-writes-weights loop has not been run under the standard")
        return rs

    def test_enough_niches_and_llm_founders(self):
        ok = False
        for _, s in self.best():
            per = {}
            for f in s["founders"]:
                per[f["niche"]] = per.get(f["niche"], 0) + 1
            if len([n for n, c in per.items() if c >= MIN_FOUNDERS_PER_NICHE]) >= MIN_NICHES:
                ok = True
        self.assertTrue(ok, ">=%d niches with >=%d LLM-authored founders each" % (MIN_NICHES, MIN_FOUNDERS_PER_NICHE))

    def test_many_distinct_weight_genomes(self):
        n = max((len({g["hash"] for g in s["genomes"]}) for _, s in self.best()), default=0)
        self.assertGreaterEqual(n, MIN_DISTINCT_GENOMES, "only %d distinct weight genomes" % n)

    def test_llm_mutation_beats_random_control(self):
        ok = False
        for _, s in self.best():
            acc = {}
            for arm in ("llm", "random"):
                xs = [g["accuracy"] for g in s["genomes"] if g["arm"] == arm]
                acc[arm] = sum(xs) / len(xs) if xs else None
            if acc["llm"] is not None and acc["random"] is not None and acc["llm"] > acc["random"]:
                ok = True
        self.assertTrue(ok, "no run shows LLM-written mutations beating the random-weight control")

    def test_crossover_children_exist_and_are_new(self):
        n = 0
        for _, s in self.best():
            parents = {g["hash"]: set() for g in s["genomes"]}
            n += sum(1 for g in s["genomes"] if g["origin"] == "crossover" and len(set(g["parents"])) >= 2)
        self.assertGreater(n, 0, "no crossover children")

    def test_accuracy_improves_over_generations(self):
        ok = False
        for _, s in self.best():
            gens = {}
            for g in s["genomes"]:
                gens.setdefault(g["generation"], []).append(g["accuracy"])
            ks = sorted(gens)
            if len(ks) >= 3:
                first = sum(gens[ks[0]]) / len(gens[ks[0]])
                last = sum(gens[ks[-1]]) / len(gens[ks[-1]])
                ok = ok or last > first
        self.assertTrue(ok, "accuracy does not rise across generations")


if __name__ == "__main__":
    unittest.main()
