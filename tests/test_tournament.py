"""Cell Wars tournament edition: duels are zero-sum, the loser's cells are out, the hard tier and sparsity break ties, and every row recomputes."""
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import arena_core as A  # noqa: E402
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402
import tournament_core as T  # noqa: E402

NICHE = N.NICHES["scale"]
PAIR = ("Ember", "Tide")


def row(house, solved, luck=0, hard=0, nz=10, total=8, **extra):
    return dict({"kind": "match", "season": 4, "round_id": "c1-final", "niche": "scale", "house": house, "gen": 1, "solved": solved, "luck": luck,
                 "solved_hard": hard, "total": total, "hard_total": total, "nonzero": nz, "score": solved + 2 * hard, "charged": 10}, **extra)


class ArenaTiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arena = T.build(NICHE, N.LIMITS["scale"], PAIR)

    def test_hard_tier_uses_fewer_demonstrations_and_is_hidden(self):
        hard, referee = self.arena["hard"], self.arena["referee"]
        self.assertTrue(hard)
        self.assertTrue(all(len(t["train"]) == T.HARD_DEMOS for t in hard.values()))
        self.assertTrue(all(len(t["train"]) == 3 for t in referee.values()))
        seen = [d["input"] for h in PAIR for t in self.arena["train"][h].values() for d in t["train"] + t["test"]]
        for tier in (hard, referee):
            for task in tier.values():
                for d in task["train"] + task["test"]:
                    self.assertNotIn(d["input"], seen, "a referee input leaked into a training set")

    def test_scored_row_carries_both_tiers_sparsity_and_everything_needed_to_recompute(self):
        entry = {p["id"]: NICHE.truth_cell(p) for p in self.arena["patterns"]}
        r = T.score_row(NICHE, "Ember", 1, entry, self.arena, "p", "c1-semi1", "Tide", 4)
        self.assertEqual(r["solved"], r["total"])
        self.assertEqual(r["score"], r["solved"] + 2 * r["solved_hard"])
        self.assertGreater(r["nonzero"], 0)
        self.assertEqual(r["referee_ids"], sorted(self.arena["referee"]))

    def test_nonzero_counts_only_nonzero_entries(self):
        shapes = S.TEMPLATES["scale_grid"]["shapes"]
        zero = S.make_cell("scale_grid", {k: np.zeros(s).tolist() for k, s in shapes.items()})
        ones = S.make_cell("scale_grid", {k: np.ones(s).tolist() for k, s in shapes.items()})
        self.assertEqual(T.nonzero({"x": zero}), 0)
        self.assertEqual(T.nonzero({"x": ones}), sum(int(np.prod(s)) for s in shapes.values()))


class Duels(unittest.TestCase):
    def test_nobody_wins_unless_they_beat_luck(self):
        self.assertEqual(T.duel(row("Ember", 3, luck=5), row("Tide", 2, luck=5)), (None, "no contest"))

    def test_the_one_house_that_beats_luck_wins_by_forfeit(self):
        self.assertEqual(T.duel(row("Ember", 8, luck=1), row("Tide", 1, luck=3)), ("Ember", "forfeit"))

    def test_the_hard_tier_counts_double_and_decides_equal_standard_scores(self):
        self.assertEqual(T.duel(row("Ember", 8, hard=3), row("Tide", 8, hard=6)), ("Tide", "score"))

    def test_equal_gauntlet_goes_to_the_sparser_tensor_and_only_identical_is_a_draw(self):
        self.assertEqual(T.duel(row("Ember", 8, hard=8, nz=40), row("Tide", 8, hard=8, nz=12)), ("Tide", "parsimony"))
        self.assertEqual(T.duel(row("Ember", 8, hard=8, nz=12), row("Tide", 8, hard=8, nz=12)), (None, "draw"))

    def test_bout_is_decided_by_niche_wins_then_score_then_sparsity_then_seed(self):
        d = lambda w, a, b: {"winner": w, "rows": {"Ember": a, "Tide": b}}  # noqa: E731
        self.assertEqual(T.bout_winner("Ember", "Tide", [d("Tide", row("Ember", 3), row("Tide", 8)), d(None, row("Ember", 8), row("Tide", 8))]), ("Tide", "niche wins"))
        self.assertEqual(T.bout_winner("Ember", "Tide", [d(None, row("Ember", 8, hard=2), row("Tide", 8, hard=5))]), ("Tide", "gauntlet score"))
        self.assertEqual(T.bout_winner("Ember", "Tide", [d(None, row("Ember", 8, nz=30), row("Tide", 8, nz=20))]), ("Tide", "parsimony"))
        self.assertEqual(T.bout_winner("Ember", "Tide", [d(None, row("Ember", 8), row("Tide", 8))]), ("Ember", "higher seed"))

    def test_best_generation_per_round_is_chosen_by_score_then_sparsity(self):
        best = T.best_by_round([row("Ember", 4, gen=1), row("Ember", 8, gen=2, nz=50), row("Ember", 8, gen=3, nz=20)])
        self.assertEqual(best[(4, "c1-final", "scale", "Ember")]["gen"], 3)


class Bracket(unittest.TestCase):
    def test_pairings_give_the_top_seed_the_bye_when_odd(self):
        self.assertEqual(T.pairings(["A", "B", "C", "D"]), ([("A", "B"), ("C", "D")], None))
        self.assertEqual(T.pairings(["A", "B", "C"]), ([("B", "C")], "A"))
        self.assertEqual(T.pairings(["A", "B"]), ([("A", "B")], None))

    def test_table_is_zero_sum_for_duels_and_pays_placings_and_merges(self):
        rows = [{"kind": "duel", "a": "Ember", "b": "Tide", "winner": "Ember", "reason": "score"},
                {"kind": "duel", "a": "Ember", "b": "Tide", "winner": None, "reason": "draw"},
                {"kind": "duel", "a": "Ember", "b": "Tide", "winner": None, "reason": "no contest"},
                {"kind": "bout", "a": "Ember", "b": "Tide", "winner": "Ember", "reason": "niche wins"},
                {"kind": "placing", "house": "Ember", "place": 1, "points": T.placing_points(1)},
                {"kind": "placing", "house": "Tide", "place": 2, "points": T.placing_points(2)},
                {"kind": "merge_duel", "winner": "Tide"}]
        t = {x["house"]: x for x in T.table(rows, ("Ember", "Tide"))}
        self.assertEqual(t["Ember"]["points"], T.DUEL_WIN + T.DUEL_DRAW + T.BOUT_WIN + 10.0)
        self.assertEqual(t["Tide"]["points"], T.DUEL_DRAW + 6.0 + T.MERGE_WIN)
        self.assertEqual((t["Ember"]["duel_wins"], t["Tide"]["duel_losses"], t["Ember"]["titles"]), (1, 1, 1))


class Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arena = T.build(NICHE, N.LIMITS["scale"], PAIR)
        truth = {p["id"]: NICHE.truth_cell(p) for p in cls.arena["patterns"]}
        junk = {p["id"]: A.rand_cell(NICHE.template, random.Random(i)) for i, p in enumerate(cls.arena["patterns"])}
        cls.entries = {("scale", "Ember", 1, "c1-final"): A.canon_entry(truth), ("scale", "Tide", 1, "c1-final"): A.canon_entry(junk)}
        cls.rows = [T.score_row(NICHE, "Ember", 1, truth, cls.arena, "p", "c1-final", "Tide", 4),
                    T.score_row(NICHE, "Tide", 1, junk, cls.arena, "p", "c1-final", "Ember", 4)]
        best = T.best_by_round(cls.rows)
        winner, reason = T.duel(best[(4, "c1-final", "scale", "Ember")], best[(4, "c1-final", "scale", "Tide")])
        cls.rows.append({"kind": "duel", "season": 4, "round_id": "c1-final", "niche": "scale", "a": "Ember", "b": "Tide", "winner": winner, "reason": reason})
        cls.rows.append({"kind": "bout", "season": 4, "round_id": "c1-final", "a": "Ember", "b": "Tide", "winner": "Ember", "reason": "niche wins"})

    def test_an_honest_tournament_ledger_recomputes_from_the_tensors(self):
        self.assertEqual(self.rows[2]["winner"], "Ember")
        self.assertEqual(T.verify(self.rows, self.entries), [])

    def test_a_flipped_duel_a_forged_score_and_edited_tensors_are_all_caught(self):
        flipped = [dict(r, winner="Tide") if r["kind"] == "duel" else r for r in self.rows]
        self.assertTrue(any("duel" in p for p in T.verify(flipped, self.entries)))
        forged = [dict(r, score=r["score"] + 5) if r["kind"] == "match" and r["house"] == "Tide" else r for r in self.rows]
        self.assertTrue(T.verify(forged, self.entries))
        cell = dict(self.entries[("scale", "Ember", 1, "c1-final")])
        first = sorted(cell)[0]
        params = {k: np.asarray(v, float) for k, v in cell[first]["params"].items()}
        params[sorted(params)[0]].flat[0] += 1.0
        edited = dict(self.entries)
        edited[("scale", "Ember", 1, "c1-final")] = dict(cell, **{first: {"template": cell[first]["template"], "params": {k: v.tolist() for k, v in params.items()}}})
        self.assertTrue(any("hash" in p for p in T.verify(self.rows, edited)))

    def test_merge_rows_recompute_and_the_group_must_be_fully_solved(self):
        referee = self.arena["referee"]
        p1, p2, p3 = self.arena["patterns"][:3]
        cell = NICHE.truth_cell(p1)
        r = T.merge_row(NICHE, "Ember", "c1-merge", "g0", [p1["id"], p2["id"]], [p3["id"]], cell, referee, "p", "Tide", 4)
        self.assertLess(r["solved_group"], 2, "one pattern's cell does not solve a different pattern")
        entries = {("scale", "Ember", "g0", "c1-merge"): A.canon_entry({"*": cell})}
        self.assertEqual(T.verify([r], entries), [])
        self.assertTrue(T.verify([dict(r, solved_group=2)], entries))
        self.assertEqual(T.merge_winner(r, dict(r, house="Tide", solved_group=2, nonzero=5)), ("Tide", "forfeit"))
        full = dict(r, solved_group=2)
        self.assertEqual(T.merge_winner(dict(full, ood_solved=1), dict(full, house="Tide", ood_solved=3)), ("Tide", "generalisation"))
        self.assertEqual(T.merge_winner(dict(full), dict(full, house="Tide")), (None, "draw"))

    def test_merge_duel_outcomes_are_recomputed_too(self):
        referee = self.arena["referee"]
        p1, p2 = self.arena["patterns"][:2]
        rows = [T.merge_row(NICHE, h, "c1-merge", "g0", [p1["id"], p2["id"]], [], None, referee, "p", "Tide" if h == "Ember" else "Ember", 4) for h in PAIR]
        entries = {("scale", h, "g0", "c1-merge"): A.canon_entry({}) for h in PAIR}
        duel_row = {"kind": "merge_duel", "season": 4, "round_id": "c1-merge", "niche": "scale", "group": "g0", "a": "Ember", "b": "Tide", "winner": None, "reason": "no contest"}
        self.assertEqual(T.verify(rows + [duel_row], entries), [])
        self.assertTrue(any("merge duel" in p for p in T.verify(rows + [dict(duel_row, winner="Ember", reason="forfeit")], entries)))


if __name__ == "__main__":
    unittest.main()
