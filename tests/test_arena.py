"""Cell Wars: the scoring core cannot be flattered. Points are margin over random luck, recomputed from the saved tensors; referee tasks are a
seed family no house trains on; tampering with either the ledger or the tensors is caught; an honest decline is free, never negative."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import arena as AR  # noqa: E402
import arena_core as A  # noqa: E402
import niches as N  # noqa: E402
import skillcore as S  # noqa: E402

NICHE = N.NICHES["scale"]  # instance niche with a cheap generator
HOUSES = A.HOUSES[:2]


def truth_entry(niche, arena):
    return {p["id"]: niche.truth_cell(p) for p in arena["patterns"]}


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arena = A.build_arena(NICHE, N.LIMITS["scale"], HOUSES)

    def test_points_are_margin_over_luck(self):
        self.assertEqual(A.points(8, 2, 10), 6.0)
        self.assertEqual(A.points(3, 5, 10), 0.0)
        self.assertEqual(A.points(0, 0, 10), 0.0)

    def test_true_tensors_beat_luck_and_random_tensors_do_not(self):
        referee = self.arena["referee"]
        good = A.score_row(NICHE, "Ember", 1, truth_entry(NICHE, self.arena), referee, "p")
        rng = np.random.RandomState(0)
        import random
        bad_entry = {p["id"]: A.rand_cell(NICHE.template, random.Random(int(rng.randint(1 << 30)))) for p in self.arena["patterns"]}
        bad = A.score_row(NICHE, "Tide", 1, bad_entry, referee, "p")
        self.assertGreater(good["points"], 0)
        self.assertEqual(good["solved"], good["total"])
        self.assertEqual(bad["points"], 0)
        self.assertEqual(good["charged"], bad["charged"], "random entries are matched in charged positions")

    def test_a_rigged_niche_pays_nothing(self):
        referee = self.arena["referee"]
        with mock.patch.object(A.S, "passes", return_value=True):
            row = A.score_row(NICHE, "Ember", 1, truth_entry(NICHE, self.arena), referee, "p")
        self.assertEqual(row["luck"], row["solved"])
        self.assertEqual(row["points"], 0.0)

    def test_houses_train_on_different_tasks_and_the_referee_is_hidden(self):
        a, b = (self.arena["train"][h] for h in HOUSES)
        differing = [pid for pid in a if a[pid]["train"] != b[pid]["train"]]
        self.assertGreaterEqual(len(differing), len(a) // 2, "houses must not be shown the same demonstrations")
        seen = [d["input"] for h in HOUSES for t in self.arena["train"][h].values() for d in t["train"] + t["test"]]
        for pid, task in self.arena["referee"].items():
            for d in task["train"] + task["test"]:
                self.assertNotIn(d["input"], seen, "referee inputs leaked into a training set")

    def test_seed_families_are_disjoint(self):
        referee = {A.referee_seed(i) for i in range(100)}
        train = {A.train_seed(h, i) for h in A.HOUSES for i in range(100)}
        base = {A.SEED * 1000 + i for i in range(100)} | {A.SEED * 1000 + 500 + i for i in range(100)}
        self.assertFalse(referee & train)
        self.assertFalse(referee & base)

    def test_ledger_recomputation_catches_tampering(self):
        referee = self.arena["referee"]
        entry = truth_entry(NICHE, self.arena)
        row = A.score_row(NICHE, "Ember", 1, entry, referee, "p")
        key = (NICHE.name, "Ember", 1)
        canon = A.canon_entry(entry)
        self.assertEqual(A.verify_ledger([row], {key: canon}), [])
        inflated = dict(row, points=row["points"] + 3, solved=row["solved"] + 1)
        self.assertTrue(A.verify_ledger([inflated], {key: canon}))
        cell = dict(canon[sorted(canon)[0]])
        edited = dict(canon)
        first = sorted(canon)[0]
        params = {k: np.asarray(v, float).copy() for k, v in cell["params"].items()}
        name = sorted(params)[0]
        params[name].flat[0] += 1.0
        edited[first] = {"template": cell["template"], "params": {k: v.tolist() for k, v in params.items()}}
        self.assertTrue(any("hash" in p for p in A.verify_ledger([row], {key: edited})))
        self.assertTrue(A.verify_ledger([row], {}))

    def test_luck_draw_is_shared_across_houses_from_season_two_and_season_one_still_verifies(self):
        referee = self.arena["referee"]
        entry = truth_entry(NICHE, self.arena)
        a, b = (A.score_row(NICHE, h, 1, entry, referee, "p", season=2) for h in HOUSES)
        self.assertEqual((a["luck"], a["points"]), (b["luck"], b["points"]), "identical entries must score identically")
        old = A.score_row(NICHE, "Tide", 1, entry, referee, "p", season=1)
        self.assertEqual(A.verify_ledger([old], {(NICHE.name, "Tide", 1): A.canon_entry(entry)}), [])
        self.assertEqual(A.verify_ledger([a], {(NICHE.name, "Ember", 1): A.canon_entry(entry)}), [])

    def test_metadata_never_enters_the_hash(self):
        entry = truth_entry(NICHE, self.arena)
        tagged = {k: dict(v, house="Ember", rationale="anything") for k, v in entry.items()}
        self.assertEqual(A.sha(A.canon_entry(entry)), A.sha(A.canon_entry(tagged)))


class StandingsTests(unittest.TestCase):
    def row(self, niche, house, gen, solved, luck, total, charged=10, declined=False):
        return {"kind": "match", "season": 1, "niche": niche, "house": house, "gen": gen, "solved": solved, "luck": luck, "total": total,
                "points": A.points(solved, luck, total), "charged": charged, "declined": declined}

    def test_winner_is_best_margin_then_fewest_positions_and_zero_pay_has_none(self):
        rows = [self.row("a", "Ember", 1, 8, 1, 10, charged=50), self.row("a", "Tide", 1, 8, 1, 10, charged=40), self.row("a", "Moss", 1, 3, 1, 10)]
        self.assertEqual(A.winners(rows), ["Tide"])
        self.assertEqual(A.winners([self.row("b", "Ember", 1, 2, 5, 10), self.row("b", "Tide", 1, 0, 5, 10)]), [])

    def test_best_generation_counts_once_and_titles_are_mechanical(self):
        rows = [self.row("a", "Ember", 1, 2, 0, 10), self.row("a", "Ember", 2, 9, 0, 10), self.row("a", "Tide", 1, 0, 0, 10),
                self.row("b", "Tide", 1, 0, 0, 10, declined=True)]
        table = {t["house"]: t for t in A.leaderboard(rows)}
        self.assertEqual(table["Ember"]["match_points"], 9.0)
        self.assertEqual(table["Ember"]["niches_won"], 1)
        self.assertIn("Sole Survivor: a", table["Ember"]["titles"])
        self.assertIn("First Blood: a", table["Ember"]["titles"])
        self.assertIn("Honest Null: b", table["Tide"]["titles"])
        self.assertEqual(table["Tide"]["points"], 0.0, "a decline is free, never negative")

    def test_same_generation_first_blood_goes_to_the_bigger_margin_and_inactive_houses_are_hidden(self):
        rows = [self.row("a", "Tide", 1, 5, 0, 10), self.row("a", "Ember", 1, 9, 0, 10)]
        table = {t["house"]: t for t in A.leaderboard(rows, houses=("Ember", "Tide"))}
        self.assertEqual(set(table), {"Ember", "Tide"})
        self.assertIn("First Blood: a", table["Ember"]["titles"])
        self.assertNotIn("First Blood: a", table["Tide"]["titles"])

    def test_exact_ties_share_first_blood_and_a_rigged_niche_draws_none(self):
        tied = [self.row("a", "Ember", 1, 9, 0, 10, charged=10), self.row("a", "Tide", 1, 9, 0, 10, charged=10)]
        table = {t["house"]: t for t in A.leaderboard(tied, houses=("Ember", "Tide"))}
        self.assertIn("First Blood: a", table["Ember"]["titles"])
        self.assertIn("First Blood: a", table["Tide"]["titles"])
        rigged = [self.row("r", "Ember", 1, 12, 9, 12)]
        self.assertGreater(rigged[0]["points"], 0)
        self.assertNotIn("First Blood: r", A.leaderboard(rigged, houses=("Ember",))[0]["titles"])

    def test_kill_points_come_from_recomputed_task_counts(self):
        table = {t["house"]: t for t in A.leaderboard([], {"Moss": 3})}
        self.assertEqual(table["Moss"]["kill_points"], 3 * A.KILL_POINTS)

    def test_decline_detection_needs_no_cells_and_a_stated_reason(self):
        self.assertTrue(A.is_decline([], ["No valid tensors express this"]))
        self.assertFalse(A.is_decline([object()], ["no valid"]))
        self.assertFalse(A.is_decline([], ["something else"]))

    def test_near_duplicate_cells(self):
        a = S.make_cell("scale_grid", {k: np.zeros(s).tolist() for k, s in S.TEMPLATES["scale_grid"]["shapes"].items()})
        b = S.make_cell("scale_grid", {k: (np.zeros(s) + 0.1).tolist() for k, s in S.TEMPLATES["scale_grid"]["shapes"].items()})
        c = S.make_cell("scale_grid", {k: (np.zeros(s) + 3).tolist() for k, s in S.TEMPLATES["scale_grid"]["shapes"].items()})
        self.assertTrue(A.near_duplicate(a, b))
        self.assertFalse(A.near_duplicate(a, c))


class BoardTests(unittest.TestCase):
    def grid(self, rows):
        return [[int(c) for c in r] for r in rows]

    def test_census_families(self):
        add = {"train": [{"input": self.grid(["000", "000"]), "output": self.grid(["000", "010"])}]}
        move = {"train": [{"input": self.grid(["100", "000"]), "output": self.grid(["000", "100"])}]}
        recolor = {"train": [{"input": self.grid(["100", "000"]), "output": self.grid(["200", "000"])}]}
        smaller = {"train": [{"input": self.grid(["100", "000"]), "output": self.grid(["1"])}]}
        self.assertEqual(A.census_category(add), "same shape, additive")
        self.assertEqual(A.census_category(move), "same shape, movement")
        self.assertEqual(A.census_category(recolor), "same shape, recolor only")
        self.assertEqual(A.census_category(smaller), "smaller output")

    def test_boss_header_states_the_strict_gain_rule_and_the_random_edit_control(self):
        text = AR.boss_header("Moss", 2, A.leaderboard([], {"Ember": 2}))
        self.assertIn("SAME items", text)
        self.assertIn("without losing any task already solved", text)
        self.assertIn("matched random edit", text)
        self.assertIn("Ember 10.0", text)
        self.assertNotIn("params", text)

    def test_boss_header_never_offers_a_free_way_out_of_the_attempt(self):
        """Season 2 round 1 and 2: 'return no entry' made all three houses skip all 30 items. A flag must come WITH an attempt."""
        text = AR.boss_header("Ember", 2, A.leaderboard([], {}))
        self.assertIn("ONE entry per item", text)
        self.assertIn("cannot_express", text)
        self.assertIn("never replaces the attempt", text)
        self.assertNotIn("return no entry", text)

    def test_prompt_header_shows_standings_but_no_tensors_or_referee_data(self):
        text = AR.header("Ember", 1, A.leaderboard([], {"Tide": 1}))
        self.assertIn("House Ember", text)
        self.assertIn("Tide 5.0", text)
        self.assertNotIn("params", text)
        self.assertIn("no valid", text)


if __name__ == "__main__":
    unittest.main()
