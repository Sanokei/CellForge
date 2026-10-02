"""Accepted structures become templates. Their tensors are renamed per structure so unpacking weights can never confuse two structures."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import skillcore as S  # noqa: E402
import structcode as X  # noqa: E402
import structlib as L  # noqa: E402

LUT = "def run(params, grid, demos):\n    lut = np.round(params['lut']).astype(int)\n    return lut[grid]\n"
SHIFT = "def run(params, grid, demos):\n    k = int(round(params['lut'][0]))\n    return (grid + k) % 10\n"
SHAPES = {"lut": [10]}
IDENTITY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TWO_TO_FIVE = [0, 1, 5, 3, 4, 5, 6, 7, 8, 9]


class Library(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lut = L.make_entry(LUT, SHAPES, house="Ember", task="t1")
        cls.shift = L.make_entry(SHIFT, {"lut": [1]}, house="Tide", task="t2")
        cls.ns = type("NS", (), {"TEMPLATES": S.TEMPLATES})
        L.register(cls.ns, [cls.lut, cls.shift])

    def test_a_registered_structure_runs_as_a_cell_and_reads_its_renamed_tensors(self):
        cell = L.cell_for(S, self.lut, {"lut": TWO_TO_FIVE})
        self.assertEqual(S.run(cell, [[2, 3]], []), [[5, 3]])
        self.assertTrue(all(k.startswith(L.prefix_of(self.lut["sha"])) for k in cell["params"]))
        self.assertEqual(S.positions(cell), 10)

    def test_weights_matter_zeroed_tensors_do_not_solve_the_task(self):
        task = {"train": [{"input": [[2, 3]], "output": [[5, 3]]}], "test": [{"input": [[3, 2]], "output": [[3, 5]]}]}
        good = L.cell_for(S, self.lut, {"lut": TWO_TO_FIVE})
        zero = L.cell_for(S, self.lut, {"lut": [0] * 10})
        self.assertTrue(S.passes(good, task))
        self.assertFalse(S.passes(zero, task))

    def test_two_structures_with_the_same_tensor_name_never_collide_when_weights_are_unpacked(self):
        a = L.cell_for(S, self.lut, {"lut": TWO_TO_FIVE})
        b = L.cell_for(S, self.shift, {"lut": [1]})
        tensors = S.to_tensors([a, b])
        cells, _ = S.from_tensors(tensors)
        self.assertEqual([c["template"] for c in cells], [self.lut["name"], self.shift["name"]])
        self.assertEqual(S.run(cells[0], [[2]], []), [[5]])
        self.assertEqual(S.run(cells[1], [[2]], []), [[3]])

    def test_structures_that_fail_the_whitelist_or_whose_hash_was_edited_are_never_registered(self):
        evil = L.make_entry("import os\n\ndef run(params, grid, demos):\n    return grid\n", {"a": [1]})
        edited = dict(self.lut, source=LUT + "# edited after hashing\n")
        ns = type("NS", (), {"TEMPLATES": {}})
        self.assertEqual(L.register(ns, [evil, edited]), 0)
        self.assertEqual(ns.TEMPLATES, {})

    def test_registration_is_idempotent_and_the_library_round_trips_through_its_file(self):
        self.assertEqual(L.register(self.ns, [self.lut]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(L.save(tmp, [self.lut, self.lut, self.shift]), 2)
            ns = type("NS", (), {"TEMPLATES": {}})
            self.assertEqual(L.load(ns, tmp), 2)
            self.assertEqual(sorted(ns.TEMPLATES), sorted([self.lut["name"], self.shift["name"]]))

    def test_a_structure_that_errors_on_a_grid_raises_so_the_scorer_counts_it_as_wrong(self):
        broken = L.make_entry("def run(params, grid, demos):\n    return grid[99]\n", {"a": [1]})
        L.register(self.ns, [broken])
        cell = L.cell_for(S, broken, {"a": [0]})
        with self.assertRaises(ValueError):
            S.run(cell, [[1]], [])


if __name__ == "__main__":
    unittest.main()
