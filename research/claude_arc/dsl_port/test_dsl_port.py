"""Unit tests on synthetic grids. Run: python -m unittest test_dsl_port -v   (from research/claude_arc/dsl_port)"""
import unittest
import numpy as np
import adapter
import objfam as O


def blank(h=9, w=9):
    return np.zeros((h, w), int)


def descs(cands, fam=None):
    return [c.desc for c in cands if fam is None or c.family == fam]


class Adapter(unittest.TestCase):
    def test_roundtrip(self):
        g = np.arange(12).reshape(3, 4) % 10
        self.assertTrue(np.array_equal(adapter.from_h(adapter.to_h(g)), g))

    def test_safe_limits(self):
        up = adapter.wrap_cell(lambda h: adapter.D.upscale(h, 4))
        self.assertIsNone(up(np.ones((10, 10), int)))        # 40x40 > cap
        self.assertIsNotNone(up(np.ones((5, 5), int)))
        bad = adapter.wrap_cell(lambda h: 1 / 0)
        self.assertIsNone(bad(np.ones((3, 3), int)))


class Families(unittest.TestCase):
    def pair_recolor_largest(self):
        def mk(big, small):
            g = blank(); g[big] = 1; g[small] = 1
            y = g.copy(); y[big] = 5
            return g, y
        return [mk((slice(0, 4), slice(0, 4)), (slice(6, 8), slice(6, 8))),
                mk((slice(4, 8), slice(3, 7)), (slice(0, 1), slice(0, 2))),
                mk((slice(1, 4), slice(5, 9)), (slice(7, 9), slice(0, 2)))]

    def test_select_recolor(self):
        tr = self.pair_recolor_largest()
        t = blank(); t[0:5, 0:5] = 3; t[7:9, 7:9] = 3
        c = O.run_all(tr, [t], 10, ["SEL_CROP"])
        self.assertTrue(c)
        want = t.copy(); want[0:5, 0:5] = 5
        self.assertTrue(any(np.array_equal(x.preds[0], want) for x in c))

    def test_crop_unique_colour(self):
        def mk(seed):
            r = np.random.RandomState(seed); g = blank(10, 10)
            g[1:3, 1:3] = 2; g[5:7, 5:7] = 2; g[6:9, 1:4] = 4
            g[r.randint(0, 2):r.randint(0, 2) + 1, 8] = 0
            return g, g[6:9, 1:4].copy()
        tr = [mk(0), mk(1)]
        g = blank(10, 10); g[0:2, 0:2] = 7; g[6:8, 6:8] = 7; g[3:6, 2:6] = 9
        c = O.run_all(tr, [g], 10, ["SEL_CROP"])
        self.assertTrue(any(np.array_equal(x.preds[0], g[3:6, 2:6]) for x in c))

    def test_rec_size_table(self):
        def mk(sizes):
            g = blank(10, 10); y = g.copy(); r = 0
            for s in sizes:
                g[r, 0:s] = 1; y[r, 0:s] = {1: 2, 2: 3, 3: 4}[s]; r += 2
            return g, y
        tr = [mk([1, 2, 3]), mk([3, 1, 2, 1])]
        t, want = mk([2, 2, 3])
        c = O.run_all(tr, [t], 10, ["REC"])
        self.assertTrue(any(np.array_equal(x.preds[0], want) for x in c))

    def test_rec_rejects_unseen_key(self):
        tr = [(np.array([[1, 0, 1, 1]]), np.array([[2, 0, 3, 3]])), (np.array([[1, 1, 0, 1]]), np.array([[3, 3, 0, 2]]))]
        t = np.array([[1, 1, 1, 0, 1]])                      # size-3 object never seen in demos
        c = O.run_all(tr, [t], 5, ["REC"])
        self.assertFalse([x for x in c if x.desc.split("|")[1] == "size" and x.desc.endswith("keep")])

    def test_line_connect(self):
        def mk(seed):
            r = np.random.RandomState(seed); g = blank(10, 10); row = r.randint(0, 10)
            a, b = sorted(r.choice(10, 2, replace=False)); g[row, a] = 3; g[row, b] = 3
            y = g.copy(); y[row, a:b + 1] = 8; y[row, a] = 3; y[row, b] = 3
            return g, y
        tr = [mk(1), mk(2), mk(3)]
        t, want = mk(4)
        c = O.run_all(tr, [t], 10, ["LINE"])
        self.assertTrue(any(np.array_equal(x.preds[0], want) for x in c))

    def test_macro_mirror_concat(self):
        g1 = np.array([[1, 2], [3, 0]]); g2 = np.array([[0, 5, 1], [4, 4, 2]])
        f = lambda g: np.hstack([g, g[:, ::-1]])
        c = O.run_all([(g1, f(g1)), (g2, f(g2))], [np.array([[7, 0], [0, 6]])], 10, ["MACRO"])
        self.assertTrue(any(np.array_equal(x.preds[0], f(np.array([[7, 0], [0, 6]]))) for x in c))

    def test_no_fit_no_candidates(self):
        r = np.random.RandomState(0)
        tr = [(r.randint(0, 4, (6, 6)), r.randint(0, 4, (6, 6))) for _ in range(3)]
        self.assertEqual(O.run_all(tr, [r.randint(0, 4, (6, 6))], 5), [])


if __name__ == "__main__":
    unittest.main()
