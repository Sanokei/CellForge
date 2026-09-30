"""Synthetic unit tests for the object/relation cells. Run: python test_objects.py"""
import numpy as np
from objects import fit_regions, fit_move, fit_lines, fit_periodic, fit_count_out, slide_color
from objects2 import fit_stamp, fit_crop_obj, fit_rays


def g(rows):
    return np.array(rows)


def test_recolor_largest():
    def mk(a, b):
        x = np.zeros((8, 8), int); x[0:a, 0:1] = 1; x[5:5 + b, 4:5] = 1
        y = x.copy(); y[x == 1] = 1
        big = (0, a) if a > b else (5, 5 + b)
        if a > b: y[0:a, 0:1] = 2
        else: y[5:5 + b, 4:5] = 2
        return x, y
    train = [mk(3, 2), mk(2, 4), mk(4, 1)]
    te = [mk(3, 1)[0]]
    r = fit_regions(train, te)
    assert r and np.array_equal(r[0][1][0], mk(3, 1)[1]), r


def test_slide():
    x = np.zeros((5, 5), int); x[0, 1] = 3; x[4, 1] = 5
    o = slide_color(x, 3, "d", 0)
    assert o[3, 1] == 3 and o[0, 1] == 0


def test_move_fit():
    def mk(c):
        x = np.zeros((6, 6), int); x[0, c] = 4; y = np.zeros((6, 6), int); y[5, c] = 4
        return x, y
    r = fit_move([mk(1), mk(3)], [mk(2)[0]])
    assert r and np.array_equal(r[0][1][0], mk(2)[1])


def test_pairs():
    def mk(r, a, b):
        x = np.zeros((6, 8), int); x[r, a] = 2; x[r, b] = 2
        y = x.copy(); y[r, a + 1:b] = 2
        return x, y
    r = fit_lines([mk(1, 0, 5), mk(3, 2, 6)], [mk(4, 1, 7)[0]])
    assert r and np.array_equal(r[0][1][0], mk(4, 1, 7)[1])


def test_periodic():
    t = np.tile(np.array([[1, 2], [3, 4]]), (4, 4)); x = t.copy(); x[2:4, 3:5] = 0
    t2 = np.tile(np.array([[5, 6, 7]]), (6, 3)); x2 = t2.copy(); x2[1:3, 4:6] = 0
    t3 = np.tile(np.array([[1, 2], [3, 4]]), (4, 4)); x3 = t3.copy(); x3[0:2, 0:2] = 0
    r = fit_periodic([(x, t), (x2, t2)], [x3])
    assert r and np.array_equal(r[0][1][0], t3)


def test_count():
    def mk(n1, n2):
        x = np.zeros((7, 7), int)
        x[0, :n1] = 1; x[2, :n2] = 2
        return x, np.array([[1], [2]] if n1 > n2 else [[2], [1]])
    r = fit_count_out([mk(4, 2), mk(1, 3)], [mk(5, 2)[0]])
    assert r and np.array_equal(r[0][1][0], mk(5, 2)[1])


def test_stamp():
    def mk(y0, x0):
        x = np.zeros((7, 7), int); x[y0, x0] = 5
        y = x.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)): y[y0 + dy, x0 + dx] = 3
        return x, y
    r = fit_stamp([mk(1, 1), mk(3, 4), mk(5, 2)], [mk(2, 2)[0]])
    assert r and np.array_equal(r[0][1][0], mk(2, 2)[1])


def test_crop():
    def mk(sz_a, sz_b):
        x = np.zeros((10, 10), int); x[0:sz_a, 0:sz_a] = 1; x[6:6 + sz_b, 6:6 + sz_b] = 2
        big = x[0:sz_a, 0:sz_a] if sz_a > sz_b else x[6:6 + sz_b, 6:6 + sz_b]
        return x, big.copy()
    r = fit_crop_obj([mk(3, 2), mk(2, 4), mk(4, 3)], [mk(3, 2)[0]])
    assert r and np.array_equal(r[0][1][0], mk(3, 2)[1])


def test_rays():
    def mk(y0, x0):
        x = np.zeros((6, 6), int); x[y0, x0] = 4
        y = x.copy(); y[:y0, x0] = 4
        return x, y
    r = fit_rays([mk(3, 1), mk(4, 4), mk(2, 0)], [mk(5, 3)[0]])
    assert r and np.array_equal(r[0][1][0], mk(5, 3)[1])


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
