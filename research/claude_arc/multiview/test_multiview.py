"""Synthetic tests. Run: python test_multiview.py"""
import numpy as np
from views import all_views, find_period, object_graph, FAM
from cells_mv import fit_pixels, fit_block, fit_projection_out, fit_multiview


def test_views_shapes():
    g = np.zeros((6, 7), int); g[1:3, 1:3] = 2; g[4, 5] = 3
    fn, F, sn, S = all_views(g)
    assert F.shape == (6, 7, len(fn)) and S.shape == (6, 7, len(sn)) and sn[0] == "self"
    assert {"pix", "obj", "proj", "hist", "sym"} <= set(FAM.values())
    assert (S[..., 0] == g).all()


def test_object_graph_relations():
    g = np.zeros((9, 9), int); g[1:8, 1:8] = 1; g[2:7, 2:7] = 0; g[4, 4] = 5
    bg, objs, lab = object_graph(g)
    ring = [o for o in objs if o["col"] == 1][0]; dot = [o for o in objs if o["col"] == 5][0]
    assert ring["holes"] == 1 and ring["contains"] == 1 and dot["encl"] >= 0


def test_period():
    t = np.array([[1, 2], [3, 4]]); g = np.tile(t, (4, 4)); g[2, 2] = 0
    py, px, m, full = find_period(g)
    assert (py, px) == (2, 2) and m == 0 and np.array_equal(full, np.tile(t, (4, 4)))


def test_cross_view_fill_enclosed():
    # rule: bg pixels inside a box take the box colour (obj view selector 'inbox', source o_owner); colours vary per demo
    def mk(c, pos):
        x = np.zeros((9, 9), int); x[pos[0]:pos[0] + 4, pos[1]:pos[1] + 4] = c; x[pos[0] + 1:pos[0] + 3, pos[1] + 1:pos[1] + 3] = 0
        y = x.copy(); y[pos[0] + 1:pos[0] + 3, pos[1] + 1:pos[1] + 3] = c
        return x, y
    tr = [mk(1, (0, 0)), mk(2, (3, 4)), mk(3, (1, 5))]
    x, y = mk(4, (2, 2))
    r = fit_pixels(tr, [x])
    assert any(np.array_equal(c["preds"][0], y) for c in r), [c["kind"] for c in r][:5]


def test_symmetry_completion():
    def mk(seed):
        rng = np.random.RandomState(seed)
        half = rng.randint(1, 5, (6, 4)); full = np.concatenate([half, half[:, ::-1]], 1)
        x = full.copy(); x[rng.randint(0, 6), rng.randint(0, 8)] = 0
        return x, full
    tr = [mk(1), mk(2), mk(3), mk(4)]
    x, y = mk(9)
    r = fit_pixels(tr, [x])
    assert any(np.array_equal(c["preds"][0], y) for c in r)


def test_block_view():
    def mk(seed):
        rng = np.random.RandomState(seed)
        o = (rng.rand(3, 3) > 0.68).astype(int) * 4
        x = np.kron(o, np.ones((2, 2), int)); x[0, 0] = 0
        return x, o
    tr = [mk(i) for i in range(4)]
    x, y = mk(11)
    r = fit_block(tr, [x])
    assert any(np.array_equal(c["preds"][0], y) for c in r), r


def test_projection_out():
    def mk(seed):
        rng = np.random.RandomState(seed)
        x = np.zeros((5, 6), int)
        for i in range(5):
            if rng.rand() > .7: x[i, :] = 3
        return x, (x.max(1, keepdims=True))
    tr = [mk(i) for i in range(4)]
    x, y = mk(12)
    r = fit_projection_out(tr, [x])
    assert any(np.array_equal(c["preds"][0], y) for c in r)


def test_ambiguity_rejected_and_identity():
    x = np.zeros((4, 4), int); x[1, 1] = 2
    assert fit_pixels([(x, x), (x, x)], [x]) == []


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
