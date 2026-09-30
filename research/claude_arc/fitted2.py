"""Task-fitted cells for tasks whose output shape differs from the input.

All choices (block orientations, crop criterion, fractal variant) are picked by exact consistency on the demo pairs;
nothing is learned by gradients. A cell is accepted only if it reproduces every demo output exactly.
"""
import numpy as np
from scipy import ndimage
from cells import bg_color

T8 = [lambda g: g, lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 3),
      lambda g: g[::-1], lambda g: g[:, ::-1], lambda g: g.T, lambda g: np.rot90(g, 2).T]


def _eq(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def fractal_cells(train, test_in):
    out = []
    variants = {
        "fractal_nz": lambda g: np.kron((g != bg_color(g)).astype(int), g) + np.kron((g == bg_color(g)).astype(int), np.full_like(g, bg_color(g))),
        "fractal_z": lambda g: np.kron((g == bg_color(g)).astype(int), g) + np.kron((g != bg_color(g)).astype(int), np.full_like(g, bg_color(g))),
    }
    for name, fn in variants.items():
        try:
            if all(_eq(fn(x), y) for x, y in train):
                out.append((name, [fn(t) for t in test_in]))
        except Exception:
            pass
    return out


def tile_orient(train, test_in):
    """Output = a x b grid of blocks, each block one of 8 orientations of the input (fitted per block position)."""
    x0, y0 = train[0]
    h, w = x0.shape
    if y0.shape[0] % h or y0.shape[1] % w:
        return []
    a, b = y0.shape[0] // h, y0.shape[1] // w
    if (a, b) == (1, 1) or a > 4 or b > 4:
        return []
    for x, y in train:
        if y.shape != (x.shape[0] * a, x.shape[1] * b):
            return []
    choice = {}
    for i in range(a):
        for j in range(b):
            found = None
            for ti, t in enumerate(T8):
                ok = True
                for x, y in train:
                    hh, ww = x.shape
                    try:
                        blk = t(x)
                    except Exception:
                        ok = False; break
                    if not _eq(blk, y[i * hh:(i + 1) * hh, j * ww:(j + 1) * ww]):
                        ok = False; break
                if ok:
                    found = ti; break
            if found is None:
                return []
            choice[(i, j)] = found
    preds = []
    for tg in test_in:
        hh, ww = tg.shape
        o = np.zeros((hh * a, ww * b), dtype=int)
        for (i, j), ti in choice.items():
            blk = T8[ti](tg)
            if blk.shape != (hh, ww):
                return []
            o[i * hh:(i + 1) * hh, j * ww:(j + 1) * ww] = blk
        preds.append(o)
    return [("tile_orient", preds)]


def _candidates(g):
    """Subgrid candidates: bbox of each color, bbox of each 8-connected multicolor object."""
    bg = bg_color(g)
    cands = []
    for c in np.unique(g):
        if c == bg:
            continue
        m = g == c
        r = np.flatnonzero(m.any(1)); cc = np.flatnonzero(m.any(0))
        cands.append(("color", g[r[0]:r[-1] + 1, cc[0]:cc[-1] + 1], (r[0], cc[0])))
    lab, n = ndimage.label(g != bg, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        m = lab == i
        r = np.flatnonzero(m.any(1)); cc = np.flatnonzero(m.any(0))
        cands.append(("obj", g[r[0]:r[-1] + 1, cc[0]:cc[-1] + 1], (r[0], cc[0])))
    return cands


def _keys(g):
    bg = bg_color(g)
    def nz(s): return int((s != bg).sum())
    return {
        "area_max": lambda c: c[1].size, "area_min": lambda c: -c[1].size,
        "nz_max": lambda c: nz(c[1]), "nz_min": lambda c: -nz(c[1]),
        "colors_max": lambda c: len(np.unique(c[1])), "colors_min": lambda c: -len(np.unique(c[1])),
        "top": lambda c: -c[2][0], "bottom": lambda c: c[2][0], "left": lambda c: -c[2][1], "right": lambda c: c[2][1],
        "dense_max": lambda c: nz(c[1]) / c[1].size, "dense_min": lambda c: -nz(c[1]) / c[1].size,
        "sym_h": lambda c: int(np.array_equal(c[1], c[1][:, ::-1])), "sym_v": lambda c: int(np.array_equal(c[1], c[1][::-1])),
    }


def crop_select(train, test_in):
    out = []
    for kind in ("color", "obj"):
        for kname in _keys(train[0][0]):
            ok = True
            for x, y in train:
                cs = [c for c in _candidates(x) if c[0] == kind]
                if not cs:
                    ok = False; break
                key = _keys(x)[kname]
                scores = [key(c) for c in cs]
                best = max(scores)
                if scores.count(best) != 1:
                    ok = False; break  # ambiguous selection is rejected
                if not _eq(cs[scores.index(best)][1], y):
                    ok = False; break
            if not ok:
                continue
            preds = []
            for tg in test_in:
                cs = [c for c in _candidates(tg) if c[0] == kind]
                if not cs:
                    preds = None; break
                key = _keys(tg)[kname]
                scores = [key(c) for c in cs]
                if scores.count(max(scores)) != 1:
                    preds = None; break
                preds.append(cs[scores.index(max(scores))][1])
            if preds:
                out.append((f"crop_{kind}_{kname}", preds))
    return out


def fit_shape_changing(train, test_in):
    res = []
    for fn in (fractal_cells, tile_orient, crop_select):
        try:
            res += fn(train, test_in)
        except Exception:
            pass
    return res
