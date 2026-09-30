"""Relational cells, batch 3: move-toward-target, count-to-uniform-output, block/cell summary grids."""
import numpy as np
from scipy import ndimage
from cells import bg_color
from objects import S8, S4, _same


# ---------------------------------------------------------------- move toward a target object until contact
def _move_toward(g, c, t, bg):
    """Each object of color c slides along the axis toward the nearest aligned object of color t until touching it."""
    o = g.copy(); h, w = g.shape
    lab, n = ndimage.label(g == c, structure=S8)
    tm = g == t
    if not tm.any() or n == 0 or n > 30:
        return o
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        best = None
        for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            k = 0; hit = False
            while True:
                ny, nx = ys + d[0] * (k + 1), xs + d[1] * (k + 1)
                if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
                    break
                blocked = (g[ny, nx] != bg) & ~m[ny, nx]
                if blocked.any():
                    hit = bool(tm[ny, nx][blocked].all())
                    break
                k += 1
            if hit and (best is None or k < best[0]):
                best = (k, d)
        if best is not None and best[0] > 0:
            k, d = best
            o[m] = bg
            o[ys + d[0] * k, xs + d[1] * k] = c
    return o


def fit_move_toward(train, test_in):
    if not all(x.shape == y.shape for x, y in train) or all(_same(x, y) for x, y in train):
        return []
    cols = sorted(set(int(c) for x, _ in train for c in np.unique(x)))
    for c in cols:
        if not all((x == c).sum() == (y == c).sum() for x, y in train):
            continue
        for t in cols:
            if t == c:
                continue
            if all(_same(_move_toward(x, c, t, bg_color(x)), y) for x, y in train):
                if all((te == c).any() and (te == t).any() for te in test_in):
                    return [(f"moveto:{c}->{t}", [_move_toward(te, c, t, bg_color(te)) for te in test_in])]
    return []


# ---------------------------------------------------------------- count -> uniform output
def _count_feats(g):
    b = bg_color(g)
    f = {"nobj_o8": ndimage.label(g != b, structure=S8)[1], "nobj_o4": ndimage.label(g != b, structure=S4)[1],
         "ncolors": len(np.unique(g[g != b]))}
    for c in range(10):
        f[f"ncell_{c}"] = int((g == c).sum())
        f[f"nobj_c{c}"] = ndimage.label(g == c, structure=S8)[1]
    return f


def _count_color(g, mode, const):
    b = bg_color(g)
    v, c = np.unique(g[g != b], return_counts=True)
    if len(v) == 0:
        return None
    if mode == "const":
        return const
    return int(v[c.argmax()]) if mode == "most" else int(v[c.argmin()])


def _shp(lay, n):
    return (1, n) if lay == "row" else (n, 1) if lay == "col" else (n, n)


def fit_count_size(train, test_in):
    if not all(len(np.unique(y)) == 1 for _, y in train) or any(y.size >= x.size for x, y in train):
        return []
    cf = [_count_feats(x) for x, _ in train]
    tf = [_count_feats(t) for t in test_in]
    for lay in ("row", "col", "sq"):
        for fname in cf[0]:
            for mode, const in (("const", int(train[0][1][0, 0])), ("most", 0), ("least", 0)):
                ok = True
                for (x, y), f in zip(train, cf):
                    n = f[fname]
                    col = _count_color(x, mode, const)
                    if n <= 0 or n > 30 or col is None or y.shape != _shp(lay, n) or y[0, 0] != col:
                        ok = False; break
                if not ok:
                    continue
                preds = []
                for t, f in zip(test_in, tf):
                    n = f[fname]; col = _count_color(t, mode, const)
                    if n <= 0 or n > 30 or col is None:
                        preds = None; break
                    preds.append(np.full(_shp(lay, n), col))
                if preds:
                    return [(f"countsize:{lay}:{fname}:{mode}", preds)]
    return []


# ---------------------------------------------------------------- block / cell summary -> small grid
def _spans(idx, n):
    out = []; st = 0
    for i in list(idx) + [n]:
        if i > st:
            out.append((st, i))
        st = i + 1
    return out


def _blocks(g, oh, ow, how):
    h, w = g.shape
    if how == "even":
        if h % oh or w % ow:
            return None
        bh, bw = h // oh, w // ow
        return [[g[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] for j in range(ow)] for i in range(oh)]
    for col in np.unique(g):
        rows = [i for i in range(h) if (g[i] == col).all()]
        cols = [j for j in range(w) if (g[:, j] == col).all()]
        if not rows and not cols:
            continue
        rs, cs = _spans(rows, h), _spans(cols, w)
        if len(rs) == oh and len(cs) == ow:
            return [[g[a:b, c:d] for (c, d) in cs] for (a, b) in rs]
    return None


def _blk_feats(blk, sep_bg):
    v, c = np.unique(blk, return_counts=True)
    nb = blk[blk != sep_bg]
    if len(nb):
        v2, c2 = np.unique(nb, return_counts=True)
        dom = int(v2[c2.argmax()])
    else:
        dom = -1
    return {"dom": dom, "n": int((blk != sep_bg).sum()), "nc": len(np.unique(nb)) if len(nb) else 0,
            "any": int((blk != sep_bg).any()), "maj": int(v[c.argmax()])}


def _key(f, fname):
    return (f["dom"], f["n"]) if fname == "dom+n" else f[fname]


def fit_block_summary(train, test_in):
    if any(y.size >= x.size for x, y in train):
        return []
    shapes = {y.shape for _, y in train}
    if len(shapes) != 1:
        return []
    oh, ow = next(iter(shapes))
    for how in ("sep", "even"):
        for sep_mode in ("bg", "zero"):
            for fname in ("dom", "maj", "any", "n", "nc", "dom+n"):
                tab = {}; ok = True
                for x, y in train:
                    bl = _blocks(x, oh, ow, how)
                    if bl is None:
                        ok = False; break
                    sb = bg_color(x) if sep_mode == "bg" else 0
                    for i in range(oh):
                        for j in range(ow):
                            if tab.setdefault(_key(_blk_feats(bl[i][j], sb), fname), int(y[i, j])) != int(y[i, j]):
                                ok = False; break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok or len(tab) < 2:
                    continue
                preds = []
                for t in test_in:
                    bl = _blocks(t, oh, ow, how)
                    if bl is None:
                        preds = None; break
                    sb = bg_color(t) if sep_mode == "bg" else 0
                    o = np.zeros((oh, ow), int)
                    for i in range(oh):
                        for j in range(ow):
                            k = _key(_blk_feats(bl[i][j], sb), fname)
                            if k not in tab:
                                preds = None; break
                            o[i, j] = tab[k]
                        if preds is None:
                            break
                    if preds is None:
                        break
                    preds.append(o)
                if preds:
                    return [(f"blocks:{how}:{sep_mode}:{fname}", preds)]
    return []


FITTERS3 = [fit_move_toward, fit_count_size, fit_block_summary]
