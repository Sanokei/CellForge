"""Alternative representations (views) of one grid. Every view returns per-pixel integer FEATURES and per-pixel colour SOURCES
(colour-valued arrays, -1 = undefined) that a rule may copy. No learning; all deterministic functions of the grid.

Families: pix (pixel grid), obj (object graph), proj (row/column projections), hist (colour histogram / colour-role view),
sym (symmetry and periodicity spectra). The block (downsampled) view lives in cells_mv.py because it changes the output lattice.
"""
import sys, os
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cells import bg_color  # noqa: E402

S8 = np.ones((3, 3), int)
FAM = {}          # feature name -> family
SRC_FAM = {}      # source name -> family


def _reg(store, fam, d, name, arr, lo=0, hi=15):
    store[name] = np.clip(arr, lo, hi).astype(int)
    d[name] = fam


def _shift(g, dy, dx, fill=-1):
    h, w = g.shape
    o = np.full((h, w), fill, dtype=int)
    ys = slice(max(0, -dy), min(h, h - dy)); xs = slice(max(0, -dx), min(w, w - dx))
    yd = slice(max(0, dy), min(h, h + dy)); xd = slice(max(0, dx), min(w, w + dx))
    o[ys, xs] = g[yd, xd]
    return o


def freq_order(g):
    bg = bg_color(g)
    vals, cnt = np.unique(g, return_counts=True)
    order = [int(v) for v, c in sorted(zip(vals, cnt), key=lambda t: (-t[1], t[0])) if v != bg]
    return bg, order, {int(v): int(c) for v, c in zip(vals, cnt)}


# ---------------------------------------------------------------- pixel view
def view_pix(g, F, S, fam):
    h, w = g.shape
    bg = bg_color(g)
    _reg(F, "pix", fam, "colour", g)
    _reg(F, "pix", fam, "is_bg", (g == bg))
    nb4 = [_shift(g, dy, dx) for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
    nb8 = nb4 + [_shift(g, dy, dx) for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]]
    _reg(F, "pix", fam, "n4nz", sum(((n != bg) & (n != -1)).astype(int) for n in nb4))
    _reg(F, "pix", fam, "n8same", sum((n == g).astype(int) for n in nb8))
    ii, jj = np.indices(g.shape)
    _reg(F, "pix", fam, "par_i", ii % 2)
    _reg(F, "pix", fam, "par_j", jj % 2)
    _reg(F, "pix", fam, "edge", (ii == 0) | (jj == 0) | (ii == h - 1) | (jj == w - 1))
    S["self"] = g.astype(int)
    SRC_FAM["self"] = "pix"
    # dominant colour among 8 neighbours (excluding bg and self colour when possible)
    best = np.full(g.shape, -1, int); bc = np.zeros(g.shape, int)
    for c in range(10):
        cnt = sum((n == c).astype(int) for n in nb8)
        upd = (cnt > bc) & (c != bg)
        best[upd] = c; bc[upd] = cnt[upd]
    S["nbdom"] = best
    SRC_FAM["nbdom"] = "pix"


# ---------------------------------------------------------------- object graph view
def objects_of(g):
    bg = bg_color(g)
    objs = []
    for col in [int(c) for c in np.unique(g) if c != bg]:
        lab, n = ndimage.label(g == col, structure=S8)
        for i in range(1, n + 1):
            m = lab == i
            r = np.flatnonzero(m.any(1)); c = np.flatnonzero(m.any(0))
            y0, y1, x0, x1 = r[0], r[-1] + 1, c[0], c[-1] + 1
            sub = m[y0:y1, x0:x1]
            pad = np.pad(~sub, 1, constant_values=True)
            hl, hn = ndimage.label(pad)
            holes = len(set(range(1, hn + 1)) - {hl[0, 0]})
            objs.append(dict(mask=m, bb=(y0, y1, x0, x1), col=col, size=int(m.sum()),
                             shape=(sub.shape, sub.tobytes()), holes=holes,
                             border=int(y0 == 0 or x0 == 0 or y1 == g.shape[0] or x1 == g.shape[1])))
    return bg, objs


def object_graph(g):
    """Objects with relations: contains / encl (enclosing object), adjacent, aligned, same-shape, same-colour."""
    bg, objs = objects_of(g)
    n = len(objs)
    for i, o in enumerate(objs):
        y0, y1, x0, x1 = o["bb"]
        o["contains"] = 0; o["encl"] = -1; o["aligned_r"] = 0; o["aligned_c"] = 0
        best = None
        for j, p in enumerate(objs):
            if i == j: continue
            a0, a1, b0, b1 = p["bb"]
            if a0 <= y0 and a1 >= y1 and b0 <= x0 and b1 >= x1 and (a1 - a0) * (b1 - b0) > (y1 - y0) * (x1 - x0):
                area = (a1 - a0) * (b1 - b0)
                if best is None or area < best[0]:
                    best = (area, j)
            if y0 <= a0 and y1 >= a1 and x0 <= b0 and x1 >= b1 and (a1 - a0) * (b1 - b0) < (y1 - y0) * (x1 - x0):
                o["contains"] += 1
            if not (a1 <= y0 or a0 >= y1): o["aligned_r"] = 1
            if not (b1 <= x0 or b0 >= x1): o["aligned_c"] = 1
        o["encl"] = -1 if best is None else best[1]
    lab = np.full(g.shape, -1, int)
    for i, o in enumerate(objs):
        lab[o["mask"]] = i
    for i, o in enumerate(objs):
        d = ndimage.binary_dilation(o["mask"], structure=S8)
        nbl = set(int(v) for v in lab[d & ~o["mask"]] if v >= 0)
        o["adj"] = nbl
    shapes = [o["shape"] for o in objs]
    for o in objs:
        o["n_same_shape"] = shapes.count(o["shape"])
        o["n_same_col"] = sum(1 for p in objs if p["col"] == o["col"])
    if objs:
        mx = max(o["size"] for o in objs); mn = min(o["size"] for o in objs)
        freq = max(shapes.count(s) for s in shapes)
        for o in objs:
            o["largest"] = int(o["size"] == mx); o["smallest"] = int(o["size"] == mn)
            o["common_shape"] = int(o["n_same_shape"] == freq and freq > 1)
    return bg, objs, lab


def view_obj(g, F, S, fam):
    bg, objs, lab = object_graph(g)
    h, w = g.shape
    keys = ["size", "n_same_shape", "n_same_col", "holes", "border", "largest", "smallest", "contains", "has_encl", "n_adj",
            "aligned_r", "aligned_c", "common_shape", "inbox"]
    arr = {k: np.zeros(g.shape, int) for k in keys}
    src_encl = np.full(g.shape, -1, int); src_adjc = np.full(g.shape, -1, int); src_ocol = np.full(g.shape, -1, int)
    owner = np.full(g.shape, -1, int)
    order = sorted(range(len(objs)), key=lambda i: -(objs[i]["bb"][1] - objs[i]["bb"][0]) * (objs[i]["bb"][3] - objs[i]["bb"][2]))
    for i in order:
        y0, y1, x0, x1 = objs[i]["bb"]
        reg = owner[y0:y1, x0:x1]
        reg[g[y0:y1, x0:x1] == bg] = i
    for i, o in enumerate(objs):
        owner[o["mask"]] = i
    for i, o in enumerate(objs):
        m = owner == i
        arr["size"][m] = o["size"]; arr["n_same_shape"][m] = o["n_same_shape"]; arr["n_same_col"][m] = o["n_same_col"]
        arr["holes"][m] = o["holes"]; arr["border"][m] = o["border"]; arr["largest"][m] = o["largest"]
        arr["smallest"][m] = o["smallest"]; arr["contains"][m] = o["contains"]
        arr["has_encl"][m] = int(o["encl"] >= 0); arr["n_adj"][m] = len(o["adj"])
        arr["aligned_r"][m] = o["aligned_r"]; arr["aligned_c"][m] = o["aligned_c"]; arr["common_shape"][m] = o["common_shape"]
        arr["inbox"][m & (g == bg)] = 1
        src_ocol[m] = o["col"]
        src_encl[m] = objs[o["encl"]]["col"] if o["encl"] >= 0 else -1
        if o["adj"]:
            cols = [objs[j]["col"] for j in o["adj"]]
            src_adjc[m] = max(set(cols), key=lambda c: (cols.count(c), -c))
    for k in keys:
        _reg(F, "obj", fam, "o_" + k, arr[k])
    # for bg pixels inside a bbox, "encl" source is the owner's colour
    bgm = (g == bg) & (owner >= 0)
    src_encl[bgm] = src_ocol[bgm]
    S["o_encl"] = src_encl; S["o_adjcol"] = src_adjc; S["o_owner"] = src_ocol
    for k in ("o_encl", "o_adjcol", "o_owner"):
        SRC_FAM[k] = "obj"


# ---------------------------------------------------------------- row/column projections
def view_proj(g, F, S, fam):
    h, w = g.shape
    bg = bg_color(g)
    nz = g != bg
    rnz = nz.sum(1); cnz = nz.sum(0)
    _reg(F, "proj", fam, "row_nz", np.repeat(rnz[:, None], w, 1), 0, 10)
    _reg(F, "proj", fam, "col_nz", np.repeat(cnz[None, :], h, 0), 0, 10)
    rnc = np.array([len(set(g[i][nz[i]].tolist())) for i in range(h)])
    cnc = np.array([len(set(g[:, j][nz[:, j]].tolist())) for j in range(w)])
    _reg(F, "proj", fam, "row_nc", np.repeat(rnc[:, None], w, 1), 0, 3)
    _reg(F, "proj", fam, "col_nc", np.repeat(cnc[None, :], h, 0), 0, 3)
    _reg(F, "proj", fam, "row_full", np.repeat((rnz == w)[:, None], w, 1))
    _reg(F, "proj", fam, "col_full", np.repeat((cnz == h)[None, :], h, 0))
    _reg(F, "proj", fam, "row_max", np.repeat(((rnz == rnz.max()) & (rnz > 0))[:, None], w, 1))
    _reg(F, "proj", fam, "col_max", np.repeat(((cnz == cnz.max()) & (cnz > 0))[None, :], h, 0))
    rs = np.zeros(g.shape, int); cs = np.zeros(g.shape, int)
    for c in range(10):
        m = g == c
        rs += m * m.sum(1, keepdims=True); cs += m * m.sum(0, keepdims=True)
    _reg(F, "proj", fam, "row_same", rs, 0, 6)
    _reg(F, "proj", fam, "col_same", cs, 0, 6)

    def dom(v):
        v = [x for x in v if x != bg]
        return max(set(v), key=lambda c: (v.count(c), -c)) if v else -1
    rd = np.array([dom(g[i].tolist()) for i in range(h)]); cd = np.array([dom(g[:, j].tolist()) for j in range(w)])
    S["row_dom"] = np.repeat(rd[:, None], w, 1); S["col_dom"] = np.repeat(cd[None, :], h, 0)
    SRC_FAM["row_dom"] = SRC_FAM["col_dom"] = "proj"


# ---------------------------------------------------------------- histogram / colour-role view
def view_hist(g, F, S, fam):
    bg, order, cnt = freq_order(g)
    rank = np.zeros(g.shape, int)
    for k, c in enumerate(order):
        rank[g == c] = k + 1
    _reg(F, "hist", fam, "crank", rank, 0, 5)
    least = order[-1] if len(order) > 1 else -1
    _reg(F, "hist", fam, "is_least", (g == least))
    _reg(F, "hist", fam, "ncolors", np.full(g.shape, len(order)), 0, 6)
    _reg(F, "hist", fam, "cnt_rank_top", (g == (order[0] if order else -1)))
    S["rank1"] = np.full(g.shape, order[0] if order else -1)
    S["rank2"] = np.full(g.shape, order[1] if len(order) > 1 else -1)
    S["least"] = np.full(g.shape, least)
    for k in ("rank1", "rank2", "least"):
        SRC_FAM[k] = "hist"


# ---------------------------------------------------------------- symmetry / periodicity spectra
def find_period(g, maxp=8):
    """Smallest (py,px) such that every residue class is constant after ignoring a mask colour m (or none). Returns
    (py, px, mask, classgrid) or None. Spectrum = the set of (py,px,m) that fit; we return the cheapest."""
    h, w = g.shape
    vals, cnt = np.unique(g, return_counts=True)
    masks = [-1] + [int(v) for v, c in sorted(zip(vals, cnt), key=lambda t: t[1])][:3]
    combos = sorted(((py, px) for py in range(1, min(h // 2, maxp) + 1) for px in range(1, min(w // 2, maxp) + 1)
                     if not (py == 1 and px == 1)), key=lambda t: (t[0] * t[1], t))
    for m in masks:
        for py, px in combos:
            cls = np.full((py, px), -1, int); ok = True
            for a in range(py):
                for b in range(px):
                    v = g[a::py, b::px].ravel()
                    if m >= 0: v = v[v != m]
                    u = np.unique(v)
                    if len(u) > 1: ok = False; break
                    if len(u) == 1: cls[a, b] = u[0]
                if not ok: break
            if ok and (cls >= 0).sum() > 0:
                if m >= 0 and (g == m).mean() > 0.5: continue
                full = np.tile(cls, (h // py + 1, w // px + 1))[:h, :w]
                return py, px, m, full
    return None


def view_sym(g, F, S, fam):
    h, w = g.shape
    bg = bg_color(g)
    parts = {"H": g[:, ::-1], "V": g[::-1], "R": g[::-1, ::-1]}
    if h == w:
        parts["T"] = g.T
    spec = 0
    for k, (nm, p) in enumerate(parts.items()):
        both = (g != bg) & (p != bg)
        agree = (g == p)
        v = np.where(p == g, 1, np.where(p == bg, 2, 3))
        _reg(F, "sym", fam, "sy_" + nm, v, 0, 3)
        S["sy_" + nm] = p.astype(int); SRC_FAM["sy_" + nm] = "sym"
        if both.any() and agree[both].all() and (g != p).sum() < 0.25 * g.size:
            spec |= 1 << k
    for nm in ("H", "V", "R", "T"):
        if nm not in parts:
            _reg(F, "sym", fam, "sy_" + nm, np.zeros(g.shape, int)); S["sy_" + nm] = np.full(g.shape, -1); SRC_FAM["sy_" + nm] = "sym"
    _reg(F, "sym", fam, "spec", np.full(g.shape, spec), 0, 15)
    per = find_period(g)
    if per is None:
        _reg(F, "sym", fam, "per_role", np.zeros(g.shape, int)); S["period"] = np.full(g.shape, -1)
    else:
        py, px, m, full = per
        role = np.where(g == full, 1, np.where((g == m) & (m >= 0), 2, 3))
        _reg(F, "sym", fam, "per_role", role, 0, 3); S["period"] = full
    SRC_FAM["period"] = "sym"


VIEWS = [view_pix, view_obj, view_proj, view_hist, view_sym]


def all_views(g, views=None):
    """Return (feature names, F[h,w,nF], source names, S[h,w,nS]); the 'self' source is always first."""
    F, S, fam = {}, {}, {}
    for v in (views or VIEWS):
        v(g, F, S, fam)
    FAM.update(fam)
    fn = list(F); sn = ["self"] + [k for k in S if k != "self"]
    return fn, np.stack([F[k] for k in fn], -1), sn, np.stack([S[k] for k in sn], -1)
