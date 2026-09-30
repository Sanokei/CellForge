"""More object/relation cells: marker stamping (offset rules learned by counting) and object crop by selector."""
import numpy as np
from cells import bg_color
from scipy import ndimage
from objects import SEGS, _bbox, _same, region_table, D8L, S8


# ---------------------------------------------------------------- stamp at markers
def _stamp_learn(train, R, per_color):
    """Learn rules (src_key, dy, dx) -> token with no contradiction on any in-bounds background target cell."""
    seen = {}
    bg0 = bg_color(train[0][0])
    for x, y in train:
        b = bg_color(x); h, w = x.shape
        ys, xs = np.nonzero(x != b)
        if len(ys) > 400:
            return None
        for sy, sx in zip(ys, xs):
            s = int(x[sy, sx]); key = s if per_color else 0
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    if dy == 0 and dx == 0:
                        continue
                    ty, tx = sy + dy, sx + dx
                    if not (0 <= ty < h and 0 <= tx < w) or x[ty, tx] != b:
                        continue
                    seen.setdefault((key, dy, dx), []).append((int(y[ty, tx]), s))
    rules = {}
    for k, obs in seen.items():
        outs = {o for o, _ in obs}
        if len(outs) == 1:
            o = outs.pop()
            if o != bg0 and len(obs) >= 2:
                rules[k] = ("abs", o)
        elif not per_color and all(o == s for o, s in obs):
            rules[k] = ("src", 0)
    return rules


def _stamp_apply(g, rules, R, per_color):
    b = bg_color(g); h, w = g.shape; o = g.copy()
    ys, xs = np.nonzero(g != b)
    for sy, sx in zip(ys, xs):
        s = int(g[sy, sx]); key = s if per_color else 0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                r = rules.get((key, dy, dx))
                if r is None:
                    continue
                ty, tx = sy + dy, sx + dx
                if 0 <= ty < h and 0 <= tx < w and g[ty, tx] == b:
                    o[ty, tx] = s if r[0] == "src" else r[1]
    return o


def fit_stamp(train, test_in):
    if not all(x.shape == y.shape for x, y in train) or all(_same(x, y) for x, y in train):
        return []
    for x, y in train:  # only additions over background
        if ((x != y) & (x != bg_color(x))).any():
            return []
    res = []
    for R in (1, 2, 3):
        for per_color in (False, True):
            rules = _stamp_learn(train, R, per_color)
            if not rules:
                continue
            if all(_same(_stamp_apply(x, rules, R, per_color), y) for x, y in train):
                if per_color:
                    seen = {k[0] for k in rules}
                    if any(int(c) not in seen for t in test_in for c in np.unique(t) if c != bg_color(t)):
                        continue
                res.append((f"stamp:R{R}:{per_color}", [_stamp_apply(t, rules, R, per_color) for t in test_in]))
                return res
    return res


# ---------------------------------------------------------------- crop an object chosen by a selector
NUMF = ["size", "holes", "ncol", "ncells_bbox", "cy", "cx", "shapefreq", "colfreq", "contentfreq"]


def _cands(g, sname):
    masks = SEGS[sname](g)
    if not masks or len(masks) > 80:
        return None
    props = region_table(g, masks)
    content = []
    for m in masks:
        y0, y1, x0, x1 = _bbox(m)
        s = g[y0:y1, x0:x1]
        content.append(s.tobytes() + bytes(s.shape))
    for p, c in zip(props, content):
        p["contentfreq"] = sum(1 for d in content if d == c)
        for cc in range(10):
            p[f"has{cc}"] = int(cc in p["colset"])
    return masks, props


def _select(props, fname, mode):
    vals = [p[fname] for p in props]
    t = max(vals) if mode == "max" else min(vals)
    idx = [i for i, v in enumerate(vals) if v == t]
    return idx[0] if len(idx) == 1 else None


def _crop(g, m, cv):
    y0, y1, x0, x1 = _bbox(m)
    sub = g[y0:y1, x0:x1]
    if cv == "box":
        return sub
    if cv == "mask":
        return np.where(m[y0:y1, x0:x1], sub, bg_color(g))
    if y1 - y0 > 2 and x1 - x0 > 2:
        return sub[1:-1, 1:-1]
    return None


def _bool_sel(props, fname, val):
    if fname == "contentfreq1":
        ids = [i for i, p in enumerate(props) if (p["contentfreq"] == 1) == bool(val)]
    else:
        ids = [i for i, p in enumerate(props) if p[fname] == val]
    return ids[0] if len(ids) == 1 else None


def fit_crop_obj(train, test_in):
    if any(y.size >= x.size for x, y in train):
        return []
    res = []
    for sname in ("obj8", "obj4", "col8", "col4", "sepcells", "holes", "holes8"):
        tr = [_cands(x, sname) for x, _ in train]
        te = [_cands(t, sname) for t in test_in]
        if any(c is None for c in tr) or any(c is None for c in te):
            continue
        for cv in ("box", "mask", "inner"):
            match = []
            for (x, y), (masks, props) in zip(train, tr):
                idx = set()
                for i, m in enumerate(masks):
                    c = _crop(x, m, cv)
                    if c is not None and _same(c, y):
                        idx.add(i)
                if not idx:
                    break
                match.append(idx)
            else:
                selectors = []
                for fname in NUMF:
                    for mode in ("max", "min"):
                        selectors.append((f"{fname}:{mode}", lambda props, f=fname, md=mode: _select(props, f, md)))
                for fname in ("uniq_shape", "uniq_col", "rect", "border", "contentfreq1") + tuple(f"has{cc}" for cc in range(10)):
                    for val in (0, 1):
                        selectors.append((f"{fname}={val}", lambda props, f=fname, v=val: _bool_sel(props, f, v)))
                for sn, sel in selectors:
                    if not all(sel(props) in idx for (masks, props), idx in zip(tr, match)):
                        continue
                    preds = []
                    for t, (masks, props) in zip(test_in, te):
                        i = sel(props)
                        c = None if i is None else _crop(t, masks[i], cv)
                        if c is None:
                            preds = None
                            break
                        preds.append(c)
                    if preds:
                        res.append((f"cropobj:{sname}:{cv}:{sn}", preds))
                        if len(res) >= 2:
                            return res
    return res


# ---------------------------------------------------------------- rays learned per (source color, direction)
def _sources_px(g, iso):
    b = bg_color(g)
    if not iso:
        return [(y, x) for y, x in zip(*np.nonzero(g != b))]
    out = []
    lab, n = ndimage.label(g != b, structure=S8)
    sizes = ndimage.sum(np.ones(g.shape), lab, range(1, n + 1))
    for i in range(1, n + 1):
        if sizes[i - 1] == 1:
            y, x = np.argwhere(lab == i)[0]
            out.append((y, x))
    return out


def _ray_cells(g, y0, x0, d, block):
    h, w = g.shape; b = bg_color(g); cells = []
    y, x = y0 + d[0], x0 + d[1]
    while 0 <= y < h and 0 <= x < w:
        if g[y, x] != b:
            if block:
                break
        else:
            cells.append((y, x))
        y += d[0]; x += d[1]
    return cells


def _rays_apply(g, rules, iso, block):
    o = g.copy()
    for y0, x0 in _sources_px(g, iso):
        s = int(g[y0, x0])
        for d in D8L:
            r = rules.get((s, d))
            if r is None:
                continue
            for (y, x) in _ray_cells(g, y0, x0, d, block):
                o[y, x] = r
    return o


def fit_rays(train, test_in):
    if not all(x.shape == y.shape for x, y in train) or all(_same(x, y) for x, y in train):
        return []
    for x, y in train:
        if ((x != y) & (x != bg_color(x))).any():
            return []
    for iso in (True, False):
        for block in (True, False):
            obs = {}
            for x, y in train:
                for y0, x0 in _sources_px(x, iso):
                    s = int(x[y0, x0])
                    for d in D8L:
                        for (yy, xx) in _ray_cells(x, y0, x0, d, block):
                            obs.setdefault((s, d), set()).add(int(y[yy, xx]))
            bg0 = bg_color(train[0][0])
            rules = {k: next(iter(v)) for k, v in obs.items() if len(v) == 1 and next(iter(v)) != bg0}
            if not rules:
                continue
            if all(_same(_rays_apply(x, rules, iso, block), y) for x, y in train):
                seen = {k[0] for k in rules}
                srcs_t = {int(t[yy, xx]) for t in test_in for yy, xx in _sources_px(t, iso)}
                if not srcs_t <= {k[0] for k in obs}:
                    continue
                return [(f"rays2:iso{iso}:blk{block}", [_rays_apply(t, rules, iso, block) for t in test_in])]
    return []


FITTERS2 = [fit_stamp, fit_crop_obj, fit_rays]
