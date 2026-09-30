"""Object/relation cells fitted by exact consistency on demo pairs (counting/enumeration only; no gradients).

Family 1  region recolor: segment the grid into regions (objects / enclosed holes / background pockets), compute a
          property key per region, fit table key -> action (keep | recolor c) by counting; strict test coverage.
Family 2  object move: per-color direction (slide until contact with anything or border).
Family 3  lines: connect same-color pixel pairs in a row/col; rays from pixels by color and direction.
Family 4  periodic completion: fill a masked color by the minimal 2D period (or output only the masked patch).
Family 5  count-to-output: output layout determined by color/object counts.
"""
import itertools
import numpy as np
from scipy import ndimage
from cells import bg_color

S8 = np.ones((3, 3), dtype=int)
S4 = ndimage.generate_binary_structure(2, 1)


def _same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


# ---------------------------------------------------------------- segmentations
def _bbox(m):
    r = np.flatnonzero(m.any(1)); c = np.flatnonzero(m.any(0))
    return r[0], r[-1] + 1, c[0], c[-1] + 1


def seg_objects(g, st, by_color):
    bg = bg_color(g)
    masks = []
    if by_color:
        for col in np.unique(g):
            if col == bg:
                continue
            lab, n = ndimage.label(g == col, structure=st)
            masks += [lab == i for i in range(1, n + 1)]
    else:
        lab, n = ndimage.label(g != bg, structure=st)
        masks = [lab == i for i in range(1, n + 1)]
    return masks


def seg_holes(g, st_bg=S4):
    """Background-colored pockets not touching the border (per color region of the most common color)."""
    bg = bg_color(g)
    lab, n = ndimage.label(g == bg, structure=st_bg)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()):
            out.append(m)
    return out


def seg_pockets(g):
    """Every 4-connected region of any single color that is not the largest region (separator-cell style)."""
    masks = []
    for col in np.unique(g):
        lab, n = ndimage.label(g == col, structure=S4)
        masks += [lab == i for i in range(1, n + 1)]
    return masks


def seg_cells_of_separators(g):
    """Regions between full-row/col separator lines of a single color (grid-of-cells layouts)."""
    h, w = g.shape
    for col in np.unique(g):
        rows = [i for i in range(h) if (g[i] == col).all()]
        cols = [j for j in range(w) if (g[:, j] == col).all()]
        if rows or cols:
            sep = np.zeros(g.shape, bool)
            sep[rows, :] = True; sep[:, cols] = True
            lab, n = ndimage.label(~sep, structure=S4)
            if n >= 2:
                return [lab == i for i in range(1, n + 1)]
    return []


SEGS = {
    "obj8": lambda g: seg_objects(g, S8, False),
    "obj4": lambda g: seg_objects(g, S4, False),
    "col8": lambda g: seg_objects(g, S8, True),
    "col4": lambda g: seg_objects(g, S4, True),
    "holes": seg_holes,
    "holes8": lambda g: seg_holes(g, S8),
    "sepcells": seg_cells_of_separators,
}


# ---------------------------------------------------------------- region features
def _shape_key(m):
    y0, y1, x0, x1 = _bbox(m)
    s = m[y0:y1, x0:x1]
    return (s.shape, s.tobytes())


def _holes_count(m):
    filled = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(filled & ~m, structure=S4)
    return n


def _dom(g, m):
    v, c = np.unique(g[m], return_counts=True)
    return int(v[c.argmax()])


def _ring_colors(g, m):
    ring = ndimage.binary_dilation(m, structure=S4) & ~m
    return ring


def region_table(g, masks):
    """Return list of dict of properties, one per region."""
    bg = bg_color(g)
    sizes = [int(m.sum()) for m in masks]
    shapes = [_shape_key(m) for m in masks]
    cols = [_dom(g, m) for m in masks]
    full = [(_shape_key(m), c) for m, c in zip(masks, cols)]
    ss = sorted(set(sizes))
    props = []
    for k, m in enumerate(masks):
        y0, y1, x0, x1 = _bbox(m)
        h, w = y1 - y0, x1 - x0
        sub = m[y0:y1, x0:x1]
        ring = _ring_colors(g, m)
        rc = g[ring]
        ringc = int(np.bincount(rc).argmax()) if len(rc) else -1
        ncol = len(np.unique(g[m]))
        border = int(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        props.append(dict(
            size=sizes[k], rank_d=len(ss) - 1 - ss.index(sizes[k]), rank_a=ss.index(sizes[k]),
            ext=(1 if sizes[k] == ss[-1] else 0) + (2 if sizes[k] == ss[0] else 0),
            shape=shapes[k], holes=_holes_count(m), border=border, color=cols[k], ncol=ncol,
            hw=(h, w), rect=int(sub.all()), sym_h=int((sub == sub[:, ::-1]).all()), sym_v=int((sub == sub[::-1]).all()),
            shapefreq=sum(1 for s in shapes if s == shapes[k]),
            fullfreq=sum(1 for s in full if s == full[k]),
            colfreq=sum(1 for c in cols if c == cols[k]),
            ring=ringc, nring=len(np.unique(rc)),
            uniq_shape=int(sum(1 for s in shapes if s == shapes[k]) == 1),
            uniq_col=int(sum(1 for c in cols if c == cols[k]) == 1),
            parity=sizes[k] % 2, taller=int(h > w) - int(h < w), ncells_bbox=h * w,
            colset=tuple(int(v) for v in np.unique(g[m])),
            cy=(y0 + y1 - 1) * 1.0, cx=(x0 + x1 - 1) * 1.0,
        ))
    # frequency ranks over colors
    cf = sorted(set(p["colfreq"] for p in props))
    for p in props:
        p["colfreq_ext"] = (1 if p["colfreq"] == cf[-1] else 0) + (2 if p["colfreq"] == cf[0] else 0)
        p["nsame"] = (p["color"], p["size"])
    return props


FEATS = [
    ("size", lambda p: p["size"]),
    ("ext", lambda p: p["ext"]),
    ("rank_d", lambda p: p["rank_d"]),
    ("rank_a", lambda p: p["rank_a"]),
    ("holes", lambda p: p["holes"]),
    ("border", lambda p: p["border"]),
    ("color", lambda p: p["color"]),
    ("shape", lambda p: p["shape"]),
    ("hw", lambda p: p["hw"]),
    ("rect", lambda p: p["rect"]),
    ("symm", lambda p: (p["sym_h"], p["sym_v"])),
    ("shapefreq", lambda p: p["shapefreq"]),
    ("uniq_shape", lambda p: p["uniq_shape"]),
    ("uniq_col", lambda p: p["uniq_col"]),
    ("colfreq_ext", lambda p: p["colfreq_ext"]),
    ("ncol", lambda p: p["ncol"]),
    ("parity", lambda p: p["parity"]),
    ("ring", lambda p: p["ring"]),
    ("nring", lambda p: p["nring"]),
    ("color+ext", lambda p: (p["color"], p["ext"])),
    ("color+size", lambda p: (p["color"], p["size"])),
    ("color+holes", lambda p: (p["color"], p["holes"])),
    ("color+border", lambda p: (p["color"], p["border"])),
    ("color+rect", lambda p: (p["color"], p["rect"])),
    ("color+uniq_shape", lambda p: (p["color"], p["uniq_shape"])),
    ("ring+size", lambda p: (p["ring"], p["size"])),
    ("ring+ext", lambda p: (p["ring"], p["ext"])),
    ("shape+color", lambda p: (p["shape"], p["color"])),
    ("holes+ext", lambda p: (p["holes"], p["ext"])),
    ("size+border", lambda p: (p["size"], p["border"])),
    ("colset", lambda p: p["colset"]),
    ("colset+ext", lambda p: (p["colset"], p["ext"])),
    ("taller", lambda p: p["taller"]),
    ("color+taller", lambda p: (p["color"], p["taller"])),
]


def _sources(g, masks, props):
    """Candidate color sources per region for relational recolor."""
    bg = bg_color(g)
    out = {"ring": [p["ring"] if p["ring"] >= 0 else None for p in props]}
    mino = []
    for m in masks:
        v, c = np.unique(g[m], return_counts=True)
        mino.append(int(v[c.argmin()]) if len(v) > 1 else None)
    out["minority"] = mino
    sz = [p["size"] for p in props]
    big = int(g[masks[int(np.argmax(sz))]][0]); small = int(g[masks[int(np.argmin(sz))]][0])
    out["largest"] = [big] * len(masks); out["smallest"] = [small] * len(masks)
    nearest = []
    for i, m in enumerate(masks):
        d = ndimage.distance_transform_edt(~m)
        best = None; bd = 1e9
        for j, m2 in enumerate(masks):
            if j != i and g[m2][0] != g[m][0]:
                dd = d[m2].min()
                if dd < bd:
                    bd, best = dd, int(g[m2][0])
        nearest.append(best)
    out["nearest"] = nearest if len(masks) <= 60 else [None] * len(masks)
    return out


def _targets(g, y, masks, srcs=None):
    # masks here are the AREAS to be painted (already restricted as needed)
    """Per region action: ('keep',) or ('set', c); None if not expressible. Also need outside-region equality."""
    cover = np.zeros(g.shape, bool)
    acts = []
    for k, m in enumerate(masks):
        cover |= m
        o = y[m]
        if len(o) == 0 or (o == g[m]).all():
            acts.append(("keep",))
        elif (o == o[0]).all():
            if srcs is not None:
                if srcs[k] is None or srcs[k] != int(o[0]):
                    return None
                acts.append(("src",))
            else:
                acts.append(("set", int(o[0])))
        else:
            return None
    if not np.array_equal(g[~cover], y[~cover]):
        return None
    return acts


def _areas(g, masks, amode):
    if amode == "self":
        return masks
    bg = bg_color(g); out = []
    for m in masks:
        if amode == "bbox":
            y0, y1, x0, x1 = _bbox(m); a = np.zeros(g.shape, bool); a[y0:y1, x0:x1] = True
        elif amode == "ring4":
            a = ndimage.binary_dilation(m, structure=S4)
        elif amode == "ring8":
            a = ndimage.binary_dilation(m, structure=S8)
        elif amode == "ring8x2":
            a = ndimage.binary_dilation(m, structure=S8, iterations=2)
        elif amode == "fillh":
            a = ndimage.binary_fill_holes(m)
        out.append(a & (g == bg) & ~m)
    return out


AMODES = ("self", "bbox", "ring4", "ring8", "fillh")


def fit_regions(train, test_in):
    """Yield (kind, preds) for region-recolor rules (absolute color or relational color source)."""
    if not all(x.shape == y.shape for x, y in train):
        return []
    if all(np.array_equal(x, y) for x, y in train):
        return []
    res = []
    for sname, sfn in SEGS.items():
        try:
            tr_masks = [sfn(x) for x, _ in train]
            te_masks = [sfn(t) for t in test_in]
            if any(len(m) == 0 for m in tr_masks) or any(len(m) == 0 for m in te_masks):
                continue
            if sum(len(m) for m in tr_masks) > 600:
                continue
            tr_props = [region_table(x, m) for (x, _), m in zip(train, tr_masks)]
            te_props = [region_table(t, m) for t, m in zip(test_in, te_masks)]
            tr_src = [_sources(x, m, p) for (x, _), m, p in zip(train, tr_masks, tr_props)]
            te_src = [_sources(t, m, p) for t, m, p in zip(test_in, te_masks, te_props)]
        except Exception:
            continue
        for amode in AMODES:
          tr_area = [_areas(x, m, amode) for (x, _), m in zip(train, tr_masks)]
          te_area = [_areas(t, m, amode) for t, m in zip(test_in, te_masks)]
          for smode in (None, "ring", "minority", "largest", "smallest", "nearest"):
              if amode != "self" and smode not in (None, "ring", "minority", "nearest"):
                  continue
              tacts = [_targets(x, y, m, None if smode is None else ss[smode])
                       for (x, y), m, ss in zip(train, tr_area, tr_src)]
              if any(a is None for a in tacts):
                  continue
              if all(a == ("keep",) for acts in tacts for a in acts):
                  continue
              for fname, ff in FEATS:
                  tab = {}
                  ok = True
                  for props, acts in zip(tr_props, tacts):
                      for p, a in zip(props, acts):
                          if tab.setdefault(ff(p), a) != a:
                              ok = False; break
                      if not ok:
                          break
                  if not ok or all(a == ("keep",) for a in tab.values()):
                      continue
                  if not all(ff(p) in tab for props in te_props for p in props):
                      continue
                  preds = []
                  for t, m, props, ss in zip(test_in, te_area, te_props, te_src):
                      o = t.copy()
                      for k, (mm, p) in enumerate(zip(m, props)):
                          a = tab[ff(p)]
                          if a[0] == "set":
                              o[mm] = a[1]
                          elif a[0] == "src":
                              c = ss[smode][k]
                              if c is None:
                                  preds = None; break
                              o[mm] = c
                      if preds is None:
                          break
                      preds.append(o)
                  if preds is None:
                      continue
                  if amode != "self":
                      chk = []
                      for (x, y), m, props, ss in zip(train, tr_area, tr_props, tr_src):
                          o = x.copy()
                          for k, (mm, p) in enumerate(zip(m, props)):
                              a = tab[ff(p)]
                              if a[0] == "set": o[mm] = a[1]
                              elif a[0] == "src": o[mm] = ss[smode][k]
                          chk.append(np.array_equal(o, y))
                      if not all(chk):
                          continue
                  res.append((f"reg:{amode}:{sname}:{smode}:{fname}", preds))
                  if len(res) >= 3:
                      return res
    return res


# ---------------------------------------------------------------- movement
DIRS = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}


def slide_color(g, color, d, bg):
    """Slide every object (8-conn component) of `color` as a rigid body in direction d until contact."""
    dy, dx = DIRS[d]
    g = g.copy()
    lab, n = ndimage.label(g == color, structure=S8)
    order = list(range(1, n + 1))
    # process objects nearest to the destination first
    def key(i):
        ys, xs = np.nonzero(lab == i)
        return -(ys.mean() * dy + xs.mean() * dx)
    order.sort(key=key)
    h, w = g.shape
    for i in order:
        m = lab == i
        ys, xs = np.nonzero(m)
        g[m] = bg
        k = 0
        while True:
            ny, nx = ys + dy * (k + 1), xs + dx * (k + 1)
            if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w or (g[ny, nx] != bg).any():
                break
            k += 1
        g[ys + dy * k, xs + dx * k] = color
    return g


def fit_move(train, test_in):
    if not all(x.shape == y.shape for x, y in train) or all(_same(x, y) for x, y in train):
        return []
    bgs = [bg_color(x) for x, _ in train]
    cols = sorted(set(int(c) for x, b in zip((x for x, _ in train), bgs) for c in np.unique(x) if c != b))
    # candidate movers: colors whose pixel count is preserved but whose position changed
    movers = []
    for c in cols:
        if all((x == c).sum() == (y == c).sum() for x, y in train) and any(not np.array_equal(x == c, y == c) for x, y in train):
            movers.append(c)
    if not movers or len(movers) > 2:
        return []
    opts = [["-"] + list(DIRS) for _ in movers]
    sols = []
    for combo in itertools.product(*opts):
        if all(c == "-" for c in combo):
            continue
        for order in itertools.permutations(range(len(movers))) if len(movers) <= 3 else [range(len(movers))]:
            ok = True
            for (x, y), b in zip(train, bgs):
                o = x
                for k in order:
                    if combo[k] != "-":
                        o = slide_color(o, movers[k], combo[k], b)
                if not np.array_equal(o, y):
                    ok = False; break
            if ok:
                preds = []
                for t in test_in:
                    o = t; b = bg_color(t)
                    for k in order:
                        if combo[k] != "-":
                            o = slide_color(o, movers[k], combo[k], b)
                    preds.append(o)
                sols.append(("move:" + "".join(combo), preds))
                break
        if sols:
            break
    return sols


# ---------------------------------------------------------------- lines
def draw_pairs(g, bg, line_color, diag=False, over=False):
    """Connect same-colored pixels sharing a row or column with a straight line over background cells."""
    o = g.copy(); h, w = g.shape
    for c in np.unique(g):
        if c == bg:
            continue
        ys, xs = np.nonzero(g == c)
        pts = list(zip(ys, xs))
        if len(pts) > 40:
            continue
        for (a, b), (c2, d) in itertools.combinations(pts, 2):
            lc = c if line_color is None else line_color
            if a == c2:
                seg = (a, slice(min(b, d) + 1, max(b, d)))
            elif b == d:
                seg = (slice(min(a, c2) + 1, max(a, c2)), b)
            else:
                continue
            sub = o[seg]
            if over:
                o[seg] = lc
            else:
                sub[sub == bg] = lc
    return o


def ray(g, bg, color, d, line_color, stop_block=True):
    o = g.copy(); h, w = g.shape
    dy, dx = d
    for y, x in zip(*np.nonzero(g == color)):
        ny, nx = y + dy, x + dx
        while 0 <= ny < h and 0 <= nx < w:
            if g[ny, nx] != bg:
                if stop_block:
                    break
            else:
                o[ny, nx] = color if line_color is None else line_color
            ny += dy; nx += dx
    return o


D8L = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def fit_lines(train, test_in):
    if not all(x.shape == y.shape for x, y in train) or all(_same(x, y) for x, y in train):
        return []
    # only additions over background
    bgs = [bg_color(x) for x, _ in train]
    for (x, y), b in zip(train, bgs):
        if ((x != y) & (x != b)).any():
            return []
    res = []
    added = set(int(c) for (x, y) in train for c in np.unique(y[x != y]))
    lcs = [None] + sorted(added)
    for lc in lcs:
        for over in (False, True):
            f = lambda g, b: draw_pairs(g, b, lc, over=over)
            if all(_same(f(x, b), y) for (x, y), b in zip(train, bgs)):
                res.append((f"pairs:{lc}:{over}", [f(t, bg_color(t)) for t in test_in]))
                return res
    # rays: choose, for each source color, a subset of directions (joint enumeration over colors with changed env)
    srcs = sorted(set(int(c) for x, b in zip((x for x, _ in train), bgs) for c in np.unique(x) if c != b))
    if len(srcs) > 4:
        return res
    for lc in lcs:
        for blk in (True, False):
            chosen = {}
            for c in srcs:
                best = None
                for nd in range(0, 3):
                    for ds in itertools.combinations(D8L, nd):
                        if nd == 0:
                            continue
                        ok = True
                        for (x, y), b in zip(train, bgs):
                            o = x
                            for d in ds:
                                o = ray(o if False else x, b, c, d, lc, blk) if False else o
                            # union of rays from original x
                            o = x.copy()
                            for d in ds:
                                r = ray(x, b, c, d, lc, blk)
                                o[r != x] = r[r != x]
                            # must be subset of target additions
                            d_ = (o != x)
                            if not (y[d_] == o[d_]).all():
                                ok = False; break
                        if ok:
                            # prefer the largest set consistent (maximal explained) -> evaluate later globally
                            best = ds if best is None or len(ds) > len(best) else best
                    if best is not None and nd >= 1 and len(best) == nd:
                        pass
                chosen[c] = best
            # verify global equality with chosen rays, trying maximal subsets per color
            cand = {c: v for c, v in chosen.items() if v}
            if not cand:
                continue
            def apply(g, b):
                o = g.copy()
                for c, ds in cand.items():
                    for d in ds:
                        r = ray(g, b, c, d, lc, blk)
                        o[r != g] = r[r != g]
                return o
            if all(_same(apply(x, b), y) for (x, y), b in zip(train, bgs)):
                res.append((f"rays:{lc}:{blk}", [apply(t, bg_color(t)) for t in test_in]))
                return res
    return res


# ---------------------------------------------------------------- periodic completion
def _period_fill(g, maskc, all_=False):
    m = g == maskc
    if not m.any() or m.all():
        return None
    h, w = g.shape
    best = None
    for py in range(1, h + 1):
        for px in range(1, w + 1):
            if py * px >= h * w // 2 and not (py == h or px == w):
                continue
            if best is not None and py * px >= best[0]:
                continue
            tile = np.full((py, px), -1)
            ok = True
            for i in range(py):
                for j in range(px):
                    v = g[i::py, j::px]; mm = ~m[i::py, j::px]
                    vals = np.unique(v[mm])
                    if len(vals) > 1:
                        ok = False; break
                    if len(vals) == 1:
                        tile[i, j] = vals[0]
                if not ok:
                    break
            if ok and (tile >= 0).all():
                best = (py * px, py, px, tile)
    if best is None:
        return None
    _, py, px, tile = best
    full = np.tile(tile, (h // py + 1, w // px + 1))[:h, :w]
    return full, m


def _diag_period_fill(g, maskc):
    """Diagonal-periodic (e.g. (i+j) mod k or (i-j) mod k) completion."""
    m = g == maskc
    if not m.any() or m.all():
        return None
    h, w = g.shape
    I, J = np.indices(g.shape)
    for k in range(1, 8):
        for sgn in (1, -1):
            key = (I + sgn * J) % k
            tile = {}
            ok = True
            for kk in range(k):
                vals = np.unique(g[(key == kk) & ~m])
                if len(vals) != 1:
                    ok = False; break
                tile[kk] = vals[0]
            if ok:
                full = np.vectorize(tile.get)(key)
                return full, m
    return None


def fit_periodic(train, test_in):
    res = []
    cols = sorted(set(int(c) for x, _ in train for c in np.unique(x)))
    for fn_name, fn in (("rect", _period_fill), ("diag", _diag_period_fill)):
        for mc in cols:
            for mode in ("full", "patch"):
                ok = True
                for x, y in train:
                    r = fn(x, mc)
                    if r is None:
                        ok = False; break
                    full, m = r
                    if mode == "full":
                        out = full
                    else:
                        if not m.any():
                            ok = False; break
                        y0, y1, x0, x1 = _bbox(m)
                        out = full[y0:y1, x0:x1]
                    if not _same(out, y):
                        ok = False; break
                if not ok:
                    continue
                preds = []
                for t in test_in:
                    r = fn(t, mc)
                    if r is None:
                        preds = None; break
                    full, m = r
                    if mode == "full":
                        preds.append(full)
                    else:
                        y0, y1, x0, x1 = _bbox(m)
                        preds.append(full[y0:y1, x0:x1])
                if preds:
                    res.append((f"period:{fn_name}:{mc}:{mode}", preds))
                    return res
    return res


# ---------------------------------------------------------------- count -> output
def _counts(g, kind):
    bg = bg_color(g)
    if kind == "cells_by_color":
        v, c = np.unique(g[g != bg], return_counts=True)
        return list(zip(v.tolist(), c.tolist()))
    if kind == "objs_by_color":
        d = {}
        for c in np.unique(g):
            if c != bg:
                d[int(c)] = ndimage.label(g == c, structure=S8)[1]
        return list(d.items())
    if kind == "objs_by_color4":
        d = {}
        for c in np.unique(g):
            if c != bg:
                d[int(c)] = ndimage.label(g == c, structure=S4)[1]
        return list(d.items())
    return []


def _layout_outputs(g, kind, order, lay):
    cs = _counts(g, kind)
    if not cs:
        return None
    cs = sorted(cs, key=lambda t: (-t[1], t[0]) if order == "desc" else (t[1], t[0]))
    cols = [c for c, _ in cs]
    if lay == "col":
        return np.array(cols).reshape(-1, 1)
    if lay == "row":
        return np.array(cols).reshape(1, -1)
    if lay == "top":
        return np.array(cols[:1]).reshape(1, 1)
    if lay == "bottom":
        return np.array(cols[-1:]).reshape(1, 1)
    if lay == "hist":  # bar chart: rows = colors, length = count, right-padded with bg
        wmax = max(n for _, n in cs)
        o = np.full((len(cs), wmax), bg_color(g))
        for i, (c, n) in enumerate(cs):
            o[i, :n] = c
        return o
    if lay == "vhist":
        wmax = max(n for _, n in cs)
        o = np.full((wmax, len(cs)), bg_color(g))
        for i, (c, n) in enumerate(cs):
            o[wmax - n:, i] = c
        return o
    return None


def _count_shape(g, kind, color_sel):
    """Output a 1xn / nx1 / sq grid whose size is a count of objects of a chosen color (or all)."""
    return None


def fit_count_out(train, test_in):
    res = []
    for kind in ("cells_by_color", "objs_by_color", "objs_by_color4"):
        for order in ("desc", "asc"):
            for lay in ("col", "row", "top", "bottom", "hist", "vhist"):
                ok = True
                for x, y in train:
                    o = _layout_outputs(x, kind, order, lay)
                    if o is None or not _same(o, y):
                        ok = False; break
                if ok:
                    preds = [_layout_outputs(t, kind, order, lay) for t in test_in]
                    if all(p is not None for p in preds):
                        res.append((f"count:{kind}:{order}:{lay}", preds))
                        return res
    return res


FITTERS = [fit_regions, fit_move, fit_lines, fit_periodic, fit_count_out]


def fit_objects(train, test_in):
    from objects2 import FITTERS2
    from objects3 import FITTERS3
    out = []
    for f in FITTERS + FITTERS2 + FITTERS3:
        try:
            for kind, preds in f(train, test_in):
                if all(p is not None and p.ndim == 2 and p.size > 0 for p in preds):
                    out.append((kind, preds))
        except Exception:
            pass
    return out
