"""Hand-authored grid cells (grid -> grid). No learning: fixed rules; a few take a small parameter.
Cells that need a task-level constant (color map) are fitted by exact matching on the demo pairs, not gradients."""
import numpy as np
from scipy import ndimage


def bg_color(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[c.argmax()])


def _bbox(mask):
    r = np.flatnonzero(mask.any(1))
    c = np.flatnonzero(mask.any(0))
    if len(r) == 0:
        return None
    return r[0], r[-1] + 1, c[0], c[-1] + 1


def crop_nonbg(g):
    b = _bbox(g != bg_color(g))
    return g if b is None else g[b[0]:b[1], b[2]:b[3]]


def components(g, diag=True, by_color=True):
    bg = bg_color(g)
    st = np.ones((3, 3)) if diag else None
    out = []
    colors = [c for c in np.unique(g) if c != bg] if by_color else [None]
    for col in colors:
        m = (g == col) if by_color else (g != bg)
        lab, n = ndimage.label(m, structure=st)
        for i in range(1, n + 1):
            mm = lab == i
            out.append((mm, _bbox(mm), int(mm.sum())))
    return out


def _pick(g, key, diag, by_color):
    comps = components(g, diag, by_color)
    if not comps:
        return g
    mm, b, _ = key(comps)
    sub = np.where(mm, g, bg_color(g))
    return sub[b[0]:b[1], b[2]:b[3]]


def obj_largest(g, diag=True, by_color=True):
    return _pick(g, lambda c: max(c, key=lambda t: t[2]), diag, by_color)


def obj_smallest(g, diag=True, by_color=True):
    return _pick(g, lambda c: min(c, key=lambda t: t[2]), diag, by_color)


def obj_unique_color(g):
    comps = components(g, True, True)
    if not comps:
        return g
    cols = [int(g[m][0]) for m, _, _ in comps]
    for (m, b, _), c in zip(comps, cols):
        if cols.count(c) == 1:
            return np.where(m, g, bg_color(g))[b[0]:b[1], b[2]:b[3]]
    return g


def gravity(g, d):
    bg = bg_color(g)
    h = np.rot90(g, d).copy()
    for j in range(h.shape[1]):
        col = h[:, j]
        nb = col[col != bg]
        h[:, j] = np.concatenate([np.full(len(col) - len(nb), bg), nb])
    return np.rot90(h, -d)


def _hole_color(g):
    nb = g[g != bg_color(g)]
    return int(np.bincount(nb).argmax()) if len(nb) else 0


def fill_holes(g, color=None):
    bg = bg_color(g)
    m = g != bg
    holes = ndimage.binary_fill_holes(m) & ~m
    if not holes.any():
        return g
    o = g.copy()
    o[holes] = color if color is not None else _hole_color(g)
    return o


def denoise(g):
    bg = bg_color(g)
    o = g.copy()
    lab, n = ndimage.label(g != bg, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        if (lab == i).sum() == 1:
            o[lab == i] = bg
    return o


def mirror_h(g):
    return np.hstack([g, g[:, ::-1]])


def mirror_v(g):
    return np.vstack([g, g[::-1]])


def mirror_4(g):
    return mirror_v(mirror_h(g))


def tile(g, a, b):
    return np.tile(g, (a, b))


def upscale(g, k):
    return np.kron(g, np.ones((k, k), dtype=g.dtype))


def downscale(g, k):
    return g[::k, ::k] if g.shape[0] % k == 0 and g.shape[1] % k == 0 else g


def sym_complete(g, mask_color):
    """Fill cells equal to mask_color from their mirror/rotation partners."""
    o = g.copy()
    m = o == mask_color
    sq = o.shape[0] == o.shape[1]
    ts = [lambda x: x[::-1], lambda x: x[:, ::-1], lambda x: np.rot90(x, 2)]
    if sq:
        ts += [lambda x: x.T, lambda x: np.rot90(x, 2).T, lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 3)]
    for t in ts:
        p = t(o)
        fill = m & (p != mask_color)
        o[fill] = p[fill]
        m = o == mask_color
    return o


def split_halves(g):
    h, w = g.shape
    seps_r = [i for i in range(h) if len(set(g[i])) == 1 and h > 2]
    seps_c = [j for j in range(w) if len(set(g[:, j])) == 1 and w > 2]
    if seps_c and not seps_r:
        j = seps_c[len(seps_c) // 2]
        a, b = g[:, :j], g[:, j + 1:]
    elif seps_r and not seps_c:
        i = seps_r[len(seps_r) // 2]
        a, b = g[:i], g[i + 1:]
    elif w % 2 == 0:
        a, b = g[:, :w // 2], g[:, w // 2:]
    elif h % 2 == 0:
        a, b = g[:h // 2], g[h // 2:]
    else:
        return None
    return (a, b) if a.shape == b.shape else None


def bool_op(g, op):
    s = split_halves(g)
    if s is None:
        return g
    a, b = s[0] != 0, s[1] != 0
    r = {"and": a & b, "or": a | b, "xor": a ^ b, "nor": ~(a | b), "a_not_b": a & ~b, "b_not_a": b & ~a}[op]
    return r.astype(int)


def outline(g):
    bg = bg_color(g)
    m = g != bg
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
    o = g.copy()
    o[er] = bg
    return o


def most_common_color_grid(g):
    nb = g[g != bg_color(g)]
    c = int(np.bincount(nb).argmax()) if len(nb) else bg_color(g)
    return np.full((1, 1), c)


def count_colors_bar(g):
    nb = g[g != bg_color(g)]
    if len(nb) == 0:
        return g
    cnt = np.bincount(nb, minlength=10)
    order = [c for c in np.argsort(-cnt) if cnt[c] > 0]
    return np.array([order]).T


def fill_row_col_lines(g):
    bg = bg_color(g)
    o = g.copy()
    for i, j in zip(*np.nonzero(g != bg)):
        o[i, :] = np.where(o[i, :] == bg, g[i, j], o[i, :])
        o[:, j] = np.where(o[:, j] == bg, g[i, j], o[:, j])
    return o


def remove_color_rare(g):
    v, c = np.unique(g, return_counts=True)
    if len(v) < 3:
        return g
    o = g.copy()
    o[g == v[c.argmin()]] = bg_color(g)
    return o


def grid_cell_pick(g):
    """Grid partitioned into blocks by separator lines: return the block that differs from the others."""
    h, w = g.shape
    rows = [i for i in range(h) if len(set(g[i])) == 1]
    cols = [j for j in range(w) if len(set(g[:, j])) == 1]
    if not rows and not cols:
        return g
    rb = np.split(np.arange(h), rows) if rows else [np.arange(h)]
    cb = np.split(np.arange(w), cols) if cols else [np.arange(w)]
    blocks = []
    for r in rb:
        for c in cb:
            rr = [i for i in r if i not in rows]
            cc = [j for j in c if j not in cols]
            if rr and cc:
                blocks.append(g[np.ix_(rr, cc)])
    if len(blocks) < 3:
        return g
    keys = [b.tobytes() + bytes(b.shape) for b in blocks]
    for b, k in zip(blocks, keys):
        if keys.count(k) == 1:
            return b
    return blocks[0]


CELLS = {"id": lambda g: g}
for _k in (1, 2, 3):
    CELLS[f"rot{_k}"] = (lambda k: lambda g: np.rot90(g, k))(_k)
CELLS.update({
    "flipud": lambda g: g[::-1], "fliplr": lambda g: g[:, ::-1], "transpose": lambda g: g.T,
    "antitranspose": lambda g: np.rot90(g, 2).T, "crop": crop_nonbg,
    "largest_obj": obj_largest, "smallest_obj": obj_smallest,
    "largest_obj_mc": lambda g: obj_largest(g, True, False), "smallest_obj_mc": lambda g: obj_smallest(g, True, False),
    "unique_color_obj": obj_unique_color, "fill_holes": fill_holes, "denoise": denoise,
    "mirror_h": mirror_h, "mirror_v": mirror_v, "mirror_4": mirror_4, "outline": outline,
    "mc_color": most_common_color_grid, "color_bar": count_colors_bar, "lines": fill_row_col_lines,
    "rare_off": remove_color_rare, "block_pick": grid_cell_pick})
for _d in range(4):
    CELLS[f"gravity{_d}"] = (lambda d: lambda g: gravity(g, d))(_d)
for _a, _b in ((1, 2), (2, 1), (2, 2), (3, 3), (1, 3), (3, 1)):
    CELLS[f"tile{_a}{_b}"] = (lambda a, b: lambda g: tile(g, a, b))(_a, _b)
for _k in (2, 3, 4):
    CELLS[f"up{_k}"] = (lambda k: lambda g: upscale(g, k))(_k)
    CELLS[f"down{_k}"] = (lambda k: lambda g: downscale(g, k))(_k)
for _op in ("and", "or", "xor", "nor", "a_not_b", "b_not_a"):
    CELLS[f"bool_{_op}"] = (lambda op: lambda g: bool_op(g, op))(_op)
for _c in range(10):
    CELLS[f"symfill{_c}"] = (lambda c: lambda g: sym_complete(g, c))(_c)
