"""Task-fitted local-rule cells for same-shape tasks.

Each cell maps a per-cell feature key to an output color. The table is built by counting on the demo pairs
(exact consistency; no gradients, no search over weights). A rule is accepted only if (a) the table is conflict-free on
every demo, (b) it is not the identity, and (c) every test-cell key was seen in the demos (strict coverage, to limit
false positives). Feature sets are tried from simplest to most specific; the first that passes wins.
"""
import numpy as np
from scipy import ndimage
from cells import bg_color

D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
D8 = D4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _nb(g, dy, dx, fill=-1):
    h, w = g.shape
    o = np.full((h, w), fill, dtype=int)
    ys = slice(max(0, -dy), min(h, h - dy)); xs = slice(max(0, -dx), min(w, w - dx))
    yd = slice(max(0, dy), min(h, h + dy)); xd = slice(max(0, dx), min(w, w + dx))
    o[ys, xs] = g[yd, xd]
    return o


def _comp_props(g):
    bg = bg_color(g)
    size = np.zeros(g.shape, dtype=int); border = np.zeros(g.shape, dtype=int); rank = np.zeros(g.shape, dtype=int)
    lab, n = ndimage.label(g != bg, structure=np.ones((3, 3)))
    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    for i in range(1, n + 1):
        m = lab == i
        size[m] = sizes[i - 1]
        border[m] = int(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        rank[m] = 1 if sizes[i - 1] == max(sizes) else (2 if sizes[i - 1] == min(sizes) else 0)
    return size, border, rank


def _ray_colors(g):
    """Nearest non-background color seen looking up/down/left/right (or -1)."""
    bg = bg_color(g); h, w = g.shape; out = np.full((4, h, w), -1, dtype=int)
    for i in range(h):
        last = -1
        for j in range(w):
            out[2, i, j] = last
            if g[i, j] != bg: last = g[i, j]
        last = -1
        for j in range(w - 1, -1, -1):
            out[3, i, j] = last
            if g[i, j] != bg: last = g[i, j]
    for j in range(w):
        last = -1
        for i in range(h):
            out[0, i, j] = last
            if g[i, j] != bg: last = g[i, j]
        last = -1
        for i in range(h - 1, -1, -1):
            out[1, i, j] = last
            if g[i, j] != bg: last = g[i, j]
    return out


def features(g, kind):
    h, w = g.shape
    if kind == "color":
        return [[(int(g[i, j]),) for j in range(w)] for i in range(h)]
    if kind == "n4":
        nbs = [_nb(g, dy, dx) for dy, dx in D4]
        return [[(int(g[i, j]),) + tuple(int(n[i, j]) for n in nbs) for j in range(w)] for i in range(h)]
    if kind == "n8count":
        same = sum((_nb(g, dy, dx) == g).astype(int) for dy, dx in D8)
        return [[(int(g[i, j]), int(same[i, j])) for j in range(w)] for i in range(h)]
    if kind == "n4count":
        bg = bg_color(g); nz = sum(((_nb(g, dy, dx) != bg) & (_nb(g, dy, dx) != -1)).astype(int) for dy, dx in D4)
        return [[(int(g[i, j]), int(nz[i, j])) for j in range(w)] for i in range(h)]
    if kind == "n8":
        nbs = [_nb(g, dy, dx) for dy, dx in D8]
        return [[(int(g[i, j]),) + tuple(int(n[i, j]) for n in nbs) for j in range(w)] for i in range(h)]
    if kind == "n8set":
        nbs = [_nb(g, dy, dx) for dy, dx in D8]
        return [[(int(g[i, j]), tuple(sorted(set(int(n[i, j]) for n in nbs)))) for j in range(w)] for i in range(h)]
    if kind == "comp":
        size, border, rank = _comp_props(g)
        return [[(int(g[i, j]), int(size[i, j]), int(border[i, j])) for j in range(w)] for i in range(h)]
    if kind == "comp_rank":
        size, border, rank = _comp_props(g)
        return [[(int(g[i, j]), int(rank[i, j])) for j in range(w)] for i in range(h)]
    if kind == "comp_size":
        size, _, _ = _comp_props(g)
        return [[(int(g[i, j]), int(size[i, j])) for j in range(w)] for i in range(h)]
    if kind == "ray":
        r = _ray_colors(g)
        return [[(int(g[i, j]),) + tuple(int(r[k, i, j]) for k in range(4)) for j in range(w)] for i in range(h)]
    if kind == "ray_pair":
        r = _ray_colors(g)
        return [[(int(g[i, j]), int(r[0, i, j]) if r[0, i, j] == r[1, i, j] else -2, int(r[2, i, j]) if r[2, i, j] == r[3, i, j] else -2) for j in range(w)] for i in range(h)]
    if kind == "parity":
        return [[(int(g[i, j]), i % 2, j % 2) for j in range(w)] for i in range(h)]
    if kind == "rowcol":
        rows = [set(int(v) for v in g[i]) for i in range(h)]; cols = [set(int(v) for v in g[:, j]) for j in range(w)]
        bg = bg_color(g)
        return [[(int(g[i, j]), tuple(sorted(rows[i] - {bg})), tuple(sorted(cols[j] - {bg}))) for j in range(w)] for i in range(h)]
    raise KeyError(kind)


KINDS = ["color", "n4count", "n8count", "comp_rank", "comp_size", "comp", "n4", "n8set", "ray_pair", "ray", "rowcol", "parity", "n8"]


def fit_local_rule(train, test_in):
    """Return list of (kind, predictions) for every accepted feature set, simplest first."""
    if not all(x.shape == y.shape for x, y in train):
        return []
    accepted = []
    for kind in KINDS:
        table, ok = {}, True
        for x, y in train:
            f = features(x, kind)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = f[i][j]
                    v = int(y[i, j])
                    if table.setdefault(k, v) != v:
                        ok = False; break
                if not ok: break
            if not ok: break
        if not ok:
            continue
        if all(k[0] == v for k, v in table.items()):  # identity rule
            continue
        preds, cover = [], True
        for tg in test_in:
            f = features(tg, kind); o = np.zeros(tg.shape, dtype=int)
            for i in range(tg.shape[0]):
                for j in range(tg.shape[1]):
                    k = f[i][j]
                    if k not in table:
                        cover = False; break
                    o[i, j] = table[k]
                if not cover: break
            if not cover: break
            preds.append(o)
        if cover:
            accepted.append((kind, preds))
    return accepted


# ---- color-invariant (role) local rules ------------------------------------------------------------------------
# Keys describe relations (background / same as center / other color / outside) instead of absolute colors, so a rule fitted
# on some colors transfers to unseen colors. Output token is either an absolute color or "same as center color".

def _roles(g, offsets):
    bg = bg_color(g); h, w = g.shape
    rel = []
    for dy, dx in offsets:
        n = _nb(g, dy, dx)
        r = np.where(n == -1, 0, np.where(n == bg, 1, np.where(n == g, 2, 3)))
        rel.append(r)
    return bg, rel


def role_features(g, kind):
    h, w = g.shape
    if kind == "role_n8":
        bg, rel = _roles(g, D8)
        return [[(int(g[i, j] == bg),) + tuple(int(r[i, j]) for r in rel) for j in range(w)] for i in range(h)]
    if kind == "role_n4":
        bg, rel = _roles(g, D4)
        return [[(int(g[i, j] == bg),) + tuple(int(r[i, j]) for r in rel) for j in range(w)] for i in range(h)]
    if kind == "role_count":
        bg, rel = _roles(g, D8)
        return [[(int(g[i, j] == bg), sum(int(r[i, j] == 2) for r in rel), sum(int(r[i, j] == 3) for r in rel)) for j in range(w)] for i in range(h)]
    if kind == "role_comp":
        bg = bg_color(g); size, border, rank = _comp_props(g)
        return [[(int(g[i, j] == bg), min(int(size[i, j]), 12), int(border[i, j]), int(rank[i, j])) for j in range(w)] for i in range(h)]
    if kind == "role_ray":
        bg = bg_color(g); r = _ray_colors(g)
        return [[(int(g[i, j] == bg),) + tuple(int(0 if r[k, i, j] == -1 else (2 if r[k, i, j] == g[i, j] else 3)) for k in range(4)) for j in range(w)] for i in range(h)]
    raise KeyError(kind)


ROLE_KINDS = ["role_count", "role_comp", "role_n4", "role_ray", "role_n8"]


def _token_ok(tok, center, out):
    return out == center if tok == "same" else out == tok


def fit_role_rule(train, test_in):
    if not all(x.shape == y.shape for x, y in train):
        return []
    accepted = []
    for kind in ROLE_KINDS:
        cand = {}
        for x, y in train:
            f = role_features(x, kind)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k, c, o = f[i][j], int(x[i, j]), int(y[i, j])
                    toks = {o, "same"} if o == c else {o}
                    cand[k] = toks if k not in cand else (cand[k] & toks)
        if any(not v for v in cand.values()):
            continue
        table = {k: ("same" if "same" in v else min(v)) for k, v in cand.items()}
        if all(v == "same" for v in table.values()):
            continue  # identity
        preds, cover = [], True
        for tg in test_in:
            f = role_features(tg, kind); o = np.zeros(tg.shape, dtype=int)
            for i in range(tg.shape[0]):
                for j in range(tg.shape[1]):
                    k = f[i][j]
                    if k not in table:
                        cover = False; break
                    o[i, j] = tg[i, j] if table[k] == "same" else table[k]
                if not cover: break
            if not cover: break
            preds.append(o)
        if cover:
            accepted.append((kind, preds))
    return accepted
