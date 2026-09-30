"""Multi-view fitted cells. A rule is a table  key(view features) -> output token, where a token is an absolute colour or a
colour SOURCE (copy the pixel's own colour / its enclosing object's colour / its mirror partner / the row-dominant colour ...).
Selector features and action sources may come from different views (cross-view rules). Parameters are chosen by exact
consistency on the demo pairs (counting, no gradients); strict coverage of test keys; ambiguous token choices are rejected.

fit_multiview(train, test_in) -> list of dict(kind, fams, nkeys, nfeat, preds), ranked simplest first.
"""
import sys, os, time, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from views import all_views, FAM, SRC_FAM  # noqa: E402

MAXKEYS = 80


def _encode(F, idx):
    k = np.zeros(F.shape[:-1], dtype=np.int64)
    for i in idx:
        k = k * 32 + (F[..., i] + 1)
    return k


def _tokens(S, y):
    """Boolean matrix (N, T): token t reproduces the output pixel. Tokens = sources then absolute colours 0..9."""
    ns = S.shape[-1]
    Sf = S.reshape(-1, ns); yf = y.reshape(-1)
    E = np.concatenate([(Sf == yf[:, None]) & (Sf >= 0), yf[:, None] == np.arange(10)[None, :]], 1)
    return E


def _fit_one(Ftr, E, idx, nsrc, maxkeys):
    keys = _encode(Ftr, idx).reshape(-1)
    uk, inv = np.unique(keys, return_inverse=True)
    if len(uk) > maxkeys or len(uk) * 3 > len(keys):
        return None
    bad = np.zeros((len(uk), E.shape[1]))
    for t in range(E.shape[1]):
        bad[:, t] = np.bincount(inv, weights=(~E[:, t]).astype(float), minlength=len(uk))
    valid = bad == 0
    if not valid.any(1).all():
        return None
    return uk, valid


def _predict(Fte, S, idx, uk, valid, pref_last):
    keys = _encode(Fte, idx).reshape(-1)
    pos = np.searchsorted(uk, keys)
    pos = np.minimum(pos, len(uk) - 1)
    if not (uk[pos] == keys).all():
        return None, None
    ns = S.shape[-1]
    T = valid.shape[1]
    if pref_last:
        tok = T - 1 - np.argmax(valid[:, ::-1], 1)
    else:
        tok = np.argmax(valid, 1)
    t = tok[pos]
    Sf = S.reshape(-1, ns)
    out = np.where(t < ns, Sf[np.arange(len(t)), np.minimum(t, ns - 1)], t - ns)
    if (out < 0).any():
        return None, None
    return out.reshape(Fte.shape[:-1]), tok


def fit_pool(pools_tr, ys, pools_te, fn, sn, max_time=3.0, singles_only=False, cross_only=True, tag="pool", ident=True):
    """pools_*: list of (F, S) per grid. Returns accepted candidates (ranked)."""
    t0 = time.time()
    Ftr = np.concatenate([F.reshape(-1, F.shape[-1]) for F, _ in pools_tr], 0)[None]  # (1,N,nF)
    Str = np.concatenate([S.reshape(-1, S.shape[-1]) for _, S in pools_tr], 0)[None]
    y = np.concatenate([yy.reshape(-1) for yy in ys])
    E = _tokens(Str[0], y)
    nsrc = Str.shape[-1]
    if ident and E[:, 0].all():  # output == input everywhere: nothing to fit
        return []
    nf = len(fn)
    subsets = [(i,) for i in range(nf)]
    if not singles_only:
        for i, j in itertools.combinations(range(nf), 2):
            if cross_only and FAM[fn[i]] == FAM[fn[j]]:
                continue
            subsets.append((i, j))
    acc, seen = [], set()
    for idx in subsets:
        if time.time() - t0 > max_time:
            break
        r = _fit_one(Ftr, E, idx, nsrc, MAXKEYS)
        if r is None:
            continue
        uk, valid = r
        first = np.argmax(valid, 1)
        if ident and (first == 0).all():  # identity rule
            continue
        preds, ok = [], True
        for (F, S) in pools_te:
            p1, _ = _predict(F, S, idx, uk, valid, False)
            p2, _ = _predict(F, S, idx, uk, valid, True)
            if p1 is None or p2 is None or not np.array_equal(p1, p2):
                ok = False; break
            preds.append(p1)
        if not ok:
            continue
        sig = tuple(p.tobytes() for p in preds)
        if sig in seen:
            continue
        seen.add(sig)
        fams = sorted({FAM[fn[i]] for i in idx})
        used = set(first.tolist())
        sfam = sorted({SRC_FAM[sn[t]] for t in used if t < nsrc and t != 0})
        acc.append(dict(kind=tag + ":" + "+".join(fn[i] for i in idx), fams=fams, sfams=sfam, nkeys=len(uk),
                        nfeat=len(idx), preds=preds))
    acc.sort(key=lambda c: (c["nfeat"], c["nkeys"]))
    return acc


# ------------------------------------------------------------------------------------------------ same-shape, per pixel
def fit_pixels(train, test_in, max_time=3.0, singles_only=False):
    if not all(x.shape == y.shape for x, y in train):
        return []
    fn = sn = None
    tr, te = [], []
    for x, _ in train:
        fn, F, sn, S = all_views(x); tr.append((F, S))
    for x in test_in:
        fn2, F, sn2, S = all_views(x)
        if fn2 != fn or sn2 != sn:
            return []
        te.append((F, S))
    return fit_pool(tr, [y for _, y in train], te, fn, sn, max_time, singles_only, tag="px")


# ------------------------------------------------------------------------------------------------ block (downsampled) view
def _blocks(g, a, b):
    h, w = g.shape
    hb, wb = h // a, w // b
    return g[:hb * a, :wb * b].reshape(hb, a, wb, b).swapaxes(1, 2).reshape(hb, wb, a * b)


def _block_pool(g, a, b):
    from cells import bg_color
    bg = bg_color(g)
    blk = _blocks(g, a, b)
    hb, wb, _ = blk.shape
    names = ["b_nnz", "b_ncol", "b_uniform", "b_maj_is_bg", "b_full"]
    F = np.zeros((hb, wb, len(names)), int)
    maj = np.full((hb, wb), -1); majnb = np.full((hb, wb), -1); minnb = np.full((hb, wb), -1)
    for i in range(hb):
        for j in range(wb):
            v = blk[i, j]
            nb = v[v != bg]
            cnt = np.bincount(v, minlength=10)
            F[i, j] = [min(len(nb), 12), min(len(set(v.tolist())), 4), int(len(set(v.tolist())) == 1), int(cnt.argmax() == bg),
                       int(len(nb) == len(v))]
            maj[i, j] = cnt.argmax()
            if len(nb):
                c2 = np.bincount(nb, minlength=10)
                majnb[i, j] = c2.argmax()
                c2[c2 == 0] = 999
                minnb[i, j] = c2.argmin()
    S = np.stack([maj, majnb, minnb], -1)
    for n in names:
        FAM[n] = "block"
    for n in ("b_maj", "b_majnb", "b_minnb"):
        SRC_FAM[n] = "block"
    return names, F, ["b_maj", "b_majnb", "b_minnb"], S


def fit_block(train, test_in, max_time=2.0):
    """Output = downsampled view: each a x b block of the input becomes one output pixel."""
    x0, y0 = train[0]
    if y0.shape == x0.shape:
        return []
    cands = set()
    for x, y in train:
        if y.shape[0] == 0 or x.shape[0] % y.shape[0] or x.shape[1] % y.shape[1]:
            return []
        cands.add((x.shape[0] // y.shape[0], x.shape[1] // y.shape[1]))
    if len(cands) != 1:
        return []
    a, b = next(iter(cands))
    if a * b == 1:
        return []
    tr = []; te = []
    for x, _ in train:
        fn, F, sn, S = _block_pool(x, a, b); tr.append((F, S))
    for x in test_in:
        if x.shape[0] % a or x.shape[1] % b:
            return []
        fn, F, sn, S = _block_pool(x, a, b); te.append((F, S))
    return fit_pool(tr, [y for _, y in train], te, fn, sn, max_time, tag="blk", cross_only=False, ident=False)


# ------------------------------------------------------------------------------------------------ row / column projection outputs
def _axis_pool(g, axis):
    from views import view_proj, view_pix, view_hist
    F, S, fam = {}, {}, {}
    view_proj(g, F, S, fam); view_pix(g, F, S, fam); view_hist(g, F, S, fam)
    names = ["row_nz", "row_nc", "row_full", "row_max", "row_same"] if axis == 0 else \
            ["col_nz", "col_nc", "col_full", "col_max", "col_same"]
    sn = ["row_dom", "rank1", "rank2", "least"] if axis == 0 else ["col_dom", "rank1", "rank2", "least"]
    sl = (slice(None), slice(0, 1)) if axis == 0 else (slice(0, 1), slice(None))
    Fa = np.stack([F[k][sl] for k in names], -1)
    Sa = np.stack([S[k][sl] for k in sn], -1)
    for n in names:
        FAM[n] = "proj"
    return names, Fa, sn, Sa


def fit_projection_out(train, test_in, max_time=1.0):
    res = []
    for axis, shp in ((0, lambda x: (x.shape[0], 1)), (1, lambda x: (1, x.shape[1]))):
        if not all(y.shape == shp(x) for x, y in train):
            continue
        tr = []; te = []
        for x, _ in train:
            fn, F, sn, S = _axis_pool(x, axis); tr.append((F, S))
        for x in test_in:
            fn, F, sn, S = _axis_pool(x, axis); te.append((F, S))
        res += fit_pool(tr, [y for _, y in train], te, fn, sn, max_time, tag="prj", cross_only=False, singles_only=False, ident=False)
    return res


def fit_multiview(train, test_in, max_time=3.0):
    """All multi-view fitters; returns candidates dict(kind, fams, sfams, nkeys, nfeat, preds, t) sorted simplest first."""
    out = []
    for fnc in (fit_pixels, fit_block, fit_projection_out):
        t0 = time.time()
        try:
            for c in fnc(train, test_in):
                c["t"] = time.time() - t0
                out.append(c)
        except Exception:
            pass
    return out
