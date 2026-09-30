"""ARGA-style object-centric fitted cell families built from Hodel's ARC-DSL (MIT).

Families (each is FITTED: parameters chosen by exact consistency on the demo pairs; accepted only if the test inputs are
covered; programs are returned as closures prog(np_grid)->np_grid|None so they can be re-applied to other examples):
  SEL   abstraction x selector x action        (same-shape actions: recolor, cover, keep, fill box/holes, outlines, halos,
                                                shoot lines, shift/move-to-border/gravitate, in-place flips, connect)
  CROP  abstraction x selector x crop variant x post (8 orientations x upscale/downscale)
  REC   abstraction x property-key -> recolor table (counted on demos; strict key coverage)
  LINE  whole-grid line drawing (connect same-colour cells / shoot rays) and enclosed-region fill
  MACRO depth<=2 compositions of whole-grid DSL macros (mirrors, concats, halves, upscales, split-overlay) + colour map
No gradients, no learning beyond counting, no LLM.
"""
import time
import itertools
import numpy as np
from scipy import ndimage
from adapter import D, to_h, ok_grid, safe_grid

ABS = {}
for _u in (True, False):
    for _d in (True, False):
        for _b in (True, False):
            ABS["obj_%s%s%s" % ("u" if _u else "m", "d" if _d else "o", "f" if _b else "a")] = (
                lambda h, u=_u, d=_d, b=_b: D.objects(h, u, d, b))
ABS["fgpart"] = lambda h: D.fgpartition(h)
ABS["part"] = lambda h: D.partition(h)
MAXOBJ = 40


class Ctx:
    """One grid seen through one abstraction: object list + lazily computed per-object keys."""

    def __init__(self, h, name):
        self.h = h
        self.bg = D.mostcolor(h)
        self.ok = True
        self.shape = (len(h), len(h[0]))
        try:
            objs = ABS[name](h)
            self.objs = sorted(objs, key=lambda o: (D.uppermost(o), D.leftmost(o), len(o), D.color(o)))
        except Exception:
            self.objs = []
            self.ok = False
        if len(self.objs) > MAXOBJ:
            self.ok = False
        self._k = {}
        self._owner = None

    def owner(self):
        if self._owner is None:
            o = {}
            for i, ob in enumerate(self.objs):
                for _, p in ob:
                    o[p] = i
            self._owner = o
        return self._owner

    def key(self, k):
        if k not in self._k:
            self._k[k] = [KEYF[k](self, o, i) for i, o in enumerate(self.objs)]
        return self._k[k]


def _shape(o):
    return D.normalize(D.toindices(o))


def _mask(ind):
    u, l = D.uppermost(ind), D.leftmost(ind)
    hh, ww = D.height(ind) + 2, D.width(ind) + 2
    m = np.zeros((hh, ww), bool)
    for a, b in ind:
        m[a - u + 1, b - l + 1] = True
    return m, u, l


def _holes(c, o, i):
    m, _, _ = _mask(D.toindices(o))
    return int(ndimage.binary_fill_holes(m).sum() - m.sum())


def _touch(c, o, i):
    ow = c.owner()
    for _, p in o:
        for q in D.dneighbors(p):
            if q in ow and ow[q] != i:
                return 1
    return 0


def _colorcount(c, o, i):
    col = D.mostcolor(o)
    return sum(r.count(col) for r in c.h)


KEYF = {
    "size": lambda c, o, i: len(o),
    "height": lambda c, o, i: D.height(o),
    "width": lambda c, o, i: D.width(o),
    "color": lambda c, o, i: D.mostcolor(o),
    "numcolors": lambda c, o, i: D.numcolors(o),
    "top": lambda c, o, i: D.uppermost(o),
    "left": lambda c, o, i: D.leftmost(o),
    "bottom": lambda c, o, i: D.lowermost(o),
    "right": lambda c, o, i: D.rightmost(o),
    "border": lambda c, o, i: int(D.bordering(o, c.h)),
    "square": lambda c, o, i: int(D.square(o)),
    "line": lambda c, o, i: int(D.height(o) == 1 or D.width(o) == 1),
    "cgrid": _colorcount,
    "shape": lambda c, o, i: _shape(o),
    "nsc": lambda c, o, i: sum(1 for x in c.key("color") if x == c.key("color")[i]),
    "nss": lambda c, o, i: sum(1 for x in c.key("shape") if x == c.key("shape")[i]),
    "holes": _holes,
    "touch": _touch,
    "symh": lambda c, o, i: int(D.hmirror(_shape(o)) == _shape(o)),
    "symv": lambda c, o, i: int(D.vmirror(_shape(o)) == _shape(o)),
    "dens": lambda c, o, i: 100 * len(o) // max(1, D.height(o) * D.width(o)),
    "sizerank": lambda c, o, i: min(3, sorted(set(c.key("size")), reverse=True).index(len(o))),
    "sizerank_a": lambda c, o, i: min(3, sorted(set(c.key("size"))).index(len(o))),
    "toprank": lambda c, o, i: min(3, sorted(set(c.key("top"))).index(D.uppermost(o))),
    "leftrank": lambda c, o, i: min(3, sorted(set(c.key("left"))).index(D.leftmost(o))),
}
NUMK = ["size", "height", "width", "numcolors", "top", "left", "bottom", "right", "cgrid", "nsc", "nss", "holes", "dens"]
BOOLK = ["border", "square", "line", "touch", "symh", "symv"]
CONSTK = ["size", "height", "width", "numcolors", "holes", "nsc", "nss"]


# ------------------------------------------------------------------ selectors
def make_selectors(ctxs):
    """List of (name, cost, fn(ctx)->tuple of selected indices) for one abstraction over the given contexts."""
    sels = [("all", 0, lambda c: tuple(range(len(c.objs))))]

    def add(name, cost, pred):
        sels.append((name, cost, lambda c, p=pred: tuple(i for i in range(len(c.objs)) if p(c, i))))
        sels.append(("not " + name, cost + 0.5, lambda c, p=pred: tuple(i for i in range(len(c.objs)) if not p(c, i))))

    for k in BOOLK:
        add(k, 1, lambda c, i, k=k: c.key(k)[i] == 1)
    add("holes>0", 1, lambda c, i: c.key("holes")[i] > 0)
    add("uniq_color", 1, lambda c, i: c.key("nsc")[i] == 1)
    add("uniq_shape", 1, lambda c, i: c.key("nss")[i] == 1)
    for k in NUMK:
        add("max_" + k, 2, lambda c, i, k=k: c.key(k)[i] == max(c.key(k)))
        add("min_" + k, 2, lambda c, i, k=k: c.key(k)[i] == min(c.key(k)))
    cols = sorted({v for c in ctxs for v in c.key("color")})
    for v in cols:
        add("color==%d" % v, 2, lambda c, i, v=v: c.key("color")[i] == v)
    for k in CONSTK:
        vals = sorted({v for c in ctxs for v in c.key(k)})[:12]
        for v in vals:
            add("%s==%d" % (k, v), 3, lambda c, i, k=k, v=v: c.key(k)[i] == v)
            if k in ("size", "height", "width"):
                add("%s<=%d" % (k, v), 3.5, lambda c, i, k=k, v=v: c.key(k)[i] <= v)
                add("%s>=%d" % (k, v), 3.5, lambda c, i, k=k, v=v: c.key(k)[i] >= v)
    return sels


# ------------------------------------------------------------------ colour specs / actions (SAME-SHAPE)
def resolve(cs, c, obj):
    kind, val = cs
    if kind == "c":
        return val
    if val == "self":
        return D.mostcolor(obj)
    cnt = {}
    for r in c.h:
        for v in r:
            if v != c.bg:
                cnt[v] = cnt.get(v, 0) + 1
    if not cnt:
        raise ValueError("no fg")
    return max(cnt, key=cnt.get) if val == "mostfg" else min(cnt, key=cnt.get)


def _paint_each(c, S, fn, cs):
    g = c.h
    for o in S:
        g = D.underfill(g, resolve(cs, c, o), fn(o))
    return g


DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
LINEMODES = {"u": [(-1, 0)], "d": [(1, 0)], "l": [(0, -1)], "r": [(0, 1)], "h": [(0, -1), (0, 1)], "v": [(-1, 0), (1, 0)],
             "hv": DIRS8[:4], "diag": DIRS8[4:], "all8": DIRS8}


def _halo4(o):
    return D.mapply(D.dneighbors, D.toindices(o)) - D.toindices(o)


def _halo8(o):
    return D.mapply(D.neighbors, D.toindices(o)) - D.toindices(o)


def _enclosed(o):
    ind = D.toindices(o)
    m, u, l = _mask(ind)
    f = ndimage.binary_fill_holes(m) & ~m
    return frozenset((int(a) + u - 1, int(b) + l - 1) for a, b in zip(*np.nonzero(f)))


def _to_border(g, shape, o, d):
    H, W = shape
    k = 0
    while k < 40:
        nxt = D.shift(o, (d[0] * (k + 1), d[1] * (k + 1)))
        if any(not (0 <= p[0] < H and 0 <= p[1] < W) for _, p in nxt):
            break
        k += 1
    return D.move(g, o, (d[0] * k, d[1] * k))


def _flip(c, S, op):
    g = c.h
    for o in S:
        g = D.paint(D.cover(g, o), op(o))
    return g


def _gravitate(c, S):
    rest = frozenset(x for o in c.objs if o not in S for x in o)
    if not rest:
        raise ValueError("no anchor")
    g = c.h
    for o in S:
        g = D.paint(D.cover(g, o), D.shift(o, D.gravitate(o, rest)))
    return g


def same_actions(cset):
    """Return list of (name, cost, fn(ctx, S)->tuple grid)."""
    A = []
    for cs in cset:
        nm = "%s%s" % cs
        A.append(("recolor " + nm, 1, lambda c, S, cs=cs: _recolor(c, S, cs)))
        A.append(("fillbox " + nm, 2, lambda c, S, cs=cs: _paint_each(c, S, D.backdrop, cs)))
        A.append(("outbox " + nm, 2, lambda c, S, cs=cs: _paint_each(c, S, D.outbox, cs)))
        A.append(("box " + nm, 2, lambda c, S, cs=cs: _paint_each(c, S, D.box, cs)))
        A.append(("halo4 " + nm, 2, lambda c, S, cs=cs: _paint_each(c, S, _halo4, cs)))
        A.append(("halo8 " + nm, 2, lambda c, S, cs=cs: _paint_each(c, S, _halo8, cs)))
        A.append(("holes " + nm, 2, lambda c, S, cs=cs: _paint_each(c, S, _enclosed, cs)))
        A.append(("inbox " + nm, 2, lambda c, S, cs=cs: _paint_each(c, S, D.inbox, cs)))
        for lm, ds in LINEMODES.items():
            A.append(("shoot_%s %s" % (lm, nm), 3, lambda c, S, cs=cs, ds=ds: _shoot(c, S, ds, cs)))
            A.append(("rays_%s %s" % (lm, nm), 3, lambda c, S, cs=cs, ds=ds: _rays(c, S, ds, cs)))
    A.append(("cover", 1, lambda c, S: _fold(c, S, D.cover)))
    A.append(("keep", 1, lambda c, S: _keep(c, S)))
    for d in DIRS8:
        for k in (1, 2, 3):
            A.append(("shift%s*%d" % (d, k), 3, lambda c, S, d=d, k=k: _fold(c, S, lambda g, o: D.move(g, o, (d[0] * k, d[1] * k)))))
    for d in DIRS8[:4]:
        A.append(("toborder%s" % (d,), 3, lambda c, S, d=d: _fold(c, S, lambda g, o: _to_border(g, c.shape, o, d))))
    A.append(("gravitate", 3, _gravitate))
    for nm, op in (("hmir", D.hmirror), ("vmir", D.vmirror), ("dmir", D.dmirror), ("cmir", D.cmirror),
                   ("rot180", lambda o: D.hmirror(D.vmirror(o)))):
        A.append(("flip_" + nm, 3, lambda c, S, op=op: _flip(c, S, op)))
    A.append(("connect_same", 4, lambda c, S: _connect_objs(c, S)))
    return A


def _recolor(c, S, cs):
    g = c.h
    for o in S:
        g = D.paint(g, D.recolor(resolve(cs, c, o), o))
    return g


def _fold(c, S, f):
    g = c.h
    for o in S:
        g = f(g, o)
    return g


def _keep(c, S):
    g = c.h
    for o in c.objs:
        if o not in S:
            g = D.cover(g, o)
    return g


def _shoot(c, S, ds, cs):
    g = c.h
    for o in S:
        ctr = D.center(o)
        ln = frozenset().union(*[D.shoot(ctr, d) for d in ds])
        g = D.underfill(g, resolve(cs, c, o), ln)
    return g


def _rays(c, S, ds, cs):
    g = c.h
    for o in S:
        ln = frozenset()
        for _, p in o:
            for d in ds:
                ln = ln | D.shoot(p, d)
        g = D.underfill(g, resolve(cs, c, o), ln - D.toindices(o))
    return g


def _connect_objs(c, S):
    g = c.h
    for a, b in itertools.combinations(S, 2):
        if D.color(a) != D.color(b):
            continue
        g = D.underfill(g, D.color(a), D.connect(D.center(a), D.center(b)))
    return g


# ------------------------------------------------------------------ CROP
def crop_variants():
    def sub(c, S):
        return D.subgrid(S[0], c.h) if len(S) == 1 else None

    def only(c, S):
        if len(S) != 1:
            return None
        return D.subgrid(S[0], D.paint(D.canvas(c.bg, c.shape), S[0]))

    def inner(c, S):
        s = sub(c, S)
        return D.trim(s) if s is not None and len(s) > 2 and len(s[0]) > 2 else None

    def usub(c, S):
        return D.subgrid(frozenset().union(*S), c.h)

    def uonly(c, S):
        o = frozenset().union(*S)
        return D.subgrid(o, D.paint(D.canvas(c.bg, c.shape), o))

    return [("sub", 1, sub), ("only", 1, only), ("inner", 2, inner), ("usub", 2, usub), ("uonly", 2, uonly)]


T8 = [("id", lambda a: a), ("r90", lambda a: np.rot90(a, 1)), ("r180", lambda a: np.rot90(a, 2)), ("r270", lambda a: np.rot90(a, 3)),
      ("fv", lambda a: a[::-1]), ("fh", lambda a: a[:, ::-1]), ("tr", lambda a: a.T), ("atr", lambda a: np.rot90(a, 2).T)]


def posts():
    P = []
    for n, f in T8:
        for k in (1, 2, 3):
            P.append(("%s_up%d" % (n, k), lambda a, f=f, k=k: np.kron(f(a), np.ones((k, k), int)) if k > 1 else f(a)))
        for k in (2, 3):
            P.append(("%s_dn%d" % (n, k), lambda a, f=f, k=k: f(a)[::k, ::k]))
    return P


# ------------------------------------------------------------------ candidate container
class Cand:
    __slots__ = ("family", "desc", "cost", "prog", "preds")

    def __init__(self, family, desc, cost, prog, preds):
        self.family, self.desc, self.cost, self.prog, self.preds = family, desc, cost, prog, preds

    def __repr__(self):
        return "%s:%s" % (self.family, self.desc)


def eqg(a, b):
    return a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b)


def colour_candidates(train):
    """Output colours at changed cells (+ roles): the only colours a same-shape action needs to try."""
    cs = set()
    for x, y in train:
        if x.shape == y.shape:
            cs |= {int(v) for v in np.unique(y[x != y])}
    L = [("c", c) for c in sorted(cs)][:6]
    L += [("r", "self"), ("r", "mostfg"), ("r", "leastfg")]
    return L


def _mk_sel_prog(absname, selfn, actfn):
    def prog(a):
        c = Ctx(to_h(a), absname)
        if not c.ok:
            return None
        S = [c.objs[i] for i in selfn(c)]
        if not S:
            return None
        return safe_grid(lambda: actfn(c, S))
    return prog


def _mk_crop_prog(absname, selfn, cvar, post):
    def prog(a):
        c = Ctx(to_h(a), absname)
        if not c.ok:
            return None
        S = [c.objs[i] for i in selfn(c)]
        if not S:
            return None
        try:
            s = cvar(c, S)
            if s is None:
                return None
            o = post(np.array(s, dtype=int))
            return o if ok_grid(o) else None
        except Exception:
            return None
    return prog


def fam_sel_crop(train, test_in, deadline, absnames=None):
    """SEL + CROP families. Returns list of Cand."""
    cands = []
    shape_same = all(x.shape == y.shape for x, y in train)
    shape_diff = any(x.shape != y.shape for x, y in train)
    if shape_same and all(np.array_equal(x, y) for x, y in train):
        return cands
    ins_h = [to_h(x) for x, _ in train]
    tst_h = [to_h(t) for t in test_in]
    cset = colour_candidates(train)
    SA = same_actions(cset) if shape_same else []
    CV = crop_variants() if shape_diff else []
    PS = posts() if shape_diff else []
    outs = [y for _, y in train]
    for an in (absnames or ABS):
        if time.time() > deadline:
            break
        dc = [Ctx(h, an) for h in ins_h]
        tc = [Ctx(h, an) for h in tst_h]
        if not all(c.ok for c in dc + tc) or not all(len(c.objs) > 0 for c in tc):
            continue
        sels = make_selectors(dc + tc)
        seen = set()
        for sname, scost, sfn in sels:
            if time.time() > deadline:
                break
            try:
                sd = [sfn(c) for c in dc]
                st = [sfn(c) for c in tc]
            except Exception:
                continue
            if any(len(s) == 0 for s in st):
                continue
            sig = (tuple(sd), tuple(st))
            if sig in seen:
                continue
            seen.add(sig)
            if shape_same:
                for aname, acost, afn in SA:
                    ok = True
                    for c, s, y in zip(dc, sd, outs):
                        S = [c.objs[i] for i in s]
                        r = safe_grid(lambda: afn(c, S)) if S else np.array(c.h)
                        if not eqg(r, y):
                            ok = False
                            break
                    if not ok:
                        continue
                    prog = _mk_sel_prog(an, sfn, afn)
                    preds = [prog(t) for t in test_in]
                    if all(p is not None for p in preds):
                        cands.append(Cand("SEL", "%s|%s|%s" % (an, sname, aname), scost + acost + 1, prog, preds))
            if shape_diff:
                for cn, ccost, cf in CV:
                    crops = []
                    for c, s in zip(dc, sd):
                        S = [c.objs[i] for i in s]
                        try:
                            r = cf(c, S) if S else None
                        except Exception:
                            r = None
                        if r is None:
                            break
                        crops.append(np.array(r, dtype=int))
                    if len(crops) != len(dc):
                        continue
                    for pn, pf in PS:
                        try:
                            if all(eqg(pf(a), y) for a, y in zip(crops, outs)):
                                prog = _mk_crop_prog(an, sfn, cf, pf)
                                preds = [prog(t) for t in test_in]
                                if all(p is not None for p in preds):
                                    cands.append(Cand("CROP", "%s|%s|%s|%s" % (an, sname, cn, pn),
                                                      scost + ccost + 1 + (pn != "id_up1"), prog, preds))
                        except Exception:
                            pass
    return cands


# ------------------------------------------------------------------ REC: recolor-by-property tables
REC_KEYS = [("size",), ("sizerank",), ("sizerank_a",), ("height",), ("width",), ("shape",), ("border",), ("holes",), ("nsc",),
            ("nss",), ("touch",), ("toprank",), ("leftrank",), ("square",), ("line",), ("symh",), ("symv",), ("numcolors",),
            ("dens",), ("color", "size"), ("color", "border"), ("color", "holes"), ("color", "shape"), ("color", "sizerank"),
            ("color", "sizerank_a"), ("color", "nsc"), ("color", "touch"), ("size", "border"), ("border", "holes"),
            ("height", "width")]


def _rec_render(c, table, keys):
    g = [list(r) for r in c.h]
    for i, o in enumerate(c.objs):
        k = tuple(c.key(n)[i] for n in keys)
        if k not in table:
            return None
        t = table[k]
        if t == "keep":
            continue
        for _, (a, b) in o:
            g[a][b] = t
    return tuple(tuple(r) for r in g)


def fam_rec(train, test_in, deadline):
    cands = []
    if not all(x.shape == y.shape for x, y in train) or all(np.array_equal(x, y) for x, y in train):
        return cands
    ins_h = [to_h(x) for x, _ in train]
    tst_h = [to_h(t) for t in test_in]
    for an in ABS:
        if time.time() > deadline:
            break
        dc = [Ctx(h, an) for h in ins_h]
        tc = [Ctx(h, an) for h in tst_h]
        if not all(c.ok for c in dc + tc) or not all(len(c.objs) > 0 for c in tc):
            continue
        rows, bad = [], False
        for c, (x, y) in zip(dc, train):
            cov = np.zeros(x.shape, bool)
            orow = []
            for o in c.objs:
                vals = {int(y[p]) for _, p in o}
                for _, p in o:
                    cov[p] = True
                if len(vals) != 1:
                    bad = True
                    break
                orow.append(vals.pop())
            if bad or not np.array_equal(x[~cov], y[~cov]):
                bad = True
                break
            rows.append(orow)
        if bad:
            continue
        for keys in REC_KEYS:
            if time.time() > deadline:
                break
            for mode in ("keep", "abs"):
                table, ok = {}, True
                for c, orow in zip(dc, rows):
                    for i, o in enumerate(c.objs):
                        k = tuple(c.key(n)[i] for n in keys)
                        oc = orow[i]
                        same = all(v == oc for v, _ in o)
                        t = "keep" if (mode == "keep" and same) else oc
                        if table.setdefault(k, t) != t:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok or all(v == "keep" for v in table.values()):
                    continue

                def prog(a, an=an, table=table, keys=keys):
                    c = Ctx(to_h(a), an)
                    if not c.ok:
                        return None
                    return safe_grid(lambda: _rec_render(c, table, keys))
                preds = [prog(t) for t in test_in]
                if all(p is not None for p in preds):
                    cands.append(Cand("REC", "%s|%s|%s" % (an, "+".join(keys), mode), 2 + len(keys) + (mode == "abs"), prog, preds))
                break
    return cands


# ------------------------------------------------------------------ LINE: whole-grid connect / shoot / enclosed fill
def _line_prog(a_col, mode, lc, under):
    def prog(a):
        h = to_h(a)
        pts = sorted(D.ofcolor(h, a_col))
        if not pts:
            return None
        col = a_col if lc == "self" else lc
        ln = frozenset()
        if mode.startswith("conn_"):
            kind = mode[5:]
            for p, q in itertools.combinations(pts, 2):
                dy, dx = q[0] - p[0], q[1] - p[1]
                ok = {"h": dy == 0, "v": dx == 0, "hv": dy == 0 or dx == 0, "diag": abs(dy) == abs(dx),
                      "all": dy == 0 or dx == 0 or abs(dy) == abs(dx)}[kind]
                if ok:
                    ln = ln | D.connect(p, q)
        else:
            for p in pts:
                for d in LINEMODES[mode[6:]]:
                    ln = ln | D.shoot(p, d)
        f = D.underfill if under else D.fill
        return safe_grid(lambda: f(h, col, ln))
    return prog


def _holes_prog(lc, conn8):
    def prog(a):
        bg = int(np.bincount(a.ravel()).argmax())
        lab, n = ndimage.label(a == bg, structure=np.ones((3, 3)) if conn8 else None)
        edge = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
        o = a.copy()
        for i in range(1, n + 1):
            if i not in edge:
                o[lab == i] = lc
        return o
    return prog


def fam_line(train, test_in, deadline):
    cands = []
    if not all(x.shape == y.shape for x, y in train) or all(np.array_equal(x, y) for x, y in train):
        return cands
    ccols = sorted({int(v) for x, y in train for v in np.unique(y[x != y])})[:4]
    incols = sorted({int(v) for x, _ in train for v in np.unique(x)})
    lcs = ccols + ["self"]
    for lc in ccols:
        for c8 in (False, True):
            p = _holes_prog(lc, c8)
            if all(eqg(p(x), y) for x, y in train):
                cands.append(Cand("LINE", "holes%d->%d" % (c8, lc), 2, p, [p(t) for t in test_in]))
    modes = ["conn_" + k for k in ("h", "v", "hv", "diag", "all")] + ["shoot_" + k for k in LINEMODES]
    for ac in incols:
        for mode in modes:
            for lc in lcs:
                for under in (True, False):
                    if time.time() > deadline:
                        return cands
                    p = _line_prog(ac, mode, lc, under)
                    if all(eqg(p(x), y) for x, y in train):
                        preds = [p(t) for t in test_in]
                        if all(q is not None for q in preds):
                            cands.append(Cand("LINE", "%s c%d lc%s u%d" % (mode, ac, lc, under), 3, p, preds))
    return cands


# ------------------------------------------------------------------ MACRO compositions
def _part(h, sp, n, pos):
    H, W = len(h), len(h[0])
    L = W if sp is D.hsplit else H
    if L % n == 0:
        return sp(h, n)[pos]
    if (L - (n - 1)) % n == 0:          # separators of width 1
        s = (L - (n - 1)) // n
        if sp is D.hsplit:
            return tuple(r[pos * (s + 1):pos * (s + 1) + s] for r in h)
        return h[pos * (s + 1):pos * (s + 1) + s]
    return None


def _split_bool(h, sp, n, op):
    parts = [_part(h, sp, n, i) for i in range(n)]
    if any(p is None for p in parts):
        return None
    ms = [np.array(p) != 0 for p in parts]
    if len({m.shape for m in ms}) != 1:
        return None
    if op == "and":
        m = np.logical_and.reduce(ms)
    elif op == "or":
        m = np.logical_or.reduce(ms)
    elif op == "xor":
        m = (sum(x.astype(int) for x in ms) == 1)
    elif op == "nor":
        m = ~np.logical_or.reduce(ms)
    elif op == "anb":
        m = ms[0] & ~np.logical_or.reduce(ms[1:])
    else:
        m = ms[-1] & ~np.logical_or.reduce(ms[:-1])
    return tuple(tuple(int(v) for v in r) for r in m.astype(int))


def _rm_frontiers(h):
    fr = D.frontiers(h)
    if not fr:
        return None
    return D.fill(h, D.mostcolor(h), D.merge(frozenset(D.toindices(f) for f in fr)))


def _dedupe_rows(h):
    out = [h[0]]
    for r in h[1:]:
        if r != out[-1]:
            out.append(r)
    return tuple(out)


def _dedupe_cols(h):
    return tuple(zip(*_dedupe_rows(tuple(zip(*h)))))


def _macro_cells():
    hm, vm, dm, cm = D.hmirror, D.vmirror, D.dmirror, D.cmirror
    r180 = D.rot180
    M = {
        "hmirror": hm, "vmirror": vm, "dmirror": dm, "cmirror": cm, "rot90": D.rot90, "rot180": D.rot180, "rot270": D.rot270,
        "tophalf": D.tophalf, "bottomhalf": D.bottomhalf, "lefthalf": D.lefthalf, "righthalf": D.righthalf,
        "trim": D.trim, "compress": D.compress,
        "up2": lambda h: D.upscale(h, 2), "up3": lambda h: D.upscale(h, 3), "dn2": lambda h: D.downscale(h, 2),
        "dn3": lambda h: D.downscale(h, 3), "hup2": lambda h: D.hupscale(h, 2), "vup2": lambda h: D.vupscale(h, 2),
        "h_g_g": lambda h: D.hconcat(h, h), "v_g_g": lambda h: D.vconcat(h, h),
        "h_g_vm": lambda h: D.hconcat(h, vm(h)), "h_vm_g": lambda h: D.hconcat(vm(h), h),
        "v_g_hm": lambda h: D.vconcat(h, hm(h)), "v_hm_g": lambda h: D.vconcat(hm(h), h),
        "q_vm_hm": lambda h: D.vconcat(D.hconcat(h, vm(h)), D.hconcat(hm(h), r180(h))),
        "q_vm_hm_b": lambda h: D.vconcat(D.hconcat(r180(h), hm(h)), D.hconcat(vm(h), h)),
        "q_g4": lambda h: D.vconcat(D.hconcat(h, h), D.hconcat(h, h)),
        "h3_g": lambda h: D.hconcat(D.hconcat(h, h), h), "v3_g": lambda h: D.vconcat(D.vconcat(h, h), h),
        "h_g_dm": lambda h: D.hconcat(h, dm(h)) if len(h) == len(h[0]) else None,
        "frontier_rm": _rm_frontiers, "sqz_rows": _dedupe_rows, "sqz_cols": _dedupe_cols,
    }
    for n in (2, 3):
        for ax in ("h", "v"):
            sp = D.hsplit if ax == "h" else D.vsplit
            for op in ("and", "or", "xor", "nor", "anb", "bna"):
                M["split_%s%d_%s" % (ax, n, op)] = (lambda h, sp=sp, n=n, op=op: _split_bool(h, sp, n, op))
            for pos in range(n):
                M["part_%s%d_%d" % (ax, n, pos)] = (lambda h, sp=sp, n=n, pos=pos: _part(h, sp, n, pos))
    return M


_MACROS = None


def macros():
    global _MACROS
    if _MACROS is None:
        _MACROS = _macro_cells()
    return _MACROS


def _fit_cmap(preds, outs):
    m = {}
    for p, y in zip(preds, outs):
        if p.shape != y.shape:
            return None
        for a, b in zip(p.ravel(), y.ravel()):
            if m.setdefault(int(a), int(b)) != int(b):
                return None
    return m


def _apply_cmap(g, m):
    o = g.copy()
    for a, b in m.items():
        o[g == a] = b
    return o


def _run_chain(M, chain, h):
    for n in chain:
        try:
            h = M[n](h)
        except Exception:
            return None
        if not isinstance(h, tuple) or len(h) == 0 or len(h) > 30 or len(h[0]) == 0 or len(h[0]) > 30:
            return None
    return h


def fam_macro(train, test_in, deadline):
    cands = []
    M = macros()
    names = list(M)
    outs = [y for _, y in train]
    ins_h = [to_h(x) for x, _ in train]
    seen = set()
    for depth in (1, 2):
        for chain in itertools.product(names, repeat=depth):
            if time.time() > deadline:
                return cands
            cur = []
            for h in ins_h:
                r = _run_chain(M, chain, h)
                if r is None:
                    break
                cur.append(np.array(r, dtype=int))
            if len(cur) != len(ins_h):
                continue
            key = tuple(c.tobytes() + bytes(c.shape) for c in cur)
            if key in seen:
                continue
            seen.add(key)
            cm = None
            if not all(eqg(c, y) for c, y in zip(cur, outs)):
                cm = _fit_cmap(cur, outs)
                if not cm or all(a == b for a, b in cm.items()):
                    continue

            def prog(a, chain=chain, cm=cm):
                r = _run_chain(M, chain, to_h(a))
                if r is None:
                    return None
                o = np.array(r, dtype=int)
                return _apply_cmap(o, cm) if cm else o
            preds = [prog(t) for t in test_in]
            if all(p is not None for p in preds):
                cands.append(Cand("MACRO", "+".join(chain) + ("+cmap" if cm else ""), depth + (cm is not None), prog, preds))
    return cands


FAMILIES = {"SEL_CROP": fam_sel_crop, "REC": fam_rec, "LINE": fam_line, "MACRO": fam_macro}


def run_all(train, test_in, tcap=12.0, families=None):
    """Run every family with a shared per-task time budget (remaining time split evenly over remaining families)."""
    fams = families or list(FAMILIES)
    t0 = time.time()
    out = []
    for i, f in enumerate(fams):
        left = tcap - (time.time() - t0)
        if left <= 0:
            break
        try:
            out += FAMILIES[f](train, test_in, time.time() + left / (len(fams) - i))
        except Exception:
            pass
    return out
