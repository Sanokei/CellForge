"""Skill niches (head-center): generated ARC-style tasks with hidden ground truth, one family per skill.

Instance niches (tiling, sequence): one cell per pattern; the maker writes an instance bank.
General niches (association, glyph): ONE cell must solve every pattern in the niche (the cell learns the skill, not the instance).
The maker only ever sees demonstration pairs. Ground truth (`apply`) is framework knowledge used to generate outputs and composite tasks.
"""
import json
import random

import numpy as np

import skillcore as S

MAXT = 4
DIMS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3), (1, 4), (4, 1), (4, 4), (2, 4)]
FAMILIES = {
    "same": lambda i, j: 0, "flip_odd_cols": lambda i, j: 1 if j % 2 else 0, "flip_odd_rows": lambda i, j: 2 if i % 2 else 0,
    "flip_lr_odd_rows": lambda i, j: 1 if i % 2 else 0, "flip_ud_odd_cols": lambda i, j: 2 if j % 2 else 0,
    "mirror4": lambda i, j: (i % 2) * 2 + (j % 2), "mirror4_rev": lambda i, j: 3 - ((i % 2) * 2 + (j % 2)),
    "checker_rot180": lambda i, j: 3 if (i + j) % 2 else 0, "rot_cycle": lambda i, j: [0, 5, 3, 6][(i + j) % 4],
    "rot_cycle_rows": lambda i, j: [0, 5, 3, 6][i % 4], "diag_transpose": lambda i, j: 4 if i == j else 0,
}
SQUARE_ONLY = {"rot_cycle", "rot_cycle_rows", "diag_transpose"}


def onehot(n, k):
    v = [0.0] * n
    v[k] = 1.0
    return v


class Niche:
    name, template, general = "", "", False

    def patterns(self, limit, seed):
        raise NotImplementedError

    verify = False
    ndemos = 3

    def discriminates(self, pattern, task):
        """Extra generation filter: the task must not be solvable by a simpler cell than the one it is meant to teach."""
        return True

    def task(self, pattern, seed, ndemos=None):
        ndemos = ndemos or self.ndemos
        for attempt in range(300):
            rng = random.Random(seed if attempt == 0 else seed * 997 + attempt)
            pairs = []
            for _ in range(ndemos + 1):
                x = pattern["sample_input"](rng)
                pairs.append({"input": x, "output": pattern["apply"](x)})
            task = {"train": pairs[:ndemos], "test": pairs[ndemos:]}
            if not self.verify or (S.passes(self.truth_cell(pattern), task) and self.discriminates(pattern, task)):
                return task
        raise RuntimeError("no verifiable task for " + pattern["id"])

    def truth_params(self, pattern):
        return pattern["truth"]

    def truth_cell(self, pattern):
        return S.make_cell(self.template, self.truth_params(pattern), niche=self.name, pattern=pattern["id"], truth=True)


# ---------------------------------------------------------------- tiling (instance niche)
class Tiling(Niche):
    name, template, general = "tiling", "tile_orient", False

    def patterns(self, limit, seed):
        seen, out = set(), []
        for (pr, pc) in DIMS:
            for fam, f in FAMILIES.items():
                ks = tuple(tuple(f(i, j) if i < pr and j < pc else 0 for j in range(MAXT)) for i in range(MAXT))
                if (pr, pc, ks) in seen:
                    continue
                seen.add((pr, pc, ks))
                square = fam in SQUARE_ONLY or any(ks[i][j] >= 4 for i in range(pr) for j in range(pc))
                truth = {"per": onehot(MAXT, pr - 1), "pec": onehot(MAXT, pc - 1),
                         "orient": [[onehot(8, ks[i][j]) for j in range(MAXT)] for i in range(MAXT)]}
                out.append({"id": "tile-%d%dx-%s" % (pr, pc, fam), "niche": self.name, "square": square, "truth": truth,
                            "apply": (lambda x, t=truth: S.tile_orient(t, x)), "sample_input": (lambda rng, sq=square: self._grid(rng, sq))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, square):
        for _ in range(500):
            n = rng.choice([2, 3])
            shape = (n, n) if square else (rng.choice([2, 3]), rng.choice([2, 3]))
            a = np.array([[rng.choice(range(10)) for _ in range(shape[1])] for _ in range(shape[0])])
            forms = [S.OPS[k](a) for k in range(8 if shape[0] == shape[1] else 4)]
            if len({f.tobytes() + bytes(f.shape) for f in forms}) == len(forms):
                return a.tolist()
        raise RuntimeError("no asymmetric grid")


# ---------------------------------------------------------------- association (general niche, LANGUAGE)
class Association(Niche):
    name, template, general = "association", "assoc_induct", True

    def patterns(self, limit, seed):
        rng, out = random.Random(seed), []
        while len(out) < limit:
            k = rng.randint(3, 6)
            mapped = rng.sample(range(10), k)
            m = {c: (rng.choice([x for x in range(10) if x != c]) if rng.random() < .85 else c) for c in mapped}
            if len(set(m.values())) < 2 or all(m[c] == c for c in m):
                continue
            ident = [c for c in range(10) if c not in m]
            truth = {"sim": (9 * np.eye(10)).tolist(), "temp": [1.0], "keep": [0.6]}
            out.append({"id": "assoc-%02d" % len(out), "niche": self.name, "map": m, "ident": ident, "truth": truth,
                        "apply": (lambda x, m=m: [[m.get(v, v) for v in row] for row in x]), "sample_input": None})
        return out

    def task(self, pattern, seed, ndemos=3):
        rng = random.Random(seed)
        mapped = sorted(pattern["map"])

        def grid(colors):
            h, w = rng.randint(3, 6), rng.randint(3, 6)
            return [[rng.choice(colors) for _ in range(w)] for _ in range(h)]
        demos = []
        for _ in range(ndemos):
            x = grid(mapped + pattern["ident"][:3])
            demos.append({"input": x, "output": pattern["apply"](x)})
        seen = sorted({v for d in demos for row in d["input"] for v in row})
        x = grid(seen)
        return {"train": demos, "test": [{"input": x, "output": pattern["apply"](x)}]}


# ---------------------------------------------------------------- sequence (instance niche, LANGUAGE)
class Sequence(Niche):
    name, template, general = "sequence", "seq_continue", False

    def patterns(self, limit, seed):
        out = []
        for axis in range(3):
            for mask in range(10):
                truth = {"mask": onehot(10, mask), "axis": onehot(3, axis), "maxp": onehot(8, 5)}
                out.append({"id": "seq-m%d-%s" % (mask, ("rows", "cols", "grid")[axis]), "niche": self.name, "mask": mask, "axis": axis,
                            "truth": truth, "apply": (lambda x, t=truth: S.seq_continue(t, x)),
                            "sample_input": (lambda rng, mk=mask, ax=axis: self._masked(rng, mk, ax))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    def task(self, pattern, seed, ndemos=3):
        rng = random.Random(seed)
        pairs = []
        for _ in range(ndemos + 1):
            x, y = self._pair(rng, pattern["mask"], pattern["axis"])
            pairs.append({"input": x, "output": y})
        return {"train": pairs[:ndemos], "test": [pairs[ndemos]]}

    def _masked(self, rng, mask, axis):
        return self._pair(rng, mask, axis)[0]

    @staticmethod
    def _pair(rng, mask, axis):
        colors = [c for c in range(10) if c != mask]
        for _ in range(2000):
            h, w = rng.randint(4, 9), rng.randint(4, 9)
            if axis == 0:
                pc, pr = rng.randint(2, 4), 1
            elif axis == 1:
                pr, pc = rng.randint(2, 4), 1
            else:
                pr, pc = rng.randint(1, 3), rng.randint(2, 3)
            tile = [[rng.choice(colors) for _ in range(pc)] for _ in range(pr)]
            if len({v for r in tile for v in r}) < 2:
                continue
            full = np.array([[tile[i % pr][j % pc] for j in range(w)] for i in range(h)])
            known = np.array([[rng.random() > .22 for _ in range(w)] for _ in range(h)])
            x = np.where(known, full, mask)
            if (x == mask).sum() < 1:
                continue
            truth = {"mask": onehot(10, mask), "axis": onehot(3, axis), "maxp": onehot(8, 5)}
            if S.seq_continue(truth, x.tolist()) == full.tolist():
                return x.tolist(), full.tolist()
        raise RuntimeError("no sequence task")


# ---------------------------------------------------------------- glyph (general niche, OCR)
SHAPES = {"plus": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], "ell": [(0, 0), (1, 0), (1, 1)], "square": [(0, 0), (0, 1), (1, 0), (1, 1)],
          "hline": [(0, 0), (0, 1), (0, 2)], "vline": [(0, 0), (1, 0), (2, 0)], "tee": [(0, 0), (0, 1), (0, 2), (1, 1)]}
CLASS_ORDER = list(SHAPES)


class Glyph(Niche):
    name, template, general = "glyph", "glyph_recolor", True

    def patterns(self, limit, seed):
        W = np.full((6, 5, 5), -1.0)
        b = np.zeros(6)
        paint = np.zeros((6, 10))
        for k, name in enumerate(CLASS_ORDER):
            for r, c in SHAPES[name]:
                W[k, r, c] = 1.0
            b[k] = -len(SHAPES[name]) + 0.5
            paint[k, k + 1] = 1.0
        truth = {"W": W.tolist(), "b": b.tolist(), "paint": paint.tolist()}
        return [{"id": "glyph-%02d" % i, "niche": self.name, "truth": truth, "seed": seed + i,
                 "apply": (lambda x, t=truth: S.glyph_recolor(t, x)), "sample_input": None} for i in range(limit)]

    def task(self, pattern, seed, ndemos=3):
        rng = random.Random(seed * 7919 + pattern["seed"])
        pairs = []
        for _ in range(ndemos + 1):
            pairs.append(self._pair(rng))
        return {"train": pairs[:ndemos], "test": [pairs[ndemos]]}

    @staticmethod
    def _pair(rng):
        for _ in range(500):
            n = rng.randint(9, 12)
            g = np.zeros((n, n), int)
            guard = np.zeros((n, n), bool)
            color = rng.choice([7, 8, 9])
            placed = 0
            for name in rng.sample(CLASS_ORDER, rng.randint(2, 4)):
                cells = SHAPES[name]
                hh, ww = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
                for _ in range(40):
                    r0, c0 = rng.randrange(0, n - hh + 1), rng.randrange(0, n - ww + 1)
                    pts = [(r0 + r, c0 + c) for r, c in cells]
                    if any(guard[p] for p in pts):
                        continue
                    for r, c in pts:
                        g[r, c] = color
                        guard[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
                    placed += 1
                    break
            if placed >= 2:
                x = g.tolist()
                truth = Glyph().patterns(1, 0)[0]["truth"]
                return {"input": x, "output": S.glyph_recolor(truth, x)}
        raise RuntimeError("no glyph task")


def rand_grid_like(rng, h, w, colors):
    return [[rng.choice(colors) for _ in range(w)] for _ in range(h)]


# ---------------------------------------------------------------- neighbour rules (instance niche over key types, in-context)
class Neighbor(Niche):
    name, template, general, verify = "neighbor", "neighbor_table", False, True

    def discriminates(self, pattern, task):
        colour_only = S.make_cell("neighbor_table", {"keysel": onehot(6, 0), "bg": onehot(10, 0)})
        return not S.passes(colour_only, task)

    @staticmethod
    def _grid(rng):
        return rand_grid_like(rng, rng.randint(4, 7), rng.randint(4, 7), [0, 0, 0, 1, 2, 3])

    @staticmethod
    def _apply(kind, rule, x):
        a = np.asarray(x)
        out = a.copy()
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                out[r, c] = rule.get(S.nkey(kind, a, r, c, 0), a[r, c])
        return out.tolist()

    def patterns(self, limit, seed):
        out = []
        for kind in range(1, 6):
            for i in range(6):
                rng = random.Random(seed * 100 + kind * 10 + i)
                freq = {}
                for _ in range(30):
                    a = np.asarray(self._grid(rng))
                    for r in range(a.shape[0]):
                        for c in range(a.shape[1]):
                            k = S.nkey(kind, a, r, c, 0)
                            freq[k] = freq.get(k, 0) + 1
                top = sorted(freq, key=lambda k: -freq[k])[:10]
                rule = {k: rng.choice([v for v in range(10) if v != k[0]]) for k in rng.sample(top, min(4, len(top)))}
                truth = {"keysel": onehot(6, kind), "bg": onehot(10, 0)}
                out.append({"id": "nbr-k%d-%d" % (kind, i), "niche": self.name, "kind": kind, "truth": truth,
                            "apply": (lambda x, k=kind, r=rule: self._apply(k, r, x)), "sample_input": self._grid})
        random.Random(seed).shuffle(out)
        return out[:limit]


# ---------------------------------------------------------------- object properties (instance niche over properties, in-context)
BLOBS = {"dot": [(0, 0)], "bar2": [(0, 0), (0, 1)], "bar3": [(0, 0), (0, 1), (0, 2)], "tall3": [(0, 0), (1, 0), (2, 0)], "sq2": [(0, 0), (0, 1), (1, 0), (1, 1)],
         "rect23": [(r, c) for r in range(2) for c in range(3)], "ell": [(0, 0), (1, 0), (1, 1)], "plus": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
         "ring": [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)], "sq3": [(r, c) for r in range(3) for c in range(3)]}


class Object(Niche):
    name, template, general, verify = "object", "object_table", False, True

    @staticmethod
    def _grid(rng):
        n = rng.randint(9, 12)
        g = np.zeros((n, n), int)
        guard = np.zeros((n, n), bool)
        for name in rng.sample(list(BLOBS), rng.randint(3, 5)):
            cells = BLOBS[name]
            hh, ww = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
            color = rng.choice([1, 2, 3])
            for _ in range(40):
                r0, c0 = rng.randrange(0, n - hh + 1), rng.randrange(0, n - ww + 1)
                pts = [(r0 + r, c0 + c) for r, c in cells]
                if any(guard[q] for q in pts):
                    continue
                for r, c in pts:
                    g[r, c] = color
                    guard[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
                break
        return g.tolist()

    @staticmethod
    def _apply(kind, rule, x):
        a = np.asarray(x)
        out = a.copy()
        objs = S.objects(a, 0)
        for o in objs:
            v = S.obj_prop(kind, o, objs, a.shape)
            if v in rule:
                for r, c in o["cells"]:
                    out[r, c] = rule[v]
        return out.tolist()

    def patterns(self, limit, seed):
        out = []
        for kind in range(8):
            for i in range(4):
                rng = random.Random(seed * 100 + kind * 10 + i)
                freq = {}
                for _ in range(25):
                    a = np.asarray(self._grid(rng))
                    objs = S.objects(a, 0)
                    for o in objs:
                        v = S.obj_prop(kind, o, objs, a.shape)
                        freq[v] = freq.get(v, 0) + 1
                top = sorted(freq, key=lambda k: -freq[k])[:6]
                chosen = rng.sample(top, min(3, len(top)))
                rule = {v: rng.choice([5, 6, 7, 8, 9]) for v in chosen}
                truth = {"prop": onehot(8, kind), "bg": onehot(10, 0)}
                out.append({"id": "obj-p%d-%d" % (kind, i), "niche": self.name, "kind": kind, "truth": truth,
                            "apply": (lambda x, k=kind, r=rule: self._apply(k, r, x)), "sample_input": self._grid})
        random.Random(seed).shuffle(out)
        return out[:limit]


# ---------------------------------------------------------------- rays (instance niche)
DIRSETS = {"up": [0], "down": [4], "right": [2], "left": [6], "vertical": [0, 4], "horizontal": [2, 6], "orthogonal": [0, 2, 4, 6],
           "diagonal": [1, 3, 5, 7], "all8": list(range(8)), "right_down": [2, 4], "up_right": [0, 2], "down_left": [4, 6]}


class Ray(Niche):
    name, template, general = "ray", "ray_cast", False

    @staticmethod
    def _grid(rng):
        n = rng.randint(7, 11)
        g = np.zeros((n, n), int)
        for _ in range(rng.randint(1, 3)):
            g[rng.randrange(n), rng.randrange(n)] = rng.choice(range(1, 8))
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randrange(n), rng.randrange(n)
            if g[r, c] == 0:
                g[r, c] = rng.choice([8, 9])
        return g.tolist()

    def patterns(self, limit, seed):
        out = []
        for dname, ds in DIRSETS.items():
            for stop in range(3):
                for fixed in (None, 4):
                    paint = [0.0] * 10
                    if fixed is not None:
                        paint[fixed] = 1.0
                    truth = {"src": [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], "dirs": [1.0 if k in ds else 0.0 for k in range(8)], "stop": onehot(3, stop), "paint": paint}
                    out.append({"id": "ray-%s-s%d-%s" % (dname, stop, "same" if fixed is None else "c%d" % fixed), "niche": self.name, "truth": truth,
                                "apply": (lambda x, t=truth: S.ray_cast(t, x)), "sample_input": self._grid})
        random.Random(seed).shuffle(out)
        return out[:limit]


# ---------------------------------------------------------------- crop by object criterion (instance niche)
class Crop(Niche):
    name, template, general = "crop", "crop_select", False

    @staticmethod
    def _grid_for(rng, truth):
        for _ in range(500):
            n = rng.randint(10, 14)
            g = np.zeros((n, n), int)
            guard = np.zeros((n, n), bool)
            colors = rng.sample(range(1, 10), 4)
            placed = 0
            for k in range(rng.randint(2, 4)):
                hh, ww = rng.randint(1, 5), rng.randint(1, 5)
                for _ in range(40):
                    r0, c0 = rng.randrange(0, n - hh + 1), rng.randrange(0, n - ww + 1)
                    if guard[r0:r0 + hh, c0:c0 + ww].any():
                        continue
                    g[r0:r0 + hh, c0:c0 + ww] = colors[k]
                    guard[max(r0 - 1, 0):r0 + hh + 1, max(c0 - 1, 0):c0 + ww + 1] = True
                    placed += 1
                    break
            if placed >= 2:
                try:
                    S.crop_select(truth, g.tolist())
                    return g.tolist()
                except ValueError:
                    continue
        raise RuntimeError("no crop grid")

    def patterns(self, limit, seed):
        out = []
        for crit in range(8):
            for conn in range(2):
                truth = {"crit": onehot(8, crit), "bg": onehot(10, 0), "conn": onehot(2, conn)}
                out.append({"id": "crop-c%d-n%d" % (crit, conn), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.crop_select(t, x)),
                            "sample_input": (lambda rng, t=truth: self._grid_for(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]



# ---------------------------------------------------------------- language niches (token sequences as 1 x N grids)
def row_task(niche, pattern, seed, ndemos, labelled=None):
    for attempt in range(80):
        rng = random.Random(seed if attempt == 0 else seed * 997 + attempt)
        pairs = []
        for i in range(ndemos + 1):
            x = pattern["sample_input"](rng) if labelled is None else labelled(rng, i % 2)
            pairs.append({"input": x, "output": pattern["apply"](x)})
        task = {"train": pairs[:ndemos], "test": pairs[ndemos:]}
        if not niche.verify or S.passes(niche.truth_cell(pattern), task):
            return task
    raise RuntimeError("no task for " + pattern["id"])


class Recall(Niche):
    name, template, general = "recall", "recall_attend", False

    def patterns(self, limit, seed):
        out = []
        for d in (7, 8, 9):
            for order in (0, 1):
                truth = {"delim": onehot(10, d), "order": onehot(2, order)}
                out.append({"id": "recall-d%d-o%d" % (d, order), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.recall_attend(t, x)),
                            "sample_input": (lambda rng, dd=d, o=order: self._seq(rng, dd, o))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _seq(rng, d, order):
        n = rng.randint(3, 5)
        keys = rng.sample(range(1, 7), n)
        vals = rng.sample(range(1, 7), n)
        pairs = [(k, v) if order == 0 else (v, k) for k, v in zip(keys, vals)]
        q = rng.choice(keys) if order == 0 else rng.choice(vals)
        return [[t for pair in pairs for t in pair] + [d, q]]


class Induction(Niche):
    name, template, general = "induction", "induction_next", False

    def patterns(self, limit, seed):
        out = []
        for which in range(3):
            for n in range(2):
                truth = {"which": onehot(3, which), "n": onehot(2, n)}
                out.append({"id": "induct-w%d-n%d" % (which, n + 1), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.induction_next(t, x)),
                            "sample_input": (lambda rng, t=truth: self._seq(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _seq(rng, truth):
        n = int(np.argmax(truth["n"])) + 1
        for _ in range(2000):
            row = [rng.randint(1, 4) for _ in range(rng.randint(9, 13))]
            ctx = row[-n:]
            followers = [row[i + n] for i in range(len(row) - n) if row[i:i + n] == ctx]
            answers = {S.induction_next(dict(truth, which=onehot(3, w)), [row])[0][0] for w in range(3)}
            if len(set(followers)) >= 2 and len(answers) >= 2:
                return [row]
        raise RuntimeError("no induction sequence")


class SeqOps(Niche):
    name, template, general = "seqops", "seq_transform", False

    def patterns(self, limit, seed):
        out = []
        for op in range(10):
            truth = {"op": onehot(10, op)}
            out.append({"id": "seqop-%d" % op, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.seq_transform(t, x)),
                        "sample_input": (lambda rng, o=op: self._seq(rng, o))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _seq(rng, op):
        truth = {"op": onehot(10, op)}
        for _ in range(500):
            row = [rng.randint(0, 5) for _ in range(rng.randint(5, 9))]
            try:
                out = S.seq_transform(truth, [row])[0]
            except ValueError:
                continue
            if out != row:
                return [row]
        raise RuntimeError("no sequence")


class SelCopy(Niche):
    name, template, general = "selcopy", "selective_copy", False

    def patterns(self, limit, seed):
        out = []
        for noise in range(10):
            truth = {"noise": onehot(10, noise)}
            out.append({"id": "selcopy-n%d" % noise, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.selective_copy(t, x)),
                        "sample_input": (lambda rng, nz=noise: self._seq(rng, nz))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _seq(rng, noise):
        colors = [c for c in range(10) if c != noise]
        for _ in range(500):
            row = [rng.choice(colors[:5]) if rng.random() > .35 else noise for _ in range(rng.randint(6, 10))]
            if noise in row and any(v != noise for v in row):
                return [row]
        raise RuntimeError("no sequence")


class Grammar(Niche):
    name, template, general, verify = "grammar", "grammar_check", False, True

    def discriminates(self, pattern, task):
        """No OTHER language (same bracket tokens) may solve the task: otherwise the maker could not tell which language is meant."""
        for lang in range(6):
            if lang == int(np.argmax(pattern["truth"]["lang"])):
                continue
            other = S.make_cell("grammar_check", dict(pattern["truth"], lang=onehot(6, lang)))
            if S.passes(other, task):
                return False
        return True

    def patterns(self, limit, seed):
        out = []
        for lang in range(6):
            for o, c in ((1, 2), (3, 4)):
                truth = {"lang": onehot(6, lang), "open": onehot(10, o), "close": onehot(10, c)}
                out.append({"id": "grammar-l%d-%d%d" % (lang, o, c), "niche": self.name, "truth": truth, "o": o, "c": c, "lang": lang,
                            "apply": (lambda x, t=truth: S.grammar_check(t, x)), "sample_input": (lambda rng, p=truth: self._any(rng, p))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _pool(truth):
        o, c = int(np.argmax(truth["open"])), int(np.argmax(truth["close"]))
        return [o, c] if int(np.argmax(truth["lang"])) in (0, 2) else [o, c, 5, 6]

    @classmethod
    def _any(cls, rng, truth):
        pool = cls._pool(truth)
        return [[rng.choice(pool) for _ in range(rng.randint(4, 8))]]

    @classmethod
    def labelled(cls, rng, truth, want):
        for _ in range(4000):
            lang = int(np.argmax(truth["lang"]))
            o, c = int(np.argmax(truth["open"])), int(np.argmax(truth["close"]))
            if lang == 0 and want:
                depth, row = 0, []
                for _ in range(rng.randint(2, 4) * 2):
                    left = rng.randint(2, 4) * 2 - len(row)
                    if depth == 0 or (rng.random() < .5 and depth < left):
                        row.append(o)
                        depth += 1
                    else:
                        row.append(c)
                        depth -= 1
                row += [c] * depth
            elif lang == 1 and want:
                half = [rng.choice(cls._pool(truth)) for _ in range(rng.randint(2, 4))]
                row = half + ([rng.choice(cls._pool(truth))] if rng.random() < .5 else []) + half[::-1]
            elif lang == 2 and want:
                n = rng.randint(1, 4)
                row = [o] * n + [c] * n
            elif lang == 5 and want:
                row = [rng.choice(cls._pool(truth))] * rng.randint(2, 7)
            elif lang == 5 and not want and rng.random() < .7:
                half = [rng.choice(cls._pool(truth)) for _ in range(rng.randint(2, 3))]
                row = half + half[::-1]
            else:
                row = cls._any(rng, truth)[0]
            if S.grammar_check(truth, [row])[0][0] == want:
                return [row]
        raise RuntimeError("no labelled sequence")


class Successor(Niche):
    name, template, general, verify, ndemos = "successor", "successor_table", False, True, 8

    def discriminates(self, pattern, task):
        """The context length must matter: the other order may not solve the task."""
        other = 1 - int(np.argmax(pattern["truth"]["order"]))
        return not S.passes(S.make_cell("successor_table", {"order": onehot(2, other)}), task)

    def patterns(self, limit, seed):
        out = []
        for order in (1, 2):
            for i in range(3):
                rng = random.Random(seed * 100 + order * 10 + i)
                perm = rng.sample(range(1, 7), 6)
                k = rng.randint(1, 5)
                if order == 1:
                    func = (lambda last, p=perm: p[last[-1] - 1])
                else:
                    func = (lambda last, kk=k: (last[-2] * kk + last[-1]) % 6 + 1)
                truth = {"order": onehot(2, order - 1)}
                out.append({"id": "succ-o%d-%d" % (order, i), "niche": self.name, "truth": truth,
                            "apply": (lambda x, f=func: [[f(x[0])]]),
                            "sample_input": (lambda rng, : [[rng.randint(1, 6) for _ in range(rng.randint(3, 6))]])})
        random.Random(seed).shuffle(out)
        return out[:limit]


class Agree(Niche):
    name, template, general, verify, ndemos = "agree", "pair_agree", False, True, 12

    def patterns(self, limit, seed):
        out = []
        for i in range(8):
            rng = random.Random(seed * 100 + i)
            nclass = 2 if i % 2 == 0 else 3
            cls = [rng.randrange(nclass) for _ in range(10)]
            rel = rng.randrange(2)
            tensor = [[1.0 if c == j else 0.0 for j in range(3)] for c in cls]
            truth = {"cls": tensor, "rel": onehot(2, rel)}
            out.append({"id": "agree-%d" % i, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.pair_agree(t, x)),
                        "sample_input": (lambda rng: [[rng.randint(1, 8), rng.randint(1, 8)]])})
        random.Random(seed).shuffle(out)
        return out[:limit]


Grammar.task = lambda self, pattern, seed, ndemos=None: row_task(self, pattern, seed, ndemos or 8, labelled=lambda rng, want: Grammar.labelled(rng, pattern["truth"], want))


# ---------------------------------------------------------------- more ARC-driven niches
class Gravity(Niche):
    name, template, general = "gravity", "gravity", False

    def patterns(self, limit, seed):
        out = []
        for d in range(4):
            truth = {"dir": onehot(4, d), "bg": onehot(10, 0)}
            out.append({"id": "gravity-d%d" % d, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.gravity(t, x)),
                        "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        for _ in range(200):
            h, w = rng.randint(4, 8), rng.randint(4, 8)
            g = [[rng.choice([1, 2, 3, 4]) if rng.random() < .25 else 0 for _ in range(w)] for _ in range(h)]
            if S.gravity(truth, g) != g:
                return g
        raise RuntimeError("no gravity grid")


class FillEnc(Niche):
    name, template, general, verify = "fillenc", "fill_enclosed", True, True

    def patterns(self, limit, seed):
        rng = random.Random(seed)
        truth = {"bg": onehot(10, 0), "conn": onehot(2, 0)}
        out = []
        for i in range(limit):
            wall, fill = rng.sample(range(1, 10), 2)
            out.append({"id": "fillenc-%02d" % i, "niche": self.name, "truth": truth, "wall": wall, "fill": fill, "seed": seed + i,
                        "apply": (lambda x, f=fill: Niche_fill(x, f)), "sample_input": None})
        return out

    def task(self, pattern, seed, ndemos=None):
        ndemos = ndemos or 3
        for attempt in range(80):
            rng = random.Random(seed if attempt == 0 else seed * 997 + attempt)
            pairs = []
            for _ in range(ndemos + 1):
                x = self._grid(rng, pattern["wall"])
                pairs.append({"input": x, "output": Niche_fill(x, pattern["fill"])})
            task = {"train": pairs[:ndemos], "test": pairs[ndemos:]}
            if S.passes(self.truth_cell(pattern), task):
                return task
        raise RuntimeError("no fillenc task")

    @staticmethod
    def _grid(rng, wall):
        n = rng.randint(8, 11)
        g = np.zeros((n, n), int)
        for _ in range(rng.randint(1, 2)):
            hh, ww = rng.randint(3, 6), rng.randint(3, 6)
            r0, c0 = rng.randrange(0, n - hh + 1), rng.randrange(0, n - ww + 1)
            g[r0, c0:c0 + ww] = wall
            g[r0 + hh - 1, c0:c0 + ww] = wall
            g[r0:r0 + hh, c0] = wall
            g[r0:r0 + hh, c0 + ww - 1] = wall
        for _ in range(rng.randint(0, 3)):
            r, c = rng.randrange(n), rng.randrange(n)
            if g[r, c] == 0:
                g[r, c] = wall
        return g.tolist()


def Niche_fill(x, fill):
    """Ground truth for enclosed-region fill (4-connected background flood from the border)."""
    a = np.asarray(x)
    h, w = a.shape
    reach = np.zeros((h, w), bool)
    stack = [(r, c) for r in range(h) for c in range(w) if (r in (0, h - 1) or c in (0, w - 1)) and a[r, c] == 0]
    for r, c in stack:
        reach[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not reach[rr, cc] and a[rr, cc] == 0:
                reach[rr, cc] = True
                stack.append((rr, cc))
    out = a.copy()
    out[(a == 0) & ~reach] = fill
    return out.tolist()


class Symmetry(Niche):
    name, template, general = "symmetry", "symmetry_complete", False

    def patterns(self, limit, seed):
        out = []
        for sym in range(4):
            for m in range(6):
                truth = {"mask": onehot(10, m), "sym": onehot(4, sym)}
                out.append({"id": "sym-s%d-m%d" % (sym, m), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.symmetry_complete(t, x)),
                            "sample_input": (lambda rng, mm=m, ss=sym: self._pair(rng, mm, ss)[0])})
        random.Random(seed).shuffle(out)
        return out[:limit]

    def task(self, pattern, seed, ndemos=None):
        rng = random.Random(seed)
        pairs = []
        for _ in range((ndemos or 3) + 1):
            x, y = self._pair(rng, int(np.argmax(pattern["truth"]["mask"])), int(np.argmax(pattern["truth"]["sym"])))
            pairs.append({"input": x, "output": y})
        n = ndemos or 3
        return {"train": pairs[:n], "test": pairs[n:]}

    @staticmethod
    def _pair(rng, m, sym):
        colors = [c for c in range(10) if c != m]
        truth = {"mask": onehot(10, m), "sym": onehot(4, sym)}
        for _ in range(2000):
            n = rng.randint(6, 10)
            h, w = (n, n) if sym == 3 else (rng.randint(6, 10), rng.randint(6, 10))
            base = np.array([[rng.choice(colors[:5]) for _ in range(w)] for _ in range(h)])
            if sym == 0:
                full = np.concatenate([base[:, :w // 2 + w % 2], base[:, :w // 2][:, ::-1]], axis=1)
            elif sym == 1:
                full = np.concatenate([base[:h // 2 + h % 2], base[:h // 2][::-1]], axis=0)
            elif sym == 2:
                top = np.concatenate([base[:h // 2 + h % 2, :w // 2 + w % 2], base[:h // 2 + h % 2, :w // 2][:, ::-1]], axis=1)
                full = np.concatenate([top, top[:h // 2][::-1]], axis=0)
            else:
                full = np.triu(base) + np.triu(base, 1).T
            hh, ww = rng.randint(1, 3), rng.randint(1, 3)
            r0, c0 = rng.randrange(0, h - hh + 1), rng.randrange(0, w - ww + 1)
            x = full.copy()
            x[r0:r0 + hh, c0:c0 + ww] = m
            try:
                if S.symmetry_complete(truth, x.tolist()) == full.tolist() and (x != full).any():
                    return x.tolist(), full.tolist()
            except ValueError:
                continue
        raise RuntimeError("no symmetry grid")


class Scale(Niche):
    name, template, general = "scale", "scale_grid", False

    def patterns(self, limit, seed):
        out = []
        for mode in (0, 1, 2, 3, 4, 5, 6, 7):
            truth = {"mode": onehot(8, mode)}
            out.append({"id": "scale-%d" % mode, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.scale_grid(t, x)),
                        "sample_input": (lambda rng, md=mode: self._grid(rng, md))})
        return out[:limit]

    @staticmethod
    def _grid(rng, mode):
        if mode in (3, 4):
            f = 2 if mode == 3 else 3
            h, w = f * rng.randint(2, 4), f * rng.randint(2, 4)
        else:
            h, w = rng.randint(2, 4), rng.randint(2, 4)
        return [[rng.randint(0, 5) for _ in range(w)] for _ in range(h)]


class HalfBool(Niche):
    name, template, general = "halfbool", "half_boolean", False

    def patterns(self, limit, seed):
        out = []
        for axis in range(2):
            for op in range(5):
                for sep in (5, 10):
                    truth = {"axis": onehot(2, axis), "sep": onehot(11, sep), "op": onehot(5, op), "outc": onehot(10, 3)}
                    out.append({"id": "halfbool-a%d-o%d-%s" % (axis, op, "line" if sep == 5 else "mid"), "niche": self.name, "truth": truth,
                                "apply": (lambda x, t=truth: S.half_boolean(t, x)), "sample_input": (lambda rng, a=axis, s=sep: self._grid(rng, a, s))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, axis, sep):
        h, w = rng.randint(3, 5), rng.randint(3, 5)
        left = np.array([[1 if rng.random() < .5 else 0 for _ in range(w)] for _ in range(h)])
        right = np.array([[2 if rng.random() < .5 else 0 for _ in range(w)] for _ in range(h)])
        parts = [left] + ([np.full((h, 1), 5)] if sep == 5 else []) + [right]
        g = np.concatenate(parts, axis=1)
        return (g.T if axis == 1 else g).tolist()


class Denoise(Niche):
    name, template, general = "denoise", "denoise_small", False

    def patterns(self, limit, seed):
        out = []
        for conn in range(2):
            for k in range(3):
                truth = {"bg": onehot(10, 0), "conn": onehot(2, conn), "maxnoise": onehot(4, k)}
                out.append({"id": "denoise-n%d-k%d" % (conn, k + 1), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.denoise_small(t, x)),
                            "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        k = int(np.argmax(truth["maxnoise"])) + 1
        for _ in range(500):
            n = rng.randint(8, 11)
            g = np.zeros((n, n), int)
            guard = np.zeros((n, n), bool)
            for _ in range(rng.randint(1, 2)):
                hh, ww = rng.randint(3, 4), rng.randint(3, 4)
                r0, c0 = rng.randrange(0, n - hh + 1), rng.randrange(0, n - ww + 1)
                if guard[r0:r0 + hh, c0:c0 + ww].any():
                    continue
                g[r0:r0 + hh, c0:c0 + ww] = rng.choice([1, 2, 3])
                guard[max(r0 - 1, 0):r0 + hh + 1, max(c0 - 1, 0):c0 + ww + 1] = True
            placed = 0
            for _ in range(rng.randint(2, 4)):
                size = rng.randint(1, k)
                r, c = rng.randrange(0, n), rng.randrange(0, n - size + 1)
                cells = [(r, c + i) for i in range(size)]
                if any(guard[q] for q in cells):
                    continue
                color = rng.choice([4, 5])
                for q in cells:
                    g[q] = color
                    guard[max(q[0] - 1, 0):q[0] + 2, max(q[1] - 1, 0):q[1] + 2] = True
                placed += 1
            if placed and (S.denoise_small(truth, g.tolist()) != g.tolist()):
                return g.tolist()
        raise RuntimeError("no denoise grid")



# ---------------------------------------------------------------- structures requested by declined mutations
class ObjMove(Niche):
    name, template, general = "objmove", "object_move", False

    def patterns(self, limit, seed):
        out = []
        for d in (0, 2, 4, 6, 1, 3, 5, 7):
            for stop in (0, 1):
                if d in (1, 3, 5, 7) and stop == 1:
                    continue
                truth = {"dir": onehot(8, d), "stop": onehot(2, stop), "bg": onehot(10, 0)}
                out.append({"id": "objmove-d%d-s%d" % (d, stop), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.object_move(t, x)),
                            "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        for _ in range(500):
            n = rng.randint(6, 9)
            g = np.zeros((n, n), int)
            guard = np.zeros((n, n), bool)
            for name in rng.sample(list(BLOBS)[:8], rng.randint(2, 3)):
                cells = BLOBS[name]
                hh, ww = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
                color = rng.choice([1, 2, 3, 4, 6])
                for _ in range(40):
                    r0, c0 = rng.randrange(0, n - hh + 1), rng.randrange(0, n - ww + 1)
                    pts = [(r0 + r, c0 + c) for r, c in cells]
                    if any(guard[q] for q in pts):
                        continue
                    for r, c in pts:
                        g[r, c] = color
                        guard[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
                    break
            if S.object_move(truth, g.tolist()) != g.tolist() and (g > 0).sum() >= 3:
                return g.tolist()
        raise RuntimeError("no move grid")


class FloodM(Niche):
    name, template, general = "floodm", "flood_marker", False

    def patterns(self, limit, seed):
        out = []
        for m in (1, 2, 3):
            for conn in (0, 1):
                for fixed in (None, 4):
                    fill = [0.0] * 10
                    if fixed is not None:
                        fill[fixed] = 1.0
                    truth = {"marker": onehot(10, m), "conn": onehot(2, conn), "fill": fill}
                    out.append({"id": "floodm-m%d-n%d-%s" % (m, conn, "same" if fixed is None else "c4"), "niche": self.name, "truth": truth,
                                "apply": (lambda x, t=truth: S.flood_marker(t, x)), "sample_input": (lambda rng, t=truth, mm=m: self._grid(rng, t, mm))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth, m):
        for _ in range(500):
            n = rng.randint(7, 10)
            g = np.zeros((n, n), int)
            for _ in range(rng.randint(2, 4)):
                if rng.random() < .5:
                    g[rng.randrange(n), :] = 8
                else:
                    g[:, rng.randrange(n)] = 8
            for _ in range(rng.randint(0, 3)):
                g[rng.randrange(n), rng.randrange(n)] = 8
            zeros = list(zip(*np.where(g == 0)))
            if not zeros:
                continue
            r, c = rng.choice(zeros)
            g[r, c] = m
            if S.flood_marker(truth, g.tolist()) != g.tolist():
                return g.tolist()
        raise RuntimeError("no flood grid")


class RayCross(Niche):
    name, template, general = "raycross", "ray_cross", False

    def patterns(self, limit, seed):
        out = []
        for dname in ("vertical", "horizontal", "orthogonal", "diagonal", "right_down", "up_right"):
            for stop in (0, 2):
                for cross in (2, 6):
                    truth = {"src": [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], "dirs": [1.0 if k in DIRSETS[dname] else 0.0 for k in range(8)], "stop": onehot(3, stop),
                             "paint": [0.0] * 10, "cross": onehot(10, cross)}
                    out.append({"id": "raycross-%s-s%d-x%d" % (dname, stop, cross), "niche": self.name, "truth": truth,
                                "apply": (lambda x, t=truth: S.ray_cross(t, x)), "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        for _ in range(500):
            n = rng.randint(7, 10)
            g = np.zeros((n, n), int)
            for _ in range(rng.randint(2, 3)):
                g[rng.randrange(n), rng.randrange(n)] = rng.choice([1, 2, 3, 4])
            for _ in range(rng.randint(0, 2)):
                r, c = rng.randrange(n), rng.randrange(n)
                if g[r, c] == 0:
                    g[r, c] = 8
            out = np.asarray(S.ray_cross(truth, g.tolist()))
            plain = np.asarray(S.ray_cross(dict(truth, cross=[0.0] * 10), g.tolist()))
            if (out != plain).any():
                return g.tolist()
        raise RuntimeError("no cross grid")


class Context2(Niche):
    name, template, general, verify, ndemos = "context2", "context_table2", False, True, 10

    @staticmethod
    def _grid(rng):
        return rand_grid_like(rng, rng.randint(4, 7), rng.randint(4, 7), [0, 0, 0, 1, 2, 3])

    def discriminates(self, pattern, task):
        colour_only = S.make_cell("neighbor_table", {"keysel": onehot(6, 0), "bg": onehot(10, 0)})
        return not S.passes(colour_only, task)

    @staticmethod
    def _apply(kind, rule, x):
        a = np.asarray(x)
        out = a.copy()
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                out[r, c] = rule.get(S.nkey2(kind, a, r, c, 0), a[r, c])
        return out.tolist()

    def patterns(self, limit, seed):
        out = []
        for kind in range(8):
            for i in range(4):
                rng = random.Random(seed * 100 + kind * 10 + i)
                freq = {}
                for _ in range(30):
                    a = np.asarray(self._grid(rng))
                    for r in range(a.shape[0]):
                        for c in range(a.shape[1]):
                            k = S.nkey2(kind, a, r, c, 0)
                            freq[k] = freq.get(k, 0) + 1
                top = sorted(freq, key=lambda k: -freq[k])[:10]
                own = (lambda k, kd=kind: k[4] if kd == 0 else k[0])
                rule = {k: rng.choice([v for v in range(10) if v != own(k)]) for k in rng.sample(top, min(4, len(top)))}
                truth = {"keysel": onehot(8, kind), "bg": onehot(10, 0)}
                out.append({"id": "ctx-k%d-%d" % (kind, i), "niche": self.name, "kind": kind, "truth": truth,
                            "apply": (lambda x, k=kind, r=rule: self._apply(k, r, x)), "sample_input": self._grid})
        random.Random(seed).shuffle(out)
        return out[:limit]



# ---------------------------------------------------------------- round-2 structures
class ObjMove2(Niche):
    name, template, general, verify = "objmove2", "object_move2", False, True

    def discriminates(self, pattern, task):
        t = pattern["truth"]
        other = dict(t, conn=onehot(2, 1 - int(np.argmax(t["conn"]))))
        return not S.passes(S.make_cell("object_move2", other), task)

    def patterns(self, limit, seed):
        out = []
        for d in (0, 2, 4, 6, 1, 3, 5, 7):
            for stop in (0, 1):
                for conn in (0, 1):
                    if d in (1, 3, 5, 7) and stop == 1:
                        continue
                    truth = {"dir": onehot(8, d), "stop": onehot(2, stop), "bg": onehot(10, 0), "conn": onehot(2, conn)}
                    out.append({"id": "objmove2-d%d-s%d-c%d" % (d, stop, conn), "niche": self.name, "truth": truth,
                                "apply": (lambda x, t=truth: S.object_move2(t, x)), "sample_input": self._grid})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng):
        n = rng.randint(6, 8)
        return [[rng.choice([1, 2, 3]) if rng.random() < .22 else 0 for _ in range(n)] for _ in range(n)]


class FloodThru(Niche):
    name, template, general = "floodthru", "flood_through", False

    def patterns(self, limit, seed):
        out = []
        for m in (1, 2, 9):
            for thr in (0, 8, 5):
                for conn in (0, 1):
                    through = [0.0] * 10
                    through[thr] = 1.0
                    truth = {"marker": onehot(10, m), "through": through, "conn": onehot(2, conn), "fill": [0.0] * 10}
                    out.append({"id": "floodthru-m%d-t%d-n%d" % (m, thr, conn), "niche": self.name, "truth": truth,
                                "apply": (lambda x, t=truth: S.flood_through(t, x)), "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        m, thr = int(np.argmax(truth["marker"])), int(np.argmax(truth["through"]))
        for _ in range(800):
            n = rng.randint(7, 10)
            g = np.zeros((n, n), int)
            if thr != 0:
                for _ in range(rng.randint(2, 3)):
                    r, c = rng.randrange(n), rng.randrange(n)
                    for _ in range(rng.randint(4, 9)):
                        g[r, c] = thr
                        dr, dc = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                        r, c = min(max(r + dr, 0), n - 1), min(max(c + dc, 0), n - 1)
            else:
                for _ in range(rng.randint(2, 4)):
                    if rng.random() < .5:
                        g[rng.randrange(n), :] = 8
                    else:
                        g[:, rng.randrange(n)] = 8
            spots = list(zip(*np.where(g == 0)))
            if not spots:
                continue
            r, c = rng.choice(spots)
            g[r, c] = m
            if S.flood_through(truth, g.tolist()) != g.tolist():
                return g.tolist()
        raise RuntimeError("no flood-through grid")


class SnapLine(Niche):
    name, template, general = "snapline", "snap_to_line", False

    def patterns(self, limit, seed):
        out = []
        for orphan in (0, 1):
            for i in range(2):
                truth = {"bg": onehot(10, 0), "orphan": onehot(2, orphan)}
                out.append({"id": "snapline-o%d-%d" % (orphan, i), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.snap_to_line(t, x)),
                            "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        for _ in range(800):
            h, w = rng.randint(10, 14), rng.randint(10, 14)
            g = np.zeros((h, w), int)
            colors = rng.sample([1, 2, 3, 6], 2)
            lines = []
            if rng.random() < .5:
                for k, col in enumerate(colors):
                    r = (3 + 5 * k) % h if k else 3
                    g[r, :] = col
                    lines.append(("r", r, col))
            else:
                for k, col in enumerate(colors):
                    c = 3 + 5 * k
                    if c < w:
                        g[:, c] = col
                        lines.append(("c", c, col))
            if not lines:
                continue
            pool = [l[2] for l in lines] + [4]
            for _ in range(rng.randint(2, 4)):
                r, c = rng.randrange(h), rng.randrange(w)
                if g[r, c] == 0 and all(not (l[0] == "r" and abs(r - l[1]) <= 1) and not (l[0] == "c" and abs(c - l[1]) <= 1) for l in lines):
                    g[r, c] = rng.choice(pool)
            if (g > 0).sum() > sum(h if l[0] == "c" else w for l in lines) and S.snap_to_line(truth, g.tolist()) != g.tolist():
                return g.tolist()
        raise RuntimeError("no snap grid")



# ---------------------------------------------------------------- OCR niches
FONT = {0: ["111", "101", "101", "101", "111"], 1: ["010", "110", "010", "010", "111"], 2: ["111", "001", "111", "100", "111"],
        3: ["111", "001", "111", "001", "111"], 4: ["101", "101", "111", "001", "001"], 5: ["111", "100", "111", "001", "111"],
        6: ["111", "100", "111", "101", "111"], 7: ["111", "001", "010", "010", "010"], 8: ["111", "101", "111", "101", "111"],
        9: ["111", "101", "111", "001", "111"]}


def font_truth(mode):
    W = np.full((10, 5, 5), -1.0)
    b = np.zeros(10)
    for k, rows in FONT.items():
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch == "1":
                    W[k, r, c] = 1.0
        b[k] = -sum(line.count("1") for line in rows)
    return {"W": W.tolist(), "b": b.tolist(), "mode": onehot(2, mode)}


def draw_glyph(g, digit, r0, c0, color, rng, noise):
    cells = [(r0 + r, c0 + c) for r, line in enumerate(FONT[digit]) for c, ch in enumerate(line) if ch == "1"]
    for r, c in cells:
        g[r, c] = color
    for _ in range(noise):  # flip one pixel inside the glyph box, never changing its bounding box
        r, c = rng.randrange(1, 4), rng.randrange(0, 3)
        g[r0 + r, c0 + c] = 0 if g[r0 + r, c0 + c] else color


class OCRDigit(Niche):
    name, template, general, verify, ndemos = "ocrdigit", "glyph_read", True, True, 10

    def patterns(self, limit, seed):
        truth = font_truth(0)
        out = []
        for i in range(limit):
            out.append({"id": "ocrdigit-%02d" % i, "niche": self.name, "truth": truth, "seed": seed + i, "color": 1 + (seed + i) % 9, "noise": (i // 3) % 2,
                        "apply": (lambda x, t=truth: S.glyph_read(t, x)), "sample_input": None})
        return out

    def task(self, pattern, seed, ndemos=None):
        for attempt in range(300):
            rng = random.Random(seed * 31 + pattern["seed"] + attempt * 7919)
            digits = list(range(10))
            rng.shuffle(digits)
            pairs = []
            for d in digits + [rng.randrange(10)]:
                n = rng.randint(7, 9)
                g = np.zeros((n, n), int)
                draw_glyph(g, d, rng.randint(0, n - 5), rng.randint(0, n - 3), pattern["color"], rng, pattern["noise"])
                x = g.tolist()
                try:
                    y = pattern["apply"](x)
                except ValueError:
                    break
                pairs.append({"input": x, "output": y})
            else:
                task = {"train": pairs[:10], "test": pairs[10:]}
                if all(p["output"] == [[d]] for p, d in zip(pairs, digits + [pairs[10]["output"][0][0]])):
                    return task
        raise RuntimeError("no ocrdigit task")


class OCRSeq(Niche):
    name, template, general, verify, ndemos = "ocrseq", "glyph_read", True, True, 8

    def patterns(self, limit, seed):
        truth = font_truth(1)
        return [{"id": "ocrseq-%02d" % i, "niche": self.name, "truth": truth, "seed": seed + i, "color": 1 + (seed + i) % 9, "noise": 0,  # one flipped pixel makes 0/8 ambiguous
                 "apply": (lambda x, t=truth: S.glyph_read(t, x)), "sample_input": None} for i in range(limit)]

    def task(self, pattern, seed, ndemos=None):
        for attempt in range(300):
            rng = random.Random(seed * 37 + pattern["seed"] + attempt * 104729)
            pairs, ok = [], True
            order = list(range(10))
            rng.shuffle(order)
            for i in range(9):
                k = rng.randint(2, 4)
                digits = order[(i * 3) % 10:(i * 3) % 10 + 3][:k]
                while len(digits) < k:
                    digits.append(rng.randrange(10))
                w = k * 4 + 1
                g = np.zeros((7, w), int)
                for j, d in enumerate(digits):
                    draw_glyph(g, d, 1, 1 + 4 * j, pattern["color"], rng, pattern["noise"])
                x = g.tolist()
                try:
                    y = pattern["apply"](x)
                except ValueError:
                    ok = False
                    break
                if y != [digits]:
                    ok = False
                    break
                pairs.append({"input": x, "output": y})
            if ok:
                return {"train": pairs[:8], "test": pairs[8:]}
        raise RuntimeError("no ocrseq task")


class Legend(Niche):
    name, template, general = "legend", "legend_map", False

    def patterns(self, limit, seed):
        out = []
        for layout in (0, 1):
            truth = {"bg": onehot(10, 0), "layout": onehot(2, layout)}
            out.append({"id": "legend-l%d" % layout, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.legend_map(t, x)),
                        "sample_input": (lambda rng, lo=layout: self._grid(rng, lo))})
        return out[:limit]

    @staticmethod
    def _grid(rng, layout):
        for _ in range(200):
            h, w = rng.randint(7, 10), rng.randint(7, 10)
            k = rng.randint(2, 3)
            srcs = rng.sample(range(1, 9), k)
            dsts = [rng.choice([c for c in range(1, 10) if c != s]) for s in srcs]
            g = np.zeros((h, w), int)
            for i, (s, d) in enumerate(zip(srcs, dsts)):
                g[0, 2 * i], g[0, 2 * i + 1] = s, d
            pool = srcs + [9]
            for r in range(2, h):
                for c in range(w):
                    if rng.random() < .4:
                        g[r, c] = rng.choice(pool)
            if layout == 1:
                g = g.T
            truth = {"bg": onehot(10, 0), "layout": onehot(2, layout)}
            try:
                if S.legend_map(truth, g.tolist()) != g.tolist():
                    return g.tolist()
            except ValueError:
                continue
        raise RuntimeError("no legend grid")



# ---------------------------------------------------------------- lexical language niches
WORD_CLASS = {}
for _cid, _words in {0: [0, 1, 2], 1: [19, 20, 21, 22, 23], 2: [3, 4, 5, 6], 3: [7, 8, 9, 10], 4: [11, 12, 13, 14], 5: [15, 16, 17, 18]}.items():
    for _w in _words:
        WORD_CLASS[_w] = _cid
CLASS_WORDS = {c: [w for w, k in WORD_CLASS.items() if k == c] for c in range(6)}
GRAMMARS = {
    "np-vp-agree": {"T": {0: [1, 2, 3], 1: [1, 2, 3], 2: [4], 3: [5]}, "start": [0, 1, 2, 3], "end": [4, 5]},
    "transitive": {"T": {0: [2, 3], 2: [4], 3: [5], 4: [0], 5: [0]}, "start": [0], "end": [2, 3]},
    "noun-phrase": {"T": {1: [1, 2, 3]}, "start": [1], "end": [2, 3]},
    "imperative": {"T": {5: [0], 0: [2, 3]}, "start": [5], "end": [2, 3]},
}


def grammar_truth(spec):
    cls = [[1.0 if WORD_CLASS[w] == c else 0.0 for c in range(6)] for w in range(24)]
    trans = [[1.0 if q in spec["T"].get(p, []) else 0.0 for q in range(6)] for p in range(6)]
    return {"cls": cls, "trans": trans, "start": [1.0 if c in spec["start"] else 0.0 for c in range(6)], "end": [1.0 if c in spec["end"] else 0.0 for c in range(6)]}


def accepts(spec, classes):
    return bool(classes) and classes[0] in spec["start"] and classes[-1] in spec["end"] and all(classes[i + 1] in spec["T"].get(classes[i], []) for i in range(len(classes) - 1))


class Grammar2(Niche):
    name, template, general, verify, ndemos = "grammar2", "grammar_net", False, True, 16

    def patterns(self, limit, seed):
        out = []
        for gname, spec in GRAMMARS.items():
            truth = grammar_truth(spec)
            out.append({"id": "grammar2-" + gname, "niche": self.name, "spec": spec, "truth": truth, "apply": (lambda x, t=truth: S.grammar_net(t, x)),
                        "sample_input": (lambda rng, sp=spec: self._sentence(rng, sp, rng.random() < .5))})
        return out[:limit]

    @staticmethod
    def _positive(rng, spec):
        for _ in range(500):
            c = rng.choice(spec["start"])
            seq = [c]
            while len(seq) < 7:
                if c in spec["end"] and rng.random() < .5:
                    break
                options = spec["T"].get(c, [])
                if not options:
                    break
                c = rng.choice(options)
                seq.append(c)
            if accepts(spec, seq):
                return seq
        raise RuntimeError("no positive sentence")

    @classmethod
    def _sentence(cls, rng, spec, positive):
        if positive:
            classes = cls._positive(rng, spec)
        else:
            for _ in range(2000):
                base = cls._positive(rng, spec)
                kind = rng.randrange(3)
                if kind == 0 and len(base) >= 2:
                    i = rng.randrange(len(base) - 1)
                    base[i], base[i + 1] = base[i + 1], base[i]
                elif kind == 1:
                    base[rng.randrange(len(base))] = rng.randrange(6)
                else:
                    base = base[:-1] if len(base) > 1 else base + [rng.randrange(6)]
                if base and not accepts(spec, base):
                    classes = base
                    break
            else:
                raise RuntimeError("no negative sentence")
        return [[rng.choice(CLASS_WORDS[c]) for c in classes]]

    def task(self, pattern, seed, ndemos=None):
        n = ndemos or self.ndemos
        for attempt in range(300):
            rng = random.Random(seed if attempt == 0 else seed * 997 + attempt)
            pairs = []
            for i in range(n + 1):
                x = self._sentence(rng, pattern["spec"], i % 2 == 0)
                pairs.append({"input": x, "output": pattern["apply"](x)})
            task = {"train": pairs[:n], "test": pairs[n:]}
            if S.passes(self.truth_cell(pattern), task):
                return task
        raise RuntimeError("no grammar2 task")


class Analogy(Niche):
    name, template, general, verify, ndemos = "analogy", "analogy_embed", True, True, 8

    @staticmethod
    def _truth():
        E = np.full((24, 4), 0.0)
        E[:, :3] = -5.0
        for fam, base in (("noun", 3), ("verb", 11)):
            for i in range(4):
                for plural in (0, 1):
                    w = base + 4 * plural + i
                    E[w] = [1 if fam == "noun" else 0, 1 if fam == "verb" else 0, plural, i]
        E[:, 3] = [E[w, 3] if 3 <= w <= 18 else float(w) for w in range(24)]
        return {"E": E.tolist()}

    def patterns(self, limit, seed):
        truth = self._truth()
        return [{"id": "analogy-%02d" % i, "niche": self.name, "truth": truth, "seed": seed + i, "apply": (lambda x, t=truth: S.analogy_embed(t, x)),
                 "sample_input": None} for i in range(limit)]

    @staticmethod
    def _tuple(rng):
        base = rng.choice([3, 11])
        p1, p2 = rng.sample(range(4), 2)
        x = rng.randrange(2)
        a, b = base + 4 * x + p1, base + 4 * (1 - x) + p1
        c = base + 4 * x + p2  # c has the same form as a, so the relation a -> b applies to it
        d = base + 4 * (1 - x) + p2
        return [[a, b, c]], [[d]]

    def task(self, pattern, seed, ndemos=None):
        n = ndemos or self.ndemos
        rng = random.Random(seed * 53 + pattern["seed"])
        pairs = []
        for _ in range(n + 1):
            x, y = self._tuple(rng)
            pairs.append({"input": x, "output": y})
        return {"train": pairs[:n], "test": pairs[n:]}


class ToolUse(Niche):
    name, template, general, verify, ndemos = "tooluse", "tool_dispatch", True, True, 12

    def patterns(self, limit, seed):
        truth = {"tools": [1.0] * 10}
        return [{"id": "tooluse-%02d" % i, "niche": self.name, "truth": truth, "seed": seed + i, "apply": None, "sample_input": None} for i in range(limit)]

    def task(self, pattern, seed, ndemos=None):
        n = ndemos or self.ndemos
        for attempt in range(300):
            rng = random.Random(seed * 59 + pattern["seed"] + attempt * 6007)
            names = rng.sample(range(1, 10), 4)
            ops = rng.sample(range(10), 4)
            binding = dict(zip(names, ops))
            pairs = []
            for i in range(n + 1):
                name = names[i % 4] if i < n else rng.choice(names)
                args = [rng.randint(0, 9) for _ in range(rng.randint(3, 6))]
                pairs.append({"input": [[name] + args], "output": [S.TOOLS[binding[name]](args)]})
            task = {"train": pairs[:n], "test": pairs[n:]}
            if S.passes(self.truth_cell(pattern), task):
                return task
        raise RuntimeError("no tooluse task")



# ---------------------------------------------------------------- niches for the structures above
class LineConn(Niche):
    name, template, general = "lineconn", "line_connect", False

    def patterns(self, limit, seed):
        out = []
        for axis in range(3):
            for fixed in (None, 4):
                for over in (0, 1):
                    paint = [0.0] * 10
                    if fixed is not None:
                        paint[fixed] = 1.0
                    truth = {"bg": onehot(10, 0), "axis": onehot(3, axis), "over": onehot(2, over), "paint": paint}
                    out.append({"id": "lineconn-a%d-%s-o%d" % (axis, "same" if fixed is None else "c4", over), "niche": self.name, "truth": truth,
                                "apply": (lambda x, t=truth: S.line_connect(t, x)), "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        axis = int(np.argmax(truth["axis"]))
        for _ in range(800):
            h, w = rng.randint(8, 12), rng.randint(8, 12)
            g = np.zeros((h, w), int)
            used_r, used_c = set(), set()
            for color in rng.sample([1, 2, 3, 6], rng.randint(2, 3)):
                horiz = axis == 0 or (axis == 2 and rng.random() < .5)
                if horiz:
                    r = rng.choice([x for x in range(h) if x not in used_r] or [0])
                    used_r.add(r)
                    c0, c1 = sorted(rng.sample(range(w), 2))
                    if c1 - c0 >= 2:
                        g[r, c0] = g[r, c1] = color
                else:
                    c = rng.choice([x for x in range(w) if x not in used_c] or [0])
                    used_c.add(c)
                    r0, r1 = sorted(rng.sample(range(h), 2))
                    if r1 - r0 >= 2:
                        g[r0, c] = g[r1, c] = color
            if S.line_connect(truth, g.tolist()) != g.tolist():
                return g.tolist()
        raise RuntimeError("no line grid")


class CropColor(Niche):
    name, template, general = "cropcolor", "crop_color", False

    def patterns(self, limit, seed):
        out = []
        specs = [(0, 0), (1, 0)] + [(2, c) for c in range(1, 9)]
        for crit, color in specs:
            truth = {"bg": onehot(10, 0), "crit": onehot(3, crit), "color": onehot(10, color)}
            out.append({"id": "cropcolor-c%d-k%d" % (crit, color), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.crop_color(t, x)),
                        "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        crit, color = int(np.argmax(truth["crit"])), int(np.argmax(truth["color"]))
        for _ in range(1000):
            h, w = rng.randint(8, 12), rng.randint(8, 12)
            g = np.zeros((h, w), int)
            colors = rng.sample(range(1, 9), 3)
            if crit == 2 and color not in colors:
                colors[0] = color
            for col, n in zip(colors, rng.sample(range(2, 14), 3)):
                for _ in range(n):
                    g[rng.randrange(h), rng.randrange(w)] = col
            try:
                if (np.asarray(S.crop_color(truth, g.tolist())).shape != g.shape):
                    return g.tolist()
            except ValueError:
                continue
        raise RuntimeError("no crop-colour grid")


class BlockPick(Niche):
    name, template, general = "blockpick", "block_pick", False

    def patterns(self, limit, seed):
        out = []
        for sep in (5, 8):
            for crit in range(3):
                truth = {"sep": onehot(10, sep), "crit": onehot(3, crit)}
                out.append({"id": "blockpick-s%d-c%d" % (sep, crit), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.block_pick(t, x)),
                            "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        sep, crit = int(np.argmax(truth["sep"])), int(np.argmax(truth["crit"]))
        for _ in range(1000):
            layout = rng.choice([(1, 3), (3, 1), (2, 2), (1, 4)])
            bh, bw = rng.randint(3, 4), rng.randint(3, 4)
            color = rng.choice([c for c in range(1, 10) if c != sep])
            if crit == 2:
                base = np.array([[color if rng.random() < .5 else 0 for _ in range(bw)] for _ in range(bh)])
                odd = np.array([[color if rng.random() < .5 else 0 for _ in range(bw)] for _ in range(bh)])
                blocks = [base.copy() for _ in range(layout[0] * layout[1])]
                blocks[rng.randrange(len(blocks))] = odd
            else:
                blocks = [np.array([[color if rng.random() < .5 else 0 for _ in range(bw)] for _ in range(bh)]) for _ in range(layout[0] * layout[1])]
            rows = []
            for i in range(layout[0]):
                parts = []
                for j in range(layout[1]):
                    parts.append(blocks[i * layout[1] + j])
                    if j < layout[1] - 1:
                        parts.append(np.full((bh, 1), sep))
                rows.append(np.concatenate(parts, axis=1))
                if i < layout[0] - 1:
                    rows.append(np.full((1, rows[-1].shape[1]), sep))
            g = np.concatenate(rows, axis=0)
            try:
                S.block_pick(truth, g.tolist())
                return g.tolist()
            except ValueError:
                continue
        raise RuntimeError("no block grid")


class SeqCrop(Niche):
    name, template, general = "seqcrop", "seq_crop", False

    def patterns(self, limit, seed):
        out = []
        for m in range(6):
            truth = {"mask": onehot(10, m), "maxp": onehot(8, 5)}
            out.append({"id": "seqcrop-m%d" % m, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.seq_crop(t, x)),
                        "sample_input": (lambda rng, mm=m, t=truth: self._grid(rng, mm, t))})
        return out[:limit]

    @staticmethod
    def _grid(rng, m, truth):
        colors = [c for c in range(1, 10) if c != m]
        for _ in range(2000):
            h, w = rng.randint(8, 12), rng.randint(8, 12)
            pr, pc = rng.randint(1, 3), rng.randint(2, 3)
            tile = [[rng.choice(colors[:5]) for _ in range(pc)] for _ in range(pr)]
            if len({v for r in tile for v in r}) < 2:
                continue
            full = np.array([[tile[i % pr][j % pc] for j in range(w)] for i in range(h)])
            hh, ww = rng.randint(2, 4), rng.randint(2, 4)
            r0, c0 = rng.randrange(0, h - hh + 1), rng.randrange(0, w - ww + 1)
            x = full.copy()
            x[r0:r0 + hh, c0:c0 + ww] = m
            try:
                if S.seq_crop(truth, x.tolist()) == full[r0:r0 + hh, c0:c0 + ww].tolist():
                    return x.tolist()
            except ValueError:
                continue
        raise RuntimeError("no seq-crop grid")


class CountOut(Niche):
    name, template, general = "countout", "count_output", False

    def patterns(self, limit, seed):
        out = []
        specs = [(0, 0, 1), (0, 0, 5), (1, 0, 1), (1, 0, 5), (2, 1, 1), (2, 2, 5), (3, 1, 1), (3, 2, 5)]
        for what, color, fill in specs:
            for form in (0, 1):
                truth = {"bg": onehot(10, 0), "what": onehot(4, what), "color": onehot(10, color), "fill": onehot(10, fill), "form": onehot(2, form)}
                out.append({"id": "countout-w%d-k%d-f%d-m%d" % (what, color, fill, form), "niche": self.name, "truth": truth,
                            "apply": (lambda x, t=truth: S.count_output(t, x)), "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        for _ in range(800):
            n = rng.randint(6, 9)
            g = np.zeros((n, n), int)
            guard = np.zeros((n, n), bool)
            for _ in range(rng.randint(2, 6)):
                r, c = rng.randrange(n), rng.randrange(n)
                if not guard[r, c]:
                    g[r, c] = rng.choice([1, 2, 3])
                    guard[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
            try:
                S.count_output(truth, g.tolist())
                return g.tolist()
            except ValueError:
                continue
        raise RuntimeError("no count grid")



# ---------------------------------------------------------------- niches from the census
class Outline(Niche):
    name, template, general = "outline", "outline", False

    def patterns(self, limit, seed):
        out = []
        for conn in (0, 1):
            for ring in (None, 4, 6):
                ringv = [0.0] * 10
                if ring is not None:
                    ringv[ring] = 1.0
                truth = {"bg": onehot(10, 0), "conn": onehot(2, conn), "src": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], "ring": ringv}
                out.append({"id": "outline-n%d-%s" % (conn, "same" if ring is None else "c%d" % ring), "niche": self.name, "truth": truth,
                            "apply": (lambda x, t=truth: S.outline(t, x)), "sample_input": self._grid})
        return out[:limit]

    @staticmethod
    def _grid(rng):
        n = rng.randint(7, 10)
        g = np.zeros((n, n), int)
        for _ in range(rng.randint(2, 4)):
            r, c = rng.randrange(1, n - 1), rng.randrange(1, n - 1)
            g[r, c] = rng.choice([1, 2, 3, 8])
        return g.tolist()


class ObjMove3(Niche):
    name, template, general = "objmove3", "object_move3", False

    def patterns(self, limit, seed):
        out = []
        for d in (0, 2, 4, 6):
            for conn in (0, 1):
                truth = {"dir": onehot(8, d), "bg": onehot(10, 0), "conn": onehot(2, conn), "mover": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]}
                out.append({"id": "objmove3-d%d-n%d" % (d, conn), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.object_move3(t, x)),
                            "sample_input": (lambda rng, t=truth: self._grid(rng, t))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, truth):
        for _ in range(800):
            n = rng.randint(7, 10)
            g = np.zeros((n, n), int)
            for _ in range(rng.randint(2, 4)):
                r, c = rng.randrange(n), rng.randrange(n)
                if g[r, c] == 0:
                    g[r, c] = rng.choice([1, 2])
            for _ in range(rng.randint(2, 5)):
                r, c = rng.randrange(n), rng.randrange(n)
                if g[r, c] == 0:
                    g[r, c] = 8
            if S.object_move3(truth, g.tolist()) != g.tolist():
                return g.tolist()
        raise RuntimeError("no move grid")


class BlockRed(Niche):
    name, template, general = "blockred", "block_reduce", False

    def patterns(self, limit, seed):
        out = []
        for kh, kw in ((2, 2), (3, 3), (1, 2), (2, 1), (2, 3)):
            for mode in range(4):
                truth = {"kh": onehot(5, kh - 1), "kw": onehot(5, kw - 1), "mode": onehot(4, mode), "mark": onehot(10, 5)}
                out.append({"id": "blockred-%dx%d-m%d" % (kh, kw, mode), "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.block_reduce(t, x)),
                            "sample_input": (lambda rng, k=(kh, kw), md=mode: self._grid(rng, k, md))})
        random.Random(seed).shuffle(out)
        return out[:limit]

    @staticmethod
    def _grid(rng, k, mode):
        kh, kw = k
        rows, cols = rng.randint(2, 4), rng.randint(2, 4)
        g = np.zeros((rows * kh, cols * kw), int)
        for i in range(rows):
            for j in range(cols):
                color = rng.choice([1, 2, 3])
                kind = rng.random()
                block = np.zeros((kh, kw), int)
                if kind < .35:
                    block[:] = color
                elif kind < .7:
                    mask = np.array([[rng.random() < .55 for _ in range(kw)] for _ in range(kh)])
                    block[mask] = color
                g[i * kh:(i + 1) * kh, j * kw:(j + 1) * kw] = block
        return g.tolist()


class SelfKron(Niche):
    name, template, general = "selfkron", "self_kron", False

    def patterns(self, limit, seed):
        out = []
        for mode in (0, 1):
            truth = {"bg": onehot(10, 0), "mode": onehot(2, mode)}
            out.append({"id": "selfkron-m%d" % mode, "niche": self.name, "truth": truth, "apply": (lambda x, t=truth: S.self_kron(t, x)),
                        "sample_input": self._grid})
        return out[:limit]

    @staticmethod
    def _grid(rng):
        n = 3
        color = rng.choice([1, 2, 3, 4])
        g = np.zeros((n, n), int)
        while g.sum() == 0 or (g != 0).all():
            g = np.array([[color if rng.random() < .5 else 0 for _ in range(n)] for _ in range(n)])
        return g.tolist()


NICHES = {n.name: n for n in (Tiling(), Association(), Sequence(), Glyph(), Neighbor(), Object(), Ray(), Crop(), Recall(), Induction(), SeqOps(), SelCopy(), Grammar(), Successor(), Agree(), Gravity(), FillEnc(), Symmetry(), Scale(), HalfBool(), Denoise(), ObjMove(), FloodM(), RayCross(), Context2(), ObjMove2(), FloodThru(), SnapLine(), OCRDigit(), OCRSeq(), Legend(), Grammar2(), Analogy(), ToolUse(), LineConn(), CropColor(), BlockPick(), SeqCrop(), CountOut(), Outline(), ObjMove3(), BlockRed(), SelfKron())}
LIMITS = {"tiling": 60, "association": 24, "sequence": 30, "glyph": 24, "neighbor": 30, "object": 32, "ray": 48, "crop": 16, "recall": 6, "induction": 6, "seqops": 10, "selcopy": 10, "grammar": 12, "successor": 6, "agree": 8, "gravity": 4, "fillenc": 12, "symmetry": 24, "scale": 8, "halfbool": 20, "denoise": 6, "objmove": 12, "floodm": 12, "raycross": 24, "context2": 32, "objmove2": 24, "floodthru": 18, "snapline": 4, "ocrdigit": 12, "ocrseq": 12, "legend": 2, "grammar2": 4, "analogy": 8, "tooluse": 12, "lineconn": 12, "cropcolor": 10, "blockpick": 6, "seqcrop": 6, "countout": 16, "outline": 6, "objmove3": 8, "blockred": 20, "selfkron": 2}


def probe_inputs(niche_name, n=6, seed=5):
    """Fixed generated inputs for behavioural fingerprints."""
    niche = NICHES[niche_name]
    pats = niche.patterns(max(n, 4), seed)
    return [niche.task(p, seed + i)["test"][0]["input"] for i, p in enumerate(pats[:n])]


# Additive structure: no existing niche/template changed.
import wall_pruned_skill as _wallpruned
import sys as _sys
NICHES['wallpruned'] = _wallpruned.make_niche(_sys.modules[__name__])
LIMITS['wallpruned'] = 24
import anchor_skill as _anchorspin
import frame_skill as _framecorner
import reference_skill as _referenceobject
from skill_niche_factory import make_niche as _structural_niche
for _module in (_anchorspin, _framecorner, _referenceobject):
    NICHES[_module.NICHENAME] = _structural_niche(_sys.modules[__name__], _module)
    LIMITS[_module.NICHENAME] = 24
import anchor_clip_skill as _anchorclip
import frame_infer_skill as _frameinfer
import reference_paint_skill as _referencepaint
for _module in (_anchorclip, _frameinfer, _referencepaint):
    NICHES[_module.NICHENAME] = _structural_niche(_sys.modules[__name__], _module)
    LIMITS[_module.NICHENAME] = 24
import anchor_assign_skill as _anchorassign
NICHES[_anchorassign.NICHENAME] = _structural_niche(_sys.modules[__name__], _anchorassign)
LIMITS[_anchorassign.NICHENAME] = 24
import corridor_skill as _corridormarker
NICHES[_corridormarker.NICHENAME] = _structural_niche(_sys.modules[__name__], _corridormarker)
LIMITS[_corridormarker.NICHENAME] = 24
