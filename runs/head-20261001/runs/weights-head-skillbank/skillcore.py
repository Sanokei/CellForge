"""Skill-cell core (head-center). Typed grid->grid cells whose behaviour lives in literal tensors written by the maker.

Templates (structure = fixed code, every behaviour-selecting number = tensor):
  tile_orient    tiling with a dihedral transform per tile                      per(4) pec(4) orient(4,4,8)
  assoc_induct   LANGUAGE: in-context token association over the demonstrations  sim(10,10) temp(1) keep(1)
  seq_continue   LANGUAGE: periodic continuation of masked tokens                mask(10) axis(3) maxp(8)
  glyph_recolor  OCR: prototype template-match of an object's bitmap -> recolour W(6,5,5) b(6) paint(6,10)
  + the frozen legacy templates of weights-pop1 (colormap, conv3x3_detect, ...), loaded from pop1_core.py at palette 10.
A chain cell (A then B) is the breeding product: its tensors are the parents' tensors, key-prefixed "A." / "B.", no new numbers.
Routing is the literal router of the joint genome: scores = route.W @ (-demonstration error) + route.b, argmax picks one cell per task.
Demonstration-aware cells see the other demonstrations only (leave-one-out), never the demonstration they are being scored on.
"""
import hashlib
import importlib.util
import json

import numpy as np

COLORS = 10


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


# ------------------------------------------------------------------ templates
OPS = [lambda a: a, np.fliplr, np.flipud, lambda a: np.rot90(a, 2), lambda a: a.T, lambda a: np.rot90(a, -1),
       lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2).T]


def tile_orient(params, grid, demos=None):
    a = np.asarray(grid)
    pr, pc = int(np.argmax(params["per"])) + 1, int(np.argmax(params["pec"])) + 1
    orient = np.asarray(params["orient"], float)
    rows = []
    for i in range(pr):
        tiles = []
        for j in range(pc):
            k = int(np.argmax(orient[i][j]))
            if k >= 4 and a.shape[0] != a.shape[1]:
                raise ValueError("transposing tile transform needs a square input")
            tiles.append(OPS[k](a))
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0).astype(int).tolist()


def assoc_induct(params, grid, demos=None):
    x = np.asarray(grid)
    sim = np.asarray(params["sim"], float)
    beta, keep = float(params["temp"][0]), float(params["keep"][0])
    counts = np.zeros((COLORS, COLORS))
    for d in demos or []:
        a, b = np.asarray(d["input"]), np.asarray(d["output"])
        if a.shape == b.shape:
            np.add.at(counts, (a.ravel(), b.ravel()), 1)
    present = counts.sum(1) > 0
    values = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    out = x.copy()
    for a in np.unique(x):
        if not present.any():
            break
        w = np.where(present, np.exp(np.clip(beta * sim[a], -50, 50)), 0.0)
        if w.sum() <= 0:
            continue
        logits = (w / w.sum()) @ values + keep * np.eye(COLORS)[a]
        out[x == a] = int(np.argmax(logits))
    return out.astype(int).tolist()


def _fill_line(v, unknown, maxp):
    n = len(v)
    for p in range(1, min(maxp, n) + 1):
        classes, ok = {}, True
        for t in range(n):
            if not unknown[t] and classes.setdefault(t % p, v[t]) != v[t]:
                ok = False
                break
        if ok:
            out = v.copy()
            for t in range(n):
                if unknown[t] and t % p in classes:
                    out[t] = classes[t % p]
            return out
    return v


def seq_continue(params, grid, demos=None):
    a = np.asarray(grid).copy()
    m, axis, maxp = int(np.argmax(params["mask"])), int(np.argmax(params["axis"])), int(np.argmax(params["maxp"])) + 1
    unknown = a == m
    if not unknown.any():
        return a.tolist()
    if axis == 0:
        for i in range(a.shape[0]):
            a[i] = _fill_line(a[i], unknown[i], maxp)
    elif axis == 1:
        for j in range(a.shape[1]):
            a[:, j] = _fill_line(a[:, j], unknown[:, j], maxp)
    else:
        h, w = a.shape
        options = sorted(((pr, pc) for pr in range(1, min(maxp, h) + 1) for pc in range(1, min(maxp, w) + 1)), key=lambda t: (t[0] * t[1], t))
        for pr, pc in options:
            classes, ok = {}, True
            for i in range(h):
                for j in range(w):
                    if not unknown[i, j] and classes.setdefault((i % pr, j % pc), a[i, j]) != a[i, j]:
                        ok = False
                        break
                if not ok:
                    break
            if ok and all((i % pr, j % pc) in classes for i in range(h) for j in range(w) if unknown[i, j]):
                out = a.copy()
                for i in range(h):
                    for j in range(w):
                        if unknown[i, j]:
                            out[i, j] = classes[(i % pr, j % pc)]
                return out.tolist()
        return a.tolist()
    return a.tolist()


def _components(fg):
    h, w = fg.shape
    seen = np.zeros_like(fg, bool)
    for i in range(h):
        for j in range(w):
            if fg[i, j] and not seen[i, j]:
                stack, cells = [(i, j)], []
                seen[i, j] = True
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w and fg[rr, cc] and not seen[rr, cc]:
                                seen[rr, cc] = True
                                stack.append((rr, cc))
                yield cells


def glyph_recolor(params, grid, demos=None):
    a = np.asarray(grid)
    W, b, paint = np.asarray(params["W"], float), np.asarray(params["b"], float), np.asarray(params["paint"], float)
    out = a.copy()
    for cells in _components(a != 0):
        rs, cs = [r for r, _ in cells], [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        if max(rs) - r0 >= 5 or max(cs) - c0 >= 5:
            continue
        canvas = np.zeros((5, 5))
        for r, c in cells:
            canvas[r - r0, c - c0] = 1.0
        scores = (W * canvas).sum((1, 2)) + b
        k = int(np.argmax(scores))
        if scores[k] > 0:
            for r, c in cells:
                out[r, c] = int(np.argmax(paint[k]))
    return out.astype(int).tolist()


TEMPLATES = {
    "tile_orient": {"shapes": {"per": (4,), "pec": (4,), "orient": (4, 4, 8)}, "fn": tile_orient, "aware": False, "niche": "tiling",
                    "doc": ("pr = argmax(per)+1 tile rows, pc = argmax(pec)+1 tile columns (each at most 4). Tile (i,j) of the output is the input "
                            "grid transformed by k = argmax(orient[i][j]) (8 numbers): 0 identity, 1 fliplr, 2 flipud, 3 rot180, 4 transpose, 5 rot90 "
                            "clockwise, 6 rot90 counter-clockwise, 7 anti-transpose. Only i<pr, j<pc are used. Transposing transforms need a square input.")},
    "assoc_induct": {"shapes": {"sim": (10, 10), "temp": (1,), "keep": (1,)}, "fn": assoc_induct, "aware": True, "niche": "association",
                     "doc": ("Reads the demonstration pairs as context (the pairs that the query is NOT part of). From them it builds, for every input colour b "
                             "seen at some pixel, the distribution V[b] over the output colours that appeared at those pixels. A query pixel of colour a "
                             "attends to the seen colours: weight w_b = exp(temp * sim[a][b]) over seen b, normalised; logits = sum_b w_b V[b] + keep * onehot(a); "
                             "the pixel becomes argmax(logits). So a large diagonal sim copies the demonstrated mapping for colour a, and keep decides "
                             "when an unseen colour stays unchanged (keep must be below the confidence of a seen mapping, which is about 1, yet above "
                             "the diluted average used for unseen colours).")},
    "seq_continue": {"shapes": {"mask": (10,), "axis": (3,), "maxp": (8,)}, "fn": seq_continue, "aware": False, "niche": "sequence",
                     "doc": ("Cells whose colour is m = argmax(mask) are unknown tokens. axis = argmax(axis): 0 treat every row as a token sequence, "
                             "1 every column, 2 the grid as a 2-D periodic pattern. maxp = argmax(maxp)+1 is the largest period tried. The smallest period "
                             "consistent with all known cells is found and every unknown cell takes the token of its period class.")},
    "glyph_recolor": {"shapes": {"W": (6, 5, 5), "b": (6,), "paint": (6, 10)}, "fn": glyph_recolor, "aware": False, "niche": "glyph",
                      "doc": ("Every 8-connected object of non-zero pixels whose bounding box fits 5x5 is drawn on a 5x5 canvas (1 where the object has a pixel, "
                              "anchored at the top-left of its bounding box). Class scores s_k = sum(W[k] * canvas) + b[k]. If max_k s_k > 0 then every pixel "
                              "of the object becomes argmax(paint[k]) for the winning k; otherwise the object is unchanged. A prototype is a template: W[k] "
                              "positive where the shape has a pixel, negative where it must be empty, b[k] about -(number of shape pixels) + 0.5.")},
}


def load_legacy(pop1_core_path):
    """Frozen weights-pop1 templates at palette 10 (colormap, conv3x3_detect, ...) plus the authored parity_row_stencil."""
    spec = importlib.util.spec_from_file_location("pop1_core_frozen", pop1_core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    k = COLORS
    core.K = k
    shapes = {"conv3x3_detect": {"conv.W": (9, k), "conv.b": (1,), "paint": (k,)}, "colormap": {"map.W": (k, k)},
              "color_gate": {"gate": (k,), "fill": (k,)}, "flip_select": {"sel": (6,), "post.W": (k, k)}, "tile_select": {"per": (3,), "pec": (3,)},
              "neighbor_count": {"self.W": (1, k), "cnt.W": (1, k), "cnt.b": (1,), "paint": (k,)}, "bbox_mask": {"pick": (k,), "fill": (k,)}}
    core.SHAPES = shapes
    for name in shapes:
        TEMPLATES[name] = {"shapes": shapes[name], "fn": (lambda p, g, d=None, n=name: core.PREDICTORS[n](p, g, False)[0]), "aware": False,
                           "niche": "legacy", "doc": ""}
    try:
        from cellforge.construction.typed_population import parity_row_stencil
        TEMPLATES["parity_row_stencil_v1"] = {"shapes": {"stencil.W": (2, 3), "foreground.W": (1, 10), "activation.b": (1,)},
                                              "fn": lambda p, g, d=None: parity_row_stencil(p, g), "aware": False, "niche": "legacy", "doc": ""}
    except ImportError:
        pass
    return core


# ------------------------------------------------------------------ cells
def template_of(params):
    keys = set(params)
    for name, t in TEMPLATES.items():
        if set(t["shapes"]) == keys and all(np.asarray(params[k]).shape == tuple(s) for k, s in t["shapes"].items()):
            return name
    raise ValueError("unknown or malformed cell tensors: %s" % sorted(keys))


def valid_params(template, params):
    t = TEMPLATES.get(template)
    if not t or not isinstance(params, dict) or set(params) != set(t["shapes"]):
        return False
    try:
        for name, shape in t["shapes"].items():
            arr = np.asarray(params[name], dtype=float)
            if arr.shape != tuple(shape) or not np.isfinite(arr).all() or np.abs(arr).max() > 9:
                return False
    except (TypeError, ValueError):
        return False
    return True


def make_cell(template, params, **meta):
    cell = {"kind": "cell", "template": template, "params": params}
    cell["id"] = digest({"t": template, "p": params})
    cell.update(meta)
    return cell


def chain_cell(a, b, **meta):
    cell = {"kind": "chain", "parts": [a, b], "parents": [a["id"], b["id"]]}
    cell["id"] = digest({"chain": [a["id"], b["id"]]})
    cell.update(meta)
    return cell


def run(cell, grid, demos):
    """Apply a cell (or chain) to a grid. demos: list of {input, output} used only by demonstration-aware templates."""
    if cell["kind"] == "chain":
        first, second = cell["parts"]
        mid = run(first, grid, demos)
        staged = [{"input": run(first, d["input"], [e for e in demos if e is not d]), "output": d["output"]} for d in demos]
        return run(second, mid, staged)
    return TEMPLATES[cell["template"]]["fn"](cell["params"], grid, demos)


def uses_context(cell):
    if cell["kind"] == "chain":
        return any(uses_context(p) for p in cell["parts"])
    return TEMPLATES[cell["template"]]["aware"]


def passes(cell, task):
    """Eligibility on one task. Context-free cells must reproduce every demonstration AND the test output. Cells that read the
    demonstrations as context are judged on the test output given all demonstrations (a colour seen in only one demonstration is
    legitimately unseen when that demonstration is held out). Invalid cells (exceptions) never pass."""
    demos = task["train"]
    try:
        if not all(run(cell, t["input"], demos) == t["output"] for t in task["test"]):
            return False
        if not uses_context(cell):
            return all(run(cell, d["input"], demos) == d["output"] for d in demos)
        return True
    except Exception:
        return False


def positions(cell):
    if cell["kind"] == "chain":
        return 0
    return int(sum(np.asarray(v).size for v in cell["params"].values()))


def depth(cell):
    return 1 if cell["kind"] == "cell" else 1 + max(depth(p) for p in cell["parts"])


# ------------------------------------------------------------------ tensors <-> cells
def _flatten(cell, prefix, out):
    if cell["kind"] == "chain":
        _flatten(cell["parts"][0], prefix + "A.", out)
        _flatten(cell["parts"][1], prefix + "B.", out)
    else:
        for k, v in cell["params"].items():
            out[prefix + k] = v


def to_tensors(cells, router=None):
    out = {}
    for i, cell in enumerate(cells):
        flat = {}
        _flatten(cell, "", flat)
        for k, v in flat.items():
            out["%d:%s" % (i, k)] = v
    n = len(cells)
    router = router or default_router(cells)
    out["route.W"], out["route.b"] = router["route.W"], router["route.b"]
    return out


def _unflatten(flat):
    if flat and all(k.startswith(("A.", "B.")) for k in flat):
        a = {k[2:]: v for k, v in flat.items() if k.startswith("A.")}
        b = {k[2:]: v for k, v in flat.items() if k.startswith("B.")}
        return chain_cell(_unflatten(a), _unflatten(b))
    return make_cell(template_of(flat), flat)


def from_tensors(tensors):
    groups = {}
    for name, value in tensors.items():
        if name.startswith("route."):
            continue
        index, key = name.split(":", 1)
        groups.setdefault(int(index), {})[key] = value
    return [_unflatten(groups[i]) for i in sorted(groups)], {"route.W": tensors["route.W"], "route.b": tensors["route.b"]}


def default_router(cells, legacy=None, tie=1e-7):
    """Identity over demonstration-error features; older cells win exact ties. `legacy` = (W, b) of the frozen authored router for the
    first len(b) cells (kept block-diagonal so the original five cells behave exactly as before)."""
    n = len(cells)
    W = np.eye(n)
    b = np.array([(n - i) * tie for i in range(n)], float)
    if legacy is not None:
        lw, lb = np.asarray(legacy[0], float), np.asarray(legacy[1], float)
        m = len(lb)
        W[:m, :m] = lw
        b[:m] = lb
    return {"route.W": W.tolist(), "route.b": b.tolist()}


# ------------------------------------------------------------------ routed scoring (cached)
class Scorer:
    """Caches demonstration error per (cell id, task id); scoring a larger registry only computes the new cells."""

    def __init__(self, rows):
        self.rows = rows
        self.err = {}

    def demo_error(self, cell, row):
        key = (cell["id"], row["id"])
        if key not in self.err:
            demos = row["task"]["train"]
            total = 0.0
            for i, d in enumerate(demos):
                try:
                    pred = np.asarray(run(cell, d["input"], demos[:i] + demos[i + 1:]))
                    exp = np.asarray(d["output"])
                    total += 1.0 if pred.shape != exp.shape else float(np.mean(pred != exp))
                except Exception:
                    total += 1.0
            self.err[key] = total / len(demos)
        return self.err[key]

    def score(self, cells, router=None, rows=None):
        rows = rows if rows is not None else self.rows
        router = router or default_router(cells)
        W, b = np.asarray(router["route.W"], float), np.asarray(router["route.b"], float)
        outcomes, picks = {}, {}
        for row in rows:
            f = -np.array([self.demo_error(c, row) for c in cells])
            k = int(np.argmax(W @ f + b))
            picks[row["id"]] = cells[k]["id"]
            ok = True
            for t in row["task"]["test"]:
                try:
                    ok &= run(cells[k], t["input"], row["task"]["train"]) == t["output"]
                except Exception:
                    ok = False
            outcomes[row["id"]] = bool(ok)
        return {"outcomes": outcomes, "solved": sum(outcomes.values()), "total": len(rows), "picks": picks, "partition": "development"}


def predict(tensors, task):
    """Evaluator entry (arc_eval.py): task has train and stripped test; returns [[grid]] per test input."""
    cells, router = from_tensors(tensors)
    demos = task["train"]
    W, b = np.asarray(router["route.W"], float), np.asarray(router["route.b"], float)
    errs = []
    for c in cells:
        total = 0.0
        for i, d in enumerate(demos):
            try:
                pred = np.asarray(run(c, d["input"], demos[:i] + demos[i + 1:]))
                exp = np.asarray(d["output"])
                total += 1.0 if pred.shape != exp.shape else float(np.mean(pred != exp))
            except Exception:
                total += 1.0
        errs.append(total / len(demos))
    k = int(np.argmax(W @ -np.array(errs) + b))
    return [[run(cells[k], t["input"], demos)] for t in task["test"]]
