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


# ------------------------------------------------------------------ object helpers
def objects(a, bg, eight=True, same_color=True):
    """Connected components of non-background pixels. Returns [{cells, color (or None), r0, c0, r1, c1}]."""
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (eight or not (dr and dc))]
    out = []
    for i in range(h):
        for j in range(w):
            if a[i, j] == bg or seen[i, j]:
                continue
            stack, cells = [(i, j)], []
            seen[i, j] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in steps:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and a[rr, cc] != bg and (not same_color or a[rr, cc] == a[i, j]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            rs, cs = [r for r, _ in cells], [c for _, c in cells]
            out.append({"cells": cells, "color": int(a[i, j]) if same_color else None, "r0": min(rs), "c0": min(cs), "r1": max(rs), "c1": max(cs)})
    return out


def holes_of(obj):
    h, w = obj["r1"] - obj["r0"] + 1, obj["c1"] - obj["c0"] + 1
    mask = np.zeros((h, w), bool)
    for r, c in obj["cells"]:
        mask[r - obj["r0"], c - obj["c0"]] = True
    seen = mask.copy()
    count = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j]:
                continue
            stack, border = [(i, j)], False
            seen[i, j] = True
            while stack:
                r, c = stack.pop()
                if r in (0, h - 1) or c in (0, w - 1):
                    border = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            count += not border
    return count


def obj_prop(kind, obj, objs, shape):
    size = len(obj["cells"])
    if kind == 0:
        return size
    if kind == 1:
        return holes_of(obj)
    if kind == 2:
        return obj["r1"] - obj["r0"] + 1
    if kind == 3:
        return obj["c1"] - obj["c0"] + 1
    if kind == 4:
        return sum(len(o["cells"]) > size for o in objs)
    if kind == 5:
        sizes = [len(o["cells"]) for o in objs]
        return 0 if size == max(sizes) else (1 if size == min(sizes) else 2)
    if kind == 6:
        return int(obj["r0"] == 0 or obj["c0"] == 0 or obj["r1"] == shape[0] - 1 or obj["c1"] == shape[1] - 1)
    return (obj["color"], holes_of(obj))


def nkey(kind, a, r, c, bg):
    h, w = a.shape
    own = int(a[r, c])

    def at(rr, cc):
        return int(a[rr, cc]) if 0 <= rr < h and 0 <= cc < w else -1
    n4 = [at(r - 1, c), at(r + 1, c), at(r, c - 1), at(r, c + 1)]
    n8 = n4 + [at(r - 1, c - 1), at(r - 1, c + 1), at(r + 1, c - 1), at(r + 1, c + 1)]
    if kind == 0:
        return (own,)
    if kind == 1:
        return (own, sum(v not in (-1, bg) for v in n4))
    if kind == 2:
        return (own, sum(v not in (-1, bg) for v in n8))
    if kind == 3:
        return (own, tuple(sorted({v for v in n8 if v not in (-1, bg)})))
    if kind == 4:
        return (own, sum(v == own for v in n4))
    return (own, tuple(n4))


# ------------------------------------------------------------------ new templates
def neighbor_table(params, grid, demos=None):
    """Demonstration-aware relational local rule: output colour = most common demonstrated output for the pixel's neighbourhood key."""
    a = np.asarray(grid)
    kind, bg = int(np.argmax(params["keysel"])), int(np.argmax(params["bg"]))
    table = {}
    for d in demos or []:
        x, y = np.asarray(d["input"]), np.asarray(d["output"])
        if x.shape != y.shape:
            continue
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                bucket = table.setdefault(nkey(kind, x, r, c, bg), {})
                bucket[int(y[r, c])] = bucket.get(int(y[r, c]), 0) + 1
    out = a.copy()
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            bucket = table.get(nkey(kind, a, r, c, bg))
            if bucket:
                out[r, c] = max(bucket, key=lambda col: (bucket[col], -col))
    return out.astype(int).tolist()


def object_table(params, grid, demos=None):
    """Demonstration-aware object recolouring: output colour of a (single-colour, 8-connected) object = learned function of one property."""
    a = np.asarray(grid)
    kind, bg = int(np.argmax(params["prop"])), int(np.argmax(params["bg"]))
    table = {}
    for d in demos or []:
        x, y = np.asarray(d["input"]), np.asarray(d["output"])
        if x.shape != y.shape:
            continue
        objs = objects(x, bg)
        for o in objs:
            cols = {int(y[r, c]) for r, c in o["cells"]}
            if len(cols) == 1:
                bucket = table.setdefault(obj_prop(kind, o, objs, x.shape), {})
                col = next(iter(cols))
                bucket[col] = bucket.get(col, 0) + 1
    out = a.copy()
    objs = objects(a, bg)
    for o in objs:
        bucket = table.get(obj_prop(kind, o, objs, a.shape))
        if bucket:
            col = max(bucket, key=lambda k: (bucket[k], -k))
            for r, c in o["cells"]:
                out[r, c] = col
    return out.astype(int).tolist()


RAY_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def ray_cast(params, grid, demos=None):
    a = np.asarray(grid)
    h, w = a.shape
    src = np.asarray(params["src"], float) > 0
    dirs = [RAY_DIRS[k] for k in range(8) if params["dirs"][k] > 0]
    stop = int(np.argmax(params["stop"]))
    paint = np.asarray(params["paint"], float)
    fixed = int(np.argmax(paint)) if paint.max() > 0 else None
    out = a.copy()
    for r in range(h):
        for c in range(w):
            if a[r, c] != 0 and src[a[r, c]]:
                color = fixed if fixed is not None else int(a[r, c])
                for dr, dc in dirs:
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < h and 0 <= cc < w:
                        blocked = a[rr, cc] != 0
                        if blocked and stop == 1:
                            break
                        if not (blocked and (stop == 2 or src[a[rr, cc]])):
                            out[rr, cc] = color
                        rr, cc = rr + dr, cc + dc
    return out.astype(int).tolist()


def crop_select(params, grid, demos=None):
    a = np.asarray(grid)
    crit, bg, eight = int(np.argmax(params["crit"])), int(np.argmax(params["bg"])), int(np.argmax(params["conn"])) == 1
    objs = objects(a, bg, eight=eight, same_color=False)
    if not objs:
        raise ValueError("no object to crop")
    area = lambda o: len(o["cells"])  # noqa: E731
    box = lambda o: (o["r1"] - o["r0"] + 1) * (o["c1"] - o["c0"] + 1)  # noqa: E731
    keys = [lambda o: -area(o), area, lambda o: -box(o), box, lambda o: o["r0"], lambda o: -o["r1"], lambda o: o["c0"], lambda o: -o["c1"]]
    ranked = sorted(objs, key=keys[crit])
    if len(ranked) > 1 and keys[crit](ranked[0]) == keys[crit](ranked[1]):
        raise ValueError("tied crop criterion")
    o = ranked[0]
    return a[o["r0"]:o["r1"] + 1, o["c0"]:o["c1"] + 1].astype(int).tolist()


TEMPLATES.update({
    "neighbor_table": {"shapes": {"keysel": (6,), "bg": (10,)}, "fn": neighbor_table, "aware": True, "niche": "neighbor",
                       "doc": ("Reads the demonstration pairs as context. For every pixel it forms a neighbourhood key chosen by kind = argmax(keysel): 0 own colour; "
                               "1 own colour + number of non-background 4-neighbours; 2 own colour + number of non-background 8-neighbours; 3 own colour + the set of "
                               "colours among the 8 neighbours; 4 own colour + number of 4-neighbours with the same colour; 5 own colour + the four 4-neighbour colours. "
                               "Background colour is argmax(bg). The demonstrations define, per key, the most common output colour at such pixels; each query pixel takes "
                               "the output of its key, or stays unchanged if the key was never seen.")},
    "object_table": {"shapes": {"prop": (8,), "bg": (10,)}, "fn": object_table, "aware": True, "niche": "object",
                     "doc": ("Reads the demonstration pairs as context. Objects are single-colour 8-connected components of non-background (background = argmax(bg)). "
                             "Property chosen by argmax(prop): 0 pixel count; 1 number of enclosed holes; 2 bounding-box height; 3 bounding-box width; 4 size rank "
                             "(number of larger objects); 5 largest / smallest / other (0/1/2); 6 touches the grid border (0/1); 7 (colour, holes). The demonstrations "
                             "define, per property value, the colour the whole object takes in the output (objects that are not recoloured uniformly are ignored); a "
                             "query object takes the colour of its property value, or stays unchanged if the value was never seen.")},
    "ray_cast": {"shapes": {"src": (10,), "dirs": (8,), "stop": (3,), "paint": (10,)}, "fn": ray_cast, "aware": False, "niche": "ray",
                 "doc": ("Every pixel whose colour c has src[c] > 0 (colour 0 is background and never a source) shoots a ray in each direction d with dirs[d] > 0, "
                         "directions in order: up, up-right, right, down-right, down, down-left, left, up-left. stop = argmax(stop): 0 the ray paints every cell to the edge "
                         "except source pixels; 1 the ray stops before the first non-background cell; 2 the ray skips over non-background cells and paints only background "
                         "cells. Paint colour: if max(paint) > 0 then argmax(paint), otherwise the source pixel's own colour.")},
    "crop_select": {"shapes": {"crit": (8,), "bg": (10,), "conn": (2,)}, "fn": crop_select, "aware": False, "niche": "crop",
                    "doc": ("Objects are connected components of non-background pixels (background = argmax(bg); connectivity 4 if argmax(conn)=0, 8 if 1; colours "
                            "may differ inside an object). crit = argmax(crit) picks one object: 0 largest pixel count, 1 smallest pixel count, 2 largest bounding box, 3 "
                            "smallest bounding box, 4 topmost, 5 bottommost, 6 leftmost, 7 rightmost. The output is the input cropped to that object's bounding box.")},
})


# ------------------------------------------------------------------ language templates (token sequences as 1 x N grids, vocabulary 10)
def _row(grid):
    a = np.asarray(grid)
    return [int(v) for v in a.ravel()]


def recall_attend(params, grid, demos=None):
    """Associative recall: k1 v1 k2 v2 ... delim q  ->  the value stored for q (last occurrence wins); 0 if absent."""
    row = _row(grid)
    d, rev = int(np.argmax(params["delim"])), int(np.argmax(params["order"])) == 1
    if d not in row:
        raise ValueError("no delimiter")
    cut = row.index(d)
    pairs, rest = row[:cut], row[cut + 1:]
    if len(pairs) % 2 or len(rest) != 1:
        raise ValueError("malformed recall sequence")
    keys, vals = pairs[0::2], pairs[1::2]
    if rev:
        keys, vals = vals, keys
    answer = 0
    for k, v in zip(keys, vals):
        if k == rest[0]:
            answer = v
    return [[answer]]


def induction_next(params, grid, demos=None):
    """Induction head: find earlier occurrences of the final n-gram; emit the token that followed them (last / first / most common)."""
    row = _row(grid)
    which, n = int(np.argmax(params["which"])), int(np.argmax(params["n"])) + 1
    if len(row) <= n:
        return [[0]]
    ctx = row[-n:]
    followers = [row[i + n] for i in range(len(row) - n) if row[i:i + n] == ctx]
    if not followers:
        return [[0]]
    if which == 0:
        return [[followers[-1]]]
    if which == 1:
        return [[followers[0]]]
    counts = {v: followers.count(v) for v in set(followers)}
    return [[max(counts, key=lambda v: (counts[v], -v))]]


def seq_transform(params, grid, demos=None):
    row = _row(grid)
    op = int(np.argmax(params["op"]))
    if op == 0:
        out = row[::-1]
    elif op == 1:
        out = sorted(row)
    elif op == 2:
        out = sorted(row, reverse=True)
    elif op == 3:
        out = [v for i, v in enumerate(row) if i == 0 or v != row[i - 1]]
    elif op == 4:
        out = [v for i, v in enumerate(row) if v not in row[:i]]
    elif op == 5:
        out = row[1:] + row[:1]
    elif op == 6:
        out = row[-1:] + row[:-1]
    elif op == 7:
        counts = {v: row.count(v) for v in set(row)}
        top = max(counts, key=lambda v: (counts[v], -v))
        out = [top] * len(row)
    elif op == 8:
        counts = {v: row.count(v) for v in set(row)}
        out = sorted(row, key=lambda v: (-counts[v], v))
    else:
        out = [v for v in row if v != 0]
    if not out:
        raise ValueError("empty output")
    return [out]


def selective_copy(params, grid, demos=None):
    noise = int(np.argmax(params["noise"]))
    out = [v for v in _row(grid) if v != noise]
    if not out:
        raise ValueError("nothing to copy")
    return [out]


def grammar_check(params, grid, demos=None):
    """Recognizer over a token row: output [[1]] if the row is in the language, else [[0]]."""
    row = _row(grid)
    lang, o, c = int(np.argmax(params["lang"])), int(np.argmax(params["open"])), int(np.argmax(params["close"]))
    if lang == 0:
        depth, ok = 0, all(v in (o, c) for v in row)
        for v in row:
            depth += 1 if v == o else -1
            ok &= depth >= 0
        ok &= depth == 0
    elif lang == 1:
        ok = row == row[::-1]
    elif lang == 2:
        n = len(row) // 2
        ok = len(row) >= 2 and len(row) % 2 == 0 and row == [o] * n + [c] * n
    elif lang == 3:
        ok = row.count(o) % 2 == 0
    elif lang == 4:
        ok = any(row[i] == o and row[i + 1] == c for i in range(len(row) - 1))
    else:
        ok = len(set(row)) == 1
    return [[1 if ok else 0]]


def successor_table(params, grid, demos=None):
    """Demonstration-aware bigram / trigram model: the next token is the most common demonstrated continuation of the last k tokens."""
    k = int(np.argmax(params["order"])) + 1
    row = _row(grid)
    if len(row) < k:
        return [[0]]
    table = {}
    for d in demos or []:
        x, y = _row(d["input"]), _row(d["output"])
        if len(x) >= k and len(y) == 1:
            bucket = table.setdefault(tuple(x[-k:]), {})
            bucket[y[0]] = bucket.get(y[0], 0) + 1
    bucket = table.get(tuple(row[-k:]))
    if not bucket:
        return [[0]]
    return [[max(bucket, key=lambda v: (bucket[v], -v))]]


def pair_agree(params, grid, demos=None):
    """Agreement between two tokens through learned categories: cls[token] one-hot over 3 classes; rel 0 = same class is valid, 1 = different."""
    row = _row(grid)
    if len(row) != 2:
        raise ValueError("a token pair is required")
    cls = np.asarray(params["cls"], float)
    same = int(np.argmax(cls[row[0]])) == int(np.argmax(cls[row[1]]))
    valid = same if int(np.argmax(params["rel"])) == 0 else not same
    return [[1 if valid else 0]]


TEMPLATES.update({
    "recall_attend": {"shapes": {"delim": (10,), "order": (2,)}, "fn": recall_attend, "aware": False, "niche": "recall",
                      "doc": ("LANGUAGE, associative recall. The input is a 1-row token sequence: pairs k1 v1 k2 v2 ... then the delimiter token d = argmax(delim), then one "
                              "query token q. order = argmax(order): 0 the first token of a pair is the key and the second the value; 1 the roles are swapped. The output is "
                              "the 1x1 grid holding the value stored for q (the last pair with that key wins), or 0 if q is not stored.")},
    "induction_next": {"shapes": {"which": (3,), "n": (2,)}, "fn": induction_next, "aware": False, "niche": "induction",
                       "doc": ("LANGUAGE, induction. The input is a 1-row token sequence. The final n tokens (n = argmax(n)+1) are searched for earlier in the sequence; the output "
                               "is the 1x1 grid holding the token that followed the earlier match: which = argmax(which): 0 the last earlier match, 1 the first earlier match, "
                               "2 the most common follower (ties to the smaller token). 0 if there is no earlier match.")},
    "seq_transform": {"shapes": {"op": (10,)}, "fn": seq_transform, "aware": False, "niche": "seqops",
                      "doc": ("LANGUAGE, sequence operations on a 1-row token sequence. op = argmax(op): 0 reverse; 1 sort ascending; 2 sort descending; 3 collapse consecutive "
                              "duplicates; 4 keep only the first occurrence of each token; 5 rotate left by one; 6 rotate right by one; 7 replace every token by the most common "
                              "token; 8 sort by frequency (most frequent first, ties by token); 9 remove all zeros.")},
    "selective_copy": {"shapes": {"noise": (10,)}, "fn": selective_copy, "aware": False, "niche": "selcopy",
                       "doc": ("LANGUAGE, selective copying. The output is the input token sequence with every token equal to noise_colour = argmax(noise) removed, order kept.")},
    "grammar_check": {"shapes": {"lang": (6,), "open": (10,), "close": (10,)}, "fn": grammar_check, "aware": False, "niche": "grammar",
                      "doc": ("LANGUAGE, grammar recognizer. Output [[1]] if the 1-row token sequence is in the language, else [[0]]. Tokens o = argmax(open), c = argmax(close). "
                              "lang = argmax(lang): 0 balanced brackets over o (open) and c (close), nothing else allowed; 1 palindrome; 2 o repeated n times then c repeated n "
                              "times (n >= 1); 3 the number of o tokens is even; 4 the sequence contains o immediately followed by c; 5 all tokens are equal.")},
    "successor_table": {"shapes": {"order": (2,)}, "fn": successor_table, "aware": True, "niche": "successor",
                        "doc": ("LANGUAGE, in-context next-token prediction. Reads the demonstration pairs (token row -> 1x1 next token) as context. context length k = argmax(order)+1: "
                                "the key is the last k tokens of the row. For each key the demonstrations give the most common next token; a query row outputs the next token of "
                                "its key, or 0 if the key was never seen.")},
    "pair_agree": {"shapes": {"cls": (10, 3), "rel": (2,)}, "fn": pair_agree, "aware": False, "niche": "agree",
                   "doc": ("LANGUAGE, agreement through learned categories. The input is a 1x2 token pair. Every token t has a class argmax(cls[t]) in 0..2. rel = argmax(rel): 0 the "
                           "pair is valid when both tokens are in the SAME class, 1 when they are in DIFFERENT classes. Output [[1]] if valid else [[0]]. The category of each "
                           "token must be inferred from the demonstrations.")},
})


# ------------------------------------------------------------------ more ARC-driven templates
def gravity(params, grid, demos=None):
    a = np.asarray(grid)
    bg, d = int(np.argmax(params["bg"])), int(np.argmax(params["dir"]))
    h, w = a.shape
    out = np.full_like(a, bg)
    if d in (0, 1):
        for c in range(w):
            col = [v for v in a[:, c] if v != bg]
            if d == 0:
                out[h - len(col):, c] = col
            else:
                out[:len(col), c] = col
    else:
        for r in range(h):
            row = [v for v in a[r] if v != bg]
            if d == 2:
                out[r, :len(row)] = row
            else:
                out[r, w - len(row):] = row
    return out.astype(int).tolist()


def fill_enclosed(params, grid, demos=None):
    """Demonstration-aware: background regions that cannot reach the border through background cells take the fill colour seen in the demonstrations."""
    a = np.asarray(grid)
    bg, eight = int(np.argmax(params["bg"])), int(np.argmax(params["conn"])) == 1
    counts = {}
    for d in demos or []:
        x, y = np.asarray(d["input"]), np.asarray(d["output"])
        if x.shape == y.shape:
            for v in y[(x == bg) & (y != bg)]:
                counts[int(v)] = counts.get(int(v), 0) + 1
    if not counts:
        return a.tolist()
    fill = max(counts, key=lambda v: (counts[v], -v))
    h, w = a.shape
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (eight or not (dr and dc))]
    reach = np.zeros((h, w), bool)
    stack = [(r, c) for r in range(h) for c in range(w) if (r in (0, h - 1) or c in (0, w - 1)) and a[r, c] == bg]
    for r, c in stack:
        reach[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not reach[rr, cc] and a[rr, cc] == bg:
                reach[rr, cc] = True
                stack.append((rr, cc))
    out = a.copy()
    out[(a == bg) & ~reach] = fill
    return out.astype(int).tolist()


def symmetry_complete(params, grid, demos=None):
    a = np.asarray(grid).copy()
    m, sym = int(np.argmax(params["mask"])), int(np.argmax(params["sym"]))
    h, w = a.shape
    if sym == 3 and h != w:
        raise ValueError("transpose symmetry needs a square grid")
    mirrors = {0: lambda r, c: [(r, w - 1 - c)], 1: lambda r, c: [(h - 1 - r, c)], 2: lambda r, c: [(r, w - 1 - c), (h - 1 - r, c), (h - 1 - r, w - 1 - c)],
               3: lambda r, c: [(c, r)]}[sym]
    src = a.copy()
    for r in range(h):
        for c in range(w):
            if src[r, c] == m:
                for rr, cc in mirrors(r, c):
                    if src[rr, cc] != m:
                        a[r, c] = src[rr, cc]
                        break
    return a.tolist()


def scale_grid(params, grid, demos=None):
    a = np.asarray(grid)
    mode = int(np.argmax(params["mode"]))
    ups = {0: (2, 2), 1: (3, 3), 2: (4, 4), 5: (2, 1), 6: (1, 2), 7: (2, 3)}
    if mode in ups:
        return np.kron(a, np.ones(ups[mode], int)).astype(int).tolist()
    f = 2 if mode == 3 else 3
    if a.shape[0] % f or a.shape[1] % f:
        raise ValueError("grid not divisible")
    return a[::f, ::f].astype(int).tolist()


def half_boolean(params, grid, demos=None):
    """Two halves of the grid (split by a separator line or in the middle) are combined cell by cell with a boolean op on 'is non-background'."""
    a = np.asarray(grid)
    axis, sep, op, outc = int(np.argmax(params["axis"])), int(np.argmax(params["sep"])), int(np.argmax(params["op"])), int(np.argmax(params["outc"]))
    if axis == 1:
        a = a.T
    h, w = a.shape
    if sep == 10:
        if w % 2:
            raise ValueError("odd width without separator")
        left, right = a[:, :w // 2], a[:, w // 2:]
    else:
        cols = [c for c in range(w) if (a[:, c] == sep).all()]
        if not cols:
            raise ValueError("no separator line")
        c = cols[0]
        left, right = a[:, :c], a[:, c + 1:]
    if left.shape != right.shape:
        raise ValueError("halves differ")
    A, B = left != 0, right != 0
    res = [A & B, A | B, A ^ B, ~(A | B), A & ~B][min(op, 4)]
    out = np.where(res, outc, 0)
    return (out.T if axis == 1 else out).astype(int).tolist()


def denoise_small(params, grid, demos=None):
    a = np.asarray(grid)
    bg, eight, limit = int(np.argmax(params["bg"])), int(np.argmax(params["conn"])) == 1, int(np.argmax(params["maxnoise"])) + 1
    out = a.copy()
    for o in objects(a, bg, eight=eight, same_color=True):
        if len(o["cells"]) <= limit:
            for r, c in o["cells"]:
                out[r, c] = bg
    return out.astype(int).tolist()


TEMPLATES.update({
    "gravity": {"shapes": {"dir": (4,), "bg": (10,)}, "fn": gravity, "aware": False, "niche": "gravity",
                "doc": ("Every non-background pixel (background = argmax(bg)) falls in direction argmax(dir): 0 down, 1 up, 2 left, 3 right. Pixels keep their order within their column "
                        "(rows for left/right) and pile up against that edge.")},
    "fill_enclosed": {"shapes": {"bg": (10,), "conn": (2,)}, "fn": fill_enclosed, "aware": True, "niche": "fillenc",
                      "doc": ("Reads the demonstration pairs as context to learn the fill colour (the most common colour that appears at background pixels that changed). Background "
                              "cells (background = argmax(bg)) that cannot reach the grid border through background cells (4-connected if argmax(conn)=0, 8-connected if 1) are "
                              "enclosed and take the fill colour; everything else is unchanged.")},
    "symmetry_complete": {"shapes": {"mask": (10,), "sym": (4,)}, "fn": symmetry_complete, "aware": False, "niche": "symmetry",
                          "doc": ("Cells equal to mask_colour = argmax(mask) are missing. sym = argmax(sym): 0 mirror left-right, 1 mirror up-down, 2 four-fold (left-right, up-down and both), "
                                  "3 transpose (square grids). Each missing cell takes the colour of its mirror partner if that partner is known.")},
    "scale_grid": {"shapes": {"mode": (8,)}, "fn": scale_grid, "aware": False, "niche": "scale",
                   "doc": ("mode = argmax(mode): 0 enlarge every pixel to a 2x2 block; 1 to 3x3; 2 to 4x4; 3 shrink by 2 keeping every second pixel in both directions (needs even sizes); "
                           "4 shrink by 3 keeping every third pixel; 5 enlarge rows by 2 (2 high, 1 wide); 6 enlarge columns by 2 (1 high, 2 wide); 7 enlarge 2 high by 3 wide.")},
    "half_boolean": {"shapes": {"axis": (2,), "sep": (11,), "op": (5,), "outc": (10,)}, "fn": half_boolean, "aware": False, "niche": "halfbool",
                     "doc": ("The grid is split into two equal halves: axis = argmax(axis) 0 left | right, 1 top | bottom. sep = argmax(sep): 0..9 a separator line of that colour sits between the "
                             "halves (the first full line of that colour); 10 no separator, split in the middle. A cell is 'on' in a half if it is non-zero. op = argmax(op): 0 both on, "
                             "1 either on, 2 exactly one on, 3 neither on, 4 first on and second off. The output has the half's shape; on cells take colour argmax(outc), others 0.")},
    "denoise_small": {"shapes": {"bg": (10,), "conn": (2,), "maxnoise": (4,)}, "fn": denoise_small, "aware": False, "niche": "denoise",
                      "doc": ("Single-colour connected objects (background = argmax(bg); 4-connected if argmax(conn)=0, 8-connected if 1) with at most k = argmax(maxnoise)+1 pixels are noise "
                              "and are erased to background; larger objects stay.")},
})


# ------------------------------------------------------------------ structures requested by the maker's declined mutations
def object_move(params, grid, demos=None):
    """Rigid object movement: every single-colour 8-connected object slides (or steps once) in a direction until blocked by the edge or another object."""
    a = np.asarray(grid)
    bg, d, step_only = int(np.argmax(params["bg"])), int(np.argmax(params["dir"])), int(np.argmax(params["stop"])) == 1
    dr, dc = RAY_DIRS[d]
    h, w = a.shape
    objs = sorted(objects(a, bg, eight=True, same_color=True), key=lambda o: -max(r * dr + c * dc for r, c in o["cells"]))
    out = np.full_like(a, bg)
    occupied = np.zeros((h, w), bool)
    for o in objs:
        cells, shift = o["cells"], 0
        while True:
            nxt = [(r + (shift + 1) * dr, c + (shift + 1) * dc) for r, c in cells]
            if any(not (0 <= r < h and 0 <= c < w) or occupied[r, c] for r, c in nxt):
                break
            shift += 1
            if step_only:
                break
        for r, c in cells:
            out[r + shift * dr, c + shift * dc] = o["color"]
            occupied[r + shift * dr, c + shift * dc] = True
    return out.astype(int).tolist()


def flood_marker(params, grid, demos=None):
    """Flood fill from marker pixels through background cells (4- or 8-connected)."""
    a = np.asarray(grid)
    m, eight = int(np.argmax(params["marker"])), int(np.argmax(params["conn"])) == 1
    fillv = np.asarray(params["fill"], float)
    fill = int(np.argmax(fillv)) if fillv.max() > 0 else m
    h, w = a.shape
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (eight or not (dr and dc))]
    out = a.copy()
    seen = np.zeros((h, w), bool)
    stack = [(r, c) for r in range(h) for c in range(w) if a[r, c] == m]
    while stack:
        r, c = stack.pop()
        for dr, dc in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and a[rr, cc] == 0:
                seen[rr, cc] = True
                out[rr, cc] = fill
                stack.append((rr, cc))
    return out.astype(int).tolist()


def ray_cross(params, grid, demos=None):
    """ray_cast plus a dedicated colour for cells that two or more rays reach."""
    a = np.asarray(grid)
    h, w = a.shape
    src = np.asarray(params["src"], float) > 0
    dirs = [RAY_DIRS[k] for k in range(8) if params["dirs"][k] > 0]
    stop = int(np.argmax(params["stop"]))
    paint, cross = np.asarray(params["paint"], float), np.asarray(params["cross"], float)
    fixed = int(np.argmax(paint)) if paint.max() > 0 else None
    out, hits = a.copy(), np.zeros((h, w), int)
    for r in range(h):
        for c in range(w):
            if a[r, c] != 0 and src[a[r, c]]:
                color = fixed if fixed is not None else int(a[r, c])
                for dr, dc in dirs:
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < h and 0 <= cc < w:
                        blocked = a[rr, cc] != 0
                        if blocked and stop == 1:
                            break
                        if not (blocked and (stop == 2 or src[a[rr, cc]])):
                            out[rr, cc] = color
                            hits[rr, cc] += 1
                        rr, cc = rr + dr, cc + dc
    if cross.max() > 0:
        out[hits >= 2] = int(np.argmax(cross))
    return out.astype(int).tolist()


def nkey2(kind, a, r, c, bg):
    h, w = a.shape
    own = int(a[r, c])
    if kind == 0:
        return tuple(int(a[rr, cc]) if 0 <= rr < h and 0 <= cc < w else -1 for rr in (r - 1, r, r + 1) for cc in (c - 1, c, c + 1))
    rowset = tuple(sorted({int(v) for v in a[r] if v != bg}))
    colset = tuple(sorted({int(v) for v in a[:, c] if v != bg}))
    if kind == 1:
        return (own, rowset)
    if kind == 2:
        return (own, colset)
    if kind == 3:
        return (own, rowset, colset)
    if kind == 4:
        return (own, r % 2, c % 2)
    if kind == 5:
        return (own, int(r in (0, h - 1) or c in (0, w - 1)))
    if kind == 6:
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc, val = r + dr, c + dc, -1
            while 0 <= rr < h and 0 <= cc < w:
                if a[rr, cc] != bg:
                    val = int(a[rr, cc])
                    break
                rr, cc = rr + dr, cc + dc
            out.append(val)
        return (own, tuple(out))
    return (own, r % 3, c % 3)


def context_table2(params, grid, demos=None):
    """Demonstration-aware table over a wider context key (full 3x3 pattern, row/column colour sets, parity, border, nearest colours, mod-3 position)."""
    a = np.asarray(grid)
    kind, bg = int(np.argmax(params["keysel"])), int(np.argmax(params["bg"]))
    table = {}
    for d in demos or []:
        x, y = np.asarray(d["input"]), np.asarray(d["output"])
        if x.shape != y.shape:
            continue
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                bucket = table.setdefault(nkey2(kind, x, r, c, bg), {})
                bucket[int(y[r, c])] = bucket.get(int(y[r, c]), 0) + 1
    out = a.copy()
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            bucket = table.get(nkey2(kind, a, r, c, bg))
            if bucket:
                out[r, c] = max(bucket, key=lambda col: (bucket[col], -col))
    return out.astype(int).tolist()


TEMPLATES.update({
    "object_move": {"shapes": {"dir": (8,), "stop": (2,), "bg": (10,)}, "fn": object_move, "aware": False, "niche": "objmove",
                    "doc": ("Rigid movement. Objects are single-colour 8-connected components of non-background (background = argmax(bg)). dir = argmax(dir) in order: up, up-right, right, "
                            "down-right, down, down-left, left, up-left. stop = argmax(stop): 0 every object slides in that direction until the grid edge or another object blocks it "
                            "(objects nearest the destination edge settle first); 1 every object moves exactly one step if it is free. Shapes are preserved.")},
    "flood_marker": {"shapes": {"marker": (10,), "conn": (2,), "fill": (10,)}, "fn": flood_marker, "aware": False, "niche": "floodm",
                     "doc": ("Flood fill from marker pixels: every pixel of colour argmax(marker) spreads through background (colour 0) cells, 4-connected if argmax(conn)=0 and 8-connected if 1. "
                             "Reached cells take colour argmax(fill) if max(fill) > 0, otherwise the marker colour. Non-background cells block the flood.")},
    "ray_cross": {"shapes": {"src": (10,), "dirs": (8,), "stop": (3,), "paint": (10,), "cross": (10,)}, "fn": ray_cross, "aware": False, "niche": "raycross",
                  "doc": ("Same as ray_cast (sources are colours c with src[c] > 0; rays in directions with dirs[d] > 0 in order up, up-right, right, down-right, down, down-left, left, up-left; "
                          "stop = argmax(stop): 0 paint to the edge except source pixels, 1 stop before the first non-background cell, 2 skip over non-background cells; paint colour is "
                          "argmax(paint) if max(paint) > 0 else the source colour). Additionally, if max(cross) > 0, every painted cell reached by two or more rays takes colour argmax(cross).")},
    "context_table2": {"shapes": {"keysel": (8,), "bg": (10,)}, "fn": context_table2, "aware": True, "niche": "context2",
                       "doc": ("Reads the demonstration pairs as context. Key kind = argmax(keysel): 0 the full 3x3 pattern around the pixel (out-of-grid = -1); 1 own colour + set of non-background "
                               "colours in the same row; 2 own colour + set in the same column; 3 own colour + both sets; 4 own colour + (row parity, column parity); 5 own colour + on-border flag; "
                               "6 own colour + the nearest non-background colour looking up, down, left, right; 7 own colour + (row mod 3, column mod 3). Background = argmax(bg). "
                               "The demonstrations give, per key, the most common output colour; a query pixel takes its key's output or stays unchanged if the key was never seen.")},
})


# ------------------------------------------------------------------ structures found by inspecting the maker's declined mutations (round 2)
def object_move2(params, grid, demos=None):
    """object_move with a connectivity choice: objects are single-colour components, 4- or 8-connected."""
    a = np.asarray(grid)
    bg, d = int(np.argmax(params["bg"])), int(np.argmax(params["dir"]))
    step_only, eight = int(np.argmax(params["stop"])) == 1, int(np.argmax(params["conn"])) == 1
    dr, dc = RAY_DIRS[d]
    h, w = a.shape
    objs = sorted(objects(a, bg, eight=eight, same_color=True), key=lambda o: -max(r * dr + c * dc for r, c in o["cells"]))
    out = np.full_like(a, bg)
    occupied = np.zeros((h, w), bool)
    for o in objs:
        cells, shift = o["cells"], 0
        while True:
            nxt = [(r + (shift + 1) * dr, c + (shift + 1) * dc) for r, c in cells]
            if any(not (0 <= r < h and 0 <= c < w) or occupied[r, c] for r, c in nxt):
                break
            shift += 1
            if step_only:
                break
        for r, c in cells:
            out[r + shift * dr, c + shift * dc] = o["color"]
            occupied[r + shift * dr, c + shift * dc] = True
    return out.astype(int).tolist()


def flood_through(params, grid, demos=None):
    """Flood from marker pixels through cells whose colour is selected by `through` (multi-hot; background 0 is selected when through[0] > 0)."""
    a = np.asarray(grid)
    m, eight = int(np.argmax(params["marker"])), int(np.argmax(params["conn"])) == 1
    through = np.asarray(params["through"], float) > 0
    fillv = np.asarray(params["fill"], float)
    fill = int(np.argmax(fillv)) if fillv.max() > 0 else m
    h, w = a.shape
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (eight or not (dr and dc))]
    out = a.copy()
    seen = np.zeros((h, w), bool)
    stack = [(r, c) for r in range(h) for c in range(w) if a[r, c] == m]
    while stack:
        r, c = stack.pop()
        for dr, dc in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and a[rr, cc] != m and through[a[rr, cc]]:
                seen[rr, cc] = True
                out[rr, cc] = fill
                stack.append((rr, cc))
    return out.astype(int).tolist()


def snap_to_line(params, grid, demos=None):
    """Attraction: full rows/columns of one non-background colour are lines. Every other pixel moves perpendicular to the line of its own colour until it touches
    it; pixels whose colour has no line are removed (orphan = 0) or kept (orphan = 1). Lines stay."""
    a = np.asarray(grid)
    bg, keep = int(np.argmax(params["bg"])), int(np.argmax(params["orphan"])) == 1
    h, w = a.shape
    rows = {r: int(a[r, 0]) for r in range(h) if a[r, 0] != bg and (a[r] == a[r, 0]).all()}
    cols = {c: int(a[0, c]) for c in range(w) if a[0, c] != bg and (a[:, c] == a[0, c]).all()}
    out = np.full_like(a, bg)
    for r, col in rows.items():
        out[r, :] = col
    for c, col in cols.items():
        out[:, c] = col
    for r in range(h):
        for c in range(w):
            v = int(a[r, c])
            if v == bg or r in rows or c in cols:
                continue
            target = None
            for lr, lc in rows.items():
                if lc == v:
                    target = ("r", lr)
            for lc_, col in cols.items():
                if col == v and target is None:
                    target = ("c", lc_)
            if target is None:
                if keep:
                    out[r, c] = v
                continue
            if target[0] == "r":
                nr = target[1] + (1 if r > target[1] else -1)
                nc = c
            else:
                nr = r
                nc = target[1] + (1 if c > target[1] else -1)
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == bg:
                out[nr, nc] = v
    return out.astype(int).tolist()


TEMPLATES.update({
    "object_move2": {"shapes": {"dir": (8,), "stop": (2,), "bg": (10,), "conn": (2,)}, "fn": object_move2, "aware": False, "niche": "objmove2",
                     "doc": ("Rigid movement, as object_move, with the object definition chosen by conn = argmax(conn): 0 single-colour 4-connected components, 1 8-connected. Objects are non-background "
                             "(background = argmax(bg)). dir = argmax(dir) in order: up, up-right, right, down-right, down, down-left, left, up-left. stop = argmax(stop): 0 every object slides "
                             "until the edge or another object blocks it (objects nearest the destination settle first); 1 every object moves one step if free. Shapes are preserved.")},
    "flood_through": {"shapes": {"marker": (10,), "through": (10,), "conn": (2,), "fill": (10,)}, "fn": flood_through, "aware": False, "niche": "floodthru",
                      "doc": ("Flood from every pixel of colour argmax(marker). The flood spreads (4-connected if argmax(conn)=0, 8-connected if 1) into cells whose colour c has through[c] > 0 "
                              "(through[0] > 0 means background cells; through[8] > 0 means colour-8 cells, and so on). Reached cells take colour argmax(fill) if max(fill) > 0, otherwise the marker colour. "
                              "Cells of other colours stop the flood.")},
    "snap_to_line": {"shapes": {"bg": (10,), "orphan": (2,)}, "fn": snap_to_line, "aware": False, "niche": "snapline",
                     "doc": ("Lines are full rows or columns of a single non-background colour (background = argmax(bg)). Every other pixel moves perpendicular to the line of its OWN colour until it "
                             "touches that line (on its own side). Pixels whose colour has no line: orphan = argmax(orphan) 0 removes them, 1 keeps them in place. Lines stay.")},
})


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
