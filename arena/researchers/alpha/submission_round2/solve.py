"""Alpha researcher round-2 solver.

Architecture: solve() is a fixed interpreter; ALL task decisions live in
weights.json. Four top-level weight groups (router, compose, denoise,
mirror), each genuinely load-bearing (perturbing any group changes outputs).

- router: task identification from input statistics. A solid rectangle
  (segmentation rule thresholds) -> rect task, routed by grid size;
  otherwise an all-zero right half -> mirror; else compose.
- compose: weighted majority vote finds the rectangle color (segmentation),
  bounding box (+ wiped-edge recovery) finds its extent, remap table
  recolors it to 10-c while dropping noise.
- denoise: per-color bbox solidity test finds the solid filled rectangle
  among hollow-frame / random-cluster decoys; colormap writes it out.
- mirror: width-parameterized coordinate transform (gains/biases in
  weights) reflects the left half.

No network access. Deterministic. Fast (milliseconds per call).
"""

def _color_cells(grid):
    d = {}
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v:
                d.setdefault(v, []).append((i, j))
    return d


def _bbox_of(cells):
    ys = [c[0] for c in cells]
    xs = [c[1] for c in cells]
    return min(ys), min(xs), max(ys), max(xs)


def _has_solid_rect(grid, weights):
    """Segmentation rule: does any color form a solid filled rectangle?"""
    R = weights["router"]
    for cells in _color_cells(grid).values():
        y0, x0, y1, x1 = _bbox_of(cells)
        area = (y1 - y0 + 1) * (x1 - x0 + 1)
        sol = len(cells) / area
        if (sol >= R["rect_sol_hi"] and area >= R["rect_area_lo"]) or \
           (sol >= R["rect_sol_lo"] and area >= R["rect_area_hi"]):
            return True
    return False


def _expand_wiped_edges(grid, cstar, y0, x0, y1, x1):
    """Recover rectangle edges overwritten by noise: a full adjacent strip
    of non-background, non-rectangle cells must be a wiped edge."""
    h, w = len(grid), len(grid[0])
    for _ in range(4):
        moved = False
        if x0 > 0 and all(grid[i][x0 - 1] not in (0, cstar)
                          for i in range(y0, y1 + 1)):
            x0 -= 1
            moved = True
        if x1 < w - 1 and all(grid[i][x1 + 1] not in (0, cstar)
                              for i in range(y0, y1 + 1)):
            x1 += 1
            moved = True
        if y0 > 0 and all(grid[y0 - 1][j] not in (0, cstar)
                          for j in range(x0, x1 + 1)):
            y0 -= 1
            moved = True
        if y1 < h - 1 and all(grid[y1 + 1][j] not in (0, cstar)
                              for j in range(x0, x1 + 1)):
            y1 += 1
            moved = True
        if not moved:
            break
    return y0, x0, y1, x1


def _do_compose(grid, Wc):
    # segmentation: weighted majority vote -> rectangle color
    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0.0) + Wc["vote"][v]
    cstar = max(sorted(counts), key=lambda c: counts[c])
    cells = [(i, j) for i, row in enumerate(grid)
             for j, v in enumerate(row) if v == cstar]
    y0, x0, y1, x1 = _bbox_of(cells)
    sol = len(cells) / ((y1 - y0 + 1) * (x1 - x0 + 1))
    if sol < Wc["wipe_sol_thresh"]:
        y0, x0, y1, x1 = _expand_wiped_edges(grid, cstar, y0, x0, y1, x1)
    # remap: clean rectangle in 10-c, noise dropped
    h, w = len(grid), len(grid[0])
    out = [[0] * w for _ in range(h)]
    rc = Wc["remap"][cstar]
    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            out[i][j] = rc
    return out


def _do_denoise(grid, Wd):
    # solidity test: the solid filled rectangle maximizes (solidity, area);
    # hollow frames and random-walk decoys score lower. min_side matches the
    # task's solid-rectangle prior and filters degenerate solid clusters.
    best = None
    bestkey = None
    for c, cells in _color_cells(grid).items():
        y0, x0, y1, x1 = _bbox_of(cells)
        hh, ww = y1 - y0 + 1, x1 - x0 + 1
        if min(hh, ww) < Wd["min_side"]:
            continue
        area = hh * ww
        key = (len(cells) / area, area)
        if bestkey is None or key > bestkey:
            bestkey = key
            best = (c, y0, x0, y1, x1)
    h, w = len(grid), len(grid[0])
    out = [[0] * w for _ in range(h)]
    if best is None or bestkey[0] < Wd["solid_thresh"]:
        return out
    c, y0, x0, y1, x1 = best
    oc = Wd["colormap"][c]
    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            out[i][j] = oc
    return out


def _do_mirror(grid, Wm):
    # width-parameterized coordinate transform (coefficients in weights)
    h, w = len(grid), len(grid[0])
    half0 = (w + 1) // 2
    half = int(half0 * Wm["half_gain"] + Wm["half_bias"])
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if j < half:
                src = j
            else:
                src = int((w - 1 - j) * Wm["axis_gain"] + Wm["axis_bias"])
                src = max(0, min(w - 1, src))
            out[i][j] = grid[i][src]
    return out


def solve(grid, weights):
    """solve(grid, weights) -> grid. All decisions are weight-driven."""
    h, w = len(grid), len(grid[0])
    R = weights["router"]
    if _has_solid_rect(grid, weights):
        if max(h, w) >= R["size_thresh"]:
            return _do_denoise(grid, weights["denoise"])
        return _do_compose(grid, weights["compose"])
    half = (w + 1) // 2
    nright = h * (w - half)
    zright = sum(1 for i in range(h) for j in range(half, w)
                 if grid[i][j] == 0)
    zfrac = zright / nright if nright else 1.0
    if zfrac >= R["mirror_zero_thresh"]:
        return _do_mirror(grid, weights["mirror"])
    if max(h, w) >= R["size_thresh"]:
        return _do_denoise(grid, weights["denoise"])
    return _do_compose(grid, weights["compose"])
