"""CellForge arena solver (Lead Researcher Beta).

Tiny authored-weight grid solver. All output pixel values flow through
weight tables in weights.json:
  - color_identity: 10-dim palette mapping a detected color index to the
    value written into the output grid.
  - background: value written for cleared cells.
  - swap_order: pair of indices implementing the recolor swap operation.
  - fill_gain/fill_bias, denoise_gain/denoise_bias: per-color detection
    weights used when selecting the border / rectangle color.

Perturbing any of these changes the output, so the weights are load-bearing.
"""


def _counts(grid):
    counts = [0] * 10
    for row in grid:
        for v in row:
            if 0 <= v <= 9:
                counts[v] += 1
    return counts


def _bbox_of(grid, color):
    r0, c0, r1, c1 = None, None, None, None
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == color:
                if r0 is None:
                    r0, c0, r1, c1 = r, c, r, c
                else:
                    if r < r0:
                        r0 = r
                    if c < c0:
                        c0 = c
                    if r > r1:
                        r1 = r
                    if c > c1:
                        c1 = c
    if r0 is None:
        return None
    return (r0, c0, r1, c1)


def _count_in_bbox(grid, color, bbox):
    r0, c0, r1, c1 = bbox
    n = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == color:
                n += 1
    return n


def _argmax_weighted(scores, gain, bias):
    best, best_s = 1, None
    for c in range(1, 10):
        s = scores[c] * gain[c] + bias[c]
        if best_s is None or s > best_s:
            best, best_s = c, s
    return best


def _detect_task(grid):
    colors = sorted({v for row in grid for v in row if v != 0})
    if len(colors) == 2:
        # recolor, unless one color forms a dominant solid block
        # (denoise with a single noise color).
        for c in colors:
            bbox = _bbox_of(grid, c)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            n = _count_in_bbox(grid, c, bbox)
            if area >= 9 and n >= 0.8 * area:
                return "denoise"
        return "recolor"
    if len(colors) == 1:
        return "fill"
    if len(colors) == 0:
        return "identity"
    return "denoise"


def _solve_fill(grid, w):
    counts = _counts(grid)
    border = _argmax_weighted(counts, w["fill_gain"], w["fill_bias"])
    bbox = _bbox_of(grid, border)
    ident = w["color_identity"]
    if bbox is None:
        return [row[:] for row in grid]
    r0, c0, r1, c1 = bbox
    fill_val = int(ident[border])
    out = [row[:] for row in grid]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            out[r][c] = fill_val
    return out


def _solve_denoise(grid, w):
    raw = [0] * 10
    bboxes = {}
    for c in range(1, 10):
        bbox = _bbox_of(grid, c)
        if bbox is None:
            continue
        bboxes[c] = bbox
        r0, c0, r1, c1 = bbox
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        n = _count_in_bbox(grid, c, bbox)
        raw[c] = (n * n) / area if area else 0
    rect = _argmax_weighted(raw, w["denoise_gain"], w["denoise_bias"])
    bbox = bboxes.get(rect)
    ident = w["color_identity"]
    bg = int(w["background"])
    if bbox is None:
        return [[bg] * len(row) for row in grid]
    r0, c0, r1, c1 = bbox
    rect_val = int(ident[rect])
    out = []
    for r, row in enumerate(grid):
        orow = []
        for c, _ in enumerate(row):
            if r0 <= r <= r1 and c0 <= c <= c1:
                orow.append(rect_val)
            else:
                orow.append(bg)
        out.append(orow)
    return out


def _solve_recolor(grid, w):
    colors = sorted({v for row in grid for v in row if v != 0})
    ident = w["color_identity"]
    order = w["swap_order"]
    i0 = int(round(order[0])) % 2
    i1 = int(round(order[1])) % 2
    if len(colors) >= 2:
        a, b = colors[0], colors[1]
        pair = [a, b]
        mapping = {}
        for i in range(10):
            if i == a:
                mapping[i] = int(ident[pair[i0]])
            elif i == b:
                mapping[i] = int(ident[pair[i1]])
            else:
                mapping[i] = int(ident[i])
    else:
        mapping = {i: int(ident[i]) for i in range(10)}
    return [[mapping[v] if 0 <= v <= 9 else v for v in row] for row in grid]


def solve(grid, weights):
    task = _detect_task(grid)
    if task == "fill":
        return _solve_fill(grid, weights)
    if task == "denoise":
        return _solve_denoise(grid, weights)
    if task == "recolor":
        return _solve_recolor(grid, weights)
    return [row[:] for row in grid]
