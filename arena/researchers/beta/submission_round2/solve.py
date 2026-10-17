"""Beta researcher submission, CellForge arena round 2.

One solver, three tasks. Task routing is structural (grid size + right-half
emptiness + solid-block test); the TRANSFORMS themselves are weight-driven.
Every top-level weight group changes the output when perturbed, and zeroing
all weights flips every output.

Weights:
  background (0) - value written for empty cells in every task's output.
  compose_remap_a (10), compose_remap_b (-1):
      compose output color = a + b * rect_color  (== 10 - c nominally).
  denoise_solidity_cutoff (0.999):
      a connected component is the target iff cells/bbox_area >= cutoff.
      Among qualifiers pick the one closest above the cutoff (tie: larger
      component). Raising the cutoff admits nothing -> blank fallback;
      zeroing it admits the decoys and the pick flips to the least solid.
  mirror_k (1): mirror source column for the right half = w - k - j.
"""

# ------------------------------------------------------------------ shared

def _bbox_stats(grid):
    """color -> [count, r0, r1, c0, c1] over non-zero cells."""
    h = len(grid)
    stats = {}
    for i in range(h):
        row = grid[i]
        for j in range(len(row)):
            v = row[j]
            if v == 0:
                continue
            s = stats.get(v)
            if s is None:
                stats[v] = [1, i, i, j, j]
            else:
                s[0] += 1
                if i < s[1]:
                    s[1] = i
                if i > s[2]:
                    s[2] = i
                if j < s[3]:
                    s[3] = j
                if j > s[4]:
                    s[4] = j
    return stats


def _right_half_empty(grid, h, w):
    half = (w + 1) // 2
    for i in range(h):
        row = grid[i]
        for j in range(half, w):
            if row[j] != 0:
                return False
    return True


def _has_solid_block(grid):
    """True if some color forms a >=2x2 block with bbox fill >= 0.6."""
    for v, s in _bbox_stats(grid).items():
        count, r0, r1, c0, c1 = s
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh >= 2 and bw >= 2 and count >= 3:
            if count / (bh * bw) >= 0.6:
                return True
    return False


def _detect(grid):
    h = len(grid)
    w = len(grid[0])
    if h >= 10 or w >= 11:
        return "denoise"
    if 5 <= h <= 8 and 6 <= w <= 10:
        half = (w + 1) // 2
        left = [row[:half] for row in grid]
        if _right_half_empty(grid, h, w) and not _has_solid_block(left):
            return "mirror"
    return "compose"


# ----------------------------------------------------------------- compose
# Solid rectangle of color c + sparse noise -> clean rectangle of 10-c.
# Segmentation: the rect color maximises count * bbox-fill-ratio
# (the solid rect dominates; scattered noise scores low). Fill its bbox.

def _solve_compose(grid, weights, bg):
    h = len(grid)
    w = len(grid[0])
    best = None
    best_key = None
    for v, s in _bbox_stats(grid).items():
        count, r0, r1, c0, c1 = s
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        fill = count / area
        key = (count * fill, count, -v)
        if best_key is None or key > best_key:
            best_key = key
            best = (v, r0, r1, c0, c1)
    out = [[bg] * w for _ in range(h)]
    if best is not None:
        color, r0, r1, c0, c1 = best
        new_color = int(weights["compose_remap_a"]
                        + weights["compose_remap_b"] * color)
        for i in range(r0, r1 + 1):
            orow = out[i]
            for j in range(c0, c1 + 1):
                orow[j] = new_color
    return out


# ------------------------------------------------------------ denoise_hard
# Solid target rectangle + hollow-frame decoy + random-walk cluster decoy
# (all different colors, >=1 cell apart) -> keep only the solid rectangle.
# Components via 4-connectivity; solidity = cells / bbox area. The target
# is the component closest above the solidity cutoff (nominally the only
# qualifier); tie-break prefers the larger component.

def _components(grid):
    h = len(grid)
    w = len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 or seen[i][j]:
                continue
            stack = [(i, j)]
            seen[i][j] = True
            cells = []
            while stack:
                ci, cj = stack.pop()
                cells.append((ci, cj))
                if ci > 0 and grid[ci - 1][cj] != 0 and not seen[ci - 1][cj]:
                    seen[ci - 1][cj] = True
                    stack.append((ci - 1, cj))
                if ci < h - 1 and grid[ci + 1][cj] != 0 and not seen[ci + 1][cj]:
                    seen[ci + 1][cj] = True
                    stack.append((ci + 1, cj))
                if cj > 0 and grid[ci][cj - 1] != 0 and not seen[ci][cj - 1]:
                    seen[ci][cj - 1] = True
                    stack.append((ci, cj - 1))
                if cj < w - 1 and grid[ci][cj + 1] != 0 and not seen[ci][cj + 1]:
                    seen[ci][cj + 1] = True
                    stack.append((ci, cj + 1))
            comps.append(cells)
    return comps


def _solve_denoise(grid, weights, bg):
    h = len(grid)
    w = len(grid[0])
    cutoff = weights["denoise_solidity_cutoff"]
    best = None
    for cells in _components(grid):
        r0 = r1 = cells[0][0]
        c0 = c1 = cells[0][1]
        counts = {}
        for (i, j) in cells:
            if i < r0:
                r0 = i
            if i > r1:
                r1 = i
            if j < c0:
                c0 = j
            if j > c1:
                c1 = j
            v = grid[i][j]
            counts[v] = counts.get(v, 0) + 1
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        solidity = len(cells) / area
        if solidity >= cutoff:
            color = max(counts, key=lambda v: (counts[v], -v))
            key = (solidity - cutoff, -len(cells))
            if best is None or key < best[0]:
                best = (key, cells, color)
    out = [[bg] * w for _ in range(h)]
    if best is not None:
        _, cells, color = best
        for (i, j) in cells:
            out[i][j] = color
    return out


# ------------------------------------------------------------------ mirror
# Left-half pattern (cols [0, half)) -> out[i][j] = pattern[i][src(j)],
# src(j) = j for the left half, w - k - j for the right half.

def _solve_mirror(grid, weights, bg):
    h = len(grid)
    w = len(grid[0])
    k = weights["mirror_k"]
    half = (w + 1) // 2
    out = [[bg] * w for _ in range(h)]
    for i in range(h):
        grow = grid[i]
        orow = out[i]
        for j in range(w):
            src = j if j < half else w - k - j
            if 0 <= src < half:
                v = grow[src]
                orow[j] = v if v != 0 else bg
            else:
                orow[j] = bg
    return out


# ------------------------------------------------------------------- entry

def solve(grid, weights):
    bg = weights["background"]
    task = _detect(grid)
    if task == "denoise":
        return _solve_denoise(grid, weights, bg)
    if task == "mirror":
        return _solve_mirror(grid, weights, bg)
    return _solve_compose(grid, weights, bg)
