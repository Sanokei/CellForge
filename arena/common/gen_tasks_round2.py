"""Generate ARC-style train/test task pairs for arena round 2.

Round 2 is harder than round 1 along the exact axes that separated the
researchers in round 1 (Alpha 8/8 weight groups load-bearing vs Beta 2/7):

- compose:      two-stage transform (segment, then remap). A single-pass
                per-pixel heuristic cannot solve it; the weights must hold
                both a segmentation rule and a color map.
- denoise_hard: decoy shapes with MORE pixels than the target break the
                blob-size heuristic that solved round-1 denoise. The weights
                must test for a *solid filled rectangle*.
- mirror:       a spatial coordinate transform keyed to grid width
                (even/odd). Weights must encode mirroring, not a color map.

Train pairs -> researchers. Test pairs -> judge only (never shown).
Different random seeds guarantee train/test separation.

Usage:
    python3 gen_tasks_round2.py --selftest   # validate generators locally
    python3 gen_tasks_round2.py              # write to /srv/cellforge-arena
"""
import json
import random
import sys

ARENA = "/srv/cellforge-arena"


# ---------------------------------------------------------------- compose
def gen_compose(rng):
    """Solid rectangle of color c + sparse noise -> clean rectangle of 10-c."""
    h, w = rng.randint(6, 9), rng.randint(6, 9)
    c = rng.randint(1, 9)
    tw, th = rng.randint(2, 4), rng.randint(2, 4)
    x0 = rng.randint(0, w - tw)
    y0 = rng.randint(0, h - th)
    x1, y1 = x0 + tw - 1, y0 + th - 1
    grid = [[0] * w for _ in range(h)]
    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            grid[i][j] = c
    # sparse noise of other colors (may land on the rectangle too)
    for _ in range(rng.randint(3, 6)):
        grid[rng.randrange(h)][rng.randrange(w)] = rng.choice(
            [v for v in range(1, 10) if v != c])
    out = [[0] * w for _ in range(h)]
    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            out[i][j] = 10 - c
    meta = {"rect": (y0, x0, y1, x1), "color": c}
    return grid, out, meta


# ---------------------------------------------------------- denoise_hard
def _rect_cells(y0, x0, y1, x1):
    return {(i, j) for i in range(y0, y1 + 1) for j in range(x0, x1 + 1)}


def _frame_cells(y0, x0, y1, x1):
    cells = set()
    for j in range(x0, x1 + 1):
        cells.add((y0, j))
        cells.add((y1, j))
    for i in range(y0, y1 + 1):
        cells.add((i, x0))
        cells.add((i, x1))
    return cells


def gen_denoise_hard(rng):
    """One SOLID filled rectangle among decoys -> keep only the solid one.

    Decoys: a hollow frame (often MORE pixels than the target) and a random
    cluster. The round-1 heuristic "keep the largest connected component"
    fails; the weights must test rectangular solidity.
    """
    h, w = rng.randint(10, 14), rng.randint(10, 14)
    colors = rng.sample(range(1, 10), 3)
    c, d_frame, d_clust = colors

    # target: solid filled rectangle
    tw, th = rng.randint(3, 5), rng.randint(3, 5)
    tx0 = rng.randint(0, w - tw - 4)
    ty0 = rng.randint(0, h - th - 4)
    tx1, ty1 = tx0 + tw - 1, ty0 + th - 1
    target = _rect_cells(ty0, tx0, ty1, tx1)

    # forbidden: target cells plus their 4-neighbourhood (keep a 1-cell gap)
    forbidden = set(target)
    for (i, j) in list(target):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            forbidden.add((i + di, j + dj))

    grid = [[0] * w for _ in range(h)]
    for (i, j) in sorted(target):
        grid[i][j] = c

    # decoy 1: hollow frame, sized so it usually out-pixels the target
    frame = None
    for _ in range(60):
        fw = rng.randint(6, 9)
        fh = rng.randint(6, 9)
        fx0 = rng.randint(0, w - fw)
        fy0 = rng.randint(0, h - fh)
        cand = _frame_cells(fy0, fx0, fy0 + fh - 1, fx0 + fw - 1)
        if not (cand & forbidden):
            frame = cand
            break
    if frame is None:  # extremely unlikely; fall back to a small frame
        frame = _frame_cells(0, 0, 3, 3) - forbidden
    for (i, j) in sorted(frame):
        if 0 <= i < h and 0 <= j < w:
            grid[i][j] = d_frame

    # decoy 2: random-walk cluster, pixel count near/above target area
    cluster = set()
    for _ in range(80):
        sx, sy = rng.randrange(w), rng.randrange(h)
        if (sy, sx) in forbidden or grid[sy][sx] != 0:
            continue
        walk = {(sy, sx)}
        ci, cj = sy, sx
        want = len(target) + rng.randint(0, 6)
        while len(walk) < want:
            di, dj = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            ci = max(0, min(h - 1, ci + di))
            cj = max(0, min(w - 1, cj + dj))
            if (ci, cj) in forbidden or grid[ci][cj] != 0:
                break
            walk.add((ci, cj))
        if len(walk) >= 4:
            cluster = walk
            break
    for (i, j) in sorted(cluster):
        grid[i][j] = d_clust

    out = [[0] * w for _ in range(h)]
    for (i, j) in sorted(target):
        out[i][j] = c
    meta = {"target": sorted(target), "target_color": c,
            "frame_n": len(frame), "cluster_n": len(cluster)}
    return grid, out, meta


# --------------------------------------------------------------- mirror
def gen_mirror(rng):
    """Left-half pattern -> horizontally completed symmetric grid.

    Width parity changes the axis handling, so weights must encode a real
    coordinate transform, not a memorised map.
    """
    h = rng.randint(5, 8)
    w = rng.randint(6, 10)
    half = (w + 1) // 2  # columns the pattern may occupy (incl. middle)
    grid = [[0] * w for _ in range(h)]
    for _ in range(200):  # ensure a non-trivial pattern
        for i in range(h):
            for j in range(half):
                if rng.random() < 0.45:
                    grid[i][j] = rng.randint(1, 9)
        if sum(1 for i in range(h) for j in range(half) if grid[i][j]) >= 3:
            break
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            src = j if j < half else w - 1 - j
            out[i][j] = grid[i][src]
    meta = {"h": h, "w": w}
    return grid, out, meta


TASKS = {"compose": gen_compose, "denoise_hard": gen_denoise_hard,
         "mirror": gen_mirror}


def make_pairs(seed, n):
    rng = random.Random(seed)
    pairs = {}
    for name, gen in TASKS.items():
        pairs[name] = []
        for _ in range(n):
            inp, out, _ = gen(rng)
            pairs[name].append({"input": inp, "output": out})
    return pairs


def make_pairs_with_meta(seed, n):
    rng = random.Random(seed)
    pairs = {}
    for name, gen in TASKS.items():
        pairs[name] = []
        for _ in range(n):
            inp, out, meta = gen(rng)
            pairs[name].append({"input": inp, "output": out, "meta": meta})
    return pairs


# ------------------------------------------------------------ self-test
def _largest_component(grid):
    """Round-1-style naive heuristic: keep the largest connected component."""
    h, w = len(grid), len(grid[0])
    seen = set()
    best = set()
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 or (i, j) in seen:
                continue
            stack = [(i, j)]
            comp = set()
            seen.add((i, j))
            while stack:
                ci, cj = stack.pop()
                comp.add((ci, cj))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = ci + di, cj + dj
                    if (0 <= ni < h and 0 <= nj < w and grid[ni][nj] != 0
                            and (ni, nj) not in seen):
                        seen.add((ni, nj))
                        stack.append((ni, nj))
            if len(comp) > len(best):
                best = comp
    out = [[0] * w for _ in range(h)]
    for (i, j) in best:
        out[i][j] = grid[i][j]
    return out


def selftest():
    # 1. determinism
    a = make_pairs(123, 4)
    b = make_pairs(123, 4)
    assert a == b, "generator not deterministic"
    # 2. train/test separation
    t = make_pairs_with_meta(20261016, 4)
    assert t != make_pairs_with_meta(135792468, 6) or True
    te = make_pairs(135792468, 6)
    assert json.dumps(t) != json.dumps(te), "train/test seeds collide"
    # 3. JSON-serialisable
    json.dumps(te)

    comp = make_pairs_with_meta(20261016, 4)
    test = make_pairs_with_meta(135792468, 6)
    for split in (comp, test):
        for name, pairs in split.items():
            for p in pairs:
                assert p["input"] != p["output"], f"{name}: degenerate pair"

    # compose: output is exactly the rect in 10-c, noise gone
    for p in test["compose"]:
        meta = p["meta"]
        y0, x0, y1, x1 = meta["rect"]
        c = meta["color"]
        for i, row in enumerate(p["output"]):
            for j, v in enumerate(row):
                want = (10 - c) if (y0 <= i <= y1 and x0 <= j <= x1) else 0
                assert v == want, "compose output != clean remapped rect"
        # naive single-pass global remap must FAIL (noise would be remapped)
        naive = [[(10 - v) if v else 0 for v in row] for row in p["input"]]
        assert naive != p["output"], "compose solvable by single-pass remap"

    # denoise_hard: output == target solid rect; decoys absent
    for p in test["denoise_hard"]:
        tgt = {tuple(c) for c in p["meta"]["target"]}
        got = {(i, j) for i, row in enumerate(p["output"])
               for j, v in enumerate(row) if v}
        assert got == tgt, "denoise_hard output != target rect"
        h, w = len(p["input"]), len(p["input"][0])
        assert 10 <= h <= 14 and 10 <= w <= 14, "grid size out of range"

    # the round-1 heuristic must fail on a majority of denoise_hard tests
    fails = sum(1 for p in test["denoise_hard"]
                if _largest_component(p["input"]) != p["output"])
    print(f"denoise_hard: round-1 heuristic fails on "
          f"{fails}/{len(test['denoise_hard'])} test pairs")
    assert fails >= len(test["denoise_hard"]) // 2 + 1, \
        "decoys do not break the old heuristic"

    # mirror: output symmetric, left half matches input
    for p in test["mirror"]:
        h = len(p["output"])
        w = len(p["output"][0])
        assert 6 <= w <= 10, "mirror width out of range"
        for i in range(h):
            for j in range(w):
                assert p["output"][i][j] == p["output"][i][w - 1 - j], \
                    "mirror output not symmetric"
        half = (w + 1) // 2
        for i in range(h):
            for j in range(half):
                assert p["output"][i][j] == p["input"][i][j], \
                    "mirror changed the given half"

    print("SELFTEST OK: determinism, separation, non-degeneracy, "
          "compose two-stage, denoise_hard decoys, mirror symmetry")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        train = make_pairs(seed=20261016, n=4)
        test = make_pairs(seed=135792468, n=6)
        with open(f"{ARENA}/common/train_round2.json", "w") as f:
            json.dump(train, f)
        with open(f"{ARENA}/judge/heldout/test_round2.json", "w") as f:
            json.dump(test, f)
        print("train tasks:", {k: len(v) for k, v in train.items()})
        print("test tasks:", {k: len(v) for k, v in test.items()})
        print("WROTE OK")
