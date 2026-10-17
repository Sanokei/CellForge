"""Generate ARC-style train/test task pairs for arena round 1.

Train pairs -> researchers. Test pairs -> judge only (never shown to researchers).
Different random seeds guarantee train/test separation.
"""
import json
import random


def gen_fill(rng):
    h, w = rng.randint(5, 7), rng.randint(5, 7)
    color = rng.randint(1, 9)
    grid = [[0] * w for _ in range(h)]
    for i in range(h):
        grid[i][0] = grid[i][w - 1] = color
    for j in range(w):
        grid[0][j] = grid[h - 1][j] = color
    out = [[color] * w for _ in range(h)]
    return grid, out


def gen_denoise(rng):
    h, w = rng.randint(6, 8), rng.randint(6, 8)
    color = rng.randint(1, 9)
    # main blob: random rectangle
    x0, y0 = rng.randint(0, 2), rng.randint(0, 2)
    x1, y1 = rng.randint(x0 + 2, w - 1), rng.randint(y0 + 2, h - 1)
    grid = [[0] * w for _ in range(h)]
    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            grid[i][j] = color
    # noise pixels of other colors
    for _ in range(rng.randint(3, 6)):
        grid[rng.randrange(h)][rng.randrange(w)] = rng.choice([c for c in range(1, 10) if c != color])
    out = [[0] * w for _ in range(h)]
    for i in range(y0, y1 + 1):
        for j in range(x0, x1 + 1):
            out[i][j] = color
    return grid, out


def gen_recolor(rng):
    h, w = rng.randint(4, 6), rng.randint(4, 6)
    c1, c2 = rng.sample(range(1, 10), 2)
    grid = [[rng.choice([0, c1, c2]) for _ in range(w)] for _ in range(h)]
    out = [[{c1: c2, c2: c1}.get(v, v) for v in row] for row in grid]
    return grid, out


TASKS = {"fill": gen_fill, "denoise": gen_denoise, "recolor": gen_recolor}


def make_pairs(seed, n):
    rng = random.Random(seed)
    pairs = {}
    for name, gen in TASKS.items():
        pairs[name] = []
        for _ in range(n):
            inp, out = gen(rng)
            pairs[name].append({"input": inp, "output": out})
    return pairs


if __name__ == "__main__":
    train = make_pairs(seed=20261015, n=3)
    test = make_pairs(seed=987654321, n=5)
    with open("/srv/cellforge-arena/common/train.json", "w") as f:
        json.dump(train, f)
    with open("/srv/cellforge-arena/judge/heldout/test.json", "w") as f:
        json.dump(test, f)
    print("train tasks:", {k: len(v) for k, v in train.items()})
    print("test tasks:", {k: len(v) for k, v in test.items()})
    print("WROTE OK")
