"""Tiny reference evaluator used only by the meta-tests. Task: recolor 1->2 on 3x3 grids with colors 0..3.
Tensor W is a 4x4 color map; prediction maps color c to argmax_j W[j][c]."""
import random


def make_tasks(seed, n):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        x = [[r.randrange(4) for _ in range(3)] for _ in range(3)]
        y = [[2 if c == 1 else c for c in row] for row in x]
        out.append((x, y))
    return out


def predict(tensors, grid):
    W = tensors["W"]
    def m(c):
        col = [W[j][c] for j in range(4)]
        return max(range(4), key=lambda j: (col[j], -j))
    return [[m(c) for c in row] for row in grid]
