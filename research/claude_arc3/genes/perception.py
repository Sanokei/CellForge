"""Visual genes. No environment objects or semantic palette labels are accepted."""
from collections import Counter, defaultdict
from math import gcd

import numpy as np


class TileGrid:
    """Infer aligned square tile scale from pixel boundary coordinates."""
    def __init__(self, frame):
        frame = np.asarray(frame)
        boundaries = [frame.shape[0], frame.shape[1]]
        boundaries += (np.flatnonzero(np.any(frame[1:] != frame[:-1], axis=1)) + 1).tolist()
        boundaries += (np.flatnonzero(np.any(frame[:, 1:] != frame[:, :-1], axis=0)) + 1).tolist()
        self.scale = boundaries[0]
        for b in boundaries:
            self.scale = gcd(self.scale, b)
        self.scale = max(1, self.scale)

    def read(self, frame):
        return np.asarray(frame, dtype=np.uint8)[::self.scale, ::self.scale].copy()

    def pixel(self, pos):
        x, y = pos
        return {"x": int(x * self.scale + self.scale // 2),
                "y": int(y * self.scale + self.scale // 2)}


def positions(grid, color):
    return [(int(x), int(y)) for y, x in np.argwhere(grid == color)]


def singletons(grid):
    counts = Counter(map(int, grid.ravel()))
    return {c: positions(grid, c)[0] for c, n in counts.items() if n == 1}


class AvatarDetection:
    """Score same-color singleton translations correlated with issued actions."""
    def __init__(self):
        self.votes = Counter()
        self.color = None
        self.effects = defaultdict(lambda: defaultdict(Counter))

    def observe(self, before, after, action=None):
        old, new = singletons(before), singletons(after)
        moved = {}
        for c in old.keys() & new.keys():
            p, q = old[c], new[c]
            d = (q[0] - p[0], q[1] - p[1])
            if abs(d[0]) + abs(d[1]) == 1:
                moved[c] = (p, q)
        # A pushed object and moving hazard can also move: vote over multiple probes.
        for c in moved:
            self.votes[c] += 1 / len(moved)
            p, q = moved[c]
            self.effects[c][action][(q[0] - p[0], q[1] - p[1])] += 1
        if self.votes:
            def score(c):
                hist = self.effects[c]
                directions = {v.most_common(1)[0][0] for v in hist.values()}
                contradiction = sum(sum(v.values()) - v.most_common(1)[0][1] for v in hist.values())
                return len(directions) * 10 + self.votes[c] - contradiction * 2
            self.color = max(self.votes, key=score)
        return moved

    def locate(self, grid):
        ps = positions(grid, self.color) if self.color is not None else []
        return ps[0] if len(ps) == 1 else None
