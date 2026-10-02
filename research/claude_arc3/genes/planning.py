"""Bounded graph, Sokoban hypothesis, and learned binary click planners."""
from collections import deque
import numpy as np


def search(start, expand, goal, cap=50000):
    queue = deque([start])
    parent = {start: None}
    while queue and len(parent) < cap:
        s = queue.popleft()
        if goal(s):
            path = []
            while parent[s] is not None:
                s, a = parent[s]
                path.append(a)
            return path[::-1]
        for nxt, a in expand(s):
            if nxt not in parent:
                parent[nxt] = s, a
                queue.append(nxt)
    return None


def navigation(start, target, mapping, allowed, blocked=()):
    def expand(p):
        for a, (dx, dy) in mapping.items():
            q = p[0] + dx, p[1] + dy
            if q in allowed and (p, q) not in blocked:
                yield q, a
    return search(start, expand, lambda p: p == target)


def push_plan(avatar, block, target, mapping, allowed, cap=30000):
    def expand(s):
        p, b = s
        for a, (dx, dy) in mapping.items():
            q = p[0] + dx, p[1] + dy
            if q not in allowed:
                continue
            b2 = b
            if q == b:
                b2 = b[0] + dx, b[1] + dy
                if b2 not in allowed:
                    continue
                # Reject static corners unless this is the hypothesized target.
                if b2 != target:
                    horiz = any((b2[0] + d, b2[1]) not in allowed for d in (-1, 1))
                    vert = any((b2[0], b2[1] + d) not in allowed for d in (-1, 1))
                    if horiz and vert:
                        continue
            yield (q, b2), a
    return search((avatar, block), expand, lambda s: s[1] == target, cap)


class ClickEffects:
    """Learn XOR toggle columns by interventions; solve either uniform-color goal."""
    def __init__(self):
        self.columns = {}
        self.targets = []
        self.colors = set()
        self.goal_index = 0
        self.tried = set()

    def discover(self, grid):
        counts = dict(zip(*np.unique(grid, return_counts=True)))
        bg = max(counts, key=counts.get)
        self.targets = [(int(x), int(y)) for y, x in np.argwhere(grid != bg)]
        self.colors.update(map(int, grid[grid != bg]))

    def observe(self, before, after, target):
        self.columns[target] = {p for p in self.targets if before[p[1], p[0]] != after[p[1], p[0]]}
        self.colors.update(int(after[p[1], p[0]]) for p in self.targets)

    def plan(self, grid):
        if len(self.columns) < len(self.targets) or len(self.colors) != 2:
            return None
        for color in sorted(self.colors):
            if color in self.tried:
                continue
            # Enumeration is bounded to small binary puzzles; no hidden rule access.
            if len(self.targets) > 16:
                return None
            wanted = {p for p in self.targets if int(grid[p[1], p[0]]) != color}
            for mask in range(1 << len(self.targets)):
                effect, actions = set(), []
                for i, p in enumerate(self.targets):
                    if mask >> i & 1:
                        effect ^= self.columns[p]
                        actions.append(p)
                if effect == wanted:
                    self.tried.add(color)
                    return actions
        return None
