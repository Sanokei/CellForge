"""Transition evidence genes; ambiguous observations remain hypotheses."""
from collections import Counter, defaultdict

from .perception import positions, singletons


class ActionEffects:
    def __init__(self):
        self.table = defaultdict(Counter)

    def observe(self, action, p, q):
        if p is not None and q is not None:
            d = q[0] - p[0], q[1] - p[1]
            if abs(d[0]) + abs(d[1]) == 1:
                self.table[action][d] += 1

    def mapping(self):
        return {a: v.most_common(1)[0][0] for a, v in self.table.items() if v}


class TerrainLearning:
    """Learn directed blocked edges, lethal cells/colors, and successful traversal."""
    def __init__(self):
        self.blocked = set()
        self.safe = set()
        self.dead = set()
        self.dead_colors = set()
        self.failures = Counter()

    def observe(self, before, p, q, action, mapping, died):
        if p is None or action not in mapping:
            return
        dx, dy = mapping[action]
        target = p[0] + dx, p[1] + dy
        if not (0 <= target[1] < before.shape[0] and 0 <= target[0] < before.shape[1]):
            return
        if died:
            self.dead.add(target)
            color = int(before[target[1], target[0]])
            if len(positions(before, color)) == 1:
                self.dead_colors.add(color)
        elif q == p:
            self.blocked.add((p, target))
            self.failures[int(before[target[1], target[0]])] += 1
        elif q is not None:
            self.safe.add(q)
            self.blocked.discard((p, q))


class GoalHypotheses:
    """Visit objects to test goals; remember failed visits and disappearing progress."""
    def __init__(self):
        self.tested = Counter()
        self.progress = []
        self.initial = {}
        self.replay = []

    def observe(self, before, after, p, q, avatar):
        old, new = singletons(before), singletons(after)
        if q is not None and q != p:
            c = int(before[q[1], q[0]])
            if c in old and c != avatar:
                self.tested[c] += 1
        disappeared = [c for c in old if c != avatar and not positions(after, c)]
        # Occlusion under the avatar is not evidence of collection.
        for c in disappeared:
            if old[c] != q and c not in self.progress:
                self.progress.append(c)


class KeyDoorLearning:
    """A collection can remove a second remote object and release blocked edges."""
    def __init__(self):
        self.pairs = set()

    def observe(self, before, after, p, q, avatar, terrain):
        if q is None or p == q:
            return
        old = singletons(before)
        touched = int(before[q[1], q[0]])
        remote = [c for c, pos in old.items()
                  if c != avatar and pos != q and not positions(after, c)]
        if remote and touched in old:
            for c in remote:
                self.pairs.add((touched, c))
            terrain.blocked.clear()


class PushDetection:
    def __init__(self):
        self.colors = Counter()

    def observe(self, moved, avatar, p, q):
        if p is None or q is None:
            return
        d = q[0] - p[0], q[1] - p[1]
        for c, (b, b2) in moved.items():
            if c != avatar and b == q and (b2[0] - b[0], b2[1] - b[1]) == d:
                self.colors[c] += 1


class HazardMotion:
    """Track non-avatar movers; predict a witnessed cycle or linear next step."""
    def __init__(self):
        self.tracks = defaultdict(list)

    def observe(self, grid, colors):
        for c in colors:
            ps = positions(grid, c)
            if len(ps) == 1:
                self.tracks[c].append(ps[0])
                self.tracks[c] = self.tracks[c][-64:]

    def predicted(self):
        out = set()
        for track in self.tracks.values():
            if len(track) < 2:
                continue
            for i in range(len(track) - 2, 0, -1):
                if track[i - 1:i + 1] == track[-2:] and i + 1 < len(track):
                    out.add(track[i + 1])
                    break
            else:
                p, q = track[-2:]
                out.add((2 * q[0] - p[0], 2 * q[1] - p[1]))
        return out
