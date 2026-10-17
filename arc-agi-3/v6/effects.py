"""Cell E1: per-action effect table and avatar detection.

Rule inferred: a keyboard action a has a displacement vector v(a) (modal move of some object class);
the avatar is the object class whose moves are explained by >=2 distinct actions with distinct vectors
(consistency = the modal vector accounts for >=70% of that action's moves of the class).
Also tracks no-op rate per action (an action that never changes anything is not a move).
"""
from __future__ import annotations

from collections import Counter, defaultdict


class EffectTable:
    def __init__(self):
        self.vec = defaultdict(lambda: defaultdict(Counter))   # cls -> act -> Counter(vector)
        self.trials = Counter()                                  # act -> tries
        self.noops = Counter()                                   # act -> tries with no pixel change
        self.cls_seen = Counter()

    def observe(self, act_id: int, moves, noop: bool):
        self.trials[act_id] += 1
        self.noops[act_id] += int(noop)
        for cls, v, _ in moves:
            if v != (0, 0):
                self.vec[cls][act_id][v] += 1

    def modal(self, cls, act_id):
        c = self.vec[cls].get(act_id)
        if not c:
            return None, 0.0
        v, n = c.most_common(1)[0]
        return v, n / sum(c.values())

    def avatar(self):
        """Return (cls, {act_id: (dy,dx)}) of the best avatar class or None."""
        best, best_score = None, 0
        for cls, per in self.vec.items():
            vecs = {}
            score = 0
            for a, c in per.items():
                v, frac = self.modal(cls, a)
                if v is not None and frac >= 0.7:
                    vecs[a] = v
                    score += c[v]
            if len(set(vecs.values())) >= 2 and score > best_score:
                best, best_score = (cls, vecs), score
        return best
