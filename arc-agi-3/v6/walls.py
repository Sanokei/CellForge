"""Cell W1: wall / hazard model learned from attempted avatar moves.

Rules inferred:
- free colours: colours seen under the avatar's destination footprint on moves that succeeded.
- blocked colours: colours under the destination footprint on moves that failed (no displacement),
  excluding colours already known free. Out-of-frame destinations count as blocked without a colour.
- hazard colours: same attribution for moves that ended in GAME_OVER.
- blocked edges: (level, anchor, act) pairs that failed, as a per-level correction for dynamic scenes.
Persisted across levels (walls keep their colour between levels in most games; wrong beliefs get
cleared when a "blocked" colour is later crossed).
"""
from __future__ import annotations

from collections import Counter


def dest_colors(frame, pix, vec):
    """Colours of destination pixels (avatar pixels translated by vec) that are not currently avatar pixels.
    Returns (set of colours, out_of_frame flag)."""
    dy, dx = vec
    h, w = frame.shape
    cols, oob = set(), False
    for (y, x) in pix:
        ny, nx = y + dy, x + dx
        if (ny, nx) in pix:
            continue
        if not (0 <= ny < h and 0 <= nx < w):
            oob = True
            continue
        cols.add(int(frame[ny, nx]))
    return cols, oob


class WallModel:
    def __init__(self):
        self.free = set()
        self.free_n = Counter()
        self.blocked = set()
        self.hazard = set()
        self.edges = set()

    def learn(self, level, anchor, act, cols, moved: bool, died: bool = False):
        if moved and not died:
            self.free |= cols
            self.free_n.update(cols)
            self.blocked -= cols
            self.hazard -= cols
        elif died:
            self.hazard |= (cols - self.free)
        else:
            self.edges.add((level, anchor, act))
            self.blocked |= (cols - self.free)

    def passable(self, cols, oob: bool, allow=frozenset()):
        if oob:
            return False
        bad = (self.blocked | self.hazard) - set(allow)
        return not (cols & bad)

    def floor(self, n=8):
        """Colours the avatar has walked over at least n times: scenery, never a goal candidate."""
        return {c for c, k in self.free_n.items() if k >= n}
