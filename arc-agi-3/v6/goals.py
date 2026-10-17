"""Cells G1/G2: win-condition hypotheses and level-to-level transfer.

G1 TouchGoals (keyboard/avatar games). Hypothesis family "touching an object of colour c is useful".
  Evidence per colour, accumulated across levels: win (level cleared within 2 steps after the touch),
  vanish (object disappeared on touch => collectible; continue with siblings of that colour),
  inert (touched, nothing changed => deprioritised), solid (touch attempted but blocked).
  Ranking: win >> collectible siblings left >> never-tried colours (rare colour first, then near) >> inert.
G2 ClickSequence (click games). Hypothesis "the winning level is a sequence of clicks on object classes
  (colour,size)". On level clear the effective-click class sequence is stored; on the next level the same
  class sequence is replayed on the nearest unclicked instance of each class, once, then control returns to search.
"""
from __future__ import annotations

from collections import Counter


class TouchGoals:
    def __init__(self):
        self.win = Counter()
        self.vanish = Counter()
        self.inert = Counter()
        self.solid = Counter()
        self.done = set()          # (level, color, bbox) already touched this level
        self.unreachable = set()   # (level, color, bbox)

    def pick(self, objs, avatar_cls, level, frame_colors_blocked, dist_fn):
        """Order candidate objects. dist_fn(obj) -> path length or None (unreachable)."""
        cnt = Counter(o["color"] for o in objs)
        cands = []
        for o in objs:
            c = o["color"]
            key = (level, c, o["bbox"])
            if o["cls"] == avatar_cls or key in self.done or key in self.unreachable:
                continue
            if c in frame_colors_blocked or o["size"] > 400:
                continue
            tried = self.win[c] + self.vanish[c] + self.inert[c] + self.solid[c]
            if self.win[c]:
                tier = 0
            elif self.vanish[c]:
                tier = 1
            elif tried == 0:
                tier = 2
            elif self.solid[c] and not self.inert[c]:
                tier = 4
            else:
                tier = 3
            cands.append((tier, cnt[c], o))
        cands.sort(key=lambda t: (t[0], t[1] if t[0] == 2 else 0))
        return cands

    def outcome(self, level, obj, kind):
        c = obj["color"]
        self.done.add((level, c, obj["bbox"]))
        {"win": self.win, "vanish": self.vanish, "inert": self.inert, "solid": self.solid}[kind][c] += 1


class ClickSequence:
    def __init__(self):
        self.winseq = []           # list of class keys from the last cleared level
        self.cur = []              # effective clicks this level: class key
        self.clicked = set()       # (x,y) clicked on this level
        self.replay = []           # remaining classes to replay on the current level
        self.replays_tried = 0

    def new_level(self, cleared: bool):
        if cleared and self.cur:
            self.winseq = list(self.cur)
        self.replay = list(self.winseq)
        self.cur = []
        self.clicked = set()
        self.replays_tried = 0

    def record(self, xy, cls, changed: bool):
        self.clicked.add(xy)
        if changed:
            self.cur.append(cls)

    def suggest(self, objs, last_xy=None):
        """Next click (x,y) replaying the stored class sequence, or None when exhausted."""
        while self.replay:
            want = self.replay[0]
            opts = [o for o in objs if (o["color"], o["size"]) == (want[0], want[1])
                    and self._rep(o) not in self.clicked]
            if not opts:
                self.replay.pop(0)
                continue
            if last_xy is not None:
                opts.sort(key=lambda o: abs(self._rep(o)[0] - last_xy[0]) + abs(self._rep(o)[1] - last_xy[1]))
            self.replay.pop(0)
            return self._rep(opts[0])
        return None

    @staticmethod
    def _rep(o):
        ys = [p[0] for p in o["pix"]]
        xs = [p[1] for p in o["pix"]]
        cy, cx = sum(ys) / len(ys), sum(xs) / len(xs)
        y, x = min(o["pix"], key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
        return (x, y)
