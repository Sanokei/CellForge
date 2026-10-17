"""Cell PL1: action-efficient planning by BFS over avatar anchor positions in the learned movement model.

Model: state = avatar anchor (top-left of its bbox); successor under action a = anchor + v(a);
an edge is legal if the destination footprint (avatar pixels shifted) stays in frame and
covers no blocked/hazard colour in the *current* frame (static-scene assumption) and the
(level, anchor, act) triple was not observed to fail. Replanned every step, so wrong predictions
cost one action, not a whole path.
"""
from __future__ import annotations

from collections import deque

from walls import WallModel, dest_colors


def plan_to(frame, avatar_pix, vecs, walls: WallModel, target_pix, level, allow=frozenset(), max_nodes=6000, unvisited=None):
    """BFS for shortest action list that makes the avatar overlap target_pix.
    avatar_pix: frozenset of (y,x). vecs: {act_id: (dy,dx)}.
    With unvisited (a set of anchors already visited) the goal is instead the nearest anchor NOT in it
    (frontier exploration of the avatar's reachable positions).
    Returns list of act ids, [] if already overlapping, None if unreachable."""
    base = (min(p[0] for p in avatar_pix), min(p[1] for p in avatar_pix))   # bbox top-left = anchor
    rel = [(y - base[0], x - base[1]) for (y, x) in avatar_pix]
    tgt = set(target_pix)
    h, w = frame.shape
    avset = set(avatar_pix)

    def overlaps(anchor):
        return any((anchor[0] + ry, anchor[1] + rx) in tgt for ry, rx in rel)

    start = base
    if unvisited is None and overlaps(start):
        return []
    prev = {start: None}
    q = deque([start])
    while q and len(prev) < max_nodes:
        a = q.popleft()
        cur_pix = frozenset((a[0] + ry, a[1] + rx) for ry, rx in rel)
        for act, (dy, dx) in vecs.items():
            n = (a[0] + dy, a[1] + dx)
            if n in prev or (level, a, act) in walls.edges:
                continue
            new_pix = [(n[0] + ry, n[1] + rx) for ry, rx in rel]
            cols, oob = set(), False
            for (y, x) in new_pix:
                if (y, x) in cur_pix:
                    continue
                if not (0 <= y < h and 0 <= x < w):
                    oob = True
                    break
                if (y, x) in avset:
                    continue            # vacated by the real avatar: treat as free
                cols.add(int(frame[y, x]))
            if oob:
                continue
            tcols = {int(frame[y, x]) for (y, x) in new_pix if (y, x) in tgt and 0 <= y < h and 0 <= x < w}
            if not walls.passable(cols - tcols, False, allow):
                continue
            prev[n] = (a, act)
            if (n not in unvisited) if unvisited is not None else any(p in tgt for p in new_pix):
                path = []
                k = n
                while prev[k] is not None:
                    k, ac = prev[k]
                    path.append(ac)
                return path[::-1]
            q.append(n)
    return None
