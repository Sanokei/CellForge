"""Cell P1: object perception and move matching (pure numpy/python, no game imports).

Rule inferred: the scene is a set of single-colour 4-connected objects on a background colour;
an object that "moves" is one whose (colour, shape) reappears displaced between two frames.
"""
from __future__ import annotations

from collections import deque

import numpy as np

MAX_OBJ = 700          # components bigger than this are treated as scenery, not objects


def background(frame: np.ndarray) -> int:
    vals, cnts = np.unique(frame, return_counts=True)
    return int(vals[np.argmax(cnts)])


def segment(frame: np.ndarray, mask: np.ndarray | None = None, bg: int | None = None):
    """Single-colour 4-connected components. Volatile pixels (mask) are dropped.
    Returns list of dict(color,size,bbox=(x0,y0,x1,y1),pix=set[(y,x)],cls=(color,size,w,h))."""
    h, w = frame.shape
    if bg is None:
        bg = background(frame)
    seen = np.zeros((h, w), bool)
    if mask is not None:
        seen |= mask
    out = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue
            c = int(frame[y0, x0])
            seen[y0, x0] = True
            q = deque([(y0, x0)])
            pix = []
            while q:
                y, x = q.popleft()
                pix.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and frame[ny, nx] == c:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if c == bg:
                continue
            ys = [p[0] for p in pix]
            xs = [p[1] for p in pix]
            bb = (min(xs), min(ys), max(xs), max(ys))
            out.append(dict(color=c, size=len(pix), bbox=bb, pix=frozenset(pix),
                            cls=(c, len(pix), bb[2] - bb[0] + 1, bb[3] - bb[1] + 1)))
    return out


def match_moves(before, after):
    """Pair unmatched same-class components between frames.
    Returns (moves, vanished, appeared); moves = list of (cls, (dy,dx), obj_before)."""
    sa = {(o["color"], o["bbox"]) for o in after}
    sb = {(o["color"], o["bbox"]) for o in before}
    ub = [o for o in before if (o["color"], o["bbox"]) not in sa and o["size"] <= MAX_OBJ]
    ua = [o for o in after if (o["color"], o["bbox"]) not in sb and o["size"] <= MAX_OBJ]
    moves, vanished = [], []
    used = set()
    for o in ub:
        best, bd = None, None
        for i, p in enumerate(ua):
            if i in used or p["cls"] != o["cls"]:
                continue
            d = abs(p["bbox"][0] - o["bbox"][0]) + abs(p["bbox"][1] - o["bbox"][1])
            if d <= 40 and (bd is None or d < bd):
                best, bd = i, d
        if best is None:
            vanished.append(o)
        else:
            used.add(best)
            p = ua[best]
            moves.append((o["cls"], (p["bbox"][1] - o["bbox"][1], p["bbox"][0] - o["bbox"][0]), o))
    appeared = [p for i, p in enumerate(ua) if i not in used]
    return moves, vanished, appeared
