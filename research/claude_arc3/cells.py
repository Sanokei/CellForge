"""Small single-purpose perception / bookkeeping cells for the pure state-graph explorer.

No learning anywhere: hashing, flood fill, counting, hand-written ordering rules.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


def last_frame(resp) -> np.ndarray:
    """Final (settled) 64x64 frame of a step response."""
    return np.asarray(resp.frame[-1], dtype=np.uint8)


def state_key(frame: np.ndarray, mask: np.ndarray | None) -> bytes:
    """Hash of the frame with volatile pixels zeroed."""
    if mask is not None and mask.any():
        frame = np.where(mask, 0, frame).astype(np.uint8)
    return hashlib.blake2b(frame.tobytes(), digest_size=10).digest()


class VolatileMask:
    """Detects volatile pixels: (1) pixels that change on most steps (blinkers), and
    (2) thin bands (progress bars / step counters) whose rows or columns see a change on most
    steps even though each individual pixel changes only once (monotone counters)."""

    def __init__(self, shape=(64, 64), rate=0.6, warmup=30, every=15, max_band=4, edge=8, band_rate=0.3):
        self.band_rate = band_rate
        self.counts = np.zeros(shape, np.int32)
        self.row_hits = np.zeros(shape[0], np.int32)
        self.col_hits = np.zeros(shape[1], np.int32)
        self.steps = 0
        self.mask = np.zeros(shape, bool)
        self.rate, self.warmup, self.every, self.max_band, self.edge = rate, warmup, every, max_band, edge

    def observe(self, prev: np.ndarray, cur: np.ndarray) -> bool:
        """Feed one transition; return True if the mask grew."""
        d = prev != cur
        if not d.any():
            return False      # no-op steps carry no information about what is volatile
        self.counts += d
        self.row_hits += d.any(axis=1)
        self.col_hits += d.any(axis=0)
        self.steps += 1
        if self.steps < self.warmup or self.steps % self.every:
            return False
        new = (self.counts > 0.9 * self.steps) if self.steps >= 100 else np.zeros_like(self.mask)
        for hits, axis in ((self.row_hits, 0), (self.col_hits, 1)):
            cand = hits > self.band_rate * self.steps
            i = 0
            while i < len(cand):
                if not cand[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(cand) and cand[j + 1]:
                    j += 1
                peripheral = i < self.edge or j >= len(cand) - self.edge
                if j - i + 1 <= self.max_band and peripheral:
                    band = np.zeros_like(self.mask)
                    if axis == 0:
                        band[i:j + 1, :] = True
                    else:
                        band[:, i:j + 1] = True
                    new |= band & (self.counts > 0)
                i = j + 1
        new &= ~self.mask
        if new.any():
            self.mask |= new
            return True
        return False


def components(frame: np.ndarray):
    """4-connected same-colour components. Returns list of dicts."""
    h, w = frame.shape
    seen = np.zeros((h, w), bool)
    out = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue
            c = frame[y0, x0]
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            pix = []
            while q:
                y, x = q.popleft()
                pix.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and frame[ny, nx] == c:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ys = [p[0] for p in pix]
            xs = [p[1] for p in pix]
            cy, cx = sum(ys) / len(ys), sum(xs) / len(xs)
            # representative pixel: member closest to centroid (robust for non-convex shapes)
            ry, rx = min(pix, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
            out.append(dict(color=int(c), size=len(pix), x=rx, y=ry,
                            bbox=(min(xs), min(ys), max(xs), max(ys))))
    return out


def salience_order(comps, bg_color: int):
    """Rank components: rare colours and small/medium objects first; background colour and
    huge regions last. Pure counting rule."""
    color_count = {}
    for c in comps:
        color_count[c["color"]] = color_count.get(c["color"], 0) + 1

    def key(c):
        is_bg = c["color"] == bg_color
        huge = c["size"] > 1024
        return (is_bg or huge, color_count[c["color"]], c["size"], c["y"], c["x"])
    return sorted(comps, key=key)


def click_targets(frame: np.ndarray, level: int = 0):
    """Ordered list of (x, y) click candidates. level 0: component representatives;
    level 1: 8-pixel lattice; level 2: 4-pixel lattice (used only when exploration stalls)."""
    vals, cnts = np.unique(frame, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    if level == 0:
        comps = components(frame)
        seen, out = set(), []
        for c in salience_order(comps, bg):
            t = (c["x"], c["y"])
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out
    step = 8 if level == 1 else 4
    off = step // 2
    return [(x, y) for y in range(off, 64, step) for x in range(off, 64, step)]
