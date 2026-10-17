"""Hand-authored perception front-end for the tiny ARC-AGI-3 model.

Fixed (non-learned) cells: color histogram, connected components on a
16x16 presence map, 8x8 spatial color map, available-action mask, and
frame-difference features. Output: 100-dim float32 vector.
"""
import numpy as np
from collections import deque

N_FEATS = 100
N_ACTIONS = 8          # GameAction ids 0..7 (0=RESET, 6=click)
N_CLICK = 64           # click coordinate resolution


def _components(pres):
    """BFS connected components on a small boolean map. Returns list of
    (size, x0, y0, x1, y1)."""
    h, w = pres.shape
    seen = np.zeros_like(pres, dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if not pres[i, j] or seen[i, j]:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            cells = 0
            x0 = y0 = 10 ** 9
            x1 = y1 = -1
            while q:
                ci, cj = q.popleft()
                cells += 1
                if cj < x0: x0 = cj
                if cj > x1: x1 = cj
                if ci < y0: y0 = ci
                if ci > y1: y1 = ci
                if ci > 0 and pres[ci-1, cj] and not seen[ci-1, cj]:
                    seen[ci-1, cj] = True; q.append((ci-1, cj))
                if ci+1 < h and pres[ci+1, cj] and not seen[ci+1, cj]:
                    seen[ci+1, cj] = True; q.append((ci+1, cj))
                if cj > 0 and pres[ci, cj-1] and not seen[ci, cj-1]:
                    seen[ci, cj-1] = True; q.append((ci, cj-1))
                if cj+1 < w and pres[ci, cj+1] and not seen[ci, cj+1]:
                    seen[ci, cj+1] = True; q.append((ci, cj+1))
            comps.append((cells, x0, y0, x1, y1))
    return comps


def extract_features(frame, prev_frame, available_actions):
    """frame: (64,64) uint8. Returns (100,) float32."""
    f = np.asarray(frame, dtype=np.uint8)
    feats = np.zeros(N_FEATS, dtype=np.float32)

    # 1. color histogram (16)
    hist = np.bincount(f.ravel(), minlength=16).astype(np.float32) / 4096.0
    feats[0:16] = hist

    # 2. connected components on 16x16 presence map (7)
    small = f.reshape(16, 4, 16, 4).max(axis=(1, 3)) > 0
    comps = _components(small)
    if comps:
        comps.sort(key=lambda c: -c[0])
        size, x0, y0, x1, y1 = comps[0]
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        feats[16] = min(len(comps) / 20.0, 1.0)
        feats[17] = size / 256.0
        feats[18] = x0 / 16.0
        feats[19] = y0 / 16.0
        feats[20] = x1 / 16.0
        feats[21] = y1 / 16.0
        feats[22] = size / max(area, 1)
    # else zeros

    # 3. 8x8 spatial max-color map (64)
    sm8 = f.reshape(8, 8, 8, 8).max(axis=(1, 3)).astype(np.float32) / 15.0
    feats[23:87] = sm8.ravel()

    # 4. available actions one-hot (8)
    for a in available_actions:
        a = int(a)
        if 0 <= a < N_ACTIONS:
            feats[87 + a] = 1.0

    # 5. frame difference vs previous (5)
    if prev_frame is not None:
        d = f != np.asarray(prev_frame, dtype=np.uint8)
        frac = d.mean()
        feats[95] = frac
        if frac > 0:
            ys, xs = np.nonzero(d)
            feats[96] = xs.min() / 64.0
            feats[97] = ys.min() / 64.0
            feats[98] = xs.max() / 64.0
            feats[99] = ys.max() / 64.0
    return feats


def batch_features(frames, avails):
    """frames: (N,64,64) uint8, avails: list of lists. Returns (N,100)."""
    n = len(frames)
    out = np.zeros((n, N_FEATS), dtype=np.float32)
    prev = None
    for i in range(n):
        out[i] = extract_features(frames[i], prev, avails[i])
        prev = frames[i]
    return out
