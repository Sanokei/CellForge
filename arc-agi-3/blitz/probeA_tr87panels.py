import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
# find 7-panels: connected components of color 7; and a-boxes (color 10)
from collections import deque
def comps(color):
    seen = np.zeros_like(f, bool); out = []
    ys, xs = np.nonzero(f == color)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]: continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; pix = []
        while q:
            y, x = q.popleft(); pix.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = y+dy, x+dx
                if 0 <= ny < 64 and 0 <= nx < 64 and not seen[ny, nx] and f[ny, nx] == color:
                    seen[ny, nx] = True; q.append((ny, nx))
        xs2 = [p[1] for p in pix]; ys2 = [p[0] for p in pix]
        out.append((min(xs2), max(xs2), min(ys2), max(ys2), len(pix)))
    return sorted(out)

print('7-panels (x0,x1,y0,y1,size):')
for p in comps(7): print('  ', p)
print('a-boxes:')
for p in comps(10): print('  ', p)
# black gap positions (color 0)
ys, xs = np.nonzero(f == 0)
print('black px bbox: x[%d..%d] y[%d..%d] n=%d' % (xs.min(), xs.max(), ys.min(), ys.max(), len(xs)))
arc.close_scorecard(card)
