import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'g50t')
f0 = frame_of(env.observation_space)
r = apply(env, [2])
f1 = frame_of(r)
r = step(env, 5)
f2 = frame_of(r)
d = f1 != f2
ys, xs = np.nonzero(d)
print('[2,5] vs [2]: ndiff=%d' % d.sum())
# cluster the diff by region
print('diff pixels by 16x16 region:')
for ry in range(0, 64, 16):
    for rx in range(0, 64, 16):
        n = d[ry:ry+16, rx:rx+16].sum()
        if n: print('  region x[%d..%d] y[%d..%d]: %d px, %s->%s' % (
            rx, rx+15, ry, ry+15, n,
            sorted(map(int, np.unique(f1[ry:ry+16, rx:rx+16][d[ry:ry+16, rx:rx+16]]))),
            sorted(map(int, np.unique(f2[ry:ry+16, rx:rx+16][d[ry:ry+16, rx:rx+16]])))))
arc.close_scorecard(card)
