import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'g50t')
r = apply(env, [2, 5])
f = frame_of(r)
print('after [2,5]: levels=%s state=%s' % (r.levels_completed, r.state))
# where are 9-pixels now?
ys, xs = np.nonzero(f == 9)
print('9-pixel bbox: x[%d..%d] y[%d..%d] n=%d' % (xs.min(), xs.max(), ys.min(), ys.max(), len(xs)))
print('top-left x[0..12] y[0..8]:')
for y in range(0, 9):
    print('  ' + ''.join('.123456789abcdef'[int(v)] for v in f[y, 0:13]))
# does the block respawn? move around and check
r = apply(env, [1, 1, 1])
f2 = frame_of(r)
ys2, xs2 = np.nonzero((f2[5:46, 10:41] == 9))
print('after more moves, block 9s in play region: n=%d' % len(xs2))
# try stamping again from a different position
r = apply(env, [4, 4, 5])
f3 = frame_of(r)
print('after [4,4,5]: levels=%s' % r.levels_completed)
for y in range(0, 9):
    print('  ' + ''.join('.123456789abcdef'[int(v)] for v in f3[y, 0:13]))
arc.close_scorecard(card)
