import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
def fresh():
    c, e = new_env(arc, 'g50t')
    return c, e, frame_of(e.observation_space)

def showdiff(f0, f1, x0, x1, y0, y1, label):
    print('--- %s' % label)
    for y in range(y0, y1+1):
        b = ''.join('.123456789abcdef'[int(v)] for v in f0[y, x0:x1+1])
        a = ''.join('.123456789abcdef'[int(v)] for v in f1[y, x0:x1+1])
        if b != a: print('  y=%2d %s -> %s' % (y, b, a))

c, e, f0 = fresh()
f1 = frame_of(apply(e, [2]))
d = f0 != f1
ys, xs = np.nonzero(d)
print('[2]: ndiff=%d bbox x[%d..%d] y[%d..%d]' % (d.sum(), xs.min(), xs.max(), ys.min(), ys.max()))
showdiff(f0, f1, 10, 28, 4, 24, 'after [2] vs initial')
f2 = frame_of(step(e, 1))
d2 = f0 != f2
ys2, xs2 = np.nonzero(d2)
print('[2,1]: ndiff vs initial=%d bbox x[%d..%d] y[%d..%d]' % (d2.sum(), xs2.min(), xs2.max(), ys2.min(), ys2.max()))
showdiff(f0, f2, 10, 28, 4, 24, 'after [2,1] vs initial')
arc.close_scorecard(c)
