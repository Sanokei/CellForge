import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
print('big panel x[14..48] y[40..62]:')
for y in range(40, 63):
    print('  %2d ' % y + ''.join('.123456789abcdef'[int(v)] for v in f[y, 14:49]))
# test selector movement: press 4 twice, 3 three times from fresh
for seq, name in (([4], 'after [4]'), ([4,4], 'after [4,4]'), ([3], 'after [3]'), ([3,3], 'after [3,3]')):
    c2, e2 = new_env(arc, 'tr87')
    r = apply(e2, seq)
    g = frame_of(r)
    ys, xs = np.nonzero(g[47:50, :] == 0)
    print('%s: gap x at y48-49: %s' % (name, sorted(set((xs).tolist()))[:6]))
    arc.close_scorecard(c2)
arc.close_scorecard(card)
