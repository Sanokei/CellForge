import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'g50t')
r = apply(env, [2])
f0 = frame_of(r)
r5 = step(env, 5)
f5 = frame_of(r5)
print('path [2] then action5: top-left x[0..25] y[0..20] BEFORE -> AFTER')
for y in range(0, 21):
    b = ''.join('.123456789abcdef'[int(v)] for v in f0[y, 0:26])
    a = ''.join('.123456789abcdef'[int(v)] for v in f5[y, 0:26])
    mark = ' <' if b != a else ''
    print('  %s  %s%s' % (b, a, mark))
arc.close_scorecard(card)
