import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def crop_show(f0, f1, x0, x1, y0, y1, label):
    print('--- %s x[%d..%d] y[%d..%d]' % (label, x0, x1, y0, y1))
    for y in range(y0, y1 + 1):
        b = ''.join('.123456789abcdef'[int(v)] for v in f0[y, x0:x1+1])
        a = ''.join('.123456789abcdef'[int(v)] for v in f1[y, x0:x1+1])
        mark = ' <' if b != a else ''
        print('  %s  %s%s' % (b, a, mark))

arc = make_arc()
# tr87: action 1,2 -> x[15..19] y[52..56]; action 3,4 -> x[15..47] y[48..60]
for a, box in ((1, (10, 24, 48, 60)), (2, (10, 24, 48, 60)), (3, (10, 52, 44, 63)), (4, (10, 52, 44, 63))):
    c1, e1 = new_env(arc, 'tr87'); f0 = frame_of(e1.observation_space)
    c2, e2 = new_env(arc, 'tr87')
    f1 = frame_of(step(e2, a))
    crop_show(f0, f1, *box, label='tr87 action %d' % a)
    print()
    arc.close_scorecard(c1); arc.close_scorecard(c2)
