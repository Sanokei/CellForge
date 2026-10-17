"""Zoomed before/after ASCII for each effective single action."""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def crop_show(f0, f1, x0, x1, y0, y1, label):
    print('--- %s x[%d..%d] y[%d..%d]' % (label, x0, x1, y0, y1))
    print('  BEFORE      AFTER')
    for y in range(y0, y1 + 1):
        b = ''.join('.123456789abcdef'[int(v)] for v in f0[y, x0:x1+1])
        a = ''.join('.123456789abcdef'[int(v)] for v in f1[y, x0:x1+1])
        mark = ' <' if b != a else ''
        print('  %s  %s%s' % (b, a, mark))

arc = make_arc()
# g50t: action 2 and 4 move the 9-glyph
for tag, acts in (('g50t', [2, 4]),):
    for a in acts:
        c1, e1 = new_env(arc, tag); f0 = frame_of(e1.observation_space)
        c2, e2 = new_env(arc, tag)
        f1 = frame_of(step(e2, a))
        crop_show(f0, f1, 12, 26, 6, 20, '%s action %d' % (tag, a))
        arc.close_scorecard(c1); arc.close_scorecard(c2)
