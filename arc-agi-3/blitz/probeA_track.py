"""Track ls20 entities across sequences with ONE env."""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def find3(f):
    ys, xs = np.nonzero(f == 3)
    return list(zip(xs.tolist(), ys.tolist()))

def cursor_pos(f):
    ys, xs = np.nonzero(f == 12)
    if len(xs) == 0: return None
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))

arc = make_arc()
card, env = new_env(arc, 'ls20')
f0 = frame_of(env.observation_space)
threes0 = find3(f0)
print('init: cursor12 bbox=%s  threes=%s' % (cursor_pos(f0), threes0[:10]))
for seqname, seq in (('2,2,2,2', [2,2,2,2]), ('R', ['R'])):
    r = apply(env, seq)
    f = frame_of(r)
    th = find3(f)
    new = [p for p in th if p not in threes0]
    gone = [p for p in threes0 if p not in th]
    print('seq %s: cursor12=%s new3=%s gone3=%s levels=%s state=%s' %
          (seqname, cursor_pos(f), new[:8], gone[:8], r.levels_completed, r.state))
    threes0 = th
arc.close_scorecard(card)
