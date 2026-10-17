"""Replay a solution seq; show cursor bbox + frame crops at key steps and the final transition."""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def cursor(f):
    ys, xs = np.nonzero(f == 12)
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) if len(xs) else None

seq = [3,3,3,1,1,1,1,4,4,4,1,1,1]
arc = make_arc()
card, env = new_env(arc, 'ls20')
r = env.observation_space
for i, a in enumerate(seq):
    r = step(env, a)
    f = frame_of(r)
    print('step %2d action %d: cursor12=%s levels=%s state=%s' % (i+1, a, cursor(f), r.levels_completed, r.state))
# show the region around the final cursor pos, before vs after last step
arc.close_scorecard(card)
