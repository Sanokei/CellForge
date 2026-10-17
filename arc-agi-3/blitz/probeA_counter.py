import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'g50t')
f0 = frame_of(env.observation_space)
print('initial (63,63) =', f0[63, 63], ' row63 tail:', f0[63, 58:64].tolist())
seq = [2,1,4,3,2,2,1,1,5,5,2,4,1,3]
for i, a in enumerate(seq):
    r = step(env, a)
    if not r.frame:
        print('step %d action %d -> EMPTY (gameover), resetting' % (i, a)); r = step(env, 'R')
    f = frame_of(r)
    print('step %2d action %d: (63,63)=%d levels=%s state=%s' % (i+1, a, f[63,63], r.levels_completed, r.state))
arc.close_scorecard(card)
