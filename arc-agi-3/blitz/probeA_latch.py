import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'g50t')
f0 = frame_of(env.observation_space)
print('init (63,63) =', f0[63,63])
r = apply(env, [2, 1])
print('after [2,1] (63,63) =', frame_of(r)[63,63])
r = step(env, 'R')
f = frame_of(r)
print('after RESET (63,63) =', f[63,63], '| frame == true init:', np.array_equal(f, f0),
      '| full_reset =', getattr(r, 'full_reset', None))
# also check: does the latch affect block movement? compare [2] from fresh vs from latched
c2, e2 = new_env(arc, 'g50t')
fa = frame_of(apply(e2, [2]))
arc.close_scorecard(c2)
c3, e3 = new_env(arc, 'g50t')
apply(e3, [2, 1])          # flip latch
step(e3, 'R')              # reset (latch maybe stays)
fb = frame_of(apply(e3, [2]))
print('[2] from fresh == [2] from post-reset-latched:', np.array_equal(fa, fb))
arc.close_scorecard(card); arc.close_scorecard(c3)
