"""From the pre-completion state (12 steps), try each single action: which completes?"""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
pre = [3,3,3,1,1,1,1,4,4,4,1,1]
for a in (1, 2, 3, 4):
    card, env = new_env(arc, 'ls20')
    r = apply(env, pre + [a])
    print('pre + action %d -> levels=%s state=%s' % (a, r.levels_completed, r.state))
    arc.close_scorecard(card)
# also: what does the target box look like -- show rows 6..18, x 30..42 of initial frame
card, env = new_env(arc, 'ls20')
f = frame_of(env.observation_space)
print('target box region x[30..42] y[6..18]:')
for y in range(6, 19):
    print('  ' + ''.join('.123456789abcdef'[int(v)] for v in f[y, 30:43]))
arc.close_scorecard(card)
