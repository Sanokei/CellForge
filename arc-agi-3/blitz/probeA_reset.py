"""Mission 1: does GameAction.RESET restart the current LEVEL or the whole game?"""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'g50t')
r0 = env.observation_space
f0 = frame_of(r0)
print('init : state=%s levels_completed=%s' % (r0.state, r0.levels_completed))
r1 = apply(env, [1, 2, 3])
f1 = frame_of(r1)
print('3 actions: frame changed =', not np.array_equal(f0, f1))
r2 = step(env, 'R')
f2 = frame_of(r2)
print('after RESET: frame == initial frame :', np.array_equal(f0, f2))
print('after RESET: levels_completed=%s full_reset=%s state=%s' %
      (r2.levels_completed, getattr(r2, 'full_reset', None), r2.state))
r3 = step(env, 'R')
print('2nd RESET: frame == initial frame :', np.array_equal(f0, frame_of(r3)),
      'full_reset=%s' % getattr(r3, 'full_reset', None))
arc.close_scorecard(card)
