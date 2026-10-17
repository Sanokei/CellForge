import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
arc = make_arc()
card, env = new_env(arc, 'g50t')
f = frame_of(env.observation_space)
print('rows 36..63, x 10..50:')
for y in range(36, 64):
    print('  %2d ' % y + ''.join('.123456789abcdef'[int(v)] for v in f[y, 10:51]))
print('top-left x[0..10] y[0..8]:')
for y in range(0, 9):
    print('  ' + ''.join('.123456789abcdef'[int(v)] for v in f[y, 0:11]))
arc.close_scorecard(card)
