import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
def pat5(f, y0, x):
    return (f[y0:y0+5, x:x+5] == 5).tobytes()
# action 2 cycle on slot0
card, env = new_env(arc, 'tr87')
cyc2 = []
r = env.observation_space
for k in range(10):
    r = step(env, 2)
    p = pat5(frame_of(r), 52, 15)
    if p in cyc2:
        print('action2 cycle period:', len(cyc2)); break
    cyc2.append(p)
arc.close_scorecard(card)
# compare action2 cycle vs action1 cycle (recompute action1)
card, env = new_env(arc, 'tr87')
cyc1 = []
r = env.observation_space
for k in range(10):
    r = step(env, 1)
    p = pat5(frame_of(r), 52, 15)
    if p in cyc1: break
    cyc1.append(p)
arc.close_scorecard(card)
print('action1 period:', len(cyc1))
print('action2 set == action1 set:', set(cyc2) == set(cyc1))
print('action2 == reverse(action1):', cyc2 == cyc1[::-1])
# target patterns from a-box at all 5 slots
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
SLOTS = [15, 22, 29, 36, 43]
allcyc = set(cyc1) | set(cyc2)
for i, x in enumerate(SLOTS):
    t = pat5(f, 41, x)
    print('slot %d target in cycle1/2:' % i, t in allcyc)
arc.close_scorecard(card)
