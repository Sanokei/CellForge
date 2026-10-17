import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

SLOTS = [15, 22, 29, 36, 43]
def pat(f, y0, i):
    g = f[y0:y0+5, SLOTS[i]:SLOTS[i]+5]
    return (g == 5).tobytes()   # foreground pattern only

arc = make_arc()
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
tgt = pat(f, 41, 0)
print('target0 pattern set; cycling action 1 on slot 0...')
r = env.observation_space
seen = set()
for k in range(1, 40):
    r = step(env, 1)
    p = pat(frame_of(r), 52, 0)
    if p == tgt:
        print('MATCH after %d presses of action 1' % k); break
    if p in seen:
        print('cycle repeats at k=%d, no match; trying action 2 direction' % k); break
    seen.add(p)
else:
    print('no match in 40 presses')
# now try action 2 from fresh
c2, e2 = new_env(arc, 'tr87')
for k in range(1, 40):
    r2 = step(e2, 2)
    p = pat(frame_of(r2), 52, 0)
    if p == tgt:
        print('MATCH after %d presses of action 2' % k); break
else:
    print('action 2: no match in 40 presses')
arc.close_scorecard(card); arc.close_scorecard(c2)
