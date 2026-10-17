import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
def pat5(f, y0, x):
    return (f[y0:y0+5, x:x+5] == 5).tobytes()
# cycle patterns available at slot0 (action 1)
cyc = []
r = env.observation_space
for k in range(8):
    r = step(env, 1)
    p = pat5(frame_of(r), 52, 15)
    if p in cyc: break
    cyc.append(p)
print('cycle has', len(cyc), 'patterns')
# scan a-box (rows 41-45, x14-48) for 5x5 windows matching any cycle pattern
hits = []
for x in range(14, 44):
    p = pat5(f, 41, x)
    if p in cyc:
        hits.append((x, cyc.index(p)))
print('a-box windows matching a cycle pattern:', hits)
# also scan the UPPER panels' 7-panel glyphs (rows 5-9 etc) for matches
for (y0, name) in ((5, 'upper1'), (14, 'upper2'), (23, 'upper3')):
    hh = []
    for x in range(12, 55):
        p = pat5(f, y0, x)
        if p in cyc: hh.append(x)
    print(name, '7-panel-ish matches at x:', hh[:10])
arc.close_scorecard(card)
