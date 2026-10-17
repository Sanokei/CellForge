import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
def pat5(f, y0, x):
    return (f[y0:y0+5, x:x+5] == 5)
def variants(a):
    out = set()
    for k in range(4):
        b = np.rot90(a, k)
        out.add(b.tobytes()); out.add(np.fliplr(b).tobytes()); out.add(np.flipud(b).tobytes())
    return out
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
SLOTS = [15, 22, 29, 36, 43]
# get cycle patterns
cyc = []
r = env.observation_space
for k in range(10):
    r = step(env, 1)
    p = pat5(frame_of(r), 52, 15).tobytes()
    if p in cyc: break
    cyc.append(p)
cycset = set(cyc)
for i, x in enumerate(SLOTS):
    t = pat5(f, 41, x)
    hit = variants(t) & cycset
    print('slot %d target matches a cycle pattern under rot/flip: %s' % (i, bool(hit)))
arc.close_scorecard(card)
