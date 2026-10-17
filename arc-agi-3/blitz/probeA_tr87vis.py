import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def patstr(p):
    a = np.frombuffer(p, dtype=np.uint8).reshape(5, 5)
    return [''.join('5' if v else '.' for v in row) for row in a]

arc = make_arc()
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
def pat(f, y0, x):
    return (f[y0:y0+5, x:x+5] == 5).tobytes()
tgt = pat(f, 41, 15)
print('TARGET0:'); [print('  '+l) for l in patstr(tgt)]
print('cycle of action 1 on slot0:')
seen = []
r = env.observation_space
for k in range(10):
    r = step(env, 1)
    p = pat(frame_of(r), 52, 15)
    if p in seen: print('  k=%d REPEAT' % (k+1)); break
    seen.append(p)
    print('  k=%d:' % (k+1)); [print('   '+l) for l in patstr(p)]
arc.close_scorecard(card)
