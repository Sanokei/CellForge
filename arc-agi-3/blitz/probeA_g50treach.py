import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np
from collections import deque
import time

def abstract(f):
    sub = f[5:46, 10:41]
    ys, xs = np.nonzero(sub == 9)
    bk = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) if len(xs) else None
    return (bk, int(f[63, 63]))

arc = make_arc()
t0 = time.time()
card, env = new_env(arc, 'g50t')
r0 = env.observation_space
acts = [a for a in r0.available_actions if a != 0]
a0 = abstract(frame_of(r0))
seen = {a0: []}
q = deque([a0])
nvis = 0
while q:
    ak = q.popleft(); path = seen[ak]; nvis += 1
    r = step(env, 'R')
    if path: r = apply(env, path)
    if abstract(frame_of(r)) != ak: continue
    for a in acts:
        r2 = step(env, a)
        if r2.levels_completed != 0 or r2.state != GameState.NOT_FINISHED:
            print('FOUND len=%d seq=%s' % (len(path)+1, path+[a])); raise SystemExit
        k2 = abstract(frame_of(r2))
        if k2 not in seen:
            seen[k2] = path+[a]; q.append(k2)
        r = step(env, 'R')
        if path: r = apply(env, path)
print('reachable abstract states: %d (%.1fs)' % (len(seen), time.time()-t0))
bks = sorted(set(k[0] for k in seen))
print('block positions (x0,x1,y0,y1) in sub-frame coords:')
for b in bks: print('  ', b, '-> abs x[%d..%d] y[%d..%d]' % (b[0]+10, b[1]+10, b[2]+5, b[3]+5))
arc.close_scorecard(card)
