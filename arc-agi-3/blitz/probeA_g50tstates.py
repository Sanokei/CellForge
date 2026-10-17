import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np, random
from collections import defaultdict

def blockkey(f):
    sub = f[5:46, 10:41]
    ys, xs = np.nonzero(sub == 9)
    if len(xs) == 0: return None
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))

arc = make_arc()
card, env = new_env(arc, 'g50t')
rng = random.Random(0)
by_block = defaultdict(list)
r = env.observation_space
for i in range(400):
    a = rng.choice([1,2,3,4,5])
    r = step(env, a)
    if not r.frame:
        r = step(env, 'R'); continue
    f = frame_of(r)
    by_block[blockkey(f)].append(f)
print('distinct block positions:', len(by_block))
# find a block position with >1 distinct frame; diff them
for bk, frames in by_block.items():
    hs = {}
    for f in frames: hs.setdefault(f.tobytes(), f)
    if len(hs) > 1:
        hlist = list(hs.values())
        d = hlist[0] != hlist[1]
        ys, xs = np.nonzero(d)
        print('block', bk, 'has', len(hs), 'variants; diff ndiff=%d bbox x[%d..%d] y[%d..%d] colors %s->%s' % (
            d.sum(), xs.min(), xs.max(), ys.min(), ys.max(),
            sorted(map(int, np.unique(hlist[0][d]))), sorted(map(int, np.unique(hlist[1][d])))))
        break
arc.close_scorecard(card)
