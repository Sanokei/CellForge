"""Volatility diagnostic: which pixels change on >80% of steps? Also determinism check."""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np, random
from arcengine import GameState

arc = make_arc()
for tag in ('g50t', 'ls20', 'tr87'):
    card, env = new_env(arc, tag, seed=0)
    acts = [a for a in env.observation_space.available_actions if a != 0]
    rng = random.Random(0)
    counts = np.zeros((64, 64), np.int32)
    prev = frame_of(env.observation_space)
    N = 150
    overs = 0
    for i in range(N):
        r = step(env, rng.choice(acts))
        if not r.frame:
            overs += 1
            r = step(env, 'R')
        cur = frame_of(r)
        counts += (prev != cur)
        prev = cur
    hot = np.nonzero(counts > 0.8 * N)
    print('%s: gameovers=%d, pixels changing on >80%% of %d steps: %d' % (tag, overs, N, len(hot[0])))
    if len(hot[0]):
        ys, xs = hot[0], hot[1]
        print('   bbox x[%d..%d] y[%d..%d] change-count %d..%d' %
              (xs.min(), xs.max(), ys.min(), ys.max(), counts[hot].min(), counts[hot].max()))
    rng2 = random.Random(1)
    c1, e1 = new_env(arc, tag, seed=0); c2, e2 = new_env(arc, tag, seed=0)
    seq = [rng2.choice(acts) for _ in range(20)]
    f1 = frame_of(apply(e1, seq)); f2 = frame_of(apply(e2, seq))
    print('   determinism (same seed+seq -> identical frame):', np.array_equal(f1, f2))
    arc.close_scorecard(card); arc.close_scorecard(c1); arc.close_scorecard(c2)
