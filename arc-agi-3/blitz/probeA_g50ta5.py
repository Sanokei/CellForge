import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

arc = make_arc()
# paths to each of the 12 block positions (from flood fill, reconstruct simply)
# we'll drive the block around and press 5 at interesting spots
tests = {
    'init': [],
    'at_slot_left': [2],          # x14-18 y14-18, left of slot
    'at_slot_below': [2, 2],      # x14-18 y20-24
    'top_right': [4, 4, 4, 4],    # x38-40 y8-12
    'mid': [4, 4, 2, 2],
}
for name, path in tests.items():
    card, env = new_env(arc, 'g50t')
    r = apply(env, path) if path else env.observation_space
    f0 = frame_of(r)
    r5 = step(env, 5)
    f5 = frame_of(r5)
    d = f0 != f5
    ys, xs = np.nonzero(d)
    info = 'no change' if not d.any() else ('ndiff=%d bbox x[%d..%d] y[%d..%d] %s->%s' % (
        d.sum(), xs.min(), xs.max(), ys.min(), ys.max(),
        sorted(map(int, np.unique(f0[d]))), sorted(map(int, np.unique(f5[d])))))
    print('%-14s path=%s : action5 -> %s | levels=%s state=%s' % (name, path, info, r5.levels_completed, r5.state))
    arc.close_scorecard(card)
