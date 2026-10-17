import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def blockbox(f):
    # movable block: 9-pixels in the play region x[10..40] y[5..45]
    sub = f[5:46, 10:41]
    ys, xs = np.nonzero(sub == 9)
    return (int(xs.min())+10, int(xs.max())+10, int(ys.min())+5, int(ys.max())+5) if len(xs) else None

arc = make_arc()
# Hypothesis A: move block onto the black slot (x20-24, y14-18)
for seq in ([2,4], [2,4,4], [4,2], [2,2,4,4]):
    card, env = new_env(arc, 'g50t')
    r = apply(env, seq)
    f = frame_of(r)
    print('seq %s -> block9=%s levels=%s state=%s' % (seq, blockbox(f), r.levels_completed, r.state))
    arc.close_scorecard(card)
