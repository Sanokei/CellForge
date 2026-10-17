import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def canvas_sig(f):
    tl = f[0:8, 0:12]
    return ''.join('.123456789abcdef'[int(v)] for v in tl.ravel())

arc = make_arc()
for seq in ([2,5,2,5], [2,5,4,5], [2,5,2,5,2,5], [4,5,2,5]):
    card, env = new_env(arc, 'g50t')
    r = apply(env, seq)
    print('seq=%s -> levels=%s state=%s canvas=%s' % (seq, r.levels_completed, r.state, canvas_sig(frame_of(r))))
    arc.close_scorecard(card)
