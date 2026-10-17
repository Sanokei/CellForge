import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

# paths to 12 block positions (from flood fill); each then + [5]
paths = [[], [2], [2,2], [2,2,2], [2,2,2,2], [4], [4,2,2], [4,4], [4,4,2],
         [4,4,4], [4,4,4,4], [2,2,2,2,3,3]]
arc = make_arc()
for path in paths:
    card, env = new_env(arc, 'g50t')
    r = apply(env, path + [5])
    f = frame_of(r)
    tl = f[0:8, 0:12]
    # compact signature of top-left canvas
    sig = ''.join('.123456789abcdef'[int(v)] for v in tl.ravel())
    print('path=%s + [5] -> levels=%s canvas=%s' % (path, r.levels_completed, sig))
    arc.close_scorecard(card)
