import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

paths = [[], [2], [2,2], [2,2,2], [2,2,2,2], [4], [4,2,2], [4,4], [4,4,2],
         [4,4,4], [4,4,4,4], [2,2,2,2,3,3]]
arc = make_arc()
# canvas B, then block to each position (no stamp)
for path in paths:
    card, env = new_env(arc, 'g50t')
    r = apply(env, [2, 5] + path)
    if r.levels_completed != 0 or r.state != GameState.NOT_FINISHED:
        print('WIN path=[2,5]+%s -> levels=%s state=%s' % (path, r.levels_completed, r.state))
    arc.close_scorecard(card)
print('done - no completion found' )
# does stamp count matter? try 5 stamps then move
card, env = new_env(arc, 'g50t')
r = apply(env, [2,5,2,5,2,5,2,5,2,5])
print('5 stamps: levels=%s' % r.levels_completed)
arc.close_scorecard(card)
