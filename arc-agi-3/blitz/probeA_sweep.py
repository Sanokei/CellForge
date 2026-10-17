"""Single-action effect sweep: for each game, apply each single action from level-0
initial frame; report frame change (n changed px, bbox, state/level changes)."""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

def diff_summary(f0, f1):
    d = (f0 != f1)
    n = int(d.sum())
    if n == 0: return 'no change'
    ys, xs = np.nonzero(d)
    return ('n=%d bbox=x[%d..%d] y[%d..%d] colors %s->%s' %
            (n, xs.min(), xs.max(), ys.min(), ys.max(),
             sorted(map(int, np.unique(f0[d]))), sorted(map(int, np.unique(f1[d])))))

arc = make_arc()
for tag in ('g50t', 'ls20', 'tr87'):
    card, env = new_env(arc, tag)
    r0 = env.observation_space
    f0 = frame_of(r0)
    print('='*16, tag, 'actions:', list(r0.available_actions), '='*16)
    for a in r0.available_actions:
        if a == 0: continue
        card2, env2 = new_env(arc, tag)
        r = step(env2, a)
        print('  action %d: %s | state=%s levels=%s' %
              (a, diff_summary(f0, frame_of(r)), r.state, r.levels_completed))
        arc.close_scorecard(card2)
    arc.close_scorecard(card)
