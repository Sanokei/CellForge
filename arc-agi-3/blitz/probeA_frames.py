"""Dump initial frames + available actions for all 3 games."""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *

arc = make_arc()
for tag in ('g50t', 'ls20', 'tr87'):
    card, env = new_env(arc, tag)
    r0 = env.observation_space
    print('='*20, tag, '='*20)
    print('available_actions:', list(r0.available_actions))
    print('state:', r0.state, 'levels_completed:', r0.levels_completed)
    show(frame_of(r0))
    arc.close_scorecard(card)
