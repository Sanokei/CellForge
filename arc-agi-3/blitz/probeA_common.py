import logging, sys
sys.path.insert(0, '/srv/arc3-claude/v6run')
lg = logging.getLogger('probeA'); lg.addHandler(logging.NullHandler()); lg.propagate = False
from arc_agi import Arcade
from arcengine import GameAction, GameState
import numpy as np

GIDS = {'g50t': 'g50t-5849a774', 'ls20': 'ls20-9607627b', 'tr87': 'tr87-cd924810'}

def make_arc():
    return Arcade(logger=lg, environments_dir='/srv/arc3-claude/environment_files',
                  recordings_dir='/tmp/probeA_rec')

def new_env(arc, tag, seed=0):
    card = arc.create_scorecard()
    env = arc.make(GIDS[tag], seed=seed, scorecard_id=card)
    return card, env

def frame_of(resp):
    f = resp.frame[-1] if isinstance(resp.frame, list) else resp.frame
    return np.asarray(f)

def show(f, maxw=64, maxh=40):
    h, w = f.shape[:2]
    print('  shape=%s colors=%s' % (f.shape, sorted(map(int, np.unique(f)))))
    for y in range(min(h, maxh)):
        row = f[y, :min(w, maxw)]
        print('  ' + ''.join('.123456789abcdef'[int(v)] if 0 <= int(v) < 16 else '?' for v in row))
    if h > maxh: print('  ... (%d more rows)' % (h - maxh))

def step(env, aid, x=None, y=None):
    if aid == 'R':
        return env.step(GameAction.RESET)
    if aid == 6:
        return env.step(GameAction.ACTION6, data={'x': x, 'y': y})
    return env.step(GameAction.from_id(aid))

def apply(env, seq):
    r = None
    for a in seq:
        r = step(env, *a) if isinstance(a, tuple) else step(env, a)
    return r
