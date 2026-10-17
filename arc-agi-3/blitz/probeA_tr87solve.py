"""Extract tr87 targets + slot glyphs; solve by matching."""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np

SLOTS = [15, 22, 29, 36, 43]  # x-starts, width 5

def glyphs(f, y0):
    # 5x5 glyphs at rows y0..y0+4, slot x-starts
    return [f[y0:y0+5, x:x+5].tobytes() for x in SLOTS]

def showg(g, label):
    a = np.frombuffer(g, dtype=np.uint8).reshape(5, 5)
    print(' %s:' % label)
    for row in a: print('   ' + ''.join('.123456789abcdef'[v] for v in row))

arc = make_arc()
card, env = new_env(arc, 'tr87')
f = frame_of(env.observation_space)
tg = glyphs(f, 41)   # a-box targets rows 41-45
sg = glyphs(f, 52)   # panel slots rows 52-56
for i in range(5):
    print('slot %d (x=%d): target==current: %s' % (i, SLOTS[i], tg[i] == sg[i]))
showg(tg[0], 'target0'); showg(sg[0], 'slot0 now')
# cycle action 1 a few times on slot 0, show glyphs
for k in range(1, 4):
    c2, e2 = new_env(arc, 'tr87')
    r = apply(e2, [1]*k)
    g = glyphs(frame_of(r), 52)[0]
    print('after %d x action1: match=%s' % (k, g == tg[0]))
    arc.close_scorecard(c2)
arc.close_scorecard(card)
