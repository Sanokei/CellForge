import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np, random

arc = make_arc()
# 1) round-trip test: move block away and back -> frame identical to initial?
card, env = new_env(arc, 'g50t')
f0 = frame_of(env.observation_space)
r = apply(env, [2, 1])   # down then up
print('g50t [2,1] roundtrip == initial:', np.array_equal(f0, frame_of(r)))
r = apply(env, [4, 3])   # right then left
print('g50t [4,3] roundtrip == initial:', np.array_equal(f0, frame_of(r)))
r = apply(env, [5])      # action 5 alone
print('g50t [5] == initial:', np.array_equal(f0, frame_of(r)), 'levels:', r.levels_completed)
# 2) long random walk: count distinct hashes + track block bbox + any gameover
rng = random.Random(0)
seen = set(); cur = f0; bboxes = set(); overs = 0; maxlvl = 0
for i in range(300):
    a = rng.choice([1,2,3,4,5])
    r = step(env, a)
    if not r.frame:
        overs += 1; r = step(env, 'R')
    cur = frame_of(r)
    seen.add(cur.tobytes())
    ys, xs = np.nonzero(cur == 9)
    bboxes.add((int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    maxlvl = max(maxlvl, r.levels_completed)
print('g50t 300 random steps: distinct frames=%d distinct 9-bboxes=%d gameovers=%d maxlevels=%d' %
      (len(seen), len(bboxes), overs, maxlvl))
print('sample bboxes:', sorted(bboxes)[:8])
arc.close_scorecard(card)
