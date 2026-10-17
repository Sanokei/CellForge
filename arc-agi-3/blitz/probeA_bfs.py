"""BFS over frame hashes for a keyboard game; report first action seq that
changes levels_completed or game state. Usage: probeA_bfs.py <tag> [maxvis] [maxdepth]"""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np
from collections import deque

def hkey(f): return f.tobytes()

def bfs(tag, maxvis=20000, maxdepth=150):
    arc = make_arc()
    card, env = new_env(arc, tag)
    r0 = env.observation_space
    acts = [a for a in r0.available_actions if a != 0]
    f0 = frame_of(r0)
    start = hkey(f0)
    prev = {start: None}
    q = deque([(start, f0, [])])
    nvis = 0
    # we need one env per expansion; reuse by replaying path (short, ok)
    while q and nvis < maxvis:
        hk, f, path = q.popleft()
        nvis += 1
        if len(path) >= maxdepth: continue
        for a in acts:
            c2, e2 = new_env(arc, tag)
            r = apply(e2, path + [a])
            arc.close_scorecard(c2)
            if r.levels_completed != 0 or r.state != GameState.NOT_FINISHED:
                print('FOUND tag=%s len=%d seq=%s -> levels=%s state=%s' %
                      (tag, len(path)+1, path+[a], r.levels_completed, r.state))
                return
            f2 = frame_of(r)
            k2 = hkey(f2)
            if k2 not in prev:
                prev[k2] = (hk, a)
                q.append((k2, f2, path+[a]))
    print('tag=%s: no solution within %d visited, depth %d' % (tag, nvis, maxdepth))

if __name__ == '__main__':
    bfs(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 20000,
        int(sys.argv[3]) if len(sys.argv)>3 else 150)
