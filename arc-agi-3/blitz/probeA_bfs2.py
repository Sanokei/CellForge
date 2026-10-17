"""Efficient single-env BFS: RESET + path replay. Usage: probeA_bfs2.py <tag> [maxvis] [maxdepth]"""
import sys; sys.path.insert(0, '/srv/arc3-claude/v6run/blitz')
from probeA_common import *
import numpy as np
from collections import deque
import time

def bfs(tag, maxvis=4000, maxdepth=60):
    t0 = time.time()
    arc = make_arc()
    card, env = new_env(arc, tag)
    r0 = env.observation_space
    acts = [a for a in r0.available_actions if a != 0]
    f0 = frame_of(r0)
    k0 = f0.tobytes()
    seen = {k0: []}
    q = deque([k0])
    nvis = 0
    unstable = 0
    while q and nvis < maxvis:
        hk = q.popleft(); path = seen[hk]; nvis += 1
        if len(path) >= maxdepth:
            continue
        r = step(env, 'R')
        if path:
            r = apply(env, path)
        if frame_of(r).tobytes() != hk:
            unstable += 1
            continue  # volatile / nondeterministic; skip
        for a in acts:
            r2 = step(env, a)
            if r2.levels_completed != 0 or r2.state != GameState.NOT_FINISHED:
                print('FOUND tag=%s len=%d seq=%s -> levels=%s state=%s (%.1fs, %d vis, %d unstable)' %
                      (tag, len(path)+1, path+[a], r2.levels_completed, r2.state,
                       time.time()-t0, nvis, unstable))
                arc.close_scorecard(card)
                return path + [a]
            k2 = frame_of(r2).tobytes()
            if k2 not in seen:
                seen[k2] = path + [a]
                q.append(k2)
            r = step(env, 'R')
            if path:
                r = apply(env, path)
    print('tag=%s: NO solution (%d vis, depth cap %d, %.1fs, %d unstable)' %
          (tag, nvis, maxdepth, time.time()-t0, unstable))
    arc.close_scorecard(card)
    return None

if __name__ == '__main__':
    bfs(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4000,
        int(sys.argv[3]) if len(sys.argv) > 3 else 60)
