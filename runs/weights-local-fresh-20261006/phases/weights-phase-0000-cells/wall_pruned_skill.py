"""New hazard-aware, dead-spur-pruned wall template. Earlier templates frozen."""
from collections import deque
import numpy as np
import skillcore as S
import wall_skill as W
import wall_avoid_skill as A

TEMPLATE = 'wall_pruned4'
NICHENAME = 'wall-pruned4'
SHAPES = dict(A.SHAPES, turn=(4,), start=(2,), bg=(10,))
schema_rows = A.schema_rows
generated = A.generated
score = A.score


def predict(params, grid, demos=None):
    a = np.asarray(grid)
    marker, bg = int(np.argmax(params['marker'])), int(np.argmax(params['bg']))
    wall, avoid = np.asarray(params['wall']) > 0, np.asarray(params['avoid']) > 0
    if wall[bg] or marker == bg:
        raise ValueError('background must be outside wall and marker roles')
    h,w = a.shape
    directions = [(-1,0),(0,1),(1,0),(0,-1)]
    exclusion = directions if int(np.argmax(params['conn'])) == 0 else [(r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c]
    marks = {tuple(p) for p in np.argwhere(a == marker)}
    allowed = set(marks)
    for r in range(h):
        for c in range(w):
            if wall[a[r,c]] and not any(0<=r+dr<h and 0<=c+dc<w and avoid[a[r+dr,c+dc]] for dr,dc in exclusion):
                allowed.add((r,c))
    neighbours = {p:{(p[0]+r,p[1]+c) for r,c in directions} & allowed for p in allowed}
    queue = deque(p for p in allowed if p not in marks and len(neighbours[p]) <= 1)
    while queue:
        p = queue.popleft()
        if p not in allowed or p in marks or len(neighbours[p]) > 1:
            continue
        allowed.remove(p)
        for nxt in list(neighbours[p]):
            neighbours[nxt].discard(p)
            if nxt not in marks and len(neighbours[nxt]) <= 1:
                queue.append(nxt)
    masked = a.copy()
    for p in neighbours:
        if p not in allowed:
            masked[p] = bg
    for r in range(h):
        for c in range(w):
            if wall[a[r,c]] and (r,c) not in allowed:
                masked[r,c] = bg
    traced = np.asarray(W.predict(params,masked.tolist()))
    out = a.copy()
    changed = traced != masked
    out[changed] = traced[changed]
    return out.tolist()


S.TEMPLATES[TEMPLATE] = {'shapes':SHAPES,'fn':predict,'aware':False,
    'doc': 'Hazard-aware wall tracing. Wall colours wall>0; marker argmax must occur twice; bg argmax must be outside wall/marker roles. '
           'Forbid walls adjacent to avoid>0 colours (conn argmax 0 four-neighbour,1 eight-neighbour). Iteratively prune all wall dead ends '
           'except endpoints. Then trace from start argmax 0 first marker or1 second in row-major order, with highest turn[4] priorities '
           '[forward,left,right,back], initial heading must be unique. Recolour path with argmax(paint) or marker if max(paint)<=0. '
           'Preserve all other original pixels, including excluded/pruned walls. Write all tensor entries in [-9,9].'}


def make_niche(base):
    class WallPruned(base.Niche):
        name, template, general, verify = 'wallpruned', TEMPLATE, True, True

        def patterns(self,limit,seed):
            truth = {'wall':[0]*8+[1,0], 'marker':[0]*9+[1], 'avoid':[0]*7+[1,0,0],
                     'paint':[0]*9+[1], 'conn':[1,0], 'bg':[1]+[0]*9, 'turn':[3,1,2,0], 'start':[1,0]}
            def apply(grid):
                a = np.asarray(grid).copy()
                h,w = a.shape
                a[1,1:w-1] = a[h-2,1:w-1] = 9
                a[1:h-1,1] = 9
                return a.tolist()
            return [{'id':'wallpruned-%d'%i,'niche':self.name,'truth':truth,'apply':apply,
                     'sample_input':lambda rng,i=i: generated({'index':i},rng.randint(1,1000000))['input']}
                    for i in range(limit)]
    return WallPruned()
