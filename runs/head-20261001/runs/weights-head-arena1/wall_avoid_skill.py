"""New wall path with exclusion zones around hazard markers; wall_follow2 frozen."""
from collections import deque
import random
import numpy as np
import skillcore as S

TEMPLATE = 'wall_avoid3'
NICHENAME = 'wall-avoid3'
SHAPES = {'wall': (10,), 'marker': (10,), 'avoid': (10,), 'paint': (10,), 'conn': (2,)}


def predict(params, grid, demos=None):
    a = np.asarray(grid)
    marker, paint = int(np.argmax(params['marker'])), int(np.argmax(params['paint']))
    marks = [tuple(p) for p in np.argwhere(a == marker)]
    if len(marks) != 2:
        raise ValueError('two endpoints required')
    wall, avoid = np.asarray(params['wall']) > 0, np.asarray(params['avoid']) > 0
    h, w = a.shape
    steps = [(-1,0),(0,1),(1,0),(0,-1)]
    exclusion = steps if int(np.argmax(params['conn'])) == 0 else [(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc]
    allowed = set(marks)
    for r in range(h):
        for c in range(w):
            if wall[a[r,c]] and not any(0 <= r+dr < h and 0 <= c+dc < w and avoid[a[r+dr,c+dc]] for dr,dc in exclusion):
                allowed.add((r,c))
    queue = deque([(marks[0], [])])
    seen = {marks[0]}
    while queue:
        pos, path = queue.popleft()
        if pos == marks[1]:
            out = a.copy()
            for r,c in path:
                if a[r,c] != marker:
                    out[r,c] = paint
            return out.tolist()
        for dr,dc in steps:
            nxt = (pos[0]+dr,pos[1]+dc)
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt,path+[nxt]))
    raise ValueError('no hazard-free wall path')


S.TEMPLATES[TEMPLATE] = {'shapes':SHAPES, 'fn':predict, 'aware':False,
    'doc':'Trace a shortest 4-connected path between exactly two pixels of argmax(marker). '
          'Traversable colours are wall>0, except wall pixels adjacent to colours with avoid>0 are forbidden. '
          'conn argmax 0 means four-neighbour exclusion, 1 means eight-neighbour exclusion. '
          'Endpoints are always allowed and preserved. Recolour intermediate path pixels to argmax(paint); preserve everything else. '
          'Write wall[10],marker[10],avoid[10],paint[10],conn[2], 42 literal entries in [-9,9].'}


def schema_rows(source):
    return [{'id':'avoid-%d'%i,'index':i} for i in range(120)]


def generated(row, seed, tier=1):
    rng = random.Random(seed*101+row['index'])
    h,w = rng.randint(8,12),rng.randint(8,12)
    a = np.zeros((h,w),int)
    a[1,1:w-1] = a[h-2,1:w-1] = 8
    a[1:h-1,1] = a[1:h-1,w-2] = 8
    a[1,w-1] = a[h-1,w-2] = 9
    for c in range(3,w-3):
        if rng.random() < .7:
            a[2:4,c] = 8
            a[4,c] = 7
    a[h//2,w-3] = 7
    expected = a.copy()
    expected[1,1:w-1] = expected[h-2,1:w-1] = 9
    expected[1:h-1,1] = 9
    return {'input':a.tolist(),'output':expected.tolist()}


def score(cell,pairs):
    hits=[]
    for pair in pairs:
        try: hits.append(S.run(cell,pair['input'],[]) == pair['output'])
        except Exception: hits.append(False)
    return sum(hits),hits
