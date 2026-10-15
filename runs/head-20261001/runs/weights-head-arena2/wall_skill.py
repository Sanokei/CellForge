"""Additive wall-following structure; fixed numeric turn priorities carry policy."""
import random
import numpy as np
import skillcore as S

TEMPLATE = 'wall_follow2'
NICHENAME = 'wall-follow2'
SHAPES = {'wall': (10,), 'marker': (10,), 'paint': (10,), 'turn': (4,), 'start': (2,)}
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def predict(params, grid, demos=None):
    a = np.asarray(grid)
    marker = int(np.argmax(params['marker']))
    marks = [tuple(x) for x in np.argwhere(a == marker)]
    if len(marks) != 2:
        raise ValueError('need exactly two endpoint markers')
    source = marks[int(np.argmax(params['start']))]
    target = marks[1 - int(np.argmax(params['start']))]
    walls = np.asarray(params['wall']) > 0
    priorities = np.asarray(params['turn'])
    paint = int(np.argmax(params['paint'])) if max(params['paint']) > 0 else marker
    h, w = a.shape

    def legal(pos):
        r, c = pos
        return 0 <= r < h and 0 <= c < w and (walls[a[r, c]] or pos == target)

    initials = [(k, (source[0]+dr, source[1]+dc)) for k, (dr, dc) in enumerate(DIRECTIONS)
                if legal((source[0]+dr, source[1]+dc))]
    if len(initials) != 1:
        raise ValueError('ambiguous initial heading')
    heading, current = initials[0]
    visited, path = {source}, []
    for _ in range(h*w):
        if current == target:
            out = a.copy()
            for r, c in path:
                out[r, c] = paint
            return out.tolist()
        if current in visited:
            raise ValueError('path loop')
        visited.add(current)
        path.append(current)
        choices = []
        # Relative directions: forward, left, right, back.
        for relative, delta in enumerate((0, -1, 1, 2)):
            nxtheading = (heading+delta) % 4
            dr, dc = DIRECTIONS[nxtheading]
            nxt = (current[0]+dr, current[1]+dc)
            if legal(nxt) and (nxt not in visited or nxt == target):
                choices.append((float(priorities[relative]), -relative, nxtheading, nxt))
        if not choices:
            raise ValueError('dead end')
        _, _, heading, current = max(choices)
    raise ValueError('path bound exceeded')


S.TEMPLATES[TEMPLATE] = {'shapes': SHAPES, 'fn': predict, 'aware': False,
    'doc': 'Grid wall tracing. wall[10]>0 names traversable wall colours; marker=argmax(marker[10]) must occur exactly twice. '
           'start[2] selects first or second marker in row-major order. It must have exactly one wall neighbour, giving initial heading. '
           'At every step choose the legal, unvisited neighbour with highest turn[4] priority for [forward,left,right,back] relative to heading; '
           'stop at the other marker. Recolour visited wall pixels by argmax(paint[10]) if max(paint)>0, otherwise use marker colour. '
           'Preserve endpoints and all unvisited pixels. Avoid interior junction detours and spurs by learning turn priorities. '
           'Write 36 literal numbers, complete shapes, finite [-9,9].'}


def schema_rows(source):
    return [{'id': 'wall-%d' % i, 'index': i} for i in range(120)]


def generated(row, seed, tier=1):
    rng = random.Random(seed * 101 + row['index'])
    h, w = rng.randint(6, 10), rng.randint(6, 10)
    a = np.zeros((h, w), int)
    a[1, 1:w-1] = 8
    a[h-2, 1:w-1] = 8
    a[1:h-1, 1] = 8
    a[1:h-1, w-2] = 8
    for c in range(2, w-2):
        if rng.random() < .7:
            a[2:h-2, c] = 8
    for r in range(2, h-2):
        for c in range(2, w-2):
            if rng.random() < .15:
                a[r, c] = 7
    a[1, w-1] = a[h-1, w-2] = 9
    expected = a.copy()
    expected[1, 1:w-1] = 9
    expected[h-2, 1:w-1] = 9
    expected[1:h-1, 1] = 9
    return {'input': a.tolist(), 'output': expected.tolist()}


def score(cell, pairs):
    outcomes = []
    for pair in pairs:
        try:
            outcomes.append(S.run(cell, pair['input'], []) == pair['output'])
        except Exception:
            outcomes.append(False)
    return sum(outcomes), outcomes
