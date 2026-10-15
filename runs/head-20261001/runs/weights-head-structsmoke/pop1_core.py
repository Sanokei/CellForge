"""Pure core of the multi-template weight cells: shapes, docs, numpy predictors, ground-truth rules and task sampling.
No dependency on niche_archive (so the run directory can ship a standalone evaluator copy).
See weight_cells.py for the WeightGenome / NicheArchive adapters."""
from __future__ import annotations

import random

import numpy as np

K = 4
OPS = ['identity', 'fliplr', 'flipud', 'transpose', 'rot90cw', 'rot180']
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

SHAPES = {
    'conv3x3_detect': {'conv.W': (9, K), 'conv.b': (1,), 'paint': (K,)},
    'colormap': {'map.W': (K, K)},
    'color_gate': {'gate': (K,), 'fill': (K,)},
    'flip_select': {'sel': (6,), 'post.W': (K, K)},
    'tile_select': {'per': (3,), 'pec': (3,)},
    'neighbor_count': {'self.W': (1, K), 'cnt.W': (1, K), 'cnt.b': (1,), 'paint': (K,)},
    'bbox_mask': {'pick': (K,), 'fill': (K,)},
}

DOCS = {
    'conv3x3_detect': (
        'For every pixel, f = conv.b[0] + sum over the 9 positions o of the 3x3 window (row-major: o=0 is up-left, '
        'o=1 up, o=2 up-right, o=3 left, o=4 centre, o=5 right, o=6 down-left, o=7 down, o=8 down-right) of '
        'conv.W[o][colour at that position]; positions outside the grid contribute 0. If f > 0 the pixel becomes '
        'argmax(paint) else it keeps its colour.'),
    'colormap': 'Each pixel of colour c becomes argmax(map.W[c]) (first index wins ties).',
    'color_gate': ('Pixel of colour c is kept if gate[c] > 0, otherwise replaced by colour argmax(fill).'),
    'flip_select': ('op = argmax(sel) over [identity, fliplr, flipud, transpose, rot90cw, rot180] (grids are square); '
                    'the grid is transformed by op and then every pixel of colour c becomes argmax(post.W[c]).'),
    'tile_select': ('pr = argmax(per)+1, pc = argmax(pec)+1; output[r][c] = input[r mod pr][c mod pc] '
                    '(tiles the top-left pr x pc block over the whole grid).'),
    'neighbor_count': ('n_k = number of the 8 neighbours with colour k. f = self.W[0][own colour] + '
                       'sum_k cnt.W[0][k]*n_k + cnt.b[0]. If f > 0 the pixel becomes argmax(paint) else unchanged.'),
    'bbox_mask': ('Let k = argmax(pick). The bounding box of all pixels of colour k is kept as is; every pixel outside '
                  'the box becomes argmax(fill). If colour k does not occur the grid is unchanged.'),
}

# ----------------------------------------------------------------------------------------
# Predictors (pure numpy). Each returns (grid as list of lists of int, fired trace).

def _fin(out, cells, pixels):
    return out.astype(int).tolist(), {'cells': cells, 'pixels': pixels}


def p_conv(params, grid, trace=False):
    a = np.array(grid)
    h, w = a.shape
    W = np.array(params['conv.W'], float)
    b = float(params['conv.b'][0])
    paint = int(np.argmax(params['paint']))
    pad = np.full((h + 2, w + 2), -1)
    pad[1:-1, 1:-1] = a
    f = np.full((h, w), b)
    for o, (dr, dc) in enumerate(OFFS):
        win = pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        f = f + np.where(win >= 0, W[o][np.clip(win, 0, K - 1)], 0.0)
    fire = f > 0
    out = np.where(fire, paint, a)
    px = {}
    if trace:
        for r in range(h):
            for c in range(w):
                px[(r, c)] = {'preact': round(float(f[r, c]), 3), 'fired': bool(fire[r, c]),
                              'window': pad[r:r + 3, c:c + 3].tolist()}
    return _fin(out, ['conv', 'paint'] if fire.any() else ['conv'], px)


def p_colormap(params, grid, trace=False):
    a = np.array(grid)
    M = np.array(params['map.W'], float)
    tgt = np.argmax(M, axis=1)
    out = tgt[np.clip(a, 0, K - 1)]
    px = {}
    if trace:
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                px[(r, c)] = {'in': int(a[r, c]), 'logits': M[a[r, c]].tolist(), 'out': int(out[r, c])}
    return _fin(out, ['map.row%d' % i for i in sorted(set(a.flatten().tolist()))], px)


def p_gate(params, grid, trace=False):
    a = np.array(grid)
    gate = np.array(params['gate'], float)
    fill = int(np.argmax(params['fill']))
    keep = gate[np.clip(a, 0, K - 1)] > 0
    out = np.where(keep, a, fill)
    px = {}
    if trace:
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                px[(r, c)] = {'in': int(a[r, c]), 'gate': float(gate[a[r, c]]), 'kept': bool(keep[r, c])}
    return _fin(out, ['gate.c%d' % i for i in sorted(set(a[~keep].tolist()))], px)


def _op(a, i):
    return [a, a[:, ::-1], a[::-1], a.T, np.rot90(a, -1), a[::-1, ::-1]][i]


def p_select(params, grid, trace=False):
    a = np.array(grid)
    sel = np.array(params['sel'], float)
    i = int(np.argmax(sel))
    t = _op(a, i) if a.shape[0] == a.shape[1] or i in (0, 1, 2, 5) else a
    post = np.argmax(np.array(params['post.W'], float), axis=1)
    out = post[np.clip(t, 0, K - 1)]
    px = {}
    if trace:
        e = np.exp(sel - sel.max())
        probs = (e / e.sum()).round(3).tolist()
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                px[(r, c)] = {'op': OPS[i], 'op_probs': probs, 'pre_colormap': int(t[r, c])}
    return _fin(out, ['op:' + OPS[i]], px)


def p_tile(params, grid, trace=False):
    a = np.array(grid)
    pr, pc = int(np.argmax(params['per'])) + 1, int(np.argmax(params['pec'])) + 1
    h, w = a.shape
    out = a[np.arange(h)[:, None] % pr, np.arange(w)[None, :] % pc]
    px = {}
    if trace:
        for r in range(h):
            for c in range(w):
                px[(r, c)] = {'period': [pr, pc], 'src': [r % pr, c % pc]}
    return _fin(out, ['tile%dx%d' % (pr, pc)], px)


def p_count(params, grid, trace=False):
    a = np.array(grid)
    h, w = a.shape
    sW = np.array(params['self.W'][0], float)
    cW = np.array(params['cnt.W'][0], float)
    b = float(params['cnt.b'][0])
    paint = int(np.argmax(params['paint']))
    pad = np.full((h + 2, w + 2), -1)
    pad[1:-1, 1:-1] = a
    n = np.zeros((K, h, w))
    for dr, dc in OFFS:
        if (dr, dc) == (0, 0):
            continue
        win = pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        for k in range(K):
            n[k] += (win == k)
    f = sW[np.clip(a, 0, K - 1)] + np.tensordot(cW, n, axes=1) + b
    fire = f > 0
    out = np.where(fire, paint, a)
    px = {}
    if trace:
        for r in range(h):
            for c in range(w):
                px[(r, c)] = {'own': int(a[r, c]), 'neighbor_counts': n[:, r, c].astype(int).tolist(),
                              'preact': round(float(f[r, c]), 3), 'fired': bool(fire[r, c])}
    return _fin(out, ['count', 'paint'] if fire.any() else ['count'], px)


def p_bbox(params, grid, trace=False):
    a = np.array(grid)
    k = int(np.argmax(params['pick']))
    fill = int(np.argmax(params['fill']))
    rs, cs = np.where(a == k)
    out = a.copy()
    box = None
    if len(rs):
        box = [int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())]
        mask = np.ones(a.shape, bool)
        mask[box[0]:box[2] + 1, box[1]:box[3] + 1] = False
        out = np.where(mask, fill, a)
    px = {}
    if trace:
        for r in range(a.shape[0]):
            for c in range(a.shape[1]):
                px[(r, c)] = {'box': box, 'inside': bool(box and box[0] <= r <= box[2] and box[1] <= c <= box[3]),
                              'picked_colour': k}
    return _fin(out, ['bbox'] if box else [], px)


PREDICTORS = {'conv3x3_detect': p_conv, 'colormap': p_colormap, 'color_gate': p_gate, 'flip_select': p_select,
              'tile_select': p_tile, 'neighbor_count': p_count, 'bbox_mask': p_bbox}


def _rand_grid(rng, h, w, probs):
    cols = list(range(K))
    return [[rng.choices(cols, probs)[0] for _ in range(w)] for _ in range(h)]


def _nb4(g, r, c):
    h, w = len(g), len(g[0])
    return [g[r + dr][c + dc] if 0 <= r + dr < h and 0 <= c + dc < w else None
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]


def _nb8(g, r, c):
    h, w = len(g), len(g[0])
    return [g[r + dr][c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w]






PREDICTORS = {'conv3x3_detect': p_conv, 'colormap': p_colormap, 'color_gate': p_gate, 'flip_select': p_select,
              'tile_select': p_tile, 'neighbor_count': p_count, 'bbox_mask': p_bbox}


def rule(kind, cfg, grid):
    k = kind
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    if k == 'fill_hole':
        for r in range(h):
            for c in range(w):
                if (grid[r][c] == 0 and _nb4(grid, r, c).count(1) >= 3 and 3 not in _nb8(grid, r, c)):
                    out[r][c] = 2
    elif k == 'isolated':
        for r in range(h):
            for c in range(w):
                nb = _nb8(grid, r, c)
                if grid[r][c] == 1 and 1 not in nb and 2 in nb:
                    out[r][c] = 3
    elif k == 'colormap':
        out = [[cfg['map'][x] for x in row] for row in grid]
    elif k == 'gate':
        out = [[cfg['fill'] if x in cfg['erase'] else x for x in row] for row in grid]
    elif k == 'select':
        a = np.array(grid)
        t = _op(a, cfg['op'])
        out = [[cfg['post'][int(x)] for x in row] for row in t.tolist()]
    elif k == 'tile':
        pr, pc = cfg['pr'], cfg['pc']
        out = [[grid[r % pr][c % pc] for c in range(w)] for r in range(h)]
    elif k == 'count_fill':
        for r in range(h):
            for c in range(w):
                nb = _nb8(grid, r, c)
                if grid[r][c] == 0 and nb.count(2) >= 3 and 1 not in nb:
                    out[r][c] = 2
    elif k == 'bbox_keep':
        cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 3]
        if cells:
            r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
            c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
            out = [[grid[r][c] if r0 <= r <= r1 and c0 <= c <= c1 else 0 for c in range(w)] for r in range(h)]
    elif k in ('teacher_conv', 'teacher_count'):
        out = PREDICTORS[KINDS[k][0]](cfg['params'], grid, False)[0]
    else:
        raise ValueError(k)
    return out


def one_grid(kind, rng, lv):
    k = kind
    h, w = rng.randint(4, 6 + min(lv, 2)), rng.randint(4, 6 + min(lv, 2))
    if k == 'fill_hole':
        g = _rand_grid(rng, h, w, [.4, .4, .1 + .03 * lv, .1 + .03 * lv])
        for _ in range(rng.randint(1, 2)):
            r, c = rng.randrange(1, h - 1), rng.randrange(1, w - 1)
            g[r][c] = 0
            nbs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for rr, cc in nbs:
                g[rr][cc] = 1
            if rng.random() < .5:                      # only three cross neighbours are 1
                rr, cc = rng.choice(nbs)
                g[rr][cc] = rng.choice([0, 2, 3])
            if rng.random() < .4:                      # a colour-3 cell somewhere in the 3x3 window
                dr, dc = rng.choice([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b])
                g[r + dr][c + dc] = 3
        return g
    if k == 'isolated':
        g = _rand_grid(rng, h, w, [.45, .3, .1, .15])
        for _ in range(rng.randint(1, 2)):
            r, c = rng.randrange(1, h - 1), rng.randrange(1, w - 1)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    g[r + dr][c + dc] = rng.choices([0, 2, 3], [.6, .1, .3])[0]
            g[r][c] = 1
            if rng.random() < .5:                      # near miss: one neighbour is also 1
                dr, dc = rng.choice([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b])
                g[r + dr][c + dc] = 1
        return g
    if k == 'colormap':
        return _rand_grid(rng, h, w, [.4, .3, .27, .03])
    if k == 'gate':
        return _rand_grid(rng, h, w, [.35, .3, .3, .05])
    if k == 'select':
        n = rng.randint(3, 5 + min(lv, 2))
        return _rand_grid(rng, n, n, [.25] * 4)
    if k == 'tile':
        return _rand_grid(rng, h, w, [.25] * 4)
    if k == 'count_fill':
        g = _rand_grid(rng, h, w, [.5, .1, .3 + .03 * lv, .1])
        r, c = rng.randrange(1, h - 1), rng.randrange(1, w - 1)    # plant a site with 2 or 3 colour-2 neighbours
        nb = [(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
        rng.shuffle(nb)
        g[r][c] = 0
        m = rng.choice([2, 3, 3, 4])
        for i, (rr, cc) in enumerate(nb):
            g[rr][cc] = 2 if i < m else rng.choices([0, 1, 3], [.5, .2, .3])[0]
        return g
    if k in ('teacher_conv', 'teacher_count'):
        return _rand_grid(rng, h, w, [.4, .25, .2, .15])
    if k == 'bbox_keep':
        g = _rand_grid(rng, h, w, [.35, .3, .35, 0.0])
        for _ in range(rng.randint(2, 3 + min(lv, 2))):
            g[rng.randrange(h)][rng.randrange(w)] = 3
        return g
    raise ValueError(k)


# TEACHERS-BEGIN
TEACHERS = {'conv_rand_a': {'conv.W': [[2.0, 0.0, 0.0, -2.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, -1.0, 0.0], [1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 2.0, -2.0], [0.0, 0.0, -1.0, -2.0], [0.0, 0.0, 0.0, 1.0], [1.0, -2.0, -2.0, 0.0], [-1.0, 1.0, 1.0, -2.0]], 'conv.b': [-3.0], 'paint': [0.0, 1.0, 0.0, 0.0]}, 'conv_rand_b': {'conv.W': [[-2.0, -1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, -2.0, 1.0, 1.0], [1.0, -2.0, -1.0, 0.0], [-1.0, 2.0, 0.0, 1.0], [0.0, 1.0, -1.0, -1.0], [-1.0, 0.0, -1.0, 1.0], [2.0, 0.0, 1.0, -2.0], [1.0, -1.0, 2.0, -2.0]], 'conv.b': [-2.0], 'paint': [0.0, 0.0, 1.0, 0.0]}, 'count_rand_a': {'self.W': [[-3.0, -3.0, 0.0, -3.0]], 'cnt.W': [[1.0, 0.0, 0.0, 0.0]], 'cnt.b': [-0.5], 'paint': [0.0, 0.0, 0.0, 1.0]}, 'count_rand_b': {'self.W': [[0.0, -1.0, -1.0, 1.0]], 'cnt.W': [[0.0, -1.0, 1.0, -1.0]], 'cnt.b': [-0.5], 'paint': [0.0, 0.0, 0.0, 1.0]}}
# TEACHERS-END

KINDS = {}

DESCRIPTIONS = {   # deliberately terse: the maker must infer details from the examples
    'fill_hole': 'Some cells change colour depending on the colours of the cells around them. Colours 0-3.',
    'isolated': 'Some cells change colour depending on the colours of the cells around them. Colours 0-3.',
    'colormap': 'Every cell of a given colour is treated the same way (by colour only, not by position).',
    'gate': 'Every cell of a given colour is treated the same way (by colour only, not by position).',
    'select': 'The output is a geometric rearrangement of the input grid (square grids), possibly with colours changed.',
    'tile': 'The output repeats a small block taken from the input.',
    'count_fill': 'Some cells change colour depending on how many of their 8 surrounding cells have a certain colour.',
    'bbox_keep': 'The output keeps part of the input and blanks the rest; a particular colour decides which part.',
    'teacher_conv': 'Some cells are recoloured depending on the colours in their 3x3 neighbourhood. Colours 0-3.',
    'teacher_count': 'Some cells are recoloured depending on their own colour and on how many of their 8 neighbours have each colour.',
}

NICHES = {   # niche -> (kind, cfg)
    'conv_fill_hole': ('fill_hole', {}),
    'conv_isolated': ('isolated', {}),
    'cmap_swap': ('colormap', {'map': [0, 2, 1, 3]}),
    'cmap_cycle': ('colormap', {'map': [0, 2, 3, 1]}),
    'gate_merge': ('gate', {'erase': [2, 3], 'fill': 1}),
    'select_fliplr': ('select', {'op': 1, 'post': [0, 1, 2, 3]}),
    'select_rot_swap': ('select', {'op': 4, 'post': [0, 3, 2, 1]}),
    'tile_2x2': ('tile', {'pr': 2, 'pc': 2}),
    'count_fill3': ('count_fill', {}),
    'bbox_keep3': ('bbox_keep', {}),
    'conv_rand_a': ('teacher_conv', {'params': TEACHERS.get('conv_rand_a', {})}),
    'conv_rand_b': ('teacher_conv', {'params': TEACHERS.get('conv_rand_b', {})}),
    'count_rand_a': ('teacher_count', {'params': TEACHERS.get('count_rand_a', {})}),
    'count_rand_b': ('teacher_count', {'params': TEACHERS.get('count_rand_b', {})}),
}
KINDS.update({
    'fill_hole': ('conv3x3_detect', 0, 0), 'isolated': ('conv3x3_detect', 0, 0), 'colormap': ('colormap', 0, 0),
    'gate': ('color_gate', 0, 0), 'select': ('flip_select', 0, 0), 'tile': ('tile_select', 0, 0),
    'count_fill': ('neighbor_count', 0, 0), 'bbox_keep': ('bbox_mask', 0, 0),
    'teacher_conv': ('conv3x3_detect', 0, 0), 'teacher_count': ('neighbor_count', 0, 0),
})


def template_of(niche):
    return KINDS[NICHES[niche][0]][0]


def sample_tasks(niche, generation, seed, n):
    kind, cfg = NICHES[niche]
    rng = random.Random(seed)
    lv = min(generation, 3)
    out = []
    for _ in range(n):
        g = one_grid(kind, rng, lv)
        out.append({'input': g, 'expected': rule(kind, cfg, g)})
    return out
