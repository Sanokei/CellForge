"""Authored object-table genes. All inference is existing IR tensor math.

Objects are supplied by the caller, not extracted by hidden Python inference.
Shapes are translated to the origin, without rotation or scale normalization.
"""
from __future__ import annotations

import numpy as np

from cellforge.ir.graph import Graph, Limits


class Wiring:
    def __init__(self):
        self.nodes = []

    def add(self, use, inputs=None, **args):
        ref = f'n{len(self.nodes)}'
        node = {'id': ref, 'use': use}
        if inputs:
            node['in'] = inputs
        if args:
            node['args'] = args
        self.nodes.append(node)
        return ref

    def op(self, name, **inputs):
        return self.add('op:' + name, inputs)

    def reshape(self, x, shape):
        return self.add('op:reshape', {'x': x}, shape=shape)

    def reduce(self, name, x, axis, keepdims=False):
        return self.add('op:' + name, {'x': x}, axis=axis, keepdims=keepdims)

    def bundle(self, inputs, outputs):
        return {'kind': 'bundle', 'role': 'cluster', 'inputs': list(inputs),
                'nodes': self.nodes, 'outputs': outputs}


def build_object_gene_bank():
    """Build a bank with independently callable, entry-reachable typed genes."""
    g = Graph(meta={'task': 'binding', 'construction': 'literal authored object genes/1',
                    'gradients': False, 'benchmark_tasks_read': 0,
                    'port_contracts': {}, 'cell_port_contracts': {},
                    'limitations': 'Requires caller-provided object tables; no extraction, '
                    'demonstration adaptation or global one-to-one assignment. '
                    'Gravity moves the selection as one rigid group, not independent bodies. '
                    'Port shapes/domains are caller preconditions, not runtime validators.'})

    def port(shape, meaning, domain='finite float64'):
        return {'dtype': 'float64', 'shape': ['B'] + shape,
                'domain': domain, 'meaning': meaning}

    valid = port([16], 'Object slots; zero slots are padding', 'binary')
    masks = port([16, 900], 'Disjoint object masks on row-major 30x30 canvas', 'binary')
    shapes = port([16, 900], 'Object masks translated to top-left origin', 'binary')
    scalar = lambda text: port([16], text, 'integer in [0,900]')
    selected = port([16], 'Selected object slots; must be zero for padding', 'binary')
    grid = port([30, 30, 10], 'One-hot valid canvas, zero outside extent', 'binary')
    canvas = port([30, 30, 1], 'Rectangular valid canvas mask', 'binary')

    def cell(name, W1, b1, act='identity', W2=None, b2=None):
        tensors = {'W1': W1, 'b1': b1}
        if W2 is not None:
            tensors.update(W2=W2, b2=b2)
        refs = {}
        for key, value in tensors.items():
            refs[key] = name + '.' + key
            g.params[refs[key]] = np.asarray(value, dtype=np.float64)
        g.defs[name] = {'kind': 'cell', 'act': act, 'params': refs}
        output_width = len(W2) if W2 is not None else len(W1)
        g.meta['cell_port_contracts'][name] = {
            'inputs': {'x': {'dtype': 'float64', 'shape': ['B', '...', len(W1[0])], 'domain': 'finite'}},
            'outputs': {'y': {'dtype': 'float64', 'shape': ['B', '...', output_width]}},
            'semantics': 'y = act(x @ W1.T + b1)' + (' @ W2.T + b2' if W2 is not None else ''),
            'activation': act, 'parameters': refs}
        return name

    # Integer equality hat and bounded binary complement are explicit tensors.
    eq = cell('object.equal', [[1], [1], [1]], [1, 0, -1], 'relu', [[1, -2, 1]], [0])
    inv = cell('object.complement', [[-1]], [1])
    identity = cell('object.identity', [[1]], [0])
    negative = cell('object.negative', [[-1]], [0])
    def equal(n, a, b):
        return n.add(eq, {'x': n.op('sub', a=a, b=b)})
    def invert(n, x):
        return n.add(inv, {'x': x})

    def register(name, n, inputs, outputs, output_types, semantics):
        g.defs[name] = n.bundle(inputs, outputs)
        g.entries[name] = name
        g.meta['port_contracts'][name] = {
            'inputs': inputs, 'outputs': output_types, 'semantics': semantics,
            'batch': 'Independent examples; cells act on the final axis.'}

    # Seven explicit correspondence costs. Shape mismatch = binary-mask L1;
    # color mismatch = exact categorical mismatch, position = Manhattan distance.
    for label, weights in (
        ('shape', [[-1, 0, 0]]), ('color', [[0, -1, 0]]),
        ('position', [[0, 0, -1]]), ('shape_color', [[-1, -1801, 0]]),
        ('shape_position', [[-61, 0, -1]]), ('color_position', [[0, -61, -1]]),
        ('shape_color_position', [[-61, -109861, -1]])):
        name = 'match_' + label
        scorer = cell(name + '.cost', weights, [0])
        n = Wiring()
        sd = n.op('abs', x=n.op('sub', a=n.reshape('$source_shapes', [16, 1, 900]),
                               b=n.reshape('$target_shapes', [1, 16, 900])))
        sd = n.reduce('sum', sd, 2, True)
        cd = equal(n, n.reshape('$source_colors', [16, 1, 1]),
                   n.reshape('$target_colors', [1, 16, 1]))
        cd = invert(n, cd)
        pd = n.op('abs', x=n.op('sub', a=n.reshape('$source_positions', [16, 1, 2]),
                               b=n.reshape('$target_positions', [1, 16, 2])))
        pd = n.reduce('sum', pd, 2, True)
        features = n.add('op:concat', {'x0': sd, 'x1': cd, 'x2': pd}, axis=2)
        scores = n.reshape(n.add(scorer, {'x': features}), [16, 16])
        missing = n.reshape(invert(n, n.reshape('$target_valid', [16, 1])), [1, 16])
        penalty = n.add('op:scale', {'x': missing}, c=-1000000000)
        ranked = n.op('add', a=scores, b=penalty)
        choice = n.add('op:onehot_argmax', {'x': ranked}, axis=1)
        anytarget = n.op('step', x=n.reduce('sum', '$target_valid', 0, True))
        choice = n.op('mul', a=choice, b=n.reshape('$source_valid', [16, 1]))
        choice = n.op('mul', a=choice, b=n.reshape(anytarget, [1, 1]))
        inputs = {'source_shapes': shapes, 'target_shapes': shapes,
                  'source_colors': port([16], 'Categorical color codes', 'integer [0,9]'),
                  'target_colors': port([16], 'Categorical color codes', 'integer [0,9]'),
                  'source_positions': port([16, 2], 'Top-left row,column', 'integer [0,29]'),
                  'target_positions': port([16, 2], 'Top-left row,column', 'integer [0,29]'),
                  'source_valid': valid, 'target_valid': valid}
        register(name, n, inputs, {'matches': choice, 'scores': scores},
                 {'matches': port([16, 16], 'One target per valid source, or zero if no targets', 'binary'),
                  'scores': port([16, 16], 'Raw weighted negative distance')},
                 'Nearest target independently per source; lowest target-slot tie. '
                 'Combined priorities: color then shape then position. Many-to-one allowed.')

    # Extremal and categorical-unique selectors; all ties or first-slot variants.
    for field in ('area', 'row', 'column', 'width', 'height'):
        for direction in ('largest', 'smallest'):
            for first in (False, True):
                name = f'select_{direction}_{field}' + ('_first' if first else '_all')
                n = Wiring()
                value = n.reshape('$values', [16, 1])
                score = n.add(identity if direction == 'largest' else negative, {'x': value})
                penalty = n.add('op:scale', {'x': invert(n, n.reshape('$valid', [16, 1]))}, c=-10000)
                score = n.op('add', a=score, b=penalty)
                if first:
                    pick = n.add('op:onehot_argmax', {'x': score}, axis=0)
                else:
                    best = n.reduce('max', score, 0, True)
                    pick = equal(n, score, best)
                pick = n.reshape(n.op('mul', a=pick, b=n.reshape('$valid', [16, 1])), [16])
                register(name, n, {'values': scalar(field), 'valid': valid}, {'selected': pick},
                         {'selected': selected}, f'{direction} {field}; ' + ('lowest-slot tie' if first else 'all ties'))

    for field, width, spec in (('color', 1, port([16, 1], 'Color code', 'integer [0,9]')),
                               ('area', 1, port([16, 1], 'Object area', 'integer [1,900]')),
                               ('shape', 900, shapes)):
        n = Wiring()
        diff = n.op('abs', x=n.op('sub', a=n.reshape('$features', [16, 1, width]),
                                 b=n.reshape('$features', [1, 16, width])))
        diff = n.reduce('sum', diff, 2, True)
        same = equal(n, diff, n.add('op:scale', {'x': diff}, c=0))
        same = n.op('mul', a=same, b=n.reshape('$valid', [1, 16, 1]))
        count = n.reduce('sum', same, 1)
        unique = equal(n, count, n.add('op:bias', {'x': n.add('op:scale', {'x': count}, c=0)}, c=1))
        pick = n.reshape(n.op('mul', a=unique, b=n.reshape('$valid', [16, 1])), [16])
        register('select_unique_' + field, n, {'features': spec, 'valid': valid}, {'selected': pick},
                 {'selected': selected}, 'Select every object whose feature occurs exactly once among valid slots.')

    # Object editing with supplied selection. Translation matrices are computed
    # in the graph, allowing arbitrary signed integer displacement (up to 29).
    for operation in ('delete', 'keep', 'copy', 'move'):
        n = Wiring()
        mask = n.op('matmul', a=n.reshape('$selected', [1, 16]), b='$masks')
        mask = n.reshape(mask, [30, 30, 1])
        foreground = n.op('mul', a='$colors', b=mask)
        base = '$colors'
        if operation in ('delete', 'move'):
            base = n.op('add', a=n.op('mul', a='$colors', b=invert(n, mask)),
                        b=n.op('mul', a=mask, b=n.reshape('$background', [1, 1, 10])))
        elif operation == 'keep':
            base = n.op('add', a=foreground,
                        b=n.op('mul', a=invert(n, mask), b=n.reshape('$background', [1, 1, 10])))
        inputs = {'colors': grid, 'masks': masks, 'selected': selected,
                  'background': port([10], 'Explicit background color', 'one-hot'), 'canvas': canvas}
        if operation in ('copy', 'move'):
            inputs['offset'] = port([2], 'Signed row,column displacement', 'integer [-29,29]')
            coords = n.add('op:coords', n=30)
            for axis in (0, 1):
                offset = n.add('op:slice', {'x': '$offset'}, axis=0, start=axis, stop=axis+1)
                target = n.op('sub', a=n.reshape(coords, [30, 1, 1]), b=n.reshape(offset, [1, 1, 1]))
                matrix = n.reshape(equal(n, target, n.reshape(coords, [1, 30, 1])), [30, 30])
                for is_mask in (False, True):
                    data = mask if is_mask else foreground
                    if axis:
                        data = n.add('op:transpose', {'x': data}, perm=[1, 0, 2])
                    channels = 1 if is_mask else 10
                    data = n.reshape(n.op('matmul', a=matrix, b=n.reshape(data, [30, 30*channels])), [30, 30, channels])
                    if axis:
                        data = n.add('op:transpose', {'x': data}, perm=[1, 0, 2])
                    if is_mask:
                        mask = data
                    else:
                        foreground = data
            base = n.op('add', a=n.op('mul', a=base, b=invert(n, mask)), b=foreground)
        base = n.op('mul', a=base, b='$canvas')
        register('object_' + operation, n, inputs, {'colors': base}, {'colors': grid},
                 'Selected disjoint masks; preserve dimensions. Move/copy clips to canvas and overwrites destination. '
                 'Move clears source before pasting; keep fills everything else with explicit background.')

    # Rigid gravity, both a certified-clearance port and complete graph search.
    for direction, W in (('down', [[1], [0]]), ('up', [[-1], [0]]),
                         ('right', [[0], [1]]), ('left', [[0], [-1]])):
        converter = cell('gravity_' + direction + '.offset', W, [0, 0])
        n = Wiring()
        offset = n.add(converter, {'x': '$clearance'})
        move_inputs = {'colors': grid, 'masks': masks, 'selected': selected,
                       'background': port([10], 'Explicit background color', 'one-hot'), 'canvas': canvas,
                       'clearance': port([1], 'Maximum collision-free rigid displacement of selected group', 'integer [0,29]')}
        call = n.add('object_move', {**{p: '$'+p for p in move_inputs if p != 'clearance'}, 'offset': offset})
        register('gravity_by_clearance_' + direction, n, move_inputs, {'colors': call+'.colors'}, {'colors': grid},
                 'Rigid group movement by certified clearance; caller computes clearance against borders and unselected obstacles.')

        n = Wiring()
        group = n.reshape(n.op('matmul', a=n.reshape('$selected', [1, 16]), b='$masks'), [30, 30, 1])
        area = n.reduce('sum', n.reshape(group, [900]), 0, True)
        background_pixels = n.reduce('sum', n.op('mul', a='$colors', b=n.reshape('$background', [1, 1, 10])), 2, True)
        occupied = n.op('mul', a=invert(n, background_pixels), b='$canvas')
        obstacles = n.op('mul', a=occupied, b=invert(n, group))
        axis = 0 if direction in ('up', 'down') else 1
        sign = 1 if direction in ('down', 'right') else -1
        blocked = n.add('op:scale', {'x': area}, c=0)
        clearance = blocked
        for step in range(1, 30):
            shifted = n.add('op:shift', {'x': group}, axis=axis, k=sign*step)
            inside = n.op('mul', a=shifted, b='$canvas')
            remaining = n.reduce('sum', n.reshape(inside, [900]), 0, True)
            escaped = n.op('sub', a=area, b=remaining)
            collisions = n.reduce('sum', n.reshape(n.op('mul', a=shifted, b=obstacles), [900]), 0, True)
            hit = n.op('step', x=n.op('add', a=escaped, b=collisions))
            blocked = n.op('maximum', a=blocked, b=hit)
            clearance = n.op('add', a=clearance, b=invert(n, blocked))
        clearance = n.op('mul', a=clearance, b=n.op('step', x=area))
        inputs = {p: spec for p, spec in move_inputs.items() if p != 'clearance'}
        call = n.add('gravity_by_clearance_' + direction,
                     {**{p: '$'+p for p in inputs}, 'clearance': clearance})
        register('gravity_' + direction, n, inputs,
                 {'colors': call+'.colors', 'clearance': clearance},
                 {'colors': grid, 'clearance': port([1], 'Computed rigid displacement', 'integer [0,29]')},
                 'Move selected objects as one rigid group until first obstacle or canvas edge. '
                 'Every intermediate displacement is checked; no tunneling. Empty selection gives zero clearance.')

    # Count encodings are exact over a declared bounded integer domain.
    # Each unary threshold is a separate tiny weighted reachable cell.
    for kind in ('objects', 'pixels'):
        n = Wiring()
        count = n.reduce('sum', '$items', 0, True)
        count = n.add(identity, {'x': count})
        register('count_' + kind, n, {'items': port([16 if kind == 'objects' else 900], 'Selected item indicators', 'binary')},
                 {'count': count}, {'count': port([1], 'Number of selected items', 'integer')}, 'Sum binary item indicators.')
    for encoding in ('strip', 'onehot'):
        n = Wiring()
        parts = {}
        for k in range(30 if encoding == 'strip' else 31):
            if encoding == 'strip':
                c = cell(f'count_strip.bit{k}', [[1], [1]], [-k, -k-1], 'relu', [[1, -1]], [0])
            else:
                c = cell(f'count_onehot.bit{k}', [[1], [1], [1]], [1-k, -k, -1-k], 'relu', [[1, -2, 1]], [0])
            parts['x'+str(k)] = n.add(c, {'x': '$count'})
        result = n.add('op:concat', parts, axis=0)
        register('count_to_' + encoding, n, {'count': port([1], 'Bounded count', 'integer [0,30]')},
                 {'output': result}, {'output': port([30 if encoding == 'strip' else 31], 'Unary prefix or categorical count encoding', 'binary')},
                 'Strip has exactly count leading ones; onehot index equals count, including zero.')

    n = Wiring()
    strip = n.add('count_to_strip', {'count': '$count'}) + '.output'
    occupied = n.reshape(strip, [1, 30, 1])
    coords = n.add('op:coords', n=30)
    # Only row zero is active; construct it with a generic onehot argmax tie.
    row = n.reshape(n.add('op:onehot_argmax', {'x': n.add('op:scale', {'x': coords}, c=0)}, axis=0), [30, 1, 1])
    occupied = n.op('mul', a=occupied, b=row)
    width = n.add('op:maximum', {'a': '$count', 'b': n.add('op:bias', {'x': n.add('op:scale', {'x': '$count'}, c=0)}, c=1)})
    diff = n.op('sub', a=n.reshape(width, [1, 1]), b=n.reshape(coords, [30, 1]))
    active = n.reshape(n.op('step', x=diff), [1, 30, 1])
    active = n.op('mul', a=active, b=row)
    colors = n.op('add', a=n.op('mul', a=occupied, b=n.reshape('$color', [1, 1, 10])),
                  b=n.op('mul', a=n.op('sub', a=active, b=occupied), b=n.reshape('$background', [1, 1, 10])))
    register('count_to_color_strip', n,
             {'count': port([1], 'Bounded count', 'integer [0,30]'),
              'color': port([10], 'Strip color', 'one-hot'),
              'background': port([10], 'Empty-output background', 'one-hot')},
             {'colors': colors, 'width': width},
             {'colors': grid, 'width': port([1], 'Strip width; height is always one', 'integer [1,30]')},
             'Render count colored pixels as a 1 x count grid; zero renders a 1 x 1 background grid.')

    for kind, size in (('objects', 16), ('pixels', 900)):
        for encoding in ('strip', 'onehot', 'color_strip'):
            n = Wiring()
            inputs = {'items': port([size], 'Binary items; total must be <=30', 'binary; sum <=30')}
            count = n.add('count_'+kind, {'items': '$items'})
            args = {'count': count+'.count'}
            encoder = 'count_to_'+encoding
            if encoding == 'color_strip':
                for key in ('color', 'background'):
                    inputs[key] = g.meta['port_contracts'][encoder]['inputs'][key]
                    args[key] = '$'+key
            encoded = n.add(encoder, args)
            output_types = g.meta['port_contracts'][encoder]['outputs']
            register('count_'+kind+'_to_'+encoding, n, inputs,
                     {key: encoded+'.'+key for key in output_types}, output_types,
                     'Count input indicators then encode; output-count domain is bounded to 30.')

    # Required binding entry is a callable count cell; every other gene has its
    # own public entry, so reachability is structural rather than an inventory claim.
    g.entries['predict'] = 'count_objects'
    g.validate(Limits(strict_microcells=True))
    return g
