"""Versioned grid-cell/circuit boundaries and ordered expert-roster contracts.

Legacy numerical IDs remain unchanged. New manifests bind those IDs together
with interface/layout versions; a matching shape is not a competence claim.
"""
import copy
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


GRID = {'type': 'grid', 'rank': 2, 'dtype': 'integer', 'encoding': 'cellforge-grid-token/1'}
DEFAULT_INTERFACE = {'schema': 'cellforge-cell-interface/1', 'version': 1,
                     'input': GRID, 'output': GRID, 'context': 'demonstration-pairs/1'}


def validate_interface(interface):
    if not isinstance(interface, dict) or set(interface) != {'schema', 'version', 'input', 'output', 'context'}:
        raise ValueError('exact versioned interface fields required')
    if interface['schema'] != 'cellforge-cell-interface/1' or type(interface['version']) is not int or interface['version'] < 1:
        raise ValueError('unsupported interface schema/version')
    if not isinstance(interface['context'], str) or not interface['context']:
        raise ValueError('explicit context encoding required')
    for key in ('input', 'output'):
        port = interface[key]
        if not isinstance(port, dict) or set(port) != {'type', 'rank', 'dtype', 'encoding'}:
            raise ValueError('typed port fields required')
        if port['type'] != 'grid' or port['rank'] != 2 or port['dtype'] != 'integer':
            raise ValueError('skillcore supports integer rank-two grid ports only')
        if not isinstance(port['encoding'], str) or not port['encoding']:
            raise ValueError('explicit grid encoding required')
    return copy.deepcopy(interface)


def reference(cell, templates):
    """Bind immutable numeric identity, typed interface and tensor layout recursively."""
    if cell.get('kind') == 'cell':
        spec = templates[cell['template']]
        expected = digest({'t': cell['template'], 'p': cell['params']})
        interface = validate_interface(spec.get('interface', DEFAULT_INTERFACE))
        layout = {name: {'shape': list(shape), 'dtype': 'float64'} for name, shape in spec['shapes'].items()}
        body = {'kind': 'cell', 'template': cell['template'],
                'implementation_version': spec.get('version', 'legacy/1'), 'layout': layout}
    elif cell.get('kind') == 'chain':
        if len(cell.get('parts', [])) != 2:
            raise ValueError('a chain requires two ordered parts')
        first, second = [reference(part, templates) for part in cell['parts']]
        require_chain(first, second)
        expected = digest({'chain': [part['id'] for part in cell['parts']]})
        interface = {'schema': 'cellforge-cell-interface/1', 'version': 1,
                     'input': first['interface']['input'], 'output': second['interface']['output'],
                     'context': first['interface']['context']}
        body = {'kind': 'chain', 'parts': [first, second]}
    else:
        raise ValueError('unknown cell kind')
    if cell.get('id') != expected:
        raise ValueError('cell identity differs from current literal weights/parts')
    body.update(schema='cellforge-cell-reference/1', weight_id=expected, interface=interface)
    body['reference_hash'] = digest(body)
    return body


def require_chain(first, second):
    a, b = first['interface'], second['interface']
    if a['version'] != b['version'] or a['output'] != b['input'] or a['context'] != b['context']:
        raise ValueError('chain boundary incompatible; an explicit authored adapter is required')


def require_crossover(first, second):
    if first['kind'] != 'cell' or second['kind'] != 'cell':
        raise ValueError('tensor crossover requires individual cells')
    for field in ('template', 'implementation_version', 'layout', 'interface'):
        if first[field] != second[field]:
            raise ValueError('tensor crossover incompatible: ' + field)


def roster_ids(cells):
    ids = [cell['id'] for cell in cells]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('router roster must contain distinct nonempty expert identities')
    return ids


def router_contract(cells, router):
    ids = roster_ids(cells)
    n = len(ids)
    if set(router) != {'route.W', 'route.b'}:
        raise ValueError('exact router tensors required')
    w, b = router['route.W'], router['route.b']
    if not isinstance(w, list) or len(w) != n or any(not isinstance(row, list) or len(row) != n for row in w):
        raise ValueError('router W must be N by N')
    if not isinstance(b, list) or len(b) != n:
        raise ValueError('router b must have N entries')
    if any(type(v) not in (int, float) or not math.isfinite(v) for row in w + [b] for v in row):
        raise ValueError('finite literal router coefficients required')
    return {'schema': 'cellforge-router-roster/1', 'expert_ids': ids,
            'router_hash': digest(router), 'features': 'negative-leave-one-out-demo-error/1'}


def remap_router(router, old_ids, new_ids):
    """Pure permutation only. Adding/removing experts needs a new authored router."""
    if len(set(old_ids)) != len(old_ids) or set(old_ids) != set(new_ids) or len(new_ids) != len(old_ids):
        raise ValueError('roster additions/removals require explicit new router tensors')
    router_contract([{'id': cid} for cid in old_ids], router)
    order = [old_ids.index(cid) for cid in new_ids]
    return {'route.W': [[router['route.W'][i][j] for j in order] for i in order],
            'route.b': [router['route.b'][i] for i in order]}


def validate_tensor_override(tensors, reference):
    """Validate an evaluator's ablation tensors without restoring saved values.

    The checkpoint supplies named topology and exact shapes. Numerical identity
    may change (including multiple identical zero experts); slots retain their
    declared meaning. No training-range bound is imposed on evaluator controls.
    """
    if not isinstance(tensors, dict) or set(tensors) != set(reference):
        raise ValueError('supplied tensors must exactly match checkpoint names')
    def check(value, shape_source):
        if isinstance(shape_source, list):
            if not isinstance(value, list) or len(value) != len(shape_source):
                raise ValueError('supplied tensor shape differs from checkpoint')
            for item, expected in zip(value, shape_source):
                check(item, expected)
        else:
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError('supplied tensors require finite literal numbers')
    for name in reference:
        check(tensors[name], reference[name])
    return copy.deepcopy(tensors)
