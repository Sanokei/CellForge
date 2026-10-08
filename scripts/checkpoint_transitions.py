"""Pure, detached phase checkpoints for alternating cell/router evolution.

Only ``validate_checkpoint`` and ``make_checkpoint`` are public APIs. Inputs
are strict finite JSON values (lists, not tuples); errors raise ValueError.
Phase indices are zero based, cells then router, with epoch == index // 2.
Rounds are positive; plan limits and per-phase call allowances belong to the
outer scheduler. worker_calls_reserved is this phase's count, not a lifetime
counter. Hashes use cell_contracts.digest: sorted compact JSON and SHA256.

Bank checks bind literal tensors, topology, declared interfaces/layouts, expert
order and router contracts without importing an evaluator or template registry.
The driver must also compare each bank with skillcore.export_bank using its
actual registry, recompute complete outcomes, and verify scope/manifest/journal
bindings. These hashes establish consistency, not authenticity or competence.

Standalone validation checks only the information in one checkpoint. In
particular, it cannot prove a carried incumbent or bank gain without the
predecessor. Construction validates the immediate transition, not an unseen
ancestral chain. Callers must authenticate and validate the persisted chain.
No report outcomes, scoring, I/O, random draws or model calls occur here.
"""

import math

import cell_contracts as contracts


__all__ = ['validate_checkpoint', 'make_checkpoint']

_SCHEMA = 'cellforge-phase-checkpoint/1'
_PANELS = ('search_fitness', 'development_selection')
_PHASE_FIELDS = {'plan_hash', 'index', 'epoch', 'kind', 'rounds'}
_EVIDENCE_FIELDS = {
    'manifest_hash', 'journal_head_hash', 'scope_hash', 'archive_cells',
    'archive_bank', 'archive_outcomes', 'incumbent_bank', 'incumbent_outcomes',
    'worker_calls_reserved', 'cell_state',
}
_CHECKPOINT_FIELDS = {
    'schema', 'phase', 'predecessor_hash', 'manifest_hash', 'journal_head_hash',
    'scope_hash', 'archive_cells', 'archive_bank', 'archive_outcomes',
    'retained_bank', 'retained_outcomes', 'retention', 'worker_calls_reserved',
    'cell_state', 'hash',
}


def _json_copy(value):
    """Reject coercions/custom objects and detach every mutable container."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_json_copy(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError('checkpoint data must be finite JSON with string object keys')


def _fields(value, required, label, optional=()):
    if (type(value) is not dict or not set(required) <= set(value)
            or set(value) - set(required) - set(optional)):
        raise ValueError(label + ' has missing or unexpected fields')


def _hash(value, label):
    if (type(value) is not str or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(label + ' must be a lowercase SHA256 hex digest')


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + ' must be an integer >= ' + str(minimum))


def _text(value, label):
    if type(value) is not str or not value:
        raise ValueError(label + ' must be a nonempty string')


def _same(first, second):
    # Ordinary Python equality conflates bool/int and integer/float literals.
    return contracts.digest(first) == contracts.digest(second)


def _check_hash(value, key, label):
    _hash(value[key], label)
    if contracts.digest({k: v for k, v in value.items() if k != key}) != value[key]:
        raise ValueError(label + ' does not match its content')


def _phase(phase):
    _fields(phase, _PHASE_FIELDS, 'phase')
    _hash(phase['plan_hash'], 'phase plan_hash')
    for name in ('index', 'epoch'):
        _integer(phase[name], 'phase ' + name)
    _integer(phase['rounds'], 'phase rounds', minimum=1)
    if phase['epoch'] != phase['index'] // 2:
        raise ValueError('phase epoch must equal index // 2')
    if phase['kind'] != ('cells' if phase['index'] % 2 == 0 else 'router'):
        raise ValueError('phase kind must alternate cells/router from index zero')


def _outcomes(outcomes):
    _fields(outcomes, _PANELS, 'outcomes')
    for panel in _PANELS:
        values = outcomes[panel]
        if type(values) is not dict or not values:
            raise ValueError('each development panel requires a complete nonempty outcome map')
        for task_id, result in values.items():
            _text(task_id, 'outcome task ID')
            if type(result) is not bool:
                raise ValueError('every outcome must be an observed boolean')
    if set(outcomes[_PANELS[0]]) & set(outcomes[_PANELS[1]]):
        raise ValueError('search_fitness and development_selection must be disjoint')


def _same_panels(first, second):
    if any(set(first[panel]) != set(second[panel]) for panel in _PANELS):
        raise ValueError('outcomes must cover identical complete development panels')


def _consistent_outcomes(first_bank, first_outcomes, second_bank, second_outcomes):
    if (first_bank['artifact_hash'] == second_bank['artifact_hash']
            and not _same(first_outcomes, second_outcomes)):
        raise ValueError('identical banks cannot have conflicting outcomes in one scope')


def _interface(interface):
    contracts.validate_interface(interface)
    if any(type(interface[port]['rank']) is not int for port in ('input', 'output')):
        raise ValueError('interface ranks must be literal integers')


def _tensor(value, shape):
    if shape:
        if type(value) is not list or len(value) != shape[0]:
            raise ValueError('expert tensor differs from its declared layout')
        for item in value:
            _tensor(item, shape[1:])
        return
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError('expert tensors require finite nonboolean literal numbers')


def _restore_reference(ref, flat):
    """Validate a reference against exactly its flattened tensors, recursively."""
    base = {'schema', 'kind', 'weight_id', 'interface', 'reference_hash'}
    if type(ref) is not dict:
        raise ValueError('expert reference must be an object')
    kind = ref.get('kind')
    if kind == 'cell':
        _fields(ref, base | {'template', 'implementation_version', 'layout'}, 'cell reference')
    elif kind == 'chain':
        _fields(ref, base | {'parts'}, 'chain reference')
    else:
        raise ValueError('expert reference kind must be cell or chain')
    if ref['schema'] != 'cellforge-cell-reference/1':
        raise ValueError('unsupported expert reference schema')
    _check_hash(ref, 'reference_hash', 'expert reference_hash')
    _hash(ref['weight_id'], 'expert weight_id')
    _interface(ref['interface'])
    if kind == 'cell':
        _text(ref['template'], 'expert template')
        _text(ref['implementation_version'], 'expert implementation_version')
        layout = ref['layout']
        if type(layout) is not dict or not layout or set(layout) != set(flat):
            raise ValueError('named expert tensors must exactly match a nonempty layout')
        for name, specification in layout.items():
            _text(name, 'tensor name')
            _fields(specification, {'shape', 'dtype'}, 'tensor layout')
            shape = specification['shape']
            if (specification['dtype'] != 'float64' or type(shape) is not list
                    or any(type(size) is not int or size < 0 for size in shape)):
                raise ValueError('tensor layout requires float64 and nonnegative integer dimensions')
            _tensor(flat[name], shape)
        expected = contracts.digest({'t': ref['template'], 'p': flat})
        cell = {'kind': 'cell', 'template': ref['template'], 'params': flat, 'id': expected}
    else:
        if type(ref['parts']) is not list or len(ref['parts']) != 2:
            raise ValueError('chain references require two ordered parts')
        if any(not key.startswith(('A.', 'B.')) for key in flat):
            raise ValueError('chain tensors require ordered A./B. prefixes')
        parts = [_restore_reference(part, {key[2:]: value for key, value in flat.items()
                                           if key.startswith(prefix)})
                 for part, prefix in zip(ref['parts'], ('A.', 'B.'))]
        first, second = ref['parts']
        contracts.require_chain(first, second)
        interface = {'schema': 'cellforge-cell-interface/1', 'version': 1,
                     'input': first['interface']['input'],
                     'output': second['interface']['output'],
                     'context': first['interface']['context']}
        if not _same(ref['interface'], interface):
            raise ValueError('chain interface differs from its ordered part boundaries')
        expected = contracts.digest({'chain': [part['id'] for part in parts]})
        cell = {'kind': 'chain', 'parts': parts, 'id': expected}
    if ref['weight_id'] != expected:
        raise ValueError('expert identity differs from its literal tensors or ordered parts')
    return cell


def _literal_matches(cell, restored):
    """Match behavior-bearing fields while leaving all JSON metadata intact."""
    if type(cell) is not dict or not set(restored) <= set(cell):
        raise ValueError('archive cell lacks required literal fields')
    if cell['kind'] != restored['kind'] or cell['id'] != restored['id']:
        raise ValueError('archive cell differs from the ordered bank expert')
    if restored['kind'] == 'cell':
        if cell['template'] != restored['template'] or not _same(cell['params'], restored['params']):
            raise ValueError('archive cell tensors/template differ from the bank')
    else:
        if type(cell['parts']) is not list or len(cell['parts']) != 2:
            raise ValueError('archive chain requires two ordered literal parts')
        for part, expected in zip(cell['parts'], restored['parts']):
            _literal_matches(part, expected)


def _bank(bank, cells=None):
    _fields(bank, {'schema', 'experts', 'router', 'tensors', 'artifact_hash'}, 'skill bank')
    if bank['schema'] != 'cellforge-skill-bank/1':
        raise ValueError('unsupported skill-bank schema')
    _check_hash(bank, 'artifact_hash', 'bank artifact_hash')
    experts, tensors = bank['experts'], bank['tensors']
    if type(experts) is not list or not experts or type(tensors) is not dict:
        raise ValueError('bank requires a nonempty ordered expert list and literal tensor map')
    if not {'route.W', 'route.b'} <= set(tensors):
        raise ValueError('bank lacks literal router tensors')
    if cells is not None and (type(cells) is not list or len(cells) != len(experts)):
        raise ValueError('archive cells must exactly cover the ordered bank roster')
    restored, consumed = [], {'route.W', 'route.b'}
    for index, ref in enumerate(experts):
        prefix = str(index) + ':'
        names = [name for name in tensors if name.startswith(prefix)]
        flat = {name[len(prefix):]: tensors[name] for name in names}
        cell = _restore_reference(ref, flat)
        restored.append(cell)
        consumed.update(names)
        if cells is not None:
            _literal_matches(cells[index], cell)
    if consumed != set(tensors):
        raise ValueError('bank contains tensors outside its exact ordered expert roster')
    router = {name: tensors[name] for name in ('route.W', 'route.b')}
    contracts.validate_tensor_override(router, router)
    if not _same(bank['router'], contracts.router_contract(restored, router)):
        raise ValueError('bank router contract differs from its tensors or ordered expert IDs')


def _bank_prefix(prefix_bank, archive_bank):
    """Retained experts may be fewer; their full router remains separate."""
    count = len(prefix_bank['experts'])
    if not _same(prefix_bank['experts'], archive_bank['experts'][:count]):
        raise ValueError('earlier expert references must remain an unchanged archive prefix')
    for name, value in prefix_bank['tensors'].items():
        if name not in ('route.W', 'route.b'):
            if name not in archive_bank['tensors'] or not _same(value, archive_bank['tensors'][name]):
                raise ValueError('earlier expert tensors must remain an unchanged archive prefix')


def _archive(bank, cells):
    if type(cells) is not list or not cells:
        raise ValueError('archive_cells must be a nonempty ordered list of literal cells')
    _bank(bank, cells)


def _ids(values, label, allowed=None):
    if type(values) is not list:
        raise ValueError(label + ' must be a list')
    for value in values:
        _text(value, label + ' entry')
    if len(set(values)) != len(values) or (allowed is not None and not set(values) <= allowed):
        raise ValueError(label + ' contains duplicate or out-of-scope IDs')


def _rng(state):
    if (type(state) is not list or len(state) != 3
            or type(state[0]) is not int or state[0] != 3
            or type(state[1]) is not list or len(state[1]) != 625):
        raise ValueError('evolution_rng must be a JSON Python Random version-3 state')
    if any(type(word) is not int or not 0 <= word <= 0xffffffff for word in state[1][:-1]):
        raise ValueError('evolution_rng requires 624 uint32 words')
    if type(state[1][-1]) is not int or not 0 <= state[1][-1] <= 624:
        raise ValueError('evolution_rng index must be in 0..624')
    if state[2] is not None:
        _tensor(state[2], [])


def _curriculum(curriculum, search_ids):
    required = {'wave', 'exposure', 'recovery_anchors'}
    optional = {'phase', 'selected_task_ids', 'scope_hash', 'outcomes'}
    _fields(curriculum, required, 'curriculum', optional)
    if set(curriculum) - required not in (set(), optional):
        raise ValueError('curriculum selection fields must occur together')
    _integer(curriculum['wave'], 'curriculum wave')
    exposure = curriculum['exposure']
    if type(exposure) is not dict or not set(exposure) <= search_ids:
        raise ValueError('curriculum exposure IDs must belong to search_fitness')
    for count in exposure.values():
        _integer(count, 'curriculum exposure')
    _ids(curriculum['recovery_anchors'], 'curriculum recovery_anchors', search_ids)
    if 'phase' not in curriculum:
        return
    if curriculum['phase'] not in ('exploration', 'challenge', 'recovery'):
        raise ValueError('unsupported curriculum phase')
    _ids(curriculum['selected_task_ids'], 'curriculum selected_task_ids', search_ids)
    _hash(curriculum['scope_hash'], 'curriculum scope_hash')
    outcomes = curriculum['outcomes']
    if type(outcomes) is not dict:
        raise ValueError('curriculum outcomes must map cell IDs to search outcomes')
    for cell_id, observations in outcomes.items():
        _hash(cell_id, 'curriculum cell ID')
        if type(observations) is not dict or not set(observations) <= search_ids:
            raise ValueError('curriculum outcome IDs must belong to search_fitness')
        if any(value is not None and type(value) is not bool for value in observations.values()):
            raise ValueError('curriculum observations must be boolean or null')


def _cell_state(state, search_ids):
    required = {'done', 'tried_pairs', 'breed_round', 'mutate_round'}
    _fields(state, required, 'cell_state', {'evolution_seed', 'evolution_rng', 'curriculum'})
    _ids(state['done'], 'cell_state done')
    for name in ('breed_round', 'mutate_round'):
        _integer(state[name], 'cell_state ' + name)
    pairs = state['tried_pairs']
    if type(pairs) is not list:
        raise ValueError('tried_pairs must be a list of ordered SHA256 pairs')
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2:
            raise ValueError('each tried pair must contain two ordered SHA256 IDs')
        for cell_id in pair:
            _hash(cell_id, 'tried pair cell ID')
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError('tried_pairs contains a repeated ordered pair')
    if ('evolution_seed' in state) != ('evolution_rng' in state):
        raise ValueError('evolution_seed and evolution_rng must occur together')
    if 'evolution_seed' in state:
        _integer(state['evolution_seed'], 'evolution_seed')
        _rng(state['evolution_rng'])
    if 'curriculum' in state:
        _curriculum(state['curriculum'], search_ids)


def _advance_state(previous, current):
    for name in ('done', 'tried_pairs'):
        if not _same(previous[name], current[name][:len(previous[name])]):
            raise ValueError('cell_state ' + name + ' must preserve its previous ordered prefix')
    for name in ('breed_round', 'mutate_round'):
        if current[name] < previous[name]:
            raise ValueError('cell_state round counters cannot decrease')
    if 'evolution_seed' in previous and current.get('evolution_seed') != previous['evolution_seed']:
        raise ValueError('cell transitions must preserve the evolution RNG seed')
    if 'curriculum' not in previous:
        return
    if 'curriculum' not in current:
        raise ValueError('cell transitions cannot discard curriculum state')
    old, new = previous['curriculum'], current['curriculum']
    if new['wave'] < old['wave']:
        raise ValueError('curriculum wave cannot decrease')
    if any(task_id not in new['exposure'] or new['exposure'][task_id] < count
           for task_id, count in old['exposure'].items()):
        raise ValueError('curriculum exposure cannot be removed or decreased')
    if not set(old['recovery_anchors']) <= set(new['recovery_anchors']):
        raise ValueError('curriculum recovery anchors cannot be lost')
    if 'phase' in old and 'phase' not in new:
        raise ValueError('curriculum selection state cannot be discarded')


def validate_checkpoint(checkpoint):
    """Validate one envelope and return a fully detached copy.

    Retained experts must form an unchanged archive prefix. ``initial`` is
    legal only at cells/index zero without a predecessor; ``bank_gain``
    requires a router phase and exact archive/retained bank and outcome equality.
    Full predecessor continuity and gain ranking require make_checkpoint.
    """
    checkpoint = _json_copy(checkpoint)
    _fields(checkpoint, _CHECKPOINT_FIELDS, 'checkpoint')
    if checkpoint['schema'] != _SCHEMA:
        raise ValueError('unsupported phase-checkpoint schema')
    _phase(checkpoint['phase'])
    _check_hash(checkpoint, 'hash', 'checkpoint hash')
    for name in ('manifest_hash', 'journal_head_hash', 'scope_hash'):
        _hash(checkpoint[name], name)
    _integer(checkpoint['worker_calls_reserved'], 'worker_calls_reserved')
    initial = checkpoint['phase']['index'] == 0
    if initial:
        if checkpoint['predecessor_hash'] is not None or checkpoint['retention'] != 'initial':
            raise ValueError('index zero requires no predecessor and initial retention')
    else:
        _hash(checkpoint['predecessor_hash'], 'predecessor_hash')
        if checkpoint['retention'] not in ('carried', 'bank_gain'):
            raise ValueError('later checkpoints require carried or bank_gain retention')
    _archive(checkpoint['archive_bank'], checkpoint['archive_cells'])
    _bank(checkpoint['retained_bank'])
    _bank_prefix(checkpoint['retained_bank'], checkpoint['archive_bank'])
    _outcomes(checkpoint['archive_outcomes'])
    _outcomes(checkpoint['retained_outcomes'])
    _same_panels(checkpoint['archive_outcomes'], checkpoint['retained_outcomes'])
    _cell_state(checkpoint['cell_state'], set(checkpoint['archive_outcomes']['search_fitness']))
    _consistent_outcomes(checkpoint['archive_bank'], checkpoint['archive_outcomes'],
                         checkpoint['retained_bank'], checkpoint['retained_outcomes'])
    if checkpoint['retention'] == 'bank_gain':
        if (checkpoint['phase']['kind'] != 'router'
                or not _same(checkpoint['archive_bank'], checkpoint['retained_bank'])
                or not _same(checkpoint['archive_outcomes'], checkpoint['retained_outcomes'])):
            raise ValueError('bank_gain must retain the complete router-phase archive result')
    return checkpoint


def make_checkpoint(phase, predecessor, evidence):
    """Validate the immediate transition, select retention and return a copy.

    Evidence has exact _EVIDENCE_FIELDS, including the required cell_state.
    The incumbent is the predecessor's full retained bank/outcomes, or the
    initial caller-supplied bank with unchanged expert prefix in the archive.
    Cells only expand the archive and carry that incumbent. Router phases keep
    the preceding cell archive unchanged (including metadata), and promote its
    bank exactly when selection successes strictly increase without losing any
    incumbent true anchor on either panel. Otherwise the incumbent is carried.
    ``bank_gain`` describes the full bank; it does not claim router learning.

    cell_state carries done/tried_pairs (append-only), nondecreasing breeding
    and mutation rounds, optional paired evolution_seed/evolution_rng, and
    optional curriculum with nondecreasing wave/exposure and preserved anchors.
    Router phases preserve this entire state exactly. Metadata remains opaque
    finite JSON, and is preserved exactly on all inherited archive cells.
    """
    phase, evidence = _json_copy(phase), _json_copy(evidence)
    _phase(phase)
    _fields(evidence, _EVIDENCE_FIELDS, 'phase evidence')
    for name in ('manifest_hash', 'journal_head_hash', 'scope_hash'):
        _hash(evidence[name], name)
    _integer(evidence['worker_calls_reserved'], 'worker_calls_reserved')
    _archive(evidence['archive_bank'], evidence['archive_cells'])
    _bank(evidence['incumbent_bank'])
    _bank_prefix(evidence['incumbent_bank'], evidence['archive_bank'])
    _outcomes(evidence['archive_outcomes'])
    _outcomes(evidence['incumbent_outcomes'])
    _same_panels(evidence['archive_outcomes'], evidence['incumbent_outcomes'])
    _cell_state(evidence['cell_state'], set(evidence['archive_outcomes']['search_fitness']))
    _consistent_outcomes(evidence['archive_bank'], evidence['archive_outcomes'],
                         evidence['incumbent_bank'], evidence['incumbent_outcomes'])
    previous = None if predecessor is None else validate_checkpoint(predecessor)
    if previous is None:
        if phase['index'] != 0:
            raise ValueError('only the first cells phase may omit its predecessor')
    else:
        if (phase['plan_hash'] != previous['phase']['plan_hash']
                or phase['index'] != previous['phase']['index'] + 1):
            raise ValueError('phase must immediately follow its predecessor in the same plan')
        if evidence['scope_hash'] != previous['scope_hash']:
            raise ValueError('development scope must remain identical across phases')
        _same_panels(evidence['archive_outcomes'], previous['archive_outcomes'])
        if (not _same(evidence['incumbent_bank'], previous['retained_bank'])
                or not _same(evidence['incumbent_outcomes'], previous['retained_outcomes'])):
            raise ValueError('incumbent must equal the predecessor retained bank and outcomes')
        old_cells, new_cells = previous['archive_cells'], evidence['archive_cells']
        if not _same(old_cells, new_cells[:len(old_cells)]):
            raise ValueError('archive must preserve every predecessor cell and all metadata in prefix order')
        _bank_prefix(previous['archive_bank'], evidence['archive_bank'])
        _consistent_outcomes(previous['archive_bank'], previous['archive_outcomes'],
                             evidence['archive_bank'], evidence['archive_outcomes'])
        if phase['kind'] == 'router':
            if not _same(old_cells, new_cells):
                raise ValueError('router phase must preserve the preceding cells archive exactly')
            if not _same(evidence['cell_state'], previous['cell_state']):
                raise ValueError('router phase must carry cell_state unchanged')
        else:
            _advance_state(previous['cell_state'], evidence['cell_state'])
    retained_bank, retained_outcomes = evidence['incumbent_bank'], evidence['incumbent_outcomes']
    retention = 'initial' if previous is None else 'carried'
    if phase['kind'] == 'router':
        candidate = evidence['archive_outcomes']
        gain = sum(candidate['development_selection'].values()) > sum(
            retained_outcomes['development_selection'].values())
        anchors = all(not solved or candidate[panel][task_id]
                      for panel in _PANELS for task_id, solved in retained_outcomes[panel].items())
        if gain and anchors:
            retained_bank, retained_outcomes = evidence['archive_bank'], candidate
            retention = 'bank_gain'
    checkpoint = {
        'schema': _SCHEMA, 'phase': phase,
        'predecessor_hash': None if previous is None else previous['hash'],
        'manifest_hash': evidence['manifest_hash'],
        'journal_head_hash': evidence['journal_head_hash'], 'scope_hash': evidence['scope_hash'],
        'archive_cells': evidence['archive_cells'], 'archive_bank': evidence['archive_bank'],
        'archive_outcomes': evidence['archive_outcomes'], 'retained_bank': retained_bank,
        'retained_outcomes': retained_outcomes, 'retention': retention,
        'worker_calls_reserved': evidence['worker_calls_reserved'], 'cell_state': evidence['cell_state'],
    }
    checkpoint['hash'] = contracts.digest(checkpoint)
    return validate_checkpoint(checkpoint)
