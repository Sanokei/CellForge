"""Development-only parent selection and curriculum; never a runtime router.

Natural-niche weighting follows the method in SakanaAI/natural_niches
(Apache-2.0), natural_niches_fn.py::sample_parents. This is an independent
stdlib implementation with explicit unknown outcomes and zero-score fallbacks.
No tensors, evaluators, acceptance thresholds or hidden partitions are edited.
"""
import copy
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


DEFAULT = {
    'schema': 'cellforge-evolution-policy/1',
    'parent_selection': 'legacy',
    'alpha': 1.0,
    'typed_chains': False,
    'niches': [],
    'router': {'enabled': False, 'task_limit': 100, 'max_experts': 64},
    'curriculum': {
        'enabled': False,
        'task_limit': 100,
        'phases': [
            {'name': 'exploration', 'rounds': 1, 'alpha': 0.0, 'pairs': 300, 'mutations': 6},
            {'name': 'challenge', 'rounds': 1, 'alpha': 1.0, 'pairs': 300, 'mutations': 6},
            {'name': 'recovery', 'rounds': 1, 'alpha': 1.0, 'pairs': 100, 'mutations': 6},
        ],
    },
    'workers': {
        'enabled': False, 'model': 'gpt-6-luna', 'reasoning': 'low',
        'parallelism': 2, 'proposals_per_prompt': 2, 'max_calls': 0,
        'timeout_seconds': 300,
    },
}


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('%s must be an integer in [%s, %s]' % (label, low, high))


def _alpha(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10:
        raise ValueError('alpha must be finite in [0, 10]')


def configuration(value=None):
    """Normalize explicitly provided options; reject typos instead of ignoring them."""
    out = copy.deepcopy(DEFAULT)
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(out):
        raise ValueError('unknown evolution policy fields')
    for key, item in value.items():
        if key in ('curriculum', 'workers', 'router'):
            if not isinstance(item, dict) or set(item) - set(out[key]):
                raise ValueError('unknown %s fields' % key)
            out[key].update(copy.deepcopy(item))
        else:
            out[key] = copy.deepcopy(item)
    if out['schema'] != DEFAULT['schema'] or out['parent_selection'] not in ('legacy', 'complementary_success'):
        raise ValueError('unsupported evolution policy version or selector')
    _alpha(out['alpha'])
    if type(out['typed_chains']) is not bool:
        raise ValueError('typed_chains must be boolean')
    if (not isinstance(out['niches'], list) or len(out['niches']) > 100
            or any(not isinstance(name, str) or not name for name in out['niches'])
            or len(set(out['niches'])) != len(out['niches'])):
        raise ValueError('niches must be an explicit bounded list of distinct names')
    curriculum = out['curriculum']
    if type(curriculum['enabled']) is not bool:
        raise ValueError('curriculum enabled must be boolean')
    _integer(curriculum['task_limit'], 1, 1000, 'curriculum task_limit')
    if not isinstance(curriculum['phases'], list) or not curriculum['phases']:
        raise ValueError('at least one curriculum phase is required')
    for phase in curriculum['phases']:
        if not isinstance(phase, dict) or set(phase) != {'name', 'rounds', 'alpha', 'pairs', 'mutations'}:
            raise ValueError('exact phase fields required')
        if phase['name'] not in ('exploration', 'challenge', 'recovery'):
            raise ValueError('unknown curriculum phase')
        _integer(phase['rounds'], 1, 100, 'phase rounds')
        _integer(phase['pairs'], 0, 300, 'phase pairs')
        _integer(phase['mutations'], 0, 6, 'phase mutations')
        _alpha(phase['alpha'])
    worker = out['workers']
    if type(worker['enabled']) is not bool:
        raise ValueError('workers enabled must be boolean')
    if worker['model'] != 'gpt-6-luna' or worker['reasoning'] not in ('low', 'medium', 'high'):
        raise ValueError('weight workers require the configured gpt-6-luna route')
    _integer(worker['parallelism'], 1, 3, 'worker parallelism')
    _integer(worker['proposals_per_prompt'], 1, 8, 'proposals_per_prompt')
    _integer(worker['max_calls'], 1 if worker['enabled'] else 0, 1000, 'max_calls')
    _integer(worker['timeout_seconds'], 1, 900, 'worker timeout_seconds')
    router = out['router']
    if type(router['enabled']) is not bool:
        raise ValueError('router enabled must be boolean')
    _integer(router['task_limit'], 1, 100, 'router task_limit')
    _integer(router['max_experts'], 1, 64, 'router max_experts')
    if router['enabled'] and (out['curriculum']['enabled'] or out['parent_selection'] != 'legacy'):
        raise ValueError('fixed-roster router waves cannot also enable cell selection or curriculum')
    return out


def validate_evidence(evidence, parent_ids):
    if evidence.get('schema') != 'cellforge-parent-success/1' or evidence.get('partition') != 'search_fitness':
        raise ValueError('parent selection requires search-fitness evidence')
    if not evidence.get('scope_hash') or not evidence.get('evaluator_hash'):
        raise ValueError('frozen task and evaluator identities required')
    tasks = evidence['task_ids']
    if not tasks or len(set(tasks)) != len(tasks) or any(not isinstance(t, str) or not t for t in tasks):
        raise ValueError('distinct nonempty task IDs required')
    records = evidence['outcomes']
    if not isinstance(records, dict) or set(records) != set(parent_ids):
        raise ValueError('evidence must cover exactly the eligible parent identities')
    for record in records.values():
        if not isinstance(record, dict) or set(record) - set(tasks):
            raise ValueError('outcome IDs exceed the frozen task scope')
        if any(value is not None and type(value) is not bool for value in record.values()):
            raise ValueError('outcomes must be true, false, or unknown (null)')
    return tasks, records


def _choose(ids, weights, rng):
    total = math.fsum(weights)
    if total <= 0:
        return rng.choice(ids), 'uniform'
    return rng.choices(ids, weights=weights, k=1)[0], 'weighted'


def complementary_pairs(parent_ids, eligible_pairs, evidence, rng, limit, alpha=1.0):
    """Sample untried compatible ordered pairs, without replacement.

    Unknown observations are excluded from the shared comparison panel; they
    never become failures. Zero complementarity falls back to the eligible
    mate's fitness, then seeded uniform choice. No pairs means mutation-only.
    """
    _alpha(alpha)
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError('distinct content-addressed parents required')
    tasks, records = validate_evidence(evidence, parent_ids)
    shared = [t for t in tasks if all(records[c].get(t) is not None for c in parent_ids)]
    support = {t: sum(records[c][t] for c in parent_ids) for t in shared}
    weighted = {c: {t: int(records[c][t]) / max(support[t], 1) ** alpha for t in shared}
                for c in parent_ids}
    fitness = {c: math.fsum(weighted[c].values()) for c in parent_ids}
    pending = set(tuple(pair) for pair in eligible_pairs)
    if any(a == b or a not in fitness or b not in fitness for a, b in pending):
        raise ValueError('eligible pairs must refer to distinct declared parents')
    selected, receipts = [], []
    eligible_count = len(pending)
    for _ in range(min(limit, eligible_count)):
        firsts = sorted({a for a, _ in pending})
        first, first_mode = _choose(firsts, [fitness[c] for c in firsts], rng)
        mates = sorted(b for a, b in pending if a == first)
        complements = [math.fsum(max(0.0, weighted[c][t] - weighted[first][t]) for t in shared) for c in mates]
        if math.fsum(complements) > 0:
            second, _ = _choose(mates, complements, rng)
            mate_mode = 'complementary_success'
        else:
            second, fallback = _choose(mates, [fitness[c] for c in mates], rng)
            mate_mode = 'fitness_fallback' if fallback == 'weighted' else 'uniform_fallback'
        pending.remove((first, second))
        selected.append((first, second))
        receipts.append({'parents': [first, second], 'first_mode': first_mode, 'mate_mode': mate_mode})
    return selected, {
        'schema': 'cellforge-parent-selection/1', 'operator': 'chain' if selected else 'mutation_only',
        'reason': 'no_eligible_pair' if not eligible_count else 'selected' if selected else 'zero_pair_limit',
        'scope_hash': evidence['scope_hash'], 'evaluator_hash': evidence['evaluator_hash'],
        'alpha': alpha, 'shared_task_ids': shared, 'eligible_pairs': eligible_count,
        'fitness': fitness, 'choices': receipts,
    }


def phase_at(config, wave):
    phases = config['phases']
    offset = wave % sum(phase['rounds'] for phase in phases)
    for phase in phases:
        if offset < phase['rounds']:
            return copy.deepcopy(phase)
        offset -= phase['rounds']
    raise AssertionError('unreachable curriculum phase')


def curriculum_tasks(config, state, evidence):
    """Select development exposure only; historical anchors/archive survive recovery."""
    tasks, records = validate_evidence(evidence, list(evidence['outcomes']))
    phase = phase_at(config, state.get('wave', 0))
    exposure = state.setdefault('exposure', {})
    anchors = set(state.get('recovery_anchors', []))
    rates = {}
    for task in tasks:
        known = [record[task] for record in records.values() if record.get(task) is not None]
        rates[task] = sum(known) / len(known) if known else None
        if any(known):
            anchors.add(task)
    def rank(task):
        rate = rates[task]
        seen = exposure.get(task, 0)
        if phase['name'] == 'challenge':
            return (rate is None, abs(rate - 0.5) if rate is not None else 1.0, seen, task)
        if phase['name'] == 'recovery':
            return (task not in anchors, rate is None, rate if rate is not None else 1.0, seen, task)
        return (seen, rate is not None, rate if rate is not None else 0.0, task)
    selected = sorted(tasks, key=rank)[:config['task_limit']]
    state['phase'] = phase['name']
    state['selected_task_ids'] = list(selected)
    state['recovery_anchors'] = sorted(anchors)
    state['scope_hash'] = evidence['scope_hash']
    state['outcomes'] = copy.deepcopy(records)
    # The driver conservatively records possible exposure before submitting a
    # prompt, so an interrupted call cannot silently erase disclosure history.
    return selected, phase


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, list) else value
