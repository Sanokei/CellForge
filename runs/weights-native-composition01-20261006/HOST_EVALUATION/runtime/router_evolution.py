"""Pure helpers for literal router-weight evolution on a frozen expert roster.

The router is ``route.W @ features + route.b``; features are the negative
leave-one-out demonstration errors in the recorded expert order. Query labels
are never inference features. The host computes those features and owns the
search/selection/report split, worker receipts, charging and evaluation.

Each proposal is a complete replacement (including an incumbent's mutation)
of N*N+N literal coefficients. Its random control replaces the same positions;
unchanged coefficients and invalid/omitted reserved proposals are not free.
The host must create and charge controls for every reservation, independently
of whether a valid proposal is returned. Record the seed and worker provenance
in the host journal: this module does not manufacture evidence of authorship.

This is evolutionary selection, not PPO/GRPO, gradient fitting or a runtime
selection policy. There is no I/O, evaluator dependency or global RNG mutation.
"""
import copy
import hashlib
import json
import math
import random


_SCHEMA = 'cellforge-router-genome/1'
_PARTITIONS = ('search_fitness', 'development_selection')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _expert_ids(expert_ids):
    # These are host-owned immutable identities, not names inferred from shape.
    # Their correspondence to actual cells is verified by the bank contract.
    if (type(expert_ids) not in (list, tuple) or not expert_ids
            or any(type(cid) is not str or not cid for cid in expert_ids)
            or len(set(expert_ids)) != len(expert_ids)):
        raise ValueError('an ordered nonempty roster of distinct expert IDs is required')
    return list(expert_ids)


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _bound(max_abs):
    if not _finite(max_abs) or max_abs < 0:
        raise ValueError('max_abs must be a finite nonnegative number')
    return float(max_abs)


def positions(expert_ids):
    """Return the full replacement charge, N*N+N, for this ordered roster."""
    n = len(_expert_ids(expert_ids))
    return n * n + n


def validate_params(params, expert_ids, max_abs=9.0):
    """Return a Python bool for exact, finite, bounded literal W[N,N]/b[N].

    Booleans, strings, array objects, ragged lists and unexpected fields are
    invalid. Invalid rosters or bounds also return False for worker callbacks.
    No values are coerced or clipped, so saved literals retain their identity.
    """
    try:
        n = len(_expert_ids(expert_ids))
        bound = _bound(max_abs)
    except ValueError:
        return False
    if type(params) is not dict or set(params) != {'route.W', 'route.b'}:
        return False
    w, b = params['route.W'], params['route.b']
    if type(w) is not list or len(w) != n or type(b) is not list or len(b) != n:
        return False
    if any(type(row) is not list or len(row) != n for row in w):
        return False
    return all(_finite(value) and -bound <= value <= bound
               for row in w + [b] for value in row)


def _parents(parents):
    if (type(parents) not in (list, tuple)
            or any(type(pid) is not str or len(pid) != 64
                   or any(char not in '0123456789abcdef' for char in pid)
                   for pid in parents)
            or len(set(parents)) != len(parents)):
        raise ValueError('parents must be distinct lowercase SHA256 router hashes')
    return list(parents)


def make_genome(expert_ids, params, parents=()):
    """Copy and bind router literals to immutable ordered expert identities.

    ``hash`` is SHA256 of exactly {schema, expert_ids, params}, using sorted
    compact JSON with nonfinite values forbidden. Parents are recorded but
    excluded from numerical identity; identical weights/order remain neutral
    across different authoring histories. A reorder requires a new genome.
    """
    ids = _expert_ids(expert_ids)
    if not validate_params(params, ids):
        raise ValueError('invalid router literals for the declared expert roster')
    body = {'schema': _SCHEMA, 'expert_ids': ids, 'params': copy.deepcopy(params)}
    return dict(body, parents=_parents(parents), hash=_digest(body))


def _genome(genome):
    if (type(genome) is not dict
            or set(genome) != {'schema', 'expert_ids', 'params', 'parents', 'hash'}
            or genome['schema'] != _SCHEMA):
        raise ValueError('exact versioned router genome fields are required')
    checked = make_genome(genome['expert_ids'], genome['params'], genome['parents'])
    if type(genome['hash']) is not str or checked['hash'] != genome['hash']:
        raise ValueError('router genome hash differs from its ordered roster or weights')
    return checked


def random_params(expert_ids, seed, max_abs=9.0):
    """Return a matched full-position control from a local seeded RNG.

    Draw order is W row-major, then b, with coefficients uniform in the given
    symmetric bounds. Persist the exact seed and returned literals in the host
    receipt; no shared RNG, completion order or candidate success is consulted.
    The normal experiment bound is 9.0, also required by make_genome.
    """
    n = len(_expert_ids(expert_ids))
    bound = _bound(max_abs)
    if type(seed) is not int:
        raise ValueError('an explicit integer random-control seed is required')
    rng = random.Random(seed)

    def draw():
        # This order avoids overflow from subtracting extreme finite bounds.
        return (2.0 * rng.random() - 1.0) * bound

    params = {'route.W': [[draw() for _ in range(n)] for _ in range(n)],
              'route.b': [draw() for _ in range(n)]}
    if not validate_params(params, expert_ids, max_abs):
        raise ValueError('random control did not produce finite bounded literals')
    return params


def _examples(examples, expert_ids):
    if type(examples) is not list or not examples:
        raise ValueError('a nonempty list of host-supplied search-fitness examples is required')
    seen, allowed = set(), set(expert_ids)
    for example in examples:
        if (type(example) is not dict
                or set(example) != {'task_id', 'features', 'successful_expert_ids'}):
            raise ValueError('examples require exact task_id/features/successful_expert_ids fields')
        task_id = example['task_id']
        if type(task_id) is not str or not task_id or task_id in seen:
            raise ValueError('search example task IDs must be nonempty and distinct')
        seen.add(task_id)
        features = example['features']
        if (type(features) is not list or len(features) != len(expert_ids)
                or any(not _finite(value) or not -1.0 <= value <= 0.0 for value in features)):
            raise ValueError('features must be aligned negative demonstration errors in [-1, 0]')
        successes = example['successful_expert_ids']
        if (type(successes) is not list
                or any(type(cid) is not str or cid not in allowed for cid in successes)
                or len(set(successes)) != len(successes)):
            raise ValueError('successful experts must be a distinct subset of the frozen roster')
    return copy.deepcopy(examples)


def proposal_prompt(expert_ids, incumbent, examples):
    """Describe a full literal replacement in the existing worker JSON schema.

    ``incumbent`` is a make_genome result on this exact roster. The host must
    supply search-fitness examples only; this function cannot establish split
    provenance from task IDs alone. Success IDs are development supervision,
    not inputs to runtime routing. The schema rejects extra fields, including
    raw query labels, and permits no expert roster or evaluator modifications.
    """
    ids = _expert_ids(expert_ids)
    parent = _genome(incumbent)
    if parent['expert_ids'] != ids:
        raise ValueError('incumbent expert order differs from the frozen router roster')
    examples = _examples(examples, ids)
    n = len(ids)
    body = {'expert_ids': ids, 'incumbent': parent,
            'shapes': {'route.W': [n, n], 'route.b': [n]},
            'search_fitness_examples': examples}
    return (
        'Write one complete replacement of the literal router tensors; no code, tools, files, '
        'expert edits, roster edits, evaluator edits or new routing features. '
        'At inference the host computes features[j] as the negative mean leave-one-out '
        'demonstration error of expert_ids[j], using visible demonstrations only. '
        'Scores[i] = sum(route.W[i][j] * features[j] for j) + route.b[i]; '
        'choose the first maximum in the immutable expert order. Query outputs and '
        'successful_expert_ids are never inference features. '
        'Each search-fitness example lists experts that succeeded on its development query; '
        'these IDs are authoring feedback only, and an empty list means no expert succeeded. '
        'Use that feedback to revise the incumbent into one shared router that selects a '
        'successful expert where possible. Return all N*N+N coefficients, even unchanged '
        'ones, as finite literal numbers in [-9, 9]. Do not return a delta or sparse edits. '
        'Return ONLY a JSON array with exactly one object in this format: '
        '[{"task_id":"router","params":{"route.W":<N by N nested list>, '
        '"route.b":<N-entry list>},"rationale":"brief explanation from the supplied examples"}]. '
        'This is literal weight evolution, not PPO/GRPO training.\n'
        + json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)
    )


def _outcomes(outcomes):
    if type(outcomes) is not dict or set(outcomes) != set(_PARTITIONS):
        raise ValueError('exact search_fitness/development_selection outcome panels are required')
    for partition in _PARTITIONS:
        panel = outcomes[partition]
        if (type(panel) is not dict or not panel
                or any(type(tid) is not str or not tid for tid in panel)
                or any(type(value) is not bool for value in panel.values())):
            raise ValueError('each outcome panel requires task IDs and complete Python booleans')
    if set(outcomes['search_fitness']) & set(outcomes['development_selection']):
        raise ValueError('search-fitness and development-selection task IDs must be disjoint')
    return copy.deepcopy(outcomes)


def select_champion(incumbent_outcomes, candidates):
    """Return a copied genome with strict selection gain and no lost anchors.

    Candidates are {genome, outcomes} records. Every panel must exactly match
    the incumbent's task IDs, and every genome must use the same ordered roster.
    The caller also checks that roster against the incumbent bank (its genome
    is deliberately absent from this outcome-only argument). Malformed records
    fail closed, even if they would not win. Missing evidence is never a failure
    outcome or a free attempt: the host accounts for invalid reservations.

    Rank eligible candidates by development-selection successes, descending,
    then content hash, ascending. Keep all incumbent True anchors across both
    panels. Selection outcomes never enter the proposal prompt or inference.
    """
    baseline = _outcomes(incumbent_outcomes)
    if type(candidates) not in (list, tuple):
        raise ValueError('candidate records must be an explicit list or tuple')
    roster, observed, eligible = None, {}, []
    incumbent_score = sum(baseline['development_selection'].values())
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != {'genome', 'outcomes'}:
            raise ValueError('candidate records require exact genome/outcomes fields')
        genome = _genome(candidate['genome'])
        outcomes = _outcomes(candidate['outcomes'])
        if roster is None:
            roster = genome['expert_ids']
        elif genome['expert_ids'] != roster:
            raise ValueError('candidate expert rosters or order differ')
        if any(set(outcomes[part]) != set(baseline[part]) for part in _PARTITIONS):
            raise ValueError('candidate outcome task IDs differ from the incumbent scope')
        if genome['hash'] in observed and observed[genome['hash']] != outcomes:
            raise ValueError('identical router weights have inconsistent recorded outcomes')
        observed[genome['hash']] = outcomes
        score = sum(outcomes['development_selection'].values())
        if score <= incumbent_score:
            continue
        if any(value and not outcomes[part][tid]
               for part in _PARTITIONS for tid, value in baseline[part].items()):
            continue
        eligible.append((score, genome))
    if not eligible:
        return None
    return copy.deepcopy(min(eligible, key=lambda item: (-item[0], item[1]['hash']))[1])
