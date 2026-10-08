"""Bounded, search-only proposal memory; no evaluator, maker, or run imports.

Integration API (all functions except FailureJournal are storage-free):

    scope = make_scope(manifest, search_rows, template_hashes)
    attempt = make_attempt(scope, task_id, template_hash=..., candidate_hash=...,
                           parent_hashes=[...], proposal_hash=..., attempt_hash=...,
                           operator='complete_tensor_replacement', hypotheses=[])
    record = make_record(scope, attempt, verify=host_verifier)
    with FailureJournal(new_path, scope, create=True) as journal:
        journal.append(record, verify=host_verifier)
        memory = retrieve(scope, journal.records(), task_id=..., template_hash=...,
                          parent_hashes=[...], verify=host_verifier)
    # Append memory['prompt'] to a later tensor proposal prompt, never the evaluator.

``manifest`` is the already verified evolution_manifest snapshot. ``search_rows``
must be the COMPLETE frozen search_fitness rows, in their original content-hash
order. Development/report rows are rejected before content hashing. Template
hashes must bind implementation, interface and layout, not weights or names alone.
The scope stores frozen task IDs and content hashes. One bounded demonstration
may be retained, only after its input/expected hashes match the search scope.
No selection/report answers, raw responses, rationales, exception messages,
paths, source code or secrets are accepted as memory fields.

``verify(request)`` is a TRUSTED HOST callback, never a model result or bool flag.
Its request is a detached JSON dict with exactly scope and attempt. The callback
must check current evaluator/task/template identities, candidate/ordered parents,
and the charged proposal/attempt hashes against immutable host artifacts. It must
recompute the search outcome or validate an existing trusted search-only receipt.
It returns exactly: binding_hash=digest(request), evidence_hash (the immutable
host search receipt digest), partition='search_fitness', category, outcome,
parent_outcome, tensor_change, demonstration, runtime_error_type. Outcomes are
bool or None; parent_outcome refers
only to the FIRST parent on the SAME task. tensor_change is from the helper below
on that parent's and candidate's literal params, or None if unavailable. Never
classify a swallowed evaluator exception as a task mistake. Raise on unavailable
or incompatible evidence; retrieval quarantines it. No callback may inspect
selection/report answers, call a model, or change the evaluator.

demonstration is None or {index, input, expected, actual, actual_status,
actual_hash}; use demonstration() below to build it. It describes ONE training
example, not the full-task outcome: leave-one-out demo failure need not imply
failure on a test input using all demonstrations. runtime_error_type is an allowlisted class
name (or 'other') for runtime_error; otherwise None. Do not store error messages.
Categories are exact observed categories, not diagnoses. Hypotheses are optional
allowlisted, explicitly unverified model labels, never scores or instructions.
The journal cannot establish host authenticity: a dishonest callback can lie.
Hashes detect inconsistency, not a complete rewrite/rollback of a local store.
Retain head_hash externally if rollback detection is required. Read snapshots
are reverified by retrieve BEFORE becoming prompt memory; records() alone is not
a verified memory API. Reverification must be deterministic and side-effect-free.

Storage is a single-owner, OS-locked append-only JSONL file plus a persistent
.lock file. Header freezes scope and caps. It NEVER evicts, rewrites, migrates or
repairs evidence; capacity raises JournalFull BEFORE writing. Failed proposals
remain as receipts/hashes and bounded deltas, not raw proposal archives. Keep the
original immutable artifacts in the host's existing provenance store. A scope
mismatch opens read-only for quarantine; append refuses it. A new authorized
scope needs a new path. Incomplete writes fail closed; a failed write can have
committed, so inspect rather than blindly retry. No folders are created. Storage
requires a trusted local parent directory/filesystem; Windows lacks portable
directory fsync. Importing this module does no I/O and starts no work.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat


SCHEMA = 'cellforge-failure-journal/1'
MAX_RECORDS = 2048
MAX_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 16384
MAX_SCOPE_BYTES = 256 * 1024
MAX_TASKS = 1000
MAX_POSITIONS = 100000
ZERO_HASH = '0' * 64
HYPOTHESES = frozenset({
    'unknown', 'representation_limit', 'insufficient_context', 'wrong_mapping',
    'wrong_geometry', 'wrong_composition', 'wrong_threshold', 'routing_error',
    'overfit_search_examples', 'no_effect', 'shape_assumption', 'runtime_assumption',
})
OPERATORS = frozenset({'complete_tensor_replacement', 'tensor_crossover',
                       'router_replacement', 'founder'})
MECHANICAL = frozenset({'invalid_tensor_shape', 'invalid_tensor_values',
                        'invalid_proposal_format', 'missing_proposal', 'refused',
                        'timeout', 'runtime_error'})
TASK_MISTAKES = frozenset({'mismatch_output_shape', 'mismatch_output_values'})
CATEGORIES = MECHANICAL | TASK_MISTAKES | {'correct', 'unobserved'}
RUNTIME_ERRORS = frozenset({'ArithmeticError', 'AssertionError', 'AttributeError',
                          'IndexError', 'KeyError', 'MemoryError', 'NameError',
                          'OverflowError', 'RecursionError', 'RuntimeError',
                          'TypeError', 'ValueError', 'ZeroDivisionError', 'other'})


class JournalError(ValueError):
    """Invalid input, incompatible append, or corrupt immutable evidence."""


class JournalFull(JournalError):
    """Frozen capacity reached; no evidence was discarded or appended."""


class JournalBusy(RuntimeError):
    """Another owner holds the persistent store lease."""


def _canonical(value, limit=MAX_BYTES):
    chunks, size = [], 0
    try:
        for part in json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).iterencode(value):
            chunk = part.encode('ascii')
            size += len(chunk)
            if size > limit:
                raise JournalError('bounded JSON size exceeded')
            chunks.append(chunk)
    except (TypeError, ValueError, RecursionError) as error:
        raise JournalError('bounded finite JSON required') from error
    return b''.join(chunks)


def digest(value):
    """SHA256 of canonical ASCII JSON, matching evolution_manifest's encoding."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fields(value, keys):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise JournalError('exact fields required: ' + keys)


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise JournalError('lowercase SHA256 required')
    return value


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise JournalError('bounded integer required')


def _hashes(value, limit, nonempty=False):
    if type(value) not in (list, tuple) or not (int(nonempty) <= len(value) <= limit):
        raise JournalError('bounded hash list required')
    result = [_hash(item) for item in value]
    if len(set(result)) != len(result):
        raise JournalError('duplicate hashes')
    return result


def _sealed(body, field):
    return dict(body, **{field: digest(body)})


def _check_hash(value, field):
    _hash(value[field])
    if digest({k: v for k, v in value.items() if k != field}) != value[field]:
        raise JournalError('immutable content hash mismatch')


def task_key(task_id):
    """Stable key for a frozen dataset-qualified task ID (never free-form text)."""
    allowed = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:/'
    if (type(task_id) is not str or not 1 <= len(task_id) <= 128
            or any(c not in allowed for c in task_id)):
        raise JournalError('bounded dataset-qualified task ID required')
    return digest({'task_id': task_id})


def _grid(value):
    if type(value) is not list or not 1 <= len(value) <= 30:
        raise JournalError('bounded ARC grid required')
    if type(value[0]) is not list or not 1 <= len(value[0]) <= 30:
        raise JournalError('bounded ARC grid row required')
    width = len(value[0])
    for row in value:
        if type(row) is not list or len(row) != width:
            raise JournalError('rectangular ARC grid required')
        if any(type(v) is not int or not 0 <= v <= 9 for v in row):
            raise JournalError('ARC grids contain only integer colors 0..9')


def demonstration(index, input_grid, expected_grid, actual_grid=None, *, actual_status='available'):
    """Build one bounded search-demo observation; runtime failures have no output.

    actual_status is available, runtime_error or unavailable. For larger/non-grid
    predictions use unavailable and None, retaining the exact category and the
    original host receipt; do not coerce or truncate an invalid output into a grid.
    """
    value = {'index': index, 'input': copy.deepcopy(input_grid),
             'expected': copy.deepcopy(expected_grid), 'actual': copy.deepcopy(actual_grid),
             'actual_status': actual_status,
             'actual_hash': digest(actual_grid) if actual_status == 'available' else None}
    _demonstration(value)
    return value


def _demonstration(value):
    _fields(value, 'index input expected actual actual_status actual_hash')
    _integer(value['index'], 0, 63)
    _grid(value['input'])
    _grid(value['expected'])
    if value['actual_status'] == 'available':
        _grid(value['actual'])
        if value['actual_hash'] != digest(value['actual']):
            raise JournalError('actual output hash mismatch')
    elif value['actual_status'] in ('runtime_error', 'unavailable'):
        if value['actual'] is not None or value['actual_hash'] is not None:
            raise JournalError('unknown/crashed prediction cannot claim an actual grid')
    else:
        raise JournalError('exact actual-output status required')


def make_scope(manifest, search_rows, template_hashes):
    """Bind the existing manifest plus exact search content and template contracts.

    This validates snapshot consistency, not its external authenticity or live
    sealed/open status. The integrator must read_manifest_status and enforce its
    run ownership/report seal BEFORE calling this function or doing research.
    """
    _fields(manifest, 'schema config config_hash source_hashes source_hashes_hash '
            'partition_ids partition_hashes partition_ids_hash manifest_hash')
    if manifest['schema'] != 'cellforge-evolution-manifest/1':
        raise JournalError('verified evolution manifest required')
    _check_hash(manifest, 'manifest_hash')
    for source, hashed in (('config', 'config_hash'), ('source_hashes', 'source_hashes_hash'),
                           ('partition_ids', 'partition_ids_hash')):
        if digest(manifest[source]) != _hash(manifest[hashed]):
            raise JournalError('manifest component hash mismatch')
    partitions = manifest['partition_ids']
    _fields(partitions, 'search_fitness development_selection report')
    _fields(manifest['partition_hashes'], 'search_fitness development_selection report')
    seen = set()
    for name, ids in partitions.items():
        if type(ids) is not list or len(ids) > 100000:
            raise JournalError('bounded manifest ID list required')
        keys = [task_key(tid) for tid in ids]
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise JournalError('overlapping or duplicate task partitions')
        seen.update(keys)
        if digest(ids) != manifest['partition_hashes'][name]:
            raise JournalError('partition hash mismatch')
    if type(search_rows) is not list or not 1 <= len(search_rows) <= MAX_TASKS:
        raise JournalError('bounded complete search rows required')
    ids = []
    for row in search_rows:
        if type(row) is not dict or 'id' not in row:
            raise JournalError('search rows require task IDs')
        task_key(row['id'])
        ids.append(row['id'])
    if len(set(ids)) != len(ids) or set(ids) != set(partitions['search_fitness']):
        raise JournalError('only the complete frozen search_fitness population is allowed')
    content_hash = digest(search_rows)
    if manifest['source_hashes'].get('search-fitness-task-content') != content_hash:
        raise JournalError('search content differs from the frozen evaluator manifest')
    tasks = {}
    for row in search_rows:
        task = row.get('task')
        if type(task) is not dict or type(task.get('train')) is not list or not 1 <= len(task['train']) <= 64:
            raise JournalError('bounded search training demonstrations required')
        demo_hashes = []
        for example in task['train']:
            if type(example) is not dict or not {'input', 'output'} <= set(example):
                raise JournalError('search training input/output pair required')
            _grid(example['input'])
            _grid(example['output'])
            demo_hashes.append(digest({'input': example['input'], 'expected': example['output']}))
        if type(task.get('test', [])) is not list or len(task.get('test', [])) > 64:
            raise JournalError('bounded task input scope required')
        test_inputs = []
        for example in task.get('test', []):
            if type(example) is not dict or 'input' not in example:
                raise JournalError('test input required')
            _grid(example['input'])
            test_inputs.append(example['input'])
        tasks[task_key(row['id'])] = {
            'task_id': row['id'], 'task_hash': digest(row),
            'input_scope_hash': digest({'train_inputs': [d['input'] for d in task['train']],
                                        'test_inputs': test_inputs}),
            'demonstration_hashes': demo_hashes}
    scope = _sealed({'schema': SCHEMA + '/scope', 'partition': 'search_fitness',
                     'manifest_hash': manifest['manifest_hash'],
                     'evaluator_hash': manifest['source_hashes_hash'],
                     'search_content_hash': content_hash,
                     'tasks': tasks,
                     'template_hashes': sorted(_hashes(template_hashes, 256, True))}, 'scope_hash')
    return _scope(scope)


def _scope(scope):
    _fields(scope, 'schema partition manifest_hash evaluator_hash search_content_hash '
            'tasks template_hashes scope_hash')
    if scope['schema'] != SCHEMA + '/scope' or scope['partition'] != 'search_fitness':
        raise JournalError('search_fitness scope required')
    for name in ('manifest_hash', 'evaluator_hash', 'search_content_hash'):
        _hash(scope[name])
    if type(scope['tasks']) is not dict or not 1 <= len(scope['tasks']) <= MAX_TASKS:
        raise JournalError('bounded search task hashes required')
    for key, value in scope['tasks'].items():
        _fields(value, 'task_id task_hash input_scope_hash demonstration_hashes')
        if _hash(key) != task_key(value['task_id']):
            raise JournalError('task ID binding mismatch')
        _hash(value['task_hash'])
        _hash(value['input_scope_hash'])
        if type(value['demonstration_hashes']) is not list or not 1 <= len(value['demonstration_hashes']) <= 64:
            raise JournalError('bounded demonstration bindings required')
        for hashed in value['demonstration_hashes']:
            _hash(hashed)
    if sorted(_hashes(scope['template_hashes'], 256, True)) != scope['template_hashes']:
        raise JournalError('canonical template hashes required')
    _canonical(scope, MAX_SCOPE_BYTES)
    _check_hash(scope, 'scope_hash')
    return copy.deepcopy(scope)


def make_attempt(scope, task_id, *, template_hash, candidate_hash, parent_hashes,
                 proposal_hash, attempt_hash, operator='complete_tensor_replacement', hypotheses=()):
    """Build a data-only attempt; proposal_hash may name an invalid raw response.

    candidate_hash is None when no valid candidate exists. attempt_hash binds the
    charged reservation/attempt, so identical tensors from different attempts
    are preserved. parent_hashes are ORDERED. No raw model rationale is accepted.
    """
    scope = _scope(scope)
    key = task_key(task_id)
    if key not in scope['tasks']:
        raise JournalError('task is outside frozen search_fitness')
    if type(hypotheses) not in (list, tuple) or len(hypotheses) > 4:
        raise JournalError('at most four hypothesis labels allowed')
    if any(type(item) is not str or item not in HYPOTHESES for item in hypotheses):
        raise JournalError('unverified hypotheses must be allowlisted labels')
    attempt = {'scope_hash': scope['scope_hash'], 'task_key': key, 'task_id': task_id,
               'task_hash': scope['tasks'][key]['task_hash'],
               'input_scope_hash': scope['tasks'][key]['input_scope_hash'],
               'evaluator_hash': scope['evaluator_hash'],
               'template_hash': template_hash, 'candidate_hash': candidate_hash,
               'parent_hashes': list(parent_hashes), 'proposal_hash': proposal_hash,
               'attempt_hash': attempt_hash, 'operator': operator,
               'hypotheses': sorted(set(hypotheses))}
    _attempt(attempt, scope)
    return attempt


def _attempt(attempt, scope=None):
    _fields(attempt, 'scope_hash task_key task_id task_hash input_scope_hash evaluator_hash template_hash '
            'candidate_hash parent_hashes proposal_hash attempt_hash operator hypotheses')
    for key in ('scope_hash', 'task_key', 'task_hash', 'input_scope_hash', 'evaluator_hash', 'template_hash',
                'proposal_hash', 'attempt_hash'):
        _hash(attempt[key])
    if task_key(attempt['task_id']) != attempt['task_key']:
        raise JournalError('attempt task ID binding mismatch')
    if attempt['candidate_hash'] is not None:
        _hash(attempt['candidate_hash'])
    _hashes(attempt['parent_hashes'], 8)
    if type(attempt['operator']) is not str or attempt['operator'] not in OPERATORS:
        raise JournalError('unsupported tensor operation')
    labels = attempt['hypotheses']
    if (type(labels) is not list or len(labels) > 4
            or any(type(v) is not str or v not in HYPOTHESES for v in labels)
            or labels != sorted(set(labels))):
        raise JournalError('canonical allowlisted hypothesis labels required')
    if scope is not None and (attempt['scope_hash'] != scope['scope_hash']
            or attempt['evaluator_hash'] != scope['evaluator_hash']
            or scope['tasks'].get(attempt['task_key'], {}).get('task_hash') != attempt['task_hash']
            or scope['tasks'].get(attempt['task_key'], {}).get('input_scope_hash') != attempt['input_scope_hash']
            or attempt['template_hash'] not in scope['template_hashes']):
        raise JournalError('attempt differs from frozen search/evaluator/template scope')


def _tensor_values(params):
    if type(params) is not dict or not 1 <= len(params) <= 256:
        raise JournalError('bounded literal tensor map required')
    if any(type(k) is not str or not k or len(k.encode('utf-8')) > 128 for k in params):
        raise JournalError('bounded tensor names required')
    values, layout, nodes = [], [], 0

    def walk(value, depth):
        nonlocal nodes
        nodes += 1
        if nodes > 2 * MAX_POSITIONS or depth > 8:
            raise JournalError('tensor traversal bound exceeded')
        if type(value) is list:
            if not value or len(value) > MAX_POSITIONS:
                raise JournalError('bounded nonempty tensor dimensions required')
            shapes = [walk(item, depth + 1) for item in value]
            if any(shape != shapes[0] for shape in shapes):
                raise JournalError('rectangular literal tensors required')
            return (len(value),) + shapes[0]
        if type(value) not in (int, float) or not -9 <= value <= 9 or not math.isfinite(value):
            raise JournalError('finite tensor numbers in [-9, 9] required')
        values.append(value)
        if len(values) > MAX_POSITIONS:
            raise JournalError('tensor position bound exceeded')
        return ()

    for key in sorted(params):
        layout.append([digest(key), list(walk(params[key], 0))])
    return values, layout


def tensor_change(parent_params, candidate_params):
    """Host helper: hashes, counts and up to eight changed scalar examples.

    Samples are [flat_index, old, new], flattening tensors in sorted name order,
    then row-major order. No names or full tensors are stored. None parent means
    a founder. Invalid tensors raise; use None change for unparseable/invalid
    proposals rather than putting malformed output into memory.
    """
    after, after_layout = _tensor_values(candidate_params)
    before, before_layout = _tensor_values(parent_params) if parent_params is not None else ([], None)
    comparable = before_layout == after_layout
    changed = [i for i, (old, new) in enumerate(zip(before, after)) if old != new] if comparable else []
    return {'status': 'comparable' if comparable else ('founder' if parent_params is None else 'layout_changed'),
            'before_hash': digest(parent_params) if parent_params is not None else None,
            'after_hash': digest(candidate_params), 'before_positions': len(before),
            'after_positions': len(after), 'changed_positions': len(changed) if comparable else None,
            'samples': [[i, before[i], after[i]] for i in changed[:8]]}


def _change(change):
    _fields(change, 'status before_hash after_hash before_positions after_positions changed_positions samples')
    if change['status'] not in ('comparable', 'layout_changed', 'founder'):
        raise JournalError('invalid tensor-change status')
    _hash(change['after_hash'])
    founder = change['status'] == 'founder'
    if founder:
        if change['before_hash'] is not None or change['before_positions'] != 0:
            raise JournalError('founder cannot claim a parent tensor')
    else:
        _hash(change['before_hash'])
    _integer(change['before_positions'], 0 if founder else 1, MAX_POSITIONS)
    _integer(change['after_positions'], 1, MAX_POSITIONS)
    if type(change['samples']) is not list or len(change['samples']) > 8:
        raise JournalError('bounded numeric change samples required')
    if change['status'] == 'comparable':
        if change['before_positions'] != change['after_positions']:
            raise JournalError('comparable layouts must have equal size')
        _integer(change['changed_positions'], 0, change['after_positions'])
        if len(change['samples']) != min(8, change['changed_positions']):
            raise JournalError('change sample count mismatch')
        last = -1
        for sample in change['samples']:
            if type(sample) is not list or len(sample) != 3:
                raise JournalError('numeric change sample required')
            index, old, new = sample
            _integer(index, last + 1, change['after_positions'] - 1)
            for value in (old, new):
                if type(value) not in (int, float) or not -9 <= value <= 9 or not math.isfinite(value):
                    raise JournalError('bounded numeric sample required')
            if old == new:
                raise JournalError('change sample must differ')
            last = index
    elif change['changed_positions'] is not None or change['samples']:
        raise JournalError('incomparable layouts cannot claim scalar deltas')


def _receipt(receipt, attempt):
    _fields(receipt, 'binding_hash evidence_hash partition category outcome parent_outcome '
            'tensor_change demonstration runtime_error_type')
    _hash(receipt['binding_hash'])
    _hash(receipt['evidence_hash'])
    if receipt['partition'] != 'search_fitness':
        raise JournalError('only host search_fitness receipts may enter memory')
    category = receipt['category']
    if type(category) is not str or category not in CATEGORIES:
        raise JournalError('exact observed category required')
    if category == 'runtime_error':
        if type(receipt['runtime_error_type']) is not str or receipt['runtime_error_type'] not in RUNTIME_ERRORS:
            raise JournalError('allowlisted runtime error class required; no error messages')
    elif receipt['runtime_error_type'] is not None:
        raise JournalError('runtime error class requires runtime_error category')
    if receipt['demonstration'] is not None:
        _demonstration(receipt['demonstration'])
    for name in ('outcome', 'parent_outcome'):
        if receipt[name] is not None and type(receipt[name]) is not bool:
            raise JournalError('host outcomes must be boolean or unknown')
    expected = True if category == 'correct' else (False if category in TASK_MISTAKES else None)
    if receipt['outcome'] is not expected:
        raise JournalError('mechanical/unknown observations cannot become task scores')
    if not attempt['parent_hashes'] and receipt['parent_outcome'] is not None:
        raise JournalError('no parent exists for a parent observation')
    if category in TASK_MISTAKES | {'correct'}:
        if attempt['candidate_hash'] is None or receipt['tensor_change'] is None:
            raise JournalError('task result requires a valid candidate and observed tensor change')
    if receipt['tensor_change'] is not None:
        _change(receipt['tensor_change'])
        if attempt['candidate_hash'] is None:
            raise JournalError('tensor change requires a valid candidate identity')
        if (receipt['tensor_change']['status'] == 'founder') != (not attempt['parent_hashes']):
            raise JournalError('tensor change disagrees with parent provenance')


def _key(attempt):
    # Hypotheses cannot give the same immutable host attempt a new identity.
    return digest({k: v for k, v in attempt.items() if k != 'hypotheses'})


def _slot(attempt):
    # One task-specific proposal per charged attempt. Changing the candidate,
    # parent or template cannot silently create a second history for that slot.
    return digest({key: attempt[key] for key in ('scope_hash', 'task_key', 'attempt_hash')})


def _demo_scope(scope, attempt, receipt):
    example = receipt['demonstration']
    if example is not None:
        hashes = scope['tasks'][attempt['task_key']]['demonstration_hashes']
        if (example['index'] >= len(hashes)
                or hashes[example['index']] != digest({'input': example['input'], 'expected': example['expected']})):
            raise JournalError('demonstration is not the exact frozen search training pair')


def make_record(scope, attempt, *, verify):
    """Obtain facts exclusively from the trusted host callback; never model scores."""
    scope = _scope(scope)
    attempt = copy.deepcopy(attempt)
    _attempt(attempt, scope)
    if not callable(verify):
        raise JournalError('trusted host verifier callback required')
    request = {'scope': scope, 'attempt': copy.deepcopy(attempt)}
    binding = digest(request)
    receipt = copy.deepcopy(verify(copy.deepcopy(request)))
    _receipt(receipt, attempt)
    _demo_scope(scope, attempt, receipt)
    if receipt['binding_hash'] != binding:
        raise JournalError('host verification is not bound to this exact request')
    record = _sealed({'schema': SCHEMA + '/record', 'record_key': _key(attempt),
                      'attempt': copy.deepcopy(attempt), 'observation': receipt}, 'record_hash')
    _record(record)
    return record


def _record(record):
    _fields(record, 'schema record_key attempt observation record_hash')
    if record['schema'] != SCHEMA + '/record':
        raise JournalError('unsupported memory record')
    _attempt(record['attempt'])
    _receipt(record['observation'], record['attempt'])
    if record['record_key'] != _key(record['attempt']):
        raise JournalError('record provenance key mismatch')
    _canonical(record, MAX_RECORD_BYTES)
    _check_hash(record, 'record_hash')


def _verified(scope, record, verify):
    if make_record(scope, record['attempt'], verify=verify) != record:
        raise JournalError('immutable host observation changed')


def retrieve(scope, records, *, task_id, template_hash, parent_hashes=(), verify,
             max_records=6, max_prompt_bytes=4096):
    """Return prompt, included hashes, quarantined hashes/reasons, excluded count.

    Scope mismatches are quarantined without invoking the verifier. Compatible
    records for other task/templates are excluded. Matching records are reverified,
    then ordered by exact ordered-parent match, newest input position, record hash.
    A prefix that fits BOTH limits is returned; hashes count against the byte cap.
    Quarantine is a returned view, never a mutation/deletion of historical evidence.
    Input records must be in journal order. Do not turn quarantine into a retry or
    rerun: investigation/authorization belongs to the host.
    """
    scope = _scope(scope)
    _integer(max_records, 1, 32)
    _integer(max_prompt_bytes, 512, 32768)
    parents = _hashes(parent_hashes, 8)
    if not callable(verify):
        raise JournalError('trusted host verifier callback required')
    key = task_key(task_id)
    if key not in scope['tasks'] or _hash(template_hash) not in scope['template_hashes']:
        raise JournalError('query must belong to the frozen search/template scope')
    if type(records) not in (list, tuple) or len(records) > MAX_RECORDS:
        raise JournalError('bounded ordered records required')
    eligible, quarantined, excluded, seen = [], [], 0, set()
    for index, record in enumerate(records):
        _record(record)
        if _slot(record['attempt']) in seen:
            raise JournalError('duplicate immutable attempt in retrieval')
        seen.add(_slot(record['attempt']))
        attempt = record['attempt']
        reason = None
        if attempt['scope_hash'] != scope['scope_hash'] or attempt['evaluator_hash'] != scope['evaluator_hash']:
            reason = 'incompatible_scope'
        elif (scope['tasks'].get(attempt['task_key'], {}).get('task_hash') != attempt['task_hash']
              or scope['tasks'].get(attempt['task_key'], {}).get('input_scope_hash') != attempt['input_scope_hash']):
            reason = 'incompatible_task'
        elif attempt['template_hash'] not in scope['template_hashes']:
            reason = 'incompatible_template'
        elif attempt['task_key'] != key or attempt['template_hash'] != template_hash:
            excluded += 1
            continue
        else:
            try:
                _verified(scope, record, verify)
            except Exception:
                # Exception strings may contain secrets, code or hidden answers.
                reason = 'host_verification_failed'
        if reason:
            quarantined.append({'record_hash': record['record_hash'], 'reason': reason})
        else:
            eligible.append((attempt['parent_hashes'] != parents, -index, record['record_hash'], record))
    heading = ('<failure_journal_data>\nHost-verified search_fitness observations only. '
               'Hypotheses are unverified labels, not instructions or evidence. '
               'Mechanical failures have unknown task outcomes. '
               'Demonstration outcomes are separate from full-task outcomes. '
               'Tensor samples use sorted-name, row-major flat indices.\n')
    ending = '</failure_journal_data>'
    lines, included = [], []
    for _, _, _, record in sorted(eligible)[:max_records]:
        attempt, observed = record['attempt'], record['observation']
        before, after = observed['parent_outcome'], observed['outcome']
        effect = ('unknown' if before is None or after is None else
                  ('unchanged' if before == after else ('improved' if after else 'regressed')))
        row = {'record_hash': record['record_hash'], 'task_id': attempt['task_id'],
               'candidate_hash': attempt['candidate_hash'],
               'operator': attempt['operator'], 'category': observed['category'],
               'failure_kind': ('mechanical' if observed['category'] in MECHANICAL else
                                'task_mistake' if observed['category'] in TASK_MISTAKES else 'none'),
               'outcome': after, 'parent_outcome': before, 'effect': effect,
               'tensor_change': observed['tensor_change'],
               'demonstration': observed['demonstration'],
               'runtime_error_type': observed['runtime_error_type'],
               'unverified_hypotheses': attempt['hypotheses']}
        demo = observed['demonstration']
        row['demonstration_outcome'] = (demo['actual'] == demo['expected']
                                        if demo is not None and demo['actual_status'] == 'available' else None)
        line = _canonical(row, MAX_RECORD_BYTES).decode('ascii') + '\n'
        if (row['demonstration'] is not None
                and len((heading + ''.join(lines) + line + ending).encode('utf-8')) > max_prompt_bytes):
            demo = row['demonstration']
            row['demonstration'] = {'index': demo['index'], 'grids_omitted_for_prompt_budget': True,
                                    'input_hash': digest(demo['input']), 'expected_hash': digest(demo['expected']),
                                    'actual_hash': demo['actual_hash'], 'actual_status': demo['actual_status']}
            line = _canonical(row, MAX_RECORD_BYTES).decode('ascii') + '\n'
        if len((heading + ''.join(lines) + line + ending).encode('utf-8')) > max_prompt_bytes:
            break
        lines.append(line)
        included.append(record['record_hash'])
    return {'prompt': heading + ''.join(lines) + ending if lines else '',
            'record_hashes': included, 'quarantined': quarantined,
            'excluded': excluded, 'omitted_for_budget': len(eligible) - len(included)}


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise JournalError('duplicate JSON key')
            result[key] = value
        return result

    def invalid(_value):
        raise JournalError('nonfinite JSON constant')

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise JournalError('invalid saved JSON') from error
    if _canonical(value) + b'\n' != data:
        raise JournalError('noncanonical or incomplete journal line')
    return value


def _open(path, flags):
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise JournalError('journal paths must be ordinary files')
    except FileNotFoundError:
        pass
    fd = os.open(path, flags | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise JournalError('ordinary journal file required')
    return fd


class FailureJournal:
    """A bounded store; explicit create=True refuses an existing journal.

    Defaults: 256 records / 1 MiB. Caps belong to the immutable header, and resume
    must supply the same limits. records() returns detached, chain-checked records;
    retrieve() performs the mandatory host reverification. head_hash/record_count
    are read-only snapshots. Close/context exit releases the OS lease. This class
    is single-threaded; do not share an instance across threads or mutate its files.
    """

    def __init__(self, path, scope, *, create=False, max_records=256, max_bytes=1024 * 1024):
        self._scope = _scope(scope)
        _integer(max_records, 1, MAX_RECORDS)
        _integer(max_bytes, 1024, MAX_BYTES)
        if type(create) is not bool:
            raise JournalError('create must be boolean')
        path = Path(path)
        self._path = path.parent.resolve(strict=True) / path.name
        self._limit, self._byte_limit = max_records, max_bytes
        self._lease, self._poisoned = None, False
        lock = self._path.with_name(self._path.name + '.lock')
        fd = _open(lock, os.O_RDWR | (os.O_CREAT if create else 0))
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            elif os.name == 'posix':
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise JournalError('supported OS locking required')
        except OSError as error:
            os.close(fd)
            raise JournalBusy('journal lease unavailable') from error
        except BaseException:
            os.close(fd)
            raise
        self._lease = fd
        try:
            if create:
                header = _sealed({'schema': SCHEMA, 'scope': self._scope,
                                  'max_records': max_records, 'max_bytes': max_bytes}, 'header_hash')
                data = _canonical(header, MAX_SCOPE_BYTES + 1024) + b'\n'
                if len(data) > max_bytes:
                    raise JournalFull('scope header exceeds storage cap')
                self._write(data, create=True)
            self._read()
        except BaseException:
            self.close()
            raise

    def _write(self, data, create=False):
        flags = os.O_WRONLY | os.O_APPEND | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = _open(self._path, flags)
        try:
            remaining = memoryview(data)
            while remaining:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise OSError('journal write incomplete')
                remaining = remaining[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if create and os.name == 'posix':
            fd = os.open(self._path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _read(self):
        self._ready()
        fd = _open(self._path, os.O_RDONLY)
        with os.fdopen(fd, 'rb') as stream:
            if os.fstat(stream.fileno()).st_size > self._byte_limit:
                raise JournalError('journal exceeds frozen byte cap')
            line = stream.readline(MAX_SCOPE_BYTES + 1025)
            header = _decode(line)
            _fields(header, 'schema scope max_records max_bytes header_hash')
            _check_hash(header, 'header_hash')
            if (header['schema'] != SCHEMA or type(header['max_records']) is not int
                    or type(header['max_bytes']) is not int or header['max_records'] != self._limit
                    or header['max_bytes'] != self._byte_limit):
                raise JournalError('journal schema/caps changed')
            saved_scope = _scope(header['scope'])
            tail, size, records, seen = header['header_hash'], len(line), [], set()
            while True:
                line = stream.readline(MAX_RECORD_BYTES + 1025)
                if not line:
                    break
                if len(records) >= self._limit or len(line) > MAX_RECORD_BYTES + 1024:
                    raise JournalError('journal exceeds frozen record cap')
                envelope = _decode(line)
                _fields(envelope, 'sequence previous_hash record event_hash')
                _check_hash(envelope, 'event_hash')
                if (type(envelope['sequence']) is not int or envelope['sequence'] != len(records)
                        or envelope['previous_hash'] != tail):
                    raise JournalError('journal chain/sequence mismatch')
                record = envelope['record']
                _record(record)
                _attempt(record['attempt'], saved_scope)
                _demo_scope(saved_scope, record['attempt'], record['observation'])
                request = {'scope': saved_scope, 'attempt': record['attempt']}
                if record['observation']['binding_hash'] != digest(request):
                    raise JournalError('saved host receipt binding mismatch')
                if _slot(record['attempt']) in seen:
                    raise JournalError('duplicate immutable attempt in journal')
                seen.add(_slot(record['attempt']))
                records.append(record)
                tail = envelope['event_hash']
                size += len(line)
        self._records, self._head, self._size = records, tail, size
        self._compatible = saved_scope == self._scope

    def _ready(self):
        if self._lease is None or self._poisoned:
            raise JournalError('journal is closed or requires inspection after a failed write')

    @property
    def head_hash(self):
        self._ready()
        return self._head

    @property
    def record_count(self):
        self._ready()
        return len(self._records)

    @property
    def compatible(self):
        self._ready()
        return self._compatible

    def records(self):
        self._read()
        return copy.deepcopy(self._records)

    def append(self, record, *, verify):
        """Reverify before appending; identical retry is idempotent, edits fail.

        Returns the stored record hash. No removal occurs on cap exhaustion.
        A changed host receipt or changed hypotheses for the same attempt is an
        immutable-provenance error, not an update. Preserve a new host attempt
        separately instead of overwriting an old failure with a later success.
        """
        self._read()
        if not self._compatible:
            raise JournalError('stale scope is quarantined read-only; use a new authorized path')
        _record(record)
        _verified(self._scope, record, verify)
        for saved in self._records:
            if _slot(saved['attempt']) == _slot(record['attempt']):
                if saved != record:
                    raise JournalError('attempt already has immutable evidence')
                return saved['record_hash']
        envelope = _sealed({'sequence': len(self._records), 'previous_hash': self._head,
                            'record': copy.deepcopy(record)}, 'event_hash')
        data = _canonical(envelope, MAX_RECORD_BYTES + 1024) + b'\n'
        if len(self._records) >= self._limit or self._size + len(data) > self._byte_limit:
            raise JournalFull('journal cap reached; existing evidence preserved')
        try:
            self._write(data)
        except BaseException:
            self._poisoned = True
            raise
        self._records.append(copy.deepcopy(record))
        self._size += len(data)
        self._head = envelope['event_hash']
        return record['record_hash']

    def close(self):
        if self._lease is not None:
            os.close(self._lease)  # Kernel releases flock/byte-range lease.
            self._lease = None

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *_args):
        self.close()
