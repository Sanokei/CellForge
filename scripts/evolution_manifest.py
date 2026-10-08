"""Frozen evolution inputs and a local, append-only provenance journal.

Only explicit calls perform I/O. ``freeze_manifest`` binds normalized config,
caller-computed SHA256 source digests, and three disjoint task-ID populations.
Call it before *any* resumed work or overwrite of saved run artifacts. Partition
order is immaterial; config array order is significant. IDs are exact strings:
the caller must use consistent dataset-qualified IDs and bind task contents in
the config/source digests as appropriate. This module never opens task data.
``freeze_manifest`` refuses a sealed journal before returning to a search
caller. ``read_manifest_status`` verifies either an open or sealed store without
writing it and returns the frozen manifest plus separate journal status; those
status fields never contribute to the manifest hash. A status check is a locked
snapshot, not a lease: the integrator must exclude concurrent research/report
sessions for the same root throughout their external scoring/maker operations.

``append_event`` accepts only these kinds (no arbitrary metadata fields):

* ``breeding_wave``: wave (nonnegative int), operator (chain, tensor_crossover,
  or mutation), parents/children (distinct SHA256 IDs, possibly empty), seed
  (nonnegative int), scope_hash (SHA256 or null), rng_state_after (the version-3
  Python Random state). Optional partition/task_ids must occur together and
  name a nonempty subset of search_fitness. This is an operation receipt, not
  a measured-fitness claim; scope_hash is an opaque caller-computed binding.
* ``fitness_evaluation``, ``development_evaluation``, ``report_evaluation``:
  exactly partition, task_ids, evaluator_hash, scope_hash, outcomes. Partition
  is respectively search_fitness, development_selection, report. Task IDs are
  a nonempty subset of that frozen population; outcomes maps candidate SHA256
  IDs to exact task-ID maps of true/false/null (unknown). evaluator_hash equals
  the manifest's source_hashes_hash; scope_hash binds caller-owned task content.
  Report must cover its entire population and seals the journal: no later
  events are admitted, including another report or renewed search/selection.
* ``mutation_competition``: wave, operator (complete_tensor_replacement),
  partition (development), task_ids (both full non-report populations),
  baseline_outcomes (partition -> task ID -> bool), attempts, and integer
  charged_positions_per_arm/invalid_or_missing totals. Each attempt has call,
  task_id (search ID), parent_id, seed, operator, charged_positions_per_arm,
  outcome (valid/neutral/invalid_or_missing), candidate_id/candidate_outcomes/
  candidate_params (all null for invalid), random_id/random_outcomes/
  random_params. Tensor params are nonempty mappings of rectangular finite
  numeric arrays/scalars. Candidate and random layouts/position charges agree.
* ``mutation_champions``: wave, candidate_id, random_id, claim. IDs can be null;
  claim must equal MUTATION_CLAIM. Requires a preceding competition in that
  wave and is accepted once. Champion IDs must maximize development_selection
  successes, breaking ties by ID, strictly improve its baseline, and preserve
  every successful baseline task across both panels. A null champion means
  no eligible entry in that arm. These checks use caller-recorded outcomes;
  the event schema lacks template names needed to recompute IDs from tensors.
* ``router_competition``: wave, operator (complete_router_replacement),
  partition (development), ordered expert_ids, features (ROUTER_FEATURES),
  baseline_id/baseline_params/baseline_outcomes, zero_id/zero_params/
  zero_outcomes, task_ids (both full non-report populations), attempts,
  charged_positions_per_arm, invalid_or_missing. Each attempt has call, seed,
  charged_positions_per_arm, outcome (valid/neutral/invalid_or_missing),
  candidate_id/candidate_params/candidate_outcomes, random_id/random_params/
  random_outcomes. Invalid candidates have all three candidate fields null.
  All params are exactly route.W[N,N]/route.b[N], finite numbers in [-9, 9];
  every reserved attempt charges N*N+N per arm, even when invalid or neutral.
  zero_params must contain only zeroes. Router IDs are recomputed from exactly
  {schema: ROUTER_SCHEMA, expert_ids, params}, excluding lineage. Identical
  router IDs must have identical recorded outcomes throughout the competition.
* ``router_champions``: wave, expert_ids, candidate_id, random_id, accepted_id,
  claim (ROUTER_CLAIM). Both arm champions follow the mutation champion rules.
  accepted_id is the candidate champion only when its development_selection
  count strictly exceeds zero AND EVERY random attempt, including randoms
  ineligible as champions; otherwise it is null. The expert roster/order is
  frozen at the first router competition; breeding/mutation cannot follow it.
  Router waves strictly increase and must settle their champions before the
  next competition. Each later baseline must match the effective incumbent's
  identity and recorded outcomes: the initial baseline until an acceptance,
  then the last accepted router. Rejected waves do not reset that incumbent.

Unknown kinds/fields, non-finite numbers, duplicate IDs/JSON keys, changed
resumes, incomplete initialization, and torn/corrupt records fail closed.
Tuple inputs (including Random states) normalize to JSON lists. The returned
event includes sequence, previous_hash, manifest_hash, and event_hash.

Cooperating processes/threads serialize through a persistent OS-locked file;
never delete/replace that lock file or modify this store while it is in use.
Lock acquisition fails promptly when busy: retry the whole call later. This
assumes a trusted local filesystem with working flock/Windows byte locks.
Files are fsynced; containing-directory fsync is also required on POSIX.
Python's stdlib cannot promise directory-entry durability on Windows or remote
filesystem behavior. Partial writes/initialization are preserved for diagnosis,
never repaired or truncated automatically. A failed call may have durably
written its record; inspect evidence before retrying an ambiguous I/O failure.

Hashes detect inconsistency, not authenticity. They cannot detect an attacker
rewriting the whole store, a rollback to a valid journal prefix, false caller
digests/outcomes, or report use outside this API. No task execution is verified
here; this journal confers no ONE_STANDARD qualification or scientific acceptance.
"""

from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time


MANIFEST_NAME = 'EVOLUTION_MANIFEST.json'
EVENTS_NAME = 'EVOLUTION_EVENTS.jsonl'
LOCK_NAME = '.evolution_manifest.lock'
MANIFEST_SCHEMA = 'cellforge-evolution-manifest/1'
EVENT_SCHEMA = 'cellforge-evolution-event/1'
PARTITIONS = ('search_fitness', 'development_selection', 'report')
ZERO_HASH = '0' * 64
MUTATION_CLAIM = 'group-relative development selection; not PPO/GRPO or independent generalisation'
ROUTER_SCHEMA = 'cellforge-router-genome/1'
ROUTER_FEATURES = 'negative-leave-one-out-demo-error/1'
ROUTER_CLAIM = 'development-only router selection; no independent generalisation claim'


class ManifestError(ValueError):
    """Invalid inputs or inconsistent persisted evidence; do not resume."""


class ManifestBusyError(RuntimeError):
    """OS lock acquisition reported contention/access denial; no event was written."""


def _json_value(value):
    """Copy strict JSON data, allowing tuples only as JSON array inputs."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_json_value(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ManifestError('only finite JSON values with string object keys are supported')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash(value, label):
    if (type(value) is not str or len(value) != 64
            or any(char not in '0123456789abcdefABCDEF' for char in value)):
        raise ManifestError(label + ' must be a SHA256 hex digest')
    return value.lower()


def _fields(value, required, optional=()):
    if type(value) is not dict or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ManifestError('exact required fields: ' + ', '.join(sorted(required)))


def _ids(value, label, hashes=False, nonempty=False):
    if type(value) is not list or (nonempty and not value):
        raise ManifestError(label + ' must be a list' + (' with at least one ID' if nonempty else ''))
    if any(type(item) is not str or not item or item != item.strip() for item in value):
        raise ManifestError(label + ' requires nonempty IDs without surrounding whitespace')
    result = [_hash(item, label) for item in value] if hashes else list(value)
    if len(set(result)) != len(result):
        raise ManifestError(label + ' contains duplicate IDs')
    return result


def _nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ManifestError(label + ' must be a nonnegative integer')


def _manifest(config, source_hashes, partition_ids):
    config = _json_value(config)
    if type(config) is not dict:
        raise ManifestError('config must be a JSON object')
    if type(source_hashes) is not dict or not source_hashes:
        raise ManifestError('source_hashes must be a nonempty mapping')
    sources = {}
    for name, digest in source_hashes.items():
        if type(name) is not str or not name or name != name.strip():
            raise ManifestError('source identities must be nonempty strings without surrounding whitespace')
        sources[name] = _hash(digest, 'source hash')
    _fields(partition_ids, PARTITIONS)
    partitions, seen = {}, set()
    for name in PARTITIONS:
        ids = _ids(partition_ids[name], name)
        if seen.intersection(ids):
            raise ManifestError('task IDs must be mutually disjoint across all partitions')
        seen.update(ids)
        partitions[name] = sorted(ids)
    body = {
        'schema': MANIFEST_SCHEMA,
        'config': config,
        'config_hash': _digest(config),
        'source_hashes': sources,
        'source_hashes_hash': _digest(sources),
        'partition_ids': partitions,
        'partition_hashes': {name: _digest(ids) for name, ids in partitions.items()},
        'partition_ids_hash': _digest(partitions),
    }
    return dict(body, manifest_hash=_digest(body))


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError('duplicate JSON object key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ManifestError('non-finite JSON constant: ' + value)


def _decode(line, label):
    if not line.endswith(b'\n'):
        raise ManifestError(label + ' is torn (missing final newline)')
    try:
        value = json.loads(line.decode('ascii'), object_pairs_hook=_no_duplicates,
                           parse_constant=_invalid_constant)
        value = _json_value(value)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ManifestError(label + ' contains invalid JSON') from error
    if line != _canonical(value) + b'\n':
        raise ManifestError(label + ' is not the canonical stored record')
    return value


def _exists(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ManifestError('store paths must be ordinary files: ' + path.name)
    return True


def _open_file(path, flags):
    _exists(path)  # Reject symlinks/reparse points and non-regular files.
    fd = os.open(path, flags | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ManifestError('store paths must be ordinary files: ' + path.name)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _sync_directory(root):
    if os.name == 'posix':
        fd = os.open(root, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def _locked(root, create=False):
    # Imports are deliberately lazy: importing this module performs no I/O.
    if os.name == 'nt':
        import msvcrt
    elif os.name == 'posix':
        import fcntl
    else:
        raise ManifestError('no supported OS lock implementation; refusing to write')
    root = Path(root)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ManifestError('run root must be an existing directory')
    root = root.resolve(strict=True)
    lock_path = root / LOCK_NAME
    if not create and not _exists(lock_path):
        raise ManifestError('existing persistent store lock required; refusing to recreate it')
    fd = _open_file(lock_path, os.O_RDWR | (os.O_CREAT if create else 0))
    acquired = False
    try:
        try:
            if os.name == 'nt':
                os.lseek(fd, 0, os.SEEK_SET)
                # Windows byte-range locks may cover a byte beyond EOF. Keep
                # the persistent lock file empty; never replace its inode.
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise ManifestBusyError('store lock unavailable; retry after its holder releases it') from error
            raise ManifestError('OS locking failed; refusing an unlocked write') from error
        acquired = True
        yield root
    finally:
        try:
            if acquired:
                if os.name == 'nt':
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_record(path, record, create=False):
    flags = os.O_WRONLY | os.O_APPEND
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    fd = _open_file(path, flags)
    try:
        remaining = memoryview(_canonical(record) + b'\n')
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError('failed to append the entire provenance record')
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if create:
        _sync_directory(path.parent)


def _read_manifest(root):
    fd = _open_file(root / MANIFEST_NAME, os.O_RDONLY)
    with os.fdopen(fd, 'rb') as stream:
        saved = _decode(stream.read(), 'manifest')
    _fields(saved, ('schema', 'config', 'config_hash', 'source_hashes', 'source_hashes_hash',
                    'partition_ids', 'partition_hashes', 'partition_ids_hash', 'manifest_hash'))
    expected = _manifest(saved['config'], saved['source_hashes'], saved['partition_ids'])
    if _canonical(saved) != _canonical(expected):
        raise ManifestError('saved manifest hashes/schema/normalization do not match its contents')
    return saved


def _rng_state(state):
    if type(state) is not list or len(state) != 3 or type(state[0]) is not int or state[0] != 3:
        raise ManifestError('rng_state_after must be a version-3 Python Random state')
    words = state[1]
    if (type(words) is not list or len(words) != 625
            or any(type(word) is not int or not 0 <= word <= 0xffffffff for word in words[:-1])
            or type(words[-1]) is not int or not 0 <= words[-1] <= 624):
        raise ManifestError('invalid Random generator words/index')
    # The JSON-normalization pass already rejects non-finite float values.
    if state[2] is not None and type(state[2]) not in (int, float):
        raise ManifestError('invalid Random Gaussian cache')


def _task_scope(payload, manifest, partition):
    if payload['partition'] != partition:
        raise ManifestError('event requires partition ' + partition + '; report outcomes cannot be fitness/selection')
    ids = _ids(payload['task_ids'], 'event task_ids', nonempty=True)
    frozen = set(manifest['partition_ids'][partition])
    if not set(ids) <= frozen:
        raise ManifestError('event task IDs exceed the declared frozen partition')
    return ids


def _partition_outcomes(value, task_ids):
    _fields(value, ('search_fitness', 'development_selection'))
    for partition, ids in task_ids.items():
        rows = value[partition]
        if type(rows) is not dict or set(rows) != set(ids):
            raise ManifestError('competition outcomes must cover exactly both declared non-report panels')
        if any(type(outcome) is not bool for outcome in rows.values()):
            raise ManifestError('competition panel outcomes must be host-scored booleans')


def _tensor_shape(value):
    if type(value) in (int, float):
        return (), 1  # _json_value has already rejected non-finite floats.
    if type(value) is not list or not value:
        raise ManifestError('tensors must be nonempty rectangular numeric arrays or numeric scalars')
    children = [_tensor_shape(item) for item in value]
    if any(shape != children[0][0] for shape, count in children):
        raise ManifestError('ragged tensor arrays are not admissible')
    return (len(value),) + children[0][0], sum(count for shape, count in children)


def _tensor_layout(params):
    if type(params) is not dict or not params:
        raise ManifestError('tensor params must be a nonempty mapping')
    layout, positions = {}, 0
    for name, value in params.items():
        if not name or name != name.strip():
            raise ManifestError('tensor names must be nonempty without surrounding whitespace')
        shape, count = _tensor_shape(value)
        layout[name] = shape
        positions += count
    return layout, positions


def _mutation_competition(payload, manifest):
    _fields(payload, ('wave', 'operator', 'partition', 'task_ids', 'baseline_outcomes',
                      'attempts', 'charged_positions_per_arm', 'invalid_or_missing'))
    _nonnegative(payload['wave'], 'wave')
    _nonnegative(payload['charged_positions_per_arm'], 'charged_positions_per_arm')
    _nonnegative(payload['invalid_or_missing'], 'invalid_or_missing')
    if payload['operator'] != 'complete_tensor_replacement' or payload['partition'] != 'development':
        raise ManifestError('mutation competition requires complete-tensor replacement on development only')
    task_ids = payload['task_ids']
    _fields(task_ids, ('search_fitness', 'development_selection'))
    for partition, ids in task_ids.items():
        ids = _ids(ids, partition, nonempty=True)
        if set(ids) != set(manifest['partition_ids'][partition]):
            raise ManifestError('mutation competitions must use both complete frozen non-report panels')
    _partition_outcomes(payload['baseline_outcomes'], task_ids)
    if type(payload['attempts']) is not list:
        raise ManifestError('mutation attempts must be a list')
    seen, charged, invalid = set(), 0, 0
    search_ids = set(task_ids['search_fitness'])
    for attempt in payload['attempts']:
        _fields(attempt, ('call', 'task_id', 'parent_id', 'seed', 'operator', 'charged_positions_per_arm',
                          'outcome', 'candidate_id', 'candidate_outcomes', 'random_id', 'random_outcomes',
                          'random_params', 'candidate_params'))
        for field in ('call', 'seed', 'charged_positions_per_arm'):
            _nonnegative(attempt[field], field)
        if type(attempt['task_id']) is not str or attempt['task_id'] not in search_ids:
            raise ManifestError('mutation target must be a frozen search_fitness ID')
        key = (attempt['call'], attempt['task_id'])
        if key in seen:
            raise ManifestError('duplicate reserved call/task mutation attempt')
        seen.add(key)
        if attempt['operator'] != 'complete_tensor_replacement':
            raise ManifestError('every reserved attempt must charge complete-tensor replacement')
        attempt['parent_id'] = _hash(attempt['parent_id'], 'parent_id')
        attempt['random_id'] = _hash(attempt['random_id'], 'random_id')
        random_layout, random_count = _tensor_layout(attempt['random_params'])
        if attempt['charged_positions_per_arm'] != random_count:
            raise ManifestError('per-arm charge differs from the full random tensor position count')
        _partition_outcomes(attempt['random_outcomes'], task_ids)
        outcome = attempt['outcome']
        if outcome == 'invalid_or_missing':
            if any(attempt[field] is not None for field in ('candidate_id', 'candidate_outcomes', 'candidate_params')):
                raise ManifestError('invalid/missing attempts must have null candidate evidence')
            invalid += 1
        elif outcome in ('valid', 'neutral'):
            attempt['candidate_id'] = _hash(attempt['candidate_id'], 'candidate_id')
            neutral = attempt['candidate_id'] == attempt['parent_id']
            if neutral != (outcome == 'neutral'):
                raise ManifestError('neutral status must match the parent/candidate identity comparison')
            layout, count = _tensor_layout(attempt['candidate_params'])
            if layout != random_layout or count != random_count:
                raise ManifestError('candidate and random controls must have equal full tensor layouts/charges')
            _partition_outcomes(attempt['candidate_outcomes'], task_ids)
        else:
            raise ManifestError('unknown mutation attempt outcome')
        charged += random_count
    if charged != payload['charged_positions_per_arm'] or invalid != payload['invalid_or_missing']:
        raise ManifestError('mutation charge/invalid totals do not reconcile with all attempts')
    return payload


def _router_identity(params, expert_ids):
    _fields(params, ('route.W', 'route.b'))
    layout, count = _tensor_layout(params)
    n = len(expert_ids)
    if layout != {'route.W': (n, n), 'route.b': (n,)} or count != n * n + n:
        raise ManifestError('router requires exactly W[N,N] and b[N] for its ordered expert roster')
    if any(not -9 <= value <= 9 for row in params['route.W'] + [params['route.b']] for value in row):
        raise ManifestError('router v1 coefficients must be finite literals in [-9, 9]')
    return _digest({'schema': ROUTER_SCHEMA, 'expert_ids': expert_ids, 'params': params})


def _router_evidence(record, arm, expert_ids, task_ids, observed):
    """Validate a complete genome/outcome binding without running any expert."""
    identity = _hash(record[arm + '_id'], arm + '_id')
    if identity != _router_identity(record[arm + '_params'], expert_ids):
        raise ManifestError(arm + ' router ID differs from its ordered expert roster or literal tensors')
    record[arm + '_id'] = identity
    outcomes = record[arm + '_outcomes']
    _partition_outcomes(outcomes, task_ids)
    if identity in observed and observed[identity] != outcomes:
        raise ManifestError('identical router genomes have inconsistent recorded outcomes')
    observed[identity] = outcomes


def _router_competition(payload, manifest):
    _fields(payload, ('wave', 'operator', 'partition', 'expert_ids', 'features', 'baseline_id',
                      'baseline_params', 'baseline_outcomes', 'task_ids', 'zero_id', 'zero_params',
                      'zero_outcomes', 'attempts', 'charged_positions_per_arm', 'invalid_or_missing'))
    for field in ('wave', 'charged_positions_per_arm', 'invalid_or_missing'):
        _nonnegative(payload[field], field)
    if payload['operator'] != 'complete_router_replacement' or payload['partition'] != 'development':
        raise ManifestError('router competition requires full-router replacement on development only')
    if payload['features'] != ROUTER_FEATURES:
        raise ManifestError('router feature semantics differ from the frozen v1 definition')
    expert_ids = _ids(payload['expert_ids'], 'expert_ids', hashes=True, nonempty=True)
    payload['expert_ids'] = expert_ids
    task_ids = payload['task_ids']
    _fields(task_ids, ('search_fitness', 'development_selection'))
    for partition, ids in task_ids.items():
        ids = _ids(ids, partition, nonempty=True)
        if set(ids) != set(manifest['partition_ids'][partition]):
            raise ManifestError('router competitions must use both complete frozen non-report panels')
    observed = {}
    _router_evidence(payload, 'baseline', expert_ids, task_ids, observed)
    _router_evidence(payload, 'zero', expert_ids, task_ids, observed)
    if any(value != 0 for row in payload['zero_params']['route.W'] + [payload['zero_params']['route.b']]
           for value in row):
        raise ManifestError('zero-router control must contain only zero coefficients')
    if type(payload['attempts']) is not list:
        raise ManifestError('router attempts must be a list')
    positions = len(expert_ids) * len(expert_ids) + len(expert_ids)
    calls, charged, invalid = set(), 0, 0
    for attempt in payload['attempts']:
        _fields(attempt, ('call', 'seed', 'charged_positions_per_arm', 'outcome', 'candidate_id',
                          'candidate_params', 'candidate_outcomes', 'random_id', 'random_params', 'random_outcomes'))
        for field in ('call', 'seed', 'charged_positions_per_arm'):
            _nonnegative(attempt[field], field)
        if attempt['call'] in calls:
            raise ManifestError('duplicate reserved router call')
        calls.add(attempt['call'])
        if attempt['charged_positions_per_arm'] != positions:
            raise ManifestError('each router attempt must charge the complete N*N+N positions per arm')
        _router_evidence(attempt, 'random', expert_ids, task_ids, observed)
        if attempt['outcome'] == 'invalid_or_missing':
            if any(attempt[field] is not None for field in ('candidate_id', 'candidate_params', 'candidate_outcomes')):
                raise ManifestError('invalid/missing router attempts must have null candidate evidence')
            invalid += 1
        elif attempt['outcome'] in ('valid', 'neutral'):
            _router_evidence(attempt, 'candidate', expert_ids, task_ids, observed)
            neutral = attempt['candidate_id'] == payload['baseline_id']
            if neutral != (attempt['outcome'] == 'neutral'):
                raise ManifestError('neutral router status must match its baseline/candidate identity comparison')
        else:
            raise ManifestError('unknown router attempt outcome')
        charged += positions
    if charged != payload['charged_positions_per_arm'] or invalid != payload['invalid_or_missing']:
        raise ManifestError('router charge/invalid totals do not reconcile with all reservations')
    return payload


def _expected_champions(competition):
    baseline = competition['baseline_outcomes']
    baseline_score = sum(baseline['development_selection'].values())
    winners = {}
    for arm in ('candidate', 'random'):
        eligible = []
        for attempt in competition['attempts']:
            outcomes = attempt[arm + '_outcomes']
            if outcomes is None:
                continue
            score = sum(outcomes['development_selection'].values())
            preserved = all(not ok or outcomes[partition][task_id]
                            for partition, rows in baseline.items() for task_id, ok in rows.items())
            if score > baseline_score and preserved:
                eligible.append((-score, attempt[arm + '_id']))
        winners[arm + '_id'] = min(eligible)[1] if eligible else None
    return winners


def _expected_router_champions(competition):
    winners = _expected_champions(competition)
    winners['accepted_id'] = None
    candidate_id = winners['candidate_id']
    if candidate_id is not None:
        score = next(sum(attempt['candidate_outcomes']['development_selection'].values())
                     for attempt in competition['attempts'] if attempt['candidate_id'] == candidate_id)
        zero_score = sum(competition['zero_outcomes']['development_selection'].values())
        if score > zero_score and all(
                score > sum(attempt['random_outcomes']['development_selection'].values())
                for attempt in competition['attempts']):
            winners['accepted_id'] = candidate_id
    return winners


def _transition(kind, payload, mutation_waves, router_waves):
    if router_waves and kind in ('breeding_wave', 'mutation_competition', 'mutation_champions'):
        raise ManifestError('fixed-roster router evolution replaces breeding and cell mutation for this campaign')
    if kind == 'mutation_competition':
        wave = payload['wave']
        if wave in mutation_waves:
            raise ManifestError('mutation wave already has a competition; do not silently retry committed work')
        mutation_waves[wave] = {'expected': _expected_champions(payload), 'settled': False}
    elif kind == 'mutation_champions':
        wave = mutation_waves.get(payload['wave'])
        if wave is None or wave['settled']:
            raise ManifestError('champions require one preceding, unsettled competition in the same wave')
        if any(payload[key] != value for key, value in wave['expected'].items()):
            raise ManifestError('champion IDs disagree with development-only ranking and baseline preservation')
        wave['settled'] = True
    elif kind == 'router_competition':
        wave = payload['wave']
        if wave in router_waves:
            raise ManifestError('router wave already has a competition; do not silently retry committed work')
        if any(not state['settled'] for state in mutation_waves.values()):
            raise ManifestError('pending cell mutation champions must settle before the router campaign')
        if router_waves:
            if wave <= max(router_waves):
                raise ManifestError('router competition waves must strictly increase in journal order')
            first = next(iter(router_waves.values()))
            if payload['expert_ids'] != first['expert_ids']:
                raise ManifestError('router campaign expert roster/order differs from its first competition')
            if any(not state['settled'] for state in router_waves.values()):
                raise ManifestError('pending router champions must settle before another competition')
            previous = router_waves[max(router_waves)]
            if payload['baseline_id'] != previous['incumbent_id']:
                raise ManifestError('router baseline must be the effective incumbent from the preceding settled wave')
            if payload['baseline_outcomes'] != previous['incumbent_outcomes']:
                raise ManifestError('unchanged router incumbent has inconsistent baseline outcomes on the frozen panels')
        expected = _expected_router_champions(payload)
        accepted_outcomes = None
        if expected['accepted_id'] is not None:
            accepted_outcomes = next(attempt['candidate_outcomes'] for attempt in payload['attempts']
                                     if attempt['candidate_id'] == expected['accepted_id'])
        # The first baseline is retained here. A later wave may only inherit
        # the previous settled incumbent; an unaccepted champion cannot reset
        # either the identity or its successful-task preservation anchors.
        router_waves[wave] = {'expected': expected, 'expert_ids': payload['expert_ids'],
                              'baseline_id': payload['baseline_id'],
                              'incumbent_id': payload['baseline_id'],
                              'incumbent_outcomes': payload['baseline_outcomes'],
                              'accepted_outcomes': accepted_outcomes, 'settled': False}
    elif kind == 'router_champions':
        wave = router_waves.get(payload['wave'])
        if wave is None or wave['settled']:
            raise ManifestError('router champions require one preceding unsettled competition in the same wave')
        if payload['expert_ids'] != wave['expert_ids']:
            raise ManifestError('router champions must preserve the competition expert roster/order')
        if any(payload[key] != value for key, value in wave['expected'].items()):
            raise ManifestError('router champions/acceptance disagree with development ranking, anchors or controls')
        if payload['accepted_id'] is not None:
            wave['incumbent_id'] = payload['accepted_id']
            wave['incumbent_outcomes'] = wave['accepted_outcomes']
        wave['settled'] = True
    elif kind == 'report_evaluation':
        if any(not state['settled'] for waves in (mutation_waves, router_waves) for state in waves.values()):
            raise ManifestError('cannot seal a report before all pending mutation/router champions are recorded')


def _event_payload(kind, payload, manifest):
    if type(kind) is not str:
        raise ManifestError('event kind must be a supported string')
    payload = _json_value(payload)
    if kind == 'router_competition':
        return _router_competition(payload, manifest)
    if kind == 'router_champions':
        _fields(payload, ('wave', 'expert_ids', 'candidate_id', 'random_id', 'accepted_id', 'claim'))
        _nonnegative(payload['wave'], 'wave')
        payload['expert_ids'] = _ids(payload['expert_ids'], 'expert_ids', hashes=True, nonempty=True)
        if payload['claim'] != ROUTER_CLAIM:
            raise ManifestError('router champions require the explicit development-only claim')
        for field in ('candidate_id', 'random_id', 'accepted_id'):
            if payload[field] is not None:
                payload[field] = _hash(payload[field], field)
        return payload
    if kind == 'mutation_competition':
        return _mutation_competition(payload, manifest)
    if kind == 'mutation_champions':
        _fields(payload, ('wave', 'candidate_id', 'random_id', 'claim'))
        _nonnegative(payload['wave'], 'wave')
        if payload['claim'] != MUTATION_CLAIM:
            raise ManifestError('mutation champions require the explicit development-only claim')
        for field in ('candidate_id', 'random_id'):
            if payload[field] is not None:
                payload[field] = _hash(payload[field], field)
        return payload
    if kind == 'breeding_wave':
        _fields(payload, ('wave', 'operator', 'parents', 'seed', 'scope_hash', 'children', 'rng_state_after'),
                ('partition', 'task_ids'))
        _nonnegative(payload['wave'], 'wave')
        _nonnegative(payload['seed'], 'seed')
        if payload['operator'] not in ('chain', 'tensor_crossover', 'mutation'):
            raise ManifestError('unsupported breeding operator')
        payload['parents'] = _ids(payload['parents'], 'parents', hashes=True)
        payload['children'] = _ids(payload['children'], 'children', hashes=True)
        if payload['scope_hash'] is not None:
            payload['scope_hash'] = _hash(payload['scope_hash'], 'scope_hash')
        if ('partition' in payload) != ('task_ids' in payload):
            raise ManifestError('partition and task_ids must be supplied together')
        if 'partition' in payload:
            _task_scope(payload, manifest, 'search_fitness')
        _rng_state(payload['rng_state_after'])
        return payload
    partitions = {'fitness_evaluation': 'search_fitness',
                  'development_evaluation': 'development_selection',
                  'report_evaluation': 'report'}
    if kind not in partitions:
        raise ManifestError('unsupported provenance event kind: ' + kind)
    _fields(payload, ('partition', 'task_ids', 'evaluator_hash', 'scope_hash', 'outcomes'))
    ids = _task_scope(payload, manifest, partitions[kind])
    payload['evaluator_hash'] = _hash(payload['evaluator_hash'], 'evaluator_hash')
    if payload['evaluator_hash'] != manifest['source_hashes_hash']:
        raise ManifestError('evaluation must bind the frozen source_hashes_hash')
    payload['scope_hash'] = _hash(payload['scope_hash'], 'scope_hash')
    if kind == 'report_evaluation' and set(ids) != set(manifest['partition_ids']['report']):
        raise ManifestError('report_evaluation must cover the entire frozen report population')
    outcomes = payload['outcomes']
    if type(outcomes) is not dict or not outcomes:
        raise ManifestError('evaluation requires at least one candidate outcome map')
    normalized = {}
    for candidate, rows in outcomes.items():
        candidate = _hash(candidate, 'candidate ID')
        if candidate in normalized:
            raise ManifestError('duplicate normalized candidate ID')
        if type(rows) is not dict or set(rows) != set(ids):
            raise ManifestError('each candidate must cover exactly the event task_ids')
        if any(value is not None and type(value) is not bool for value in rows.values()):
            raise ManifestError('outcomes must be true, false, or null (unknown)')
        normalized[candidate] = rows
    payload['outcomes'] = normalized
    return payload


def _genesis_payload(manifest):
    return {key: manifest[key] for key in ('config_hash', 'source_hashes_hash', 'partition_ids_hash')}


def _record(manifest, sequence, previous_hash, kind, payload):
    body = {'schema': EVENT_SCHEMA, 'sequence': sequence, 'previous_hash': previous_hash,
            'manifest_hash': manifest['manifest_hash'], 'kind': kind, 'payload': payload,
            'recorded_at_ns': time.time_ns()}
    return dict(body, event_hash=_digest(body))


def _read_events(root, manifest):
    """Validate every record before trusting a tail; no repair or truncation."""
    previous, count, sealed = ZERO_HASH, 0, False
    mutation_waves, router_waves = {}, {}
    fd = _open_file(root / EVENTS_NAME, os.O_RDONLY)
    with os.fdopen(fd, 'rb') as stream:
        for line in stream:
            record = _decode(line, 'event %d' % count)
            _fields(record, ('schema', 'sequence', 'previous_hash', 'manifest_hash',
                             'kind', 'payload', 'recorded_at_ns', 'event_hash'))
            if sealed:
                raise ManifestError('journal contains an event after the terminal report')
            if (record['schema'] != EVENT_SCHEMA or type(record['sequence']) is not int
                    or record['sequence'] != count or record['previous_hash'] != previous
                    or record['manifest_hash'] != manifest['manifest_hash']):
                raise ManifestError('event sequence, chain, schema, or manifest binding is corrupt')
            _nonnegative(record['recorded_at_ns'], 'recorded_at_ns')
            body = {key: value for key, value in record.items() if key != 'event_hash'}
            if record['event_hash'] != _digest(body):
                raise ManifestError('event hash does not match the saved record')
            if count == 0:
                if record['kind'] != 'manifest_frozen' or record['payload'] != _genesis_payload(manifest):
                    raise ManifestError('journal must start with the matching frozen manifest')
            else:
                payload = _event_payload(record['kind'], record['payload'], manifest)
                if _canonical(payload) != _canonical(record['payload']):
                    raise ManifestError('saved event payload is not normalized')
                _transition(record['kind'], payload, mutation_waves, router_waves)
            sealed = record['kind'] == 'report_evaluation'
            previous = record['event_hash']
            count += 1
    if not count:
        raise ManifestError('empty event journal: incomplete initialization or deleted evidence')
    return count, previous, sealed, mutation_waves, router_waves


def _require_store(root):
    if not _exists(root / MANIFEST_NAME) or not _exists(root / EVENTS_NAME):
        raise ManifestError('complete frozen manifest and event journal required; refuse automatic recovery')
    manifest = _read_manifest(root)
    return manifest, _read_events(root, manifest)


def read_manifest_status(root) -> dict:
    """Verify the existing store, including sealed reports, without writing it.

    Requires its existing lock file and acquires the same exclusive OS lock as
    writers. Creates no directories/files, appends no events, and does not alter
    the manifest. ``event_count`` includes the genesis record. ``sealed`` is a
    boolean; ``pending_mutation_waves`` and ``pending_router_waves`` name their
    respective competitions awaiting champions. ``latest_accepted_router_id``
    is the last settled, non-null router acceptance in journal order, or null
    if none exists. Rejected later waves never erase that accepted identity.
    It must match the saved active router before constructing workers; when
    null, the caller verifies its recomputed initial baseline router instead.
    ``router_expert_ids`` is the first competition's ordered roster or null;
    ``latest_router_wave`` is the maximum competition wave or null, including
    any pending wave which the caller must reject before resuming work.
    Status belongs to ``head_event_hash`` only: it is not a reservation for
    future work. Search callers must refuse sealed status before any scoring,
    maker call or saved-artifact mutation and retain exclusive run ownership.
    """
    with _locked(root) as root:
        manifest, (count, head_hash, sealed, mutation_waves, router_waves) = _require_store(root)
        accepted = [state['expected']['accepted_id'] for state in router_waves.values()
                    if state['settled'] and state['expected']['accepted_id'] is not None]
        router_experts = list(next(iter(router_waves.values()))['expert_ids']) if router_waves else None
        return {'schema': 'cellforge-evolution-status/1', 'manifest': manifest,
                'manifest_hash': manifest['manifest_hash'], 'event_count': count,
                'head_event_hash': head_hash, 'sealed': sealed,
                'pending_mutation_waves': sorted(wave for wave, state in mutation_waves.items()
                                                 if not state['settled']),
                'pending_router_waves': sorted(wave for wave, state in router_waves.items()
                                               if not state['settled']),
                'router_expert_ids': router_experts,
                'latest_router_wave': max(router_waves, default=None),
                'latest_accepted_router_id': accepted[-1] if accepted else None}


def freeze_manifest(root, config, source_hashes: dict[str, str],
                    partition_ids: dict[str, list[str]]) -> dict:
    """Bind the first explicit use, or verify an identical resume under lock.

    Empty partition lists are allowed to describe an unused population, but
    evaluation events need nonempty scopes. Existing manifests/journals are
    never overwritten. Both files must either be absent or validate together.
    The return value is the frozen manifest, independent of event-tail changes.
    A terminal report refuses a search resume even with identical inputs; use
    ``read_manifest_status`` for read-only verification of that closed store.
    """
    expected = _manifest(config, source_hashes, partition_ids)
    with _locked(root, create=True) as root:
        manifest_exists, events_exist = _exists(root / MANIFEST_NAME), _exists(root / EVENTS_NAME)
        if manifest_exists or events_exist:
            saved, (_count, _head, sealed, _mutations, _routers) = _require_store(root)
            if _canonical(saved) != _canonical(expected):
                raise ManifestError('resume changed config, source hashes, or partition IDs; use a new authorized root')
            if sealed:
                raise ManifestError('terminal report sealed this store; search resume is forbidden')
            return saved
        _write_record(root / MANIFEST_NAME, expected, create=True)
        genesis = _record(expected, 0, ZERO_HASH, 'manifest_frozen', _genesis_payload(expected))
        _write_record(root / EVENTS_NAME, genesis, create=True)
        return expected


def append_event(root, kind, payload) -> dict:
    """Append one validated event after verifying the complete frozen store.

    ``freeze_manifest`` must already have succeeded and must be called again
    by the integrator with current inputs before each resumed session. This
    API cannot inspect the caller's live source/config. Calls are serialized,
    not deduplicated; retain returned event_hash as an external audit anchor.
    """
    with _locked(root) as root:
        manifest, (sequence, previous_hash, sealed, mutation_waves, router_waves) = _require_store(root)
        if sealed:
            raise ManifestError('report evaluation sealed this store; no search/selection/reports may follow')
        payload = _event_payload(kind, payload, manifest)
        _transition(kind, payload, mutation_waves, router_waves)
        record = _record(manifest, sequence, previous_hash, kind, payload)
        _write_record(root / EVENTS_NAME, record)
        return record
