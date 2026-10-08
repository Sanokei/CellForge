"""Opt-in, finite cell/router epochs; one explicitly requested phase at a time.

``configuration`` and ``plan`` neither write nor import the driver. Execution
freezes the whole plan and reserves each child's *full* allowance before driver
construction. Unused calls are never transferred. An incomplete reservation,
child or receipt requires manual reconciliation; this module never repairs,
replaces or retries uncertain work. Existing campaigns cannot change plans.

The campaign lock serializes cooperating invocations across the complete phase.
The child worker and evolution journals retain their own locks and accounting.
Hashes detect inconsistency, not malicious rewriting or rollback of the entire
campaign. Filesystem locking must work; Windows lacks portable directory fsync.
STOP is cooperative and worker close cannot cancel already running remote work.
These are operational receipts, not scientific acceptance or execution consent.
"""

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

# A direct CLI preview must not create a policy __pycache__ as an incidental
# import write. Restore the caller's setting before any explicit execution.
_bytecode_setting = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    if __package__:
        from . import evolution_policy as EP
        from . import fresh_inputs as FI
    else:
        import evolution_policy as EP
        import fresh_inputs as FI
finally:
    sys.dont_write_bytecode = _bytecode_setting
    del _bytecode_setting


SCHEMA = 'cellforge-epochs/1'
PLAN_NAME = 'EPOCH_PLAN.json'
INDEX_NAME = 'PHASE_INDEX.json'
LOCK_NAME = '.evolution_epochs.lock'
CHECKPOINT_NAME = 'PHASE_CHECKPOINT.json'
COMPLETION_NAME = 'PHASE_COMPLETION.json'
KINDS = ('cells', 'router')


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('%s must be an integer in [%d, %d]' % (label, low, high))


def _options(value, keys, label):
    if type(value) is not dict or set(value) - set(keys):
        raise ValueError('unknown or invalid ' + label + ' fields')


def configuration(value=None):
    """Return a detached strict plan; disabled pools must have zero allowance."""
    value = {} if value is None else value
    _options(value, ('schema', 'enabled', 'epochs', 'max_total_calls', 'fresh_inputs', *KINDS), 'epoch plan')
    out = {'schema': SCHEMA, 'enabled': False, 'epochs': 1, 'max_total_calls': 0}
    for key in out:
        if key in value:
            out[key] = copy.deepcopy(value[key])
    if out['schema'] != SCHEMA or type(out['enabled']) is not bool:
        raise ValueError('unsupported epoch schema or nonboolean enabled flag')
    _integer(out['epochs'], 1, 8, 'epochs')
    _integer(out['max_total_calls'], 0, 16000, 'max_total_calls')
    if 'fresh_inputs' in value:
        out['fresh_inputs'] = FI.configuration(value['fresh_inputs'])
    for kind in KINDS:
        spec = value.get(kind, {})
        _options(spec, ('rounds', 'policy'), kind)
        rounds = spec.get('rounds', 1)
        _integer(rounds, 1, 12, kind + ' rounds')
        policy = copy.deepcopy(spec.get('policy', {}))
        _options(policy, EP.DEFAULT, kind + ' policy')
        # Merge the router-specific default before EP's normal defaults, while
        # preserving an explicitly contradictory flag so it is rejected below.
        if kind == 'router':
            router = policy.get('router', {})
            if type(router) is dict:
                policy['router'] = {'enabled': True, **router}
        policy = EP.configuration(policy)
        if policy['router']['enabled'] != (kind == 'router'):
            raise ValueError(kind + ' phase has the wrong router enabled flag')
        if kind == 'router' and (policy['niches'] or policy['curriculum']['enabled']
                                 or policy['parent_selection'] != 'legacy'):
            raise ValueError('router phases require no niches, curriculum or cell selection')
        worker = policy['workers']
        if not worker['enabled'] and worker['max_calls'] != 0:
            raise ValueError('disabled workers require max_calls=0')
        out[kind] = {'rounds': rounds, 'policy': policy}
    if _allocated(out) > out['max_total_calls']:
        raise ValueError('all phase allowances must fit max_total_calls without recycling')
    return out


def _allocated(config):
    return config['epochs'] * sum(config[k]['policy']['workers']['max_calls'] for k in KINDS)


def _phase(config, index):
    _integer(index, 0, 2 * config['epochs'] - 1, 'phase index')
    kind = KINDS[index % 2]
    return {'plan_hash': EP.digest(config), 'index': index, 'epoch': index // 2,
            'kind': kind, 'rounds': config[kind]['rounds']}


def _child_name(phase):
    return 'weights-phase-%04d-%s' % (phase['index'], phase['kind'])


def _root(value):
    root = Path(value).expanduser().absolute()
    if '..' in root.parts or not root.name.startswith('weights-') or root.name == 'weights-':
        raise ValueError('campaign must have a weights-* basename and no parent traversal')
    # Reject aliases, including Windows junctions, before resolving the path.
    for path in reversed((root, *root.parents)):
        _ordinary(path, directory=True, missing=True)
    return root.resolve()


def plan(root, value=None):
    """Describe the finite schedule without changing files or importing Driver.

    This is a configuration preview, not a campaign-status/reconciliation API.
    Pass the same explicit configuration to execute_phase for every invocation.
    """
    root, config = _root(root), configuration(value)
    phases = []
    for index in range(2 * config['epochs']):
        phase = _phase(config, index)
        phases.append({'phase': phase, 'child': 'phases/' + _child_name(phase),
                       'allocated_calls': config[phase['kind']]['policy']['workers']['max_calls']})
    return {'schema': 'cellforge-epoch-plan/1', 'campaign': str(root),
            'plan_hash': EP.digest(config), 'plan': config,
            'allocated_calls': _allocated(config), 'phases': phases}


def _ordinary(path, directory=False, missing=False):
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing:
            return False
        raise ValueError('missing campaign evidence: ' + str(path)) from None
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('ordinary %s required: %s' % ('directory' if directory else 'file', path))
    return True


def _present(path):
    # Even a dangling symlink named STOP blocks work; never follow it.
    return os.path.lexists(path)


def _stop(root, child=None):
    if _present(root / 'STOP') or (child is not None and _present(child / 'STOP')):
        raise RuntimeError('STOP prevents starting this phase')


def _unique(pairs):
    out = {}
    for key, item in pairs:
        if key in out:
            raise ValueError('duplicate JSON key: ' + key)
        out[key] = item
    return out


def _nonfinite(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def _decode(data):
    value = json.loads(data, object_pairs_hook=_unique, parse_constant=_nonfinite)
    EP.digest(value)  # Reject overflowed JSON numbers as well as NaN constants.
    return value


def _read(path):
    _ordinary(path)
    return _decode(path.read_bytes())


def _signed(body):
    return dict(body, hash=EP.digest(body))


def _expect(actual, expected, label):
    if EP.digest(actual) != EP.digest(expected):
        raise ValueError(label + ' does not match frozen campaign evidence')


def _sync_directory(path):
    if os.name == 'posix':
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic(path, value, replace=False):
    if _present(path):
        _ordinary(path)
        if not replace:
            raise ValueError('refusing to replace existing evidence: ' + str(path))
    temporary = path.with_name('.' + path.name + '.pending')
    # Leave an uncertain temporary write for manual reconciliation. Never erase
    # it to make a repeated invocation look like a first attempt.
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False) + '\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


@contextmanager
def _locked(root):
    if os.name == 'nt':
        import msvcrt
    elif os.name == 'posix':
        import fcntl
    else:
        raise RuntimeError('campaign locking is unsupported on this platform')
    root.mkdir(parents=True, exist_ok=True)
    _ordinary(root, directory=True)
    lock = root / LOCK_NAME
    exists = _ordinary(lock, missing=True)
    if not exists and any(root.iterdir()):
        raise ValueError('existing campaign lacks its persistent lock; manual reconciliation required')
    flags = os.O_RDWR | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(lock, flags | (0 if exists else os.O_CREAT | os.O_EXCL), 0o600)
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('campaign lock is not an ordinary file')
        try:
            if os.name == 'nt':
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('campaign is locked or locking failed; no phase may start') from error
        acquired = True
        if not exists:
            os.fsync(descriptor)
            _sync_directory(root)
        yield
    finally:
        try:
            if acquired:
                if os.name == 'nt':
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _index(config, reservations):
    return _signed({'schema': 'cellforge-phase-index/1', 'plan_hash': EP.digest(config),
                    'reservations': reservations})


def _reservation(config, index, predecessor):
    phase = _phase(config, index)
    return _signed({'schema': 'cellforge-phase-reservation/1', 'phase': phase,
                    'child': 'phases/' + _child_name(phase),
                    'predecessor_hash': None if predecessor is None else predecessor['hash'],
                    'allocated_calls': config[phase['kind']]['policy']['workers']['max_calls']})


def _store(root, config):
    frozen = _signed({'schema': 'cellforge-epoch-plan/1', 'plan_hash': EP.digest(config),
                      'plan': config, 'allocated_calls': _allocated(config)})
    names = {path.name for path in root.iterdir()}
    if names == {LOCK_NAME}:
        _atomic(root / PLAN_NAME, frozen)
        _atomic(root / INDEX_NAME, _index(config, []))
        (root / 'phases').mkdir()
        _sync_directory(root)
    elif names - {LOCK_NAME, PLAN_NAME, INDEX_NAME, 'phases', 'STOP'}:
        raise ValueError('unrecognized campaign contents; refusing retrofit or partial-write recovery')
    _expect(_read(root / PLAN_NAME), frozen, 'plan')
    index = _read(root / INDEX_NAME)
    if type(index) is not dict or type(index.get('reservations')) is not list:
        raise ValueError('invalid phase index')
    reservations = index['reservations']
    if len(reservations) > 2 * config['epochs']:
        raise ValueError('too many phase reservations')
    _expect(index, _index(config, reservations), 'phase index')
    _ordinary(root / 'phases', directory=True)
    return reservations


def _transitions():
    if __package__:
        from . import checkpoint_transitions
    else:
        import checkpoint_transitions
    return checkpoint_transitions


def _checkpoint(value, phase, predecessor):
    transitions = _transitions()
    checkpoint = transitions.validate_checkpoint(value)
    _expect(checkpoint['phase'], phase, 'checkpoint phase')
    # Reapply the transition, including carried cell_state and incumbent
    # anchors, rather than accepting an independently rehashed checkpoint.
    evidence = {key: checkpoint[key] for key in (
        'manifest_hash', 'journal_head_hash', 'scope_hash', 'archive_cells',
        'archive_bank', 'archive_outcomes', 'cell_state', 'worker_calls_reserved')}
    incumbent = predecessor if predecessor is not None else checkpoint
    evidence['incumbent_bank'] = incumbent['retained_bank']
    evidence['incumbent_outcomes'] = incumbent['retained_outcomes']
    expected = transitions.make_checkpoint(phase, predecessor, evidence)
    _expect(checkpoint, expected, 'checkpoint transition')
    return checkpoint


def _workers(child, policy):
    """Require settled, metered accounting without importing the maker route.

    A durable halt cannot be cleared by starting the next phase's fresh pool.
    Ordinary max_calls exhaustion records no halt and remains completable.
    """
    path = child / 'WORKER_PROPOSALS.jsonl'
    works = {p.name for p in child.glob('AUTHOR-WORKER-*')}
    if not _present(path):
        if works:
            raise ValueError('worker directories without a reservation ledger')
        return 0, None
    _ordinary(path)
    tail, count, attempts = '0' * 64, 0, {}
    meter_baseline = None
    file_hash = hashlib.sha256()
    with path.open('rb') as stream:
        for sequence in range(1, 4003):
            line = stream.readline(1024 * 1024 + 2)
            if not line:
                break
            if sequence > 4001 or len(line) > 1024 * 1024 + 1 or not line.endswith(b'\n'):
                raise ValueError('torn or oversized worker accounting')
            file_hash.update(line)
            row = _decode(line)
            if type(row) is not dict:
                raise ValueError('invalid worker accounting record')
            signature = row.pop('record_sha256', None)
            if (row.get('schema') != 'cellforge-weight-worker/2'
                    or type(row.get('sequence')) is not int or row['sequence'] != sequence
                    or row.get('previous_sha256') != tail or signature != EP.digest(row)):
                raise ValueError('worker accounting hash chain mismatch')
            tail = signature
            event, call = row.get('event'), row.get('call')
            if event == 'reserved':
                _integer(call, count + 1, count + 1, 'worker reservation')
                if row.get('config_sha256') != EP.digest(policy['workers']):
                    raise ValueError('worker reservation changed its frozen configuration')
                for key, low, high in (('usage_start', 0, 100), ('resets_at', 1, 1e12)):
                    value = row.get(key)
                    if type(value) not in (int, float) or not low <= value <= high:
                        raise ValueError('invalid worker reservation meter baseline')
                baseline = (row['usage_start'], row['resets_at'])
                if meter_baseline is not None and baseline != meter_baseline:
                    raise ValueError('worker reservations changed the pinned meter baseline')
                meter_baseline = baseline
                work = row.get('work')
                if (type(work) is not str or not work.startswith('AUTHOR-WORKER-')
                        or Path(work).name != work or '/' in work or '\\' in work):
                    raise ValueError('invalid worker directory identity')
                count = call
                _integer(count, 0, policy['workers']['max_calls'], 'reserved worker calls')
                attempts[call] = {'work': work, 'finished': False, 'dispatched': False}
            elif event in ('dispatched', 'finished'):
                _integer(call, 1, count, 'worker event call')
                attempt = attempts[call]
                if attempt['finished']:
                    raise ValueError('post-completion worker event')
                if event == 'dispatched':
                    if attempt['dispatched']:
                        raise ValueError('duplicate worker dispatch')
                    attempt['dispatched'] = True
                else:
                    if type(row.get('submitted')) is not bool:
                        raise ValueError('uncertain worker completion requires reconciliation')
                    if row.get('submission_uncertain', False) is not False or 'meter_error_type' in row:
                        raise ValueError('uncertain or failed worker metering requires reconciliation')
                    if row['submitted']:
                        if (not attempt['dispatched'] or row.get('metered') is not True
                                or row.get('outcome') not in ('valid', 'refused', 'invalid_or_failed')):
                            raise ValueError('submitted worker attempt lacks a settled metered completion')
                        meter = row.get('meter')
                        if (type(meter) is not dict or set(meter) != {'used_percent', 'resets_at'}
                                or type(meter['used_percent']) not in (int, float)
                                or not meter_baseline[0] <= meter['used_percent'] <= 100
                                or type(meter['resets_at']) not in (int, float)
                                or meter['resets_at'] != meter_baseline[1]):
                            raise ValueError('worker completion meter differs from its pinned baseline')
                    elif (attempt['dispatched'] or row.get('outcome') != 'not_submitted'
                          or row.get('metered') is not False or 'meter' in row):
                        raise ValueError('only a conclusively undispatched reservation may lack a meter')
                    attempt['finished'] = True
            elif event == 'halted':
                if call is not None:
                    _integer(call, 1, count, 'halted worker call')
                # STOP, missing/stale meter, budget refusal, timeout, executor
                # or ledger safety failures all need manual reconciliation.
                # Ordinary allowance exhaustion emits no halted record.
                raise ValueError('durable worker safety halt blocks phase completion and successors')
            else:
                raise ValueError('unknown worker accounting event')
    expected = {attempt['work'] for attempt in attempts.values()}
    if len(expected) != count or works != expected or any(not a['finished'] for a in attempts.values()):
        raise ValueError('incomplete or mismatched worker accounting; manual reconciliation required')
    for work in expected:
        _ordinary(child / work, directory=True)
    return count, file_hash.hexdigest()


def _completion(child, config, reservation, checkpoint, predecessor):
    if __package__:
        from . import evolution_manifest as manifest
    else:
        import evolution_manifest as manifest
    policy = config[reservation['phase']['kind']]['policy']
    _expect(_read(child / 'EVOLUTION_POLICY.json'), policy, 'child policy')
    context = {'schema': 'cellforge-phase-context/1', 'phase': reservation['phase'],
               'predecessor': predecessor}
    _expect(_read(child / 'PHASE_CONTEXT.json'), context, 'child phase context')
    status = manifest.read_manifest_status(child)
    frozen = status['manifest']
    _expect(frozen['config'], policy, 'manifest policy')
    if frozen['source_hashes'].get('phase/context.json') != EP.digest(context):
        raise ValueError('child manifest does not bind the exact phase context and predecessor')
    scope = {'sources': {key: value for key, value in frozen['source_hashes'].items()
                         if not key.startswith('phase/')}, 'partitions': frozen['partition_ids']}
    if checkpoint['scope_hash'] != EP.digest(scope):
        raise ValueError('checkpoint scope differs from frozen sources and task partitions')
    for panel in ('search_fitness', 'development_selection'):
        for field in ('archive_outcomes', 'retained_outcomes'):
            if sorted(checkpoint[field][panel]) != frozen['partition_ids'][panel]:
                raise ValueError('checkpoint must cover the full frozen development panels')
    if status['sealed'] or status['pending_mutation_waves'] or status['pending_router_waves']:
        raise ValueError('sealed or incomplete child evolution journal')
    if (checkpoint['manifest_hash'] != status['manifest_hash']
            or checkpoint['journal_head_hash'] != status['head_event_hash']):
        raise ValueError('checkpoint does not bind the current child manifest and journal')
    calls, ledger_hash = _workers(child, policy)
    if checkpoint['worker_calls_reserved'] != calls:
        raise ValueError('checkpoint calls differ from durable worker reservations')
    return _signed({'schema': 'cellforge-phase-completion/1',
                    'phase': reservation['phase'], 'child': reservation['child'],
                    'reservation_hash': reservation['hash'],
                    'allocated_calls': reservation['allocated_calls'],
                    'manifest_hash': checkpoint['manifest_hash'],
                    'journal_head_hash': checkpoint['journal_head_hash'],
                    'checkpoint_hash': checkpoint['hash'],
                    'worker_calls_reserved': calls, 'worker_ledger_sha256': ledger_hash})


def _history(root, config, reservations):
    children = {path.name for path in (root / 'phases').iterdir()}
    expected = {_child_name(_phase(config, index)) for index in range(len(reservations))}
    if children != expected:
        raise ValueError('reserved children missing or unexpected children present; reconcile manually')
    predecessor, allocated, calls = None, 0, 0
    for index, reservation in enumerate(reservations):
        _expect(reservation, _reservation(config, index, predecessor), 'phase reservation')
        allocated += reservation['allocated_calls']
        _integer(allocated, 0, config['max_total_calls'], 'campaign reserved allowances')
        child = root / reservation['child']
        _ordinary(child, directory=True)
        if not _present(child / CHECKPOINT_NAME) or not _present(child / COMPLETION_NAME):
            raise ValueError('incomplete reserved phase; no automatic retry or replacement: ' + child.name)
        checkpoint = _checkpoint(_read(child / CHECKPOINT_NAME), reservation['phase'], predecessor)
        completion = _completion(child, config, reservation, checkpoint, predecessor)
        _expect(_read(child / COMPLETION_NAME), completion, 'phase completion')
        calls += completion['worker_calls_reserved']
        _integer(calls, 0, allocated, 'campaign actual reserved worker calls')
        predecessor = checkpoint
    return predecessor, allocated, calls


def execute_phase(root, value, index):
    """Execute exactly the next requested phase, returning a checkpoint or None.

    None/exception leaves the reservation in place and blocks every later
    invocation. There is deliberately no resume, force, overwrite or retry flag.
    """
    config, root = configuration(value), _root(root)
    phase = _phase(config, index)
    if not config['enabled']:
        raise ValueError('explicit execution also requires enabled=true in the frozen plan')
    _stop(root)
    # Execution modules use sibling imports. Keep that repository convention
    # confined to this opt-in path; configuration/plan never alter sys.path.
    scripts_path = str(Path(__file__).resolve().parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    if 'fresh_inputs' in config:
        # Missing or altered local task/seed inputs fail before a reservation.
        FI.load(config['fresh_inputs'])
    with _locked(root):
        _stop(root)
        # Do not create even a frozen campaign when a later phase was requested
        # without its predecessors.
        if not _present(root / PLAN_NAME) and index != 0:
            raise ValueError('a new campaign must begin with phase index 0')
        reservations = _store(root, config)
        predecessor, allocated, calls = _history(root, config, reservations)
        if index != len(reservations):
            raise ValueError('execute only the next unreserved index; completed phases cannot be replayed')
        reservation = _reservation(config, index, predecessor)
        _integer(allocated + reservation['allocated_calls'], 0, config['max_total_calls'],
                 'campaign reserved allowances including this phase')
        child = root / reservation['child']
        _stop(root, child)
        # Reserve before mkdir/import/constructor: failures cannot erase or
        # release this allowance, even when actual dispatch is uncertain.
        _atomic(root / INDEX_NAME, _index(config, reservations + [reservation]), replace=True)
        child.mkdir()
        _sync_directory(child.parent)
        _stop(root, child)
        if __package__:
            from .head_driver import Driver
        else:
            from head_driver import Driver
        _stop(root, child)
        driver = None
        try:
            driver = Driver(child, evolution=copy.deepcopy(config[phase['kind']]['policy']),
                            fresh_inputs=copy.deepcopy(config.get('fresh_inputs')),
                            phase_context={'schema': 'cellforge-phase-context/1',
                                           'phase': copy.deepcopy(phase),
                                           'predecessor': copy.deepcopy(predecessor)})
            result = driver.run_phase()
        finally:
            if driver is not None and getattr(driver, 'worker_pool', None) is not None:
                driver.worker_pool.close()
        if result is None:
            return None
        checkpoint = _checkpoint(result, phase, predecessor)
        completion = _completion(child, config, reservation, checkpoint, predecessor)
        if getattr(driver, 'worker_pool', None) is not None:
            _integer(driver.worker_pool.calls, 0, reservation['allocated_calls'], 'live worker calls')
            if driver.worker_pool.calls != completion['worker_calls_reserved']:
                raise ValueError('live worker calls differ from durable completion accounting')
        _integer(calls + completion['worker_calls_reserved'], 0, config['max_total_calls'],
                 'campaign actual reserved worker calls')
        _atomic(child / CHECKPOINT_NAME, checkpoint)
        # A crash between these atomic writes remains incomplete, never an
        # implicit permission to rerun a paid phase or start its successor.
        _atomic(child / COMPLETION_NAME, completion)
        return copy.deepcopy(checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path, help='campaign directory with a weights-* basename')
    parser.add_argument('--config', type=Path, help='strict epoch-plan JSON; defaults are disabled')
    parser.add_argument('--execute-phase', type=int, metavar='INDEX',
                        help='explicitly execute only this next phase; omitted means read-only preview')
    args = parser.parse_args(argv)
    try:
        value = _read(args.config) if args.config is not None else None
        if args.execute_phase is None:
            result = plan(args.campaign, value)
        else:
            result = execute_phase(args.campaign, value, args.execute_phase)
            if result is None:
                print('Phase incomplete; allowance remains reserved. Manual reconciliation required.')
                return 2
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, 'evolution_epochs: %s\n' % error)


if __name__ == '__main__':
    raise SystemExit(main())
