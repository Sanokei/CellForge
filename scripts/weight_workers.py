"""Bounded literal-weight proposals through the existing metered maker route.

One pool owns a run directory at a time. Every fsynced reservation consumes a
call and all requested positions, even if dispatch, parsing or validation fails.
attempts_for(tag) exposes this accounting independently of released tensors.
Unfinished reservations are never retried automatically after restart.
Host-provided stop_paths add exact STOP-file paths to root/STOP; ancestors are
never searched. status() distinguishes call-limit exhaustion from safety halts.

Only the host callback knows tensor shapes; this module never evaluates cells.
Timeouts limit caller waiting and are passed to maker.ask, but that API cannot
forcibly cancel Python threads, host callbacks or remote activity. A timed-out
thread retains its slot and directory lease until it actually exits; interpreter
shutdown may also wait for it. maker.ask parses responses before returning;
these size limits cannot bound its parsing memory, output files, filesystem
latency or subprocess descendants. Shared usage is a retrospective meter, not
an atomic account-wide quota reservation: an in-flight call can cross a cap.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import maker


_SCHEMA = 'cellforge-weight-worker/2'
_CONFIG_FIELDS = {'enabled', 'model', 'reasoning', 'parallelism',
                  'proposals_per_prompt', 'max_calls', 'timeout_seconds'}
_MAX_PROMPT_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_POSITIONS = 1000000
_REFUSALS = ('no valid', 'no satisfying', 'no exact', 'no revision', 'no tensors',
             'no such', 'cannot', "can't", 'unable to', 'i cannot', "i can't",
             'i am unable', 'i refuse')


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('%s must be an integer in [%s, %s]' % (label, low, high))


def _text(value, limit, label):
    if type(value) is not str or not value or len(value.encode('utf-8')) > limit:
        raise ValueError('%s must be a nonempty bounded string' % label)


def _finite(value, low, high):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def _encoded(value, limit):
    chunks, size = [], 0
    for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        data = chunk.encode('utf-8')
        size += len(data)
        if size > limit:
            raise ValueError('JSON exceeds size limit')
        chunks.append(data)
    return b''.join(chunks)


def _digest(value):
    return hashlib.sha256(_encoded(value, _MAX_RECORD_BYTES)).hexdigest()


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate ledger key')
        out[key] = value
    return out


def _no_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def _sync_directory(path):
    # Windows has no portable directory fsync in the standard library.
    if os.name != 'nt':
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_once(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _templates(value):
    if type(value) is not dict or not 1 <= len(value) <= 256:
        raise ValueError('one to 256 trusted task/template bindings required')
    for task, template in value.items():
        _text(task, 256, 'task ID')
        _text(template, 256, 'template name')


class _Refusal(ValueError):
    pass


class WeightWorkers:
    def __init__(self, root, config, stop_paths=()):
        if type(config) is not dict or set(config) != _CONFIG_FIELDS:
            raise ValueError('exact worker configuration fields required')
        if type(config['enabled']) is not bool:
            raise ValueError('enabled must be boolean')
        if config['model'] != 'gpt-6-luna' or config['reasoning'] not in ('low', 'medium', 'high'):
            raise ValueError('workers require the configured gpt-6-luna route')
        for key, low, high in (('parallelism', 1, 3), ('proposals_per_prompt', 1, 8),
                               ('max_calls', 1 if config['enabled'] else 0, 1000),
                               ('timeout_seconds', 1, 900)):
            _integer(config[key], low, high, key)
        self.root = Path(root).resolve()
        if isinstance(stop_paths, (str, bytes, os.PathLike)):
            raise ValueError('stop_paths must be an iterable of explicit paths')
        paths = [self.root / 'STOP']
        for index, path in enumerate(stop_paths):
            if index >= 16 or not isinstance(path, (str, os.PathLike)):
                raise ValueError('at most 16 explicit STOP-file paths allowed')
            _text(os.fspath(path), 32768, 'STOP-file path')
            # Resolve the parent only: the marker itself may be a dangling
            # symlink, whose presence must still stop the pool.
            marker = Path(path)
            paths.append(marker.parent.resolve() / marker.name)
        self.stop_paths = tuple(dict.fromkeys(paths))
        self.config = copy.deepcopy(config)
        self._config_hash = _digest(self.config)
        self.lock = threading.RLock()
        self._wave_lock = threading.Lock()
        self.ledger = self.root / 'WORKER_PROPOSALS.jsonl'
        self.calls = 0
        self.start_usage = self.resets_at = self._usage_peak = None
        self._attempts = {}
        self._sequence, self._tail = 0, '0' * 64
        self._halt_reason = None
        self._halt_durable = False
        self._halt_stop_path = None
        self._ledger_failed = False
        self._closed, self._inflight = False, 0
        self._lease = None
        self.root.mkdir(parents=True, exist_ok=True)
        # Never delete/steal the lock file based on a PID or age. The kernel
        # releases this lease on process exit, including crashes.
        lease = (self.root / 'WORKER_PROPOSALS.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                lease.seek(0)
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lease = lease
            self._load()
            if any(not row['finished'] for row in self._attempts.values()):
                self._halt('interrupted_reservation')
            elif any(row['submitted'] is None or row.get('submission_uncertain', False)
                     for row in self._attempts.values()):
                self._halt('uncertain_submission')
            self._executor = ThreadPoolExecutor(max_workers=config['parallelism'],
                                                thread_name_prefix='weight-worker')
        except BaseException:
            lease.close()
            self._lease = None
            raise

    def _apply(self, row):
        event = row.get('event')
        if event == 'reserved':
            call = row.get('call')
            _integer(call, 1, 1000, 'reserved call')
            if call != self.calls + 1 or row.get('config_sha256') != self._config_hash:
                raise ValueError('nonsequential reservation or changed worker configuration')
            _templates(row.get('templates'))
            _integer(row.get('charged_positions'), 1, _MAX_POSITIONS, 'charged positions')
            _text(row.get('wave'), 256, 'wave')
            saved_paths = row.get('stop_paths', [str(self.root / 'STOP')])
            if (type(saved_paths) is not list or not 1 <= len(saved_paths) <= 17
                    or any(type(path) is not str for path in saved_paths)
                    or not set(saved_paths).issubset({str(path) for path in self.stop_paths})):
                raise ValueError('resume cannot remove recorded STOP paths')
            _text(row.get('work'), 100, 'worker directory')
            if Path(row['work']).name != row['work'] or not row['work'].startswith('AUTHOR-WORKER-'):
                raise ValueError('invalid worker directory name')
            if not _finite(row.get('usage_start'), 0, 100) or not _finite(row.get('resets_at'), 1, 1e12):
                raise ValueError('invalid reservation meter period')
            if self.start_usage is not None and (row['usage_start'], row['resets_at']) != (self.start_usage, self.resets_at):
                raise ValueError('reservation changed the pinned meter baseline')
            self.start_usage, self.resets_at = row['usage_start'], row['resets_at']
            if self._usage_peak is None:
                self._usage_peak = self.start_usage
            self.calls = call
            self._attempts[call] = dict(copy.deepcopy(row), submitted=False,
                                        finished=False, outcome='reserved')
        elif event in ('dispatched', 'finished'):
            call = row.get('call')
            _integer(call, 1, self.calls, 'event call')
            attempt = self._attempts[call]
            if attempt['finished']:
                raise ValueError('duplicate or post-completion attempt event')
            if event == 'dispatched':
                if attempt['submitted'] is not False:
                    raise ValueError('duplicate dispatch')
                attempt.update(submitted=None, outcome='dispatching')
            else:
                if row.get('submitted') is not None and type(row.get('submitted')) is not bool:
                    raise ValueError('invalid submission state')
                _text(row.get('outcome'), 100, 'outcome')
                if row.get('metered') is True:
                    meter = row.get('meter')
                    if (type(meter) is not dict or not _finite(meter.get('used_percent'), self.start_usage, 100)
                            or meter.get('resets_at') != self.resets_at):
                        raise ValueError('invalid completed-attempt meter')
                    self._usage_peak = max(self._usage_peak, meter['used_percent'])
                attempt.update({k: copy.deepcopy(v) for k, v in row.items()
                                if k not in ('event', 'schema', 'sequence', 'previous_sha256', 'record_sha256')})
                attempt['finished'] = True
        elif event == 'halted':
            _text(row.get('reason'), 100, 'halt reason')
            self._halt_reason = row['reason']
            self._halt_durable = True
            self._halt_stop_path = row.get('stop_path')
            call = row.get('call')
            if call is not None:
                _integer(call, 1, self.calls, 'halted call')
                if not self._attempts[call]['finished']:
                    self._attempts[call]['outcome'] = row['reason']
        else:
            raise ValueError('unknown worker ledger event')

    def _load(self):
        if not self.ledger.exists():
            if any(self.root.glob('AUTHOR-WORKER-*')):
                raise ValueError('worker directories without a reservation ledger; refusing a budget reset')
            return
        with self.ledger.open('rb') as stream:
            while True:
                line = stream.readline(_MAX_RECORD_BYTES + 2)
                if not line:
                    break
                if len(line) > _MAX_RECORD_BYTES + 1 or not line.endswith(b'\n') or self._sequence >= 4001:
                    raise ValueError('torn or oversized worker ledger; manual review required')
                row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_no_constant)
                if type(row) is not dict or row.get('schema') != _SCHEMA:
                    raise ValueError('unsupported worker ledger; no automatic migration/reset')
                signature = row.pop('record_sha256', None)
                if (type(row.get('sequence')) is not int or row['sequence'] != self._sequence + 1
                        or row.get('previous_sha256') != self._tail or signature != _digest(row)):
                    raise ValueError('worker ledger chain mismatch')
                row['record_sha256'] = signature
                self._apply(row)
                self._sequence, self._tail = row['sequence'], signature
        expected = {row['work'] for row in self._attempts.values()}
        actual = {path.name for path in self.root.glob('AUTHOR-WORKER-*')}
        if actual != expected:
            raise ValueError('worker directories differ from ledger; refusing ambiguous recovery')

    def _append(self, event):
        # A failed fsync is ambiguous: halt in this process and release nothing.
        if self._ledger_failed:
            raise RuntimeError('worker ledger is unavailable')
        row = dict(event, schema=_SCHEMA, sequence=self._sequence + 1,
                   previous_sha256=self._tail)
        row['record_sha256'] = _digest(row)
        try:
            data = _encoded(row, _MAX_RECORD_BYTES) + b'\n'
            with self.ledger.open('ab') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(self.root)
            self._apply(row)
            self._sequence, self._tail = row['sequence'], row['record_sha256']
        except BaseException:
            self._ledger_failed = True
            self._halt_reason = 'ledger_failure'
            self._halt_durable = False
            raise

    def _halt(self, reason, call=None, stop_path=None):
        if self._halt_reason is None:
            if self._lease is None:
                self._halt_reason = reason
                self._halt_stop_path = stop_path
                self._halt_durable = False
            else:
                self._append({'event': 'halted', 'reason': reason, 'call': call,
                              'stop_path': stop_path, 'utc': maker.now()})

    def _blocked(self):
        for path in self.stop_paths:
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                self._halt('stop_check_failed', stop_path=str(path))
                break
            self._halt('stop_requested', stop_path=str(path))
            break
        return self._closed or self._halt_reason is not None

    def stopped(self):
        return self.status()['stopped']

    def safety_halt(self):
        """Return the persistent safety reason, or None for normal exhaustion."""
        return self.status()['halt_reason']

    def status(self):
        """Snapshot after checking every explicitly registered STOP path.

        Call exhaustion alone is normal (safety_halt=False). A halted ledger
        event survives restart/removal of the STOP file. When safety_halt=True,
        halt_durable=False identifies a storage failure or a STOP first observed
        after close(); neither is evidence of safe phase completion. Calls can still
        fail after this snapshot, so completion also requires in_flight == 0.
        """
        with self.lock:
            blocked = self._blocked()
            exhausted = self.calls >= self.config['max_calls']
            return {'calls': self.calls, 'max_calls': self.config['max_calls'],
                    'exhausted': exhausted, 'in_flight': self._inflight,
                    'closed': self._closed,
                    'stopped': bool(not self.config['enabled'] or blocked or exhausted),
                    'safety_halt': self._halt_reason is not None,
                    'halt_reason': self._halt_reason, 'halt_durable': self._halt_durable,
                    'stop_path': self._halt_stop_path}

    def attempts_for(self, tag):
        """Detached snapshots for ALL reservations in this wave, in call order.

        submitted=False means no submission is known; None means dispatch may
        have occurred (including crashes/timeouts); True means ask returned.
        None never means a free attempt. Charge every reservation and match
        controls for every task in templates, including omitted responses.
        outcome='valid' means schema-valid, not improved or even changed.
        Live timed-out snapshots can later acquire completion details.
        """
        with self.lock:
            return [copy.deepcopy(row) for row in self._attempts.values() if row['wave'] == tag]

    def close(self):
        """Stop new work without claiming running-thread cancellation."""
        with self.lock:
            self._blocked()
            self._closed = True
            if not self._inflight and self._lease is not None:
                self._lease.close()
                self._lease = None
        self._executor.shutdown(wait=False)

    def _budget(self):
        used, reset = maker.last_used()
        # maker.cap() silently defaults after malformed input; a worker must
        # instead refuse if an existing owner cap file cannot be read strictly.
        if maker.CAP_FILE.exists():
            with maker.CAP_FILE.open('rb') as stream:
                data = stream.read(4097)
            if len(data) > 4096:
                raise maker.BudgetExhausted('oversized owner budget cap')
            cap = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_no_constant)['stop_at_percent']
        else:
            cap = maker.DEFAULT_CAP
        if (not _finite(used, 0, 100) or not _finite(reset, 1, 1e12)
                or not _finite(cap, 0, 100) or reset <= time.time() or used >= cap):
            raise maker.BudgetExhausted('a current finite shared meter below cap is required')
        if self.resets_at is not None and (reset != self.resets_at or used < self._usage_peak):
            raise maker.BudgetExhausted('meter period changed or usage moved backwards')
        if self.start_usage is not None and used - self.start_usage >= 1.0:
            raise maker.BudgetExhausted('worker pool usage-increase ceiling reached')
        return used, reset

    def _reserve(self, prompt, tag, templates, positions):
        with self.lock:
            if self.stopped():
                return None
            try:
                used, reset = self._budget()
            except Exception:
                self._halt('budget_unavailable')
                return None
            if self._blocked():
                return None
            call = self.calls + 1
            work = self.root / ('AUTHOR-WORKER-%05d-%s' % (call, uuid.uuid4().hex))
            work.mkdir()
            _write_once(work / 'PROMPT.txt', prompt.encode('utf-8'))
            reservation = {
                'event': 'reserved', 'call': call, 'wave': tag, 'utc': maker.now(),
                'operator': 'literal_weight_proposal', 'config_sha256': self._config_hash,
                'model': self.config['model'], 'reasoning': self.config['reasoning'],
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'templates': copy.deepcopy(templates), 'charged_positions': positions,
                'stop_paths': [str(path) for path in self.stop_paths],
                'usage_start': used if self.start_usage is None else self.start_usage,
                'resets_at': reset, 'work': work.name,
            }
            self._append(reservation)
            return reservation

    def _meter_receipt(self, reservation):
        # A last_used() value from another concurrent call is insufficient.
        rows = [row for row in maker.readings()
                if row.get('work') == reservation['work']
                and row.get('prompt_sha256') == reservation['prompt_sha256']
                and row.get('track') == 'cellforge-weight-pool' and row.get('role') == 'writer']
        if len(rows) != 1:
            raise ValueError('missing or ambiguous attempt-specific maker meter')
        row = rows[0]
        if (not _finite(row.get('used_percent'), self.start_usage, 100)
                or not _finite(row.get('resets_at'), 1, 1e12) or row['resets_at'] != self.resets_at):
            raise ValueError('unmetered attempt or changed meter period')
        return {'used_percent': row['used_percent'], 'resets_at': row['resets_at']}

    def _response(self, work, raw):
        # maker.ask has already parsed JSON permissively. Re-read its saved
        # response to reject duplicate keys and nonfinite constants lost during
        # that parse. Only literal JSON is accepted here, without code fences.
        with (work / 'response.json').open('rb') as stream:
            source = stream.read(_MAX_RESPONSE_BYTES + 1)
        if len(source) > _MAX_RESPONSE_BYTES:
            raise ValueError('source response exceeds size limit')
        strict = json.loads(source, object_pairs_hook=_unique_object, parse_constant=_no_constant)
        data = _encoded(strict, _MAX_RESPONSE_BYTES)
        if data != _encoded(raw, _MAX_RESPONSE_BYTES):
            raise ValueError('maker return differs from saved response')
        return strict, data, hashlib.sha256(source).hexdigest()

    def _validate(self, raw, templates, validate, positions):
        if raw is None or raw == [] or type(raw) is str or (type(raw) is dict and 'refusal' in raw):
            raise _Refusal('no tensor proposals')
        if type(raw) is not list or not 1 <= len(raw) <= len(templates):
            raise ValueError('bounded JSON array of task-specific tensor proposals required')
        seen, leaves, nodes = set(), 0, 0
        for entry in raw:
            if type(entry) is not dict or set(entry) != {'task_id', 'params', 'rationale'}:
                raise ValueError('exact task_id/params/rationale fields required')
            tid, params, rationale = entry['task_id'], entry['params'], entry['rationale']
            if type(tid) is not str or tid not in templates or tid in seen:
                raise ValueError('unknown or duplicate proposal task ID')
            seen.add(tid)
            if type(rationale) is not str or len(rationale.encode('utf-8')) > 4096:
                raise ValueError('bounded rationale string required')
            if rationale.lstrip().lower().startswith(_REFUSALS):
                raise _Refusal('maker declined a tensor proposal')
            if type(params) is not dict or not 1 <= len(params) <= 256:
                raise ValueError('bounded nonempty tensor map required')
            stack = []
            for name, value in params.items():
                _text(name, 256, 'tensor name')
                if type(value) is not list:
                    raise ValueError('each tensor must be a nested JSON list')
                stack.append((value, 1))
            while stack:
                value, depth = stack.pop()
                nodes += 1
                if depth > 8 or nodes > positions * 10 + 256 * len(templates):
                    raise ValueError('tensor nesting/node limit exceeded')
                if type(value) is list:
                    if not value or len(value) > positions or len(stack) + len(value) > positions * 10:
                        raise ValueError('empty or oversized tensor list')
                    stack.extend((item, depth + 1) for item in value)
                else:
                    if not _finite(value, -9, 9):
                        raise ValueError('only finite literal numbers in [-9, 9] allowed')
                    leaves += 1
                    if leaves > positions:
                        raise ValueError('response exceeds reserved tensor positions')
            # The callback sees a copy and must explicitly return a Python bool.
            if validate(templates[tid], copy.deepcopy(params)) is not True:
                raise ValueError('proposal failed trusted tensor schema validation')
        return copy.deepcopy(raw), leaves

    def _attempt(self, prompt, reservation, validate, deadline):
        call, work = reservation['call'], self.root / reservation['work']
        proposals, submitted, dispatched = [], False, False
        outcome = {'event': 'finished', 'call': call, 'outcome': 'not_submitted',
                   'proposals': 0, 'submitted': False, 'metered': False}
        try:
            with self.lock:
                if self._blocked() or time.monotonic() >= deadline:
                    return []
                try:
                    self._budget()
                except Exception:
                    self._halt('budget_unavailable', call)
                    return []
                if self._blocked():
                    return []
                self._append({'event': 'dispatched', 'call': call, 'utc': maker.now()})
                dispatched, submitted = True, None
                if self._blocked():
                    submitted = False
                    return []
            try:
                raw = maker.ask(prompt, work, role='writer', track='cellforge-weight-pool',
                                model=self.config['model'], effort=self.config['reasoning'],
                                timeout=max(0.001, deadline - time.monotonic()))
                submitted = True
                raw, data, source_hash = self._response(work, raw)
                _write_once(work / 'RAW_PROPOSAL.json', data)
                outcome['raw_response_sha256'] = hashlib.sha256(data).hexdigest()
                outcome['source_response_sha256'] = source_hash
                proposals, count = self._validate(raw, reservation['templates'], validate,
                                                   reservation['charged_positions'])
                outcome.update(outcome='valid', returned_positions=count,
                               returned_task_ids=[entry['task_id'] for entry in proposals])
            except maker.BudgetExhausted:
                if submitted is True:
                    # A host validator exception cannot erase a real dispatch.
                    outcome.update(outcome='invalid_or_failed', error_type='BudgetExhausted')
                else:
                    submitted = False  # maker.ask refuses before its route.
                    outcome['outcome'] = 'budget_refused'
                    with self.lock:
                        self._halt('budget_refused', call)
            except subprocess.TimeoutExpired:
                outcome['outcome'] = 'timed_out'
                with self.lock:
                    self._halt('timed_out', call)
            except _Refusal:
                outcome['outcome'] = 'refused'
            except Exception as error:
                outcome.update(outcome='invalid_or_failed', error_type=type(error).__name__)
            if submitted is not False:
                try:
                    outcome['meter'] = self._meter_receipt(reservation)
                    outcome['metered'] = True
                    with self.lock:
                        self._usage_peak = max(self._usage_peak, outcome['meter']['used_percent'])
                        self._budget()
                except Exception as error:
                    proposals = []
                    outcome.update(outcome='unmetered_or_budget_stopped', meter_error_type=type(error).__name__)
                    with self.lock:
                        self._halt('unmetered_or_budget_stopped', call)
        finally:
            with self.lock:
                try:
                    if time.monotonic() >= deadline:
                        proposals = []
                        outcome['outcome'] = 'timed_out'
                        self._halt('timed_out', call)
                    elif self._blocked():
                        proposals = []
                        if outcome['outcome'] == 'valid':
                            outcome['outcome'] = 'stopped'
                    if outcome['outcome'] != 'valid':
                        proposals = []
                    outcome.update(submitted=submitted, proposals=len(proposals), utc=maker.now())
                    if dispatched and submitted is None:
                        outcome['submission_uncertain'] = True
                        self._halt('uncertain_submission', call)
                    self._append(outcome)
                finally:
                    self._inflight -= 1
                    if self._closed and not self._inflight and self._lease is not None:
                        self._lease.close()
                        self._lease = None
        for entry in proposals:
            entry['_worker_receipt'] = {
                'call': call, 'wave': reservation['wave'], 'model': self.config['model'],
                'prompt_sha256': reservation['prompt_sha256'],
                'reservation_sha256': self._attempts[call]['record_sha256'],
                'raw_response': str(Path(reservation['work']) / 'RAW_PROPOSAL.json'),
                'raw_response_sha256': outcome['raw_response_sha256'],
            }
        return proposals

    def _submit(self, prompt, tag, templates, validate, positions):
        # Reservation and executor submission share the lock with close(), so
        # the directory lease cannot be released between these operations.
        with self.lock:
            reservation = self._reserve(prompt, tag, templates, positions)
            if reservation is None:
                return None
            deadline = time.monotonic() + self.config['timeout_seconds']
            self._inflight += 1
            try:
                future = self._executor.submit(self._attempt, prompt, reservation, validate, deadline)
            except Exception as error:
                self._inflight -= 1
                self._append({'event': 'finished', 'call': reservation['call'],
                              'outcome': 'submission_failed', 'submitted': False,
                              'proposals': 0, 'error_type': type(error).__name__})
                self._halt('executor_failure', reservation['call'])
                raise
            return future, reservation['call'], deadline

    def propose(self, prompt, tag, templates, validate, positions):
        if self.stopped():
            return []
        _text(prompt, _MAX_PROMPT_BYTES, 'prompt')
        _text(tag, 256, 'wave')
        _templates(templates)
        _integer(positions, 1, _MAX_POSITIONS, 'positions')
        if not callable(validate):
            raise ValueError('trusted host validator required')
        templates = copy.deepcopy(templates)
        # Serialize host waves: concurrent callers cannot grow executor queues.
        # Waiting for a wave slot creates no charged reservation.
        queue_deadline = time.monotonic() + self.config['timeout_seconds']
        while not self._wave_lock.acquire(timeout=0.1):
            if self.stopped() or time.monotonic() >= queue_deadline:
                return []
        pending, results, scheduled = {}, {}, 0
        try:
            while True:
                # Drain a complete batch, including its meter receipts, before
                # admitting another batch. Failed/unmetered calls cannot leave
                # a growing queue of submissions behind them.
                if not pending:
                    count = min(self.config['parallelism'], self.config['proposals_per_prompt'] - scheduled)
                    for _ in range(count):
                        job = self._submit(prompt, tag, templates, validate, positions)
                        if job is None:
                            break
                        future, call, deadline = job
                        pending[future] = (call, deadline)
                        scheduled += 1
                if not pending:
                    break
                with self.lock:
                    if self._blocked():
                        return []
                    expired = [call for future, (call, deadline) in pending.items()
                               if not future.done() and time.monotonic() >= deadline]
                    if expired:
                        self._halt('timed_out', min(expired))
                        return []
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    call, _ = pending.pop(future)
                    results[call] = future.result()
            with self.lock:
                if self._blocked():
                    return []
            # Reservation order, never completion order, determines result order.
            return [entry for call in sorted(results) for entry in results[call]]
        except BaseException:
            with self.lock:
                self._halt('host_interrupted')
            raise
        finally:
            self._wave_lock.release()
