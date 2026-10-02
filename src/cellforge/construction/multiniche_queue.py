"""Durable independent niche services; callbacks own provenance and budget ledgers.

A callback must bound its external work. Threads cannot terminate a hung provider.
Checkpointed unknown author outcomes require explicit reconciliation, never replay.
This queue does not infer learning, admission, or free proposals from cache hits.
"""
import copy
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

from cellforge.construction.action_cell_continuation import atomic


class NicheWaiting(Exception):
    def __init__(self, reason, retry_after=60):
        super().__init__(reason)
        self.reason = str(reason)
        self.retry_after = max(0, float(retry_after))


class MultiNicheQueue:
    """Two author workers and one evaluation worker, fair persistent service.

    author(niche_id, payload) returns a JSON-compatible authored proposal.
    evaluate(niche_id, proposal) returns {payload?, saturated?, numeric_evaluation}.
    numeric_evaluation must contain development traces with expected/predicted,
    correct, total and solved. Execution receipts are validated for internal
    consistency; the queue cannot independently prove callback execution.
    Each callback must persist its own charged ledger before external work; queue
    dispatches/counts are never substituted for charged positions or evaluations.
    tick() never waits for unfinished callbacks. Call close() after draining.
    """
    def __init__(self, checkpoint, author, evaluate, *, author_workers=2,
                 eval_workers=1, clock=time.time):
        if not 1 <= author_workers <= 2 or eval_workers != 1:
            raise ValueError('server limits: author workers 1..2, evaluation workers 1')
        self.path = pathlib.Path(checkpoint)
        self.author, self.evaluate, self.clock = author, evaluate, clock
        self.limits = {'author': author_workers, 'evaluate': eval_workers}
        self.pools = {stage: ThreadPoolExecutor(max_workers=count,
                      thread_name_prefix='niche-'+stage) for stage, count in self.limits.items()}
        self.pending = {}
        self.state = {'schema': 'cellforge-multiniche-queue/1', 'cursor': 0, 'niches': {}}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('schema') != 'cellforge-multiniche-queue/1':
                raise ValueError('unknown queue checkpoint')
            for row in self.state['niches'].values():
                if row['status'] == 'running':
                    row.update(status='waiting', reason='interrupted_'+row['stage']+'_outcome_unknown', retry_at=None)
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic(self.path, self.state)

    def add_niche(self, niche_id, payload, *, saturated=False):
        if niche_id in self.state['niches']:
            raise ValueError('niche already registered; reconcile it explicitly')
        self.state['niches'][niche_id] = {'payload': copy.deepcopy(payload),
            'status': 'saturated' if saturated else 'ready', 'stage': 'author',
            'generation': 0, 'dispatches': {'author': 0, 'evaluate': 0},
            'failures': 0, 'reason': 'perfect_parent' if saturated else None,
            'retry_at': None, 'proposal': None, 'last_receipt': None}
        self._save()

    def reconcile(self, niche_id, *, proposal=None, payload=None, saturated=False):
        """Operator/callback reconciliation after ledger inspection, no silent replay."""
        if niche_id in self.pending:
            raise ValueError('cannot reconcile an active callback')
        row = self.state['niches'][niche_id]
        if payload is not None:
            row['payload'] = copy.deepcopy(payload)
        if proposal is not None:
            row['proposal'] = copy.deepcopy(proposal)
        row.update(stage='evaluate' if row['proposal'] is not None else 'author',
                   status='saturated' if saturated else 'ready', reason=None, retry_at=None)
        self._save()

    @staticmethod
    def _validate_evaluation(result):
        if result.get('status') in ('unscored', 'inapplicable', 'waiting', 'failed'):
            raise ValueError('evaluation did not execute scored development tasks')
        if 'scored_tasks' in result and (type(result['scored_tasks']) is not int or result['scored_tasks'] <= 0):
            raise ValueError('evaluation has no scored tasks')
        evidence = result.get('numeric_evaluation')
        if not isinstance(evidence, dict) or evidence.get('partition') != 'development':
            raise ValueError('executed development numeric_evaluation required')
        traces = evidence.get('traces')
        if not isinstance(traces, list) or not traces:
            raise ValueError('nonempty execution traces required')
        solved = 0
        for trace in traces:
            if not isinstance(trace, dict) or not all(k in trace for k in ('expected', 'predicted', 'correct')):
                raise ValueError('execution trace requires expected, predicted and correct')
            correct = trace['predicted'] == trace['expected']
            if type(trace['correct']) is not bool or trace['correct'] != correct:
                raise ValueError('execution trace correctness differs from numeric outputs')
            solved += correct
        if type(evidence.get('total')) is not int or evidence['total'] != len(traces):
            raise ValueError('execution trace total differs')
        if type(evidence.get('solved')) is not int or evidence['solved'] != solved:
            raise ValueError('execution trace solved count differs')
        if 'scored_tasks' in result and result['scored_tasks'] != len(traces):
            raise ValueError('scored_tasks differs from execution trace count')

    def tick(self):
        now = self.clock()
        for niche_id, (stage, future) in list(self.pending.items()):
            if not future.done():
                continue
            del self.pending[niche_id]
            row = self.state['niches'][niche_id]
            try:
                result = future.result()
                # Refuse results that cannot be durably represented before transitioning.
                json.dumps(result, allow_nan=False)
                if stage == 'author':
                    row.update(proposal=result, stage='evaluate', status='ready', reason=None)
                else:
                    if not isinstance(result, dict):
                        raise ValueError('evaluation must return a receipt object')
                    row['last_receipt'] = result
                    self._validate_evaluation(result)
                    row['generation'] += 1
                    row['payload'] = result.get('payload', row['payload'])
                    row.update(proposal=None, stage='author',
                        status='saturated' if result.get('saturated', False) else 'ready',
                        reason='perfect_parent' if result.get('saturated', False) else None)
            except NicheWaiting as error:
                row.update(status='waiting', reason=error.reason, retry_at=now+error.retry_after)
            except Exception as error:
                row['failures'] += 1
                row.update(status='waiting', reason=type(error).__name__+': '+str(error), retry_at=None)
            self._save()
        names = list(self.state['niches'])
        if not names:
            return self.status()
        start = self.state['cursor'] % len(names)
        used = {stage: sum(s == stage for s, _ in self.pending.values()) for stage in self.limits}
        for offset in range(len(names)):
            index = (start+offset) % len(names)
            niche_id = names[index]
            row = self.state['niches'][niche_id]
            if row['status'] == 'waiting' and row['retry_at'] is not None and now >= row['retry_at']:
                row.update(status='ready', reason=None, retry_at=None)
            stage = row['stage']
            if row['status'] != 'ready' or used[stage] >= self.limits[stage]:
                continue
            row.update(status='running', reason=None)
            row['dispatches'][stage] += 1
            self.state['cursor'] = index+1
            self._save()  # Persist uncertain work before handing it to the callback.
            callback = self.author if stage == 'author' else self.evaluate
            value = row['payload'] if stage == 'author' else row['proposal']
            self.pending[niche_id] = (stage, self.pools[stage].submit(callback, niche_id, copy.deepcopy(value)))
            used[stage] += 1
        self._save()
        return self.status()

    def status(self):
        return copy.deepcopy(self.state)

    def close(self, *, wait=True):
        for pool in self.pools.values():
            pool.shutdown(wait=wait, cancel_futures=True)
