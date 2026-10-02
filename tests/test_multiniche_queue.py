"""Scheduler fault-injection tests, not learning or authorship evidence."""
import json
import pathlib
import tempfile
import threading
import time
import unittest
from cellforge.construction.multiniche_queue import MultiNicheQueue, NicheWaiting


def evaluated(**fields):
    # Deliberate fault-injection evidence, not actual ARC execution/authorship.
    return dict(numeric_evaluation={'partition':'development', 'total':1,
                'solved':0, 'traces':[{'expected':1,'predicted':0,'correct':False}]}, **fields)


class TestMultiNicheQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name)/'queue.json'

    def queue(self, author, evaluate, **kwargs):
        queue = MultiNicheQueue(self.path, author, evaluate, **kwargs)
        self.addCleanup(queue.close)
        return queue

    def advance(self, queue, predicate):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            state = queue.tick()
            if predicate(state['niches']):
                return state
            time.sleep(.002)
        self.fail('queue did not reach expected state: '+repr(queue.status()))

    def test_stalled_author_does_not_block_other_niche_or_perfect_parent(self):
        release = threading.Event()
        self.addCleanup(release.set)
        calls = []
        def author(niche, payload):
            calls.append(niche)
            if niche == 'stalled':
                release.wait(2)
            return payload
        queue = self.queue(author, lambda n,p: evaluated(payload=p))
        queue.add_niche('stalled', {})
        queue.add_niche('hard', {})
        queue.add_niche('perfect', {}, saturated=True)
        state = self.advance(queue, lambda rows: rows['hard']['generation'] >= 8)
        self.assertEqual(state['niches']['stalled']['generation'], 0)
        self.assertNotIn('perfect', calls)
        self.assertGreater(state['niches']['hard']['dispatches']['author'], 6)
        release.set()

    def test_rate_limit_reason_is_durable_and_other_niche_advances(self):
        now = [100.0]
        def author(niche, payload):
            if niche == 'limited':
                raise NicheWaiting('subscription_rate_limit', retry_after=60)
            return payload
        queue = self.queue(author, lambda n,p: evaluated(saturated=True), clock=lambda:now[0])
        queue.add_niche('limited', {})
        queue.add_niche('hard', {})
        self.advance(queue, lambda rows: rows['limited']['status']=='waiting' and rows['hard']['generation']==1)
        saved = json.loads(self.path.read_text())['niches']['limited']
        self.assertEqual(saved['reason'], 'subscription_rate_limit')
        self.assertEqual(saved['retry_at'], 160)
        self.assertEqual(saved['dispatches']['author'], 1)

    def test_failure_isolated_and_evaluation_receipt_retained(self):
        def author(niche, payload):
            if niche == 'failed':
                raise RuntimeError('provider failed after charge')
            return {'literal':3, 'charged_positions':1}
        queue = self.queue(author, lambda n,p: evaluated(saturated=True,receipt=p))
        queue.add_niche('failed', {})
        queue.add_niche('hard', {})
        state = self.advance(queue, lambda rows: rows['hard']['generation']==1 and rows['failed']['status']=='waiting')
        self.assertIsNone(state['niches']['failed']['retry_at'])
        self.assertEqual(state['niches']['hard']['last_receipt']['receipt']['charged_positions'], 1)

    def test_resume_unknown_author_never_automatically_replays(self):
        queue = self.queue(lambda n,p:p, lambda n,p:evaluated(saturated=True))
        queue.add_niche('hard', {})
        snapshot = queue.status()
        snapshot['niches']['hard'].update(status='running', stage='author')
        self.path.write_text(json.dumps(snapshot))
        calls = []
        restored = self.queue(lambda n,p:calls.append(n) or p, lambda n,p:evaluated(saturated=True))
        self.assertEqual(restored.tick()['niches']['hard']['reason'], 'interrupted_author_outcome_unknown')
        self.assertEqual(calls, [])
        restored.reconcile('hard', proposal={'charged_positions':39})
        state = self.advance(restored, lambda rows: rows['hard']['generation']==1)
        self.assertEqual(calls, [])
        self.assertEqual(state['niches']['hard']['dispatches']['author'], 0)

    def test_author_and_evaluation_worker_limits(self):
        lock = threading.Lock()
        active = {'author':0, 'evaluate':0}
        peak = dict(active)
        def work(stage, result):
            def callback(niche, payload):
                with lock:
                    active[stage] += 1
                    peak[stage] = max(peak[stage], active[stage])
                time.sleep(.015)
                with lock:
                    active[stage] -= 1
                return result
            return callback
        queue = self.queue(work('author', {}), work('evaluate', evaluated(saturated=True)))
        for i in range(6):
            queue.add_niche(str(i), {})
        self.advance(queue, lambda rows: all(row['generation']==1 for row in rows.values()))
        self.assertEqual(peak, {'author':2, 'evaluate':1})

    def test_duplicate_proposals_are_not_free_or_deduplicated_across_niches(self):
        calls = []
        queue = self.queue(lambda n,p:{'weights':[1], 'charged_positions':1},
                           lambda n,p:calls.append(n) or evaluated(saturated=True,receipt=p))
        for niche in ('a','b'):
            queue.add_niche(niche, {})
        state = self.advance(queue, lambda rows: all(row['generation']==1 for row in rows.values()))
        self.assertCountEqual(calls, ['a','b'])
        self.assertTrue(all(row['dispatches']=={'author':1,'evaluate':1} for row in state['niches'].values()))

    def test_unscored_evaluation_pauses_without_progress_and_preserves_other_lane(self):
        def evaluate(niche, proposal):
            if niche == 'unscored':
                return {'status':'unscored','scored_tasks':0,'payload':{'changed':True}}
            return evaluated(saturated=True)
        queue = self.queue(lambda n,p:p, evaluate)
        queue.add_niche('unscored', {'original':True})
        queue.add_niche('hard', {})
        state = self.advance(queue, lambda rows: rows['unscored']['status']=='waiting' and rows['hard']['generation']==1)
        row = state['niches']['unscored']
        self.assertEqual(row['generation'], 0)
        self.assertEqual(row['payload'], {'original':True})
        self.assertEqual(row['proposal'], {'original':True})
        self.assertIsNone(row['retry_at'])
        saved = json.loads(self.path.read_text())['niches']['unscored']
        self.assertEqual(saved['last_receipt']['status'], 'unscored')
        self.assertEqual(saved['status'], 'waiting')

    def test_claim_only_and_inconsistent_execution_receipts_cannot_progress(self):
        for receipt in ({'saturated':True}, evaluated(nothing=True), evaluated(scored_tasks=0)):
            if 'nothing' in receipt:
                receipt['numeric_evaluation']['traces'][0]['correct'] = True
            with self.subTest(receipt=receipt):
                with tempfile.TemporaryDirectory() as tmp:
                    queue = MultiNicheQueue(pathlib.Path(tmp)/'queue.json', lambda n,p:p, lambda n,p:receipt)
                    try:
                        queue.add_niche('hard', {})
                        state = self.advance(queue, lambda rows: rows['hard']['status']=='waiting')
                        self.assertEqual(state['niches']['hard']['generation'], 0)
                    finally:
                        queue.close()

if __name__ == '__main__':
    unittest.main()
