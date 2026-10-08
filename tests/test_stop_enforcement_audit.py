"""Offline safety audit: temporary receipts and fake makers only, no model calls."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import weight_workers as W
import evolution_manifest as M
import head_driver as H


class StopEnforcementAudit(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='cellforge-stop-audit-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = dict(enabled=True, model='gpt-6-luna', reasoning='low',
                           parallelism=1, proposals_per_prompt=1, max_calls=3, timeout_seconds=10)
        self.reset = time.time() + 3600
        self.rows = []
        for patch in (mock.patch.object(W.maker, 'last_used', return_value=(10, self.reset)),
                      mock.patch.object(W.maker, 'readings', side_effect=lambda: self.rows),
                      mock.patch.object(W.maker, 'CAP_FILE', self.root / 'absent-cap.json')):
            patch.start()
            self.addCleanup(patch.stop)
        self.backend = mock.patch.object(W.maker, 'ask', side_effect=AssertionError('real maker forbidden')).start()
        self.addCleanup(mock.patch.stopall)

    def pool(self):
        pool = W.WeightWorkers(self.root / 'pool', self.config)
        self.addCleanup(pool.close)
        return pool

    def metered(self, prompt, work, **kwargs):
        self.rows.append(dict(work=work.name, prompt_sha256=W.hashlib.sha256(prompt.encode()).hexdigest(),
                              track=kwargs['track'], role=kwargs['role'], used_percent=10, resets_at=self.reset))

    def valid(self, prompt, work, **kwargs):
        self.metered(prompt, work, **kwargs)
        result = [{'task_id': 'task', 'params': {'x': [1.]}, 'rationale': 'fixture'}]
        (work / 'response.json').write_text(json.dumps(result))
        return result

    def propose(self, pool, tag='wave'):
        return pool.propose('fixture only', tag, {'task': 'template'}, lambda *_: True, 1)

    def test_stop_before_reservation_charges_nothing(self):
        pool = self.pool()
        (pool.root / 'STOP').symlink_to(pool.root / 'missing')
        self.assertEqual(self.propose(pool), [])
        self.assertEqual(pool.calls, 0)
        self.backend.assert_not_called()

    def test_stop_after_reservation_prevents_dispatch_and_remains_charged(self):
        pool = self.pool()
        append = pool._append
        def stopped_reservation(event):
            append(event)
            if event['event'] == 'reserved':
                (pool.root / 'STOP').write_text('stop')
        with mock.patch.object(pool, '_append', side_effect=stopped_reservation):
            self.assertEqual(self.propose(pool), [])
        pool._executor.shutdown(wait=True)
        self.assertEqual(pool.calls, 1)
        self.assertEqual(pool.attempts_for('wave')[0]['charged_positions'], 1)
        self.backend.assert_not_called()
        pool.close()
        (pool.root / 'STOP').unlink()
        resumed = self.pool()
        self.assertEqual(self.propose(resumed), [])
        self.assertEqual(resumed.calls, 1)

    def test_pending_reservation_is_never_replaced_on_resume(self):
        pool = self.pool()
        pool._reserve('fixture only', 'wave', {'task': 'template'}, 1)
        pool.close()
        resumed = self.pool()
        self.assertEqual(self.propose(resumed), [])
        self.assertEqual(resumed.calls, 1)
        self.assertEqual(resumed.safety_halt(), 'interrupted_reservation')
        self.backend.assert_not_called()

    def test_stop_at_dispatch_receipt_prevents_backend_call(self):
        pool = self.pool()
        append = pool._append
        def stopped_dispatch(event):
            append(event)
            if event['event'] == 'dispatched':
                (pool.root / 'STOP').write_text('stop')
        with mock.patch.object(pool, '_append', side_effect=stopped_dispatch):
            self.assertEqual(self.propose(pool), [])
        pool._executor.shutdown(wait=True)
        attempt = pool.attempts_for('wave')[0]
        self.assertEqual(attempt['charged_positions'], 1)
        self.assertIs(attempt['submitted'], False)
        self.backend.assert_not_called()

    def test_uncertain_metered_dispatch_blocks_current_and_resumed_work(self):
        pool = self.pool()
        def uncertain(prompt, work, **kwargs):
            self.metered(prompt, work, **kwargs)
            raise RuntimeError('route exited without a releasable result')
        self.backend.side_effect = uncertain
        self.assertEqual(self.propose(pool), [])
        attempt = pool.attempts_for('wave')[0]
        self.assertIsNone(attempt['submitted'])
        self.assertTrue(attempt['submission_uncertain'])
        self.assertTrue(attempt['metered'])
        self.assertIsNotNone(pool.safety_halt())
        self.backend.side_effect = self.valid
        self.assertEqual(self.propose(pool, 'next'), [])
        pool.close()
        resumed = self.pool()
        self.assertEqual(self.propose(resumed, 'next'), [])
        self.assertEqual(resumed.calls, 1)
        self.assertEqual(self.backend.call_count, 1)

    def test_invalid_and_duplicate_valid_proposals_each_remain_charged(self):
        pool = self.pool()
        def invalid(prompt, work, **kwargs):
            self.metered(prompt, work, **kwargs)
            (work / 'response.json').write_text('[]')
            return []
        self.backend.side_effect = invalid
        self.assertEqual(self.propose(pool, 'invalid'), [])
        self.backend.side_effect = self.valid
        self.assertEqual(len(self.propose(pool, 'first')), 1)
        self.assertEqual(len(self.propose(pool, 'duplicate')), 1)
        self.assertEqual(pool.calls, 3)
        self.assertEqual(sum(a['charged_positions'] for tag in ('invalid', 'first', 'duplicate')
                             for a in pool.attempts_for(tag)), 3)

    def test_preexisting_finished_uncertain_receipt_halts_on_resume(self):
        pool = self.pool()
        def uncertain(prompt, work, **kwargs):
            self.metered(prompt, work, **kwargs)
            raise ValueError('fake malformed response after metering')
        self.backend.side_effect = uncertain
        # Reproduce a pre-fix ledger without rewriting any historical receipt.
        with mock.patch.object(pool, '_halt'):
            self.assertEqual(self.propose(pool), [])
        self.assertTrue(pool.attempts_for('wave')[0]['finished'])
        pool.close()
        resumed = self.pool()
        self.assertEqual(resumed.safety_halt(), 'uncertain_submission')
        self.assertEqual(self.propose(resumed, 'next'), [])
        self.assertEqual(resumed.calls, 1)
        self.assertEqual(self.backend.call_count, 1)

    def test_disabled_explicit_driver_policy_never_invokes_legacy(self):
        driver = H.Driver.__new__(H.Driver)
        driver.evolution_active, driver.fake, driver.worker_pool = True, False, None
        with mock.patch.object(H.H, 'ask_maker') as legacy:
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                driver.ask('fixture', 'wave', [])
            legacy.assert_not_called()
        self.config.update(enabled=False, max_calls=0)
        pool = self.pool()
        self.assertEqual(self.propose(pool), [])
        self.backend.assert_not_called()

    def test_manifest_resume_rejects_source_task_content_and_population_drift(self):
        sources = {'worker.py': 'a' * 64, 'search-fitness-task-content': 'b' * 64}
        partitions = {'search_fitness': ['search'], 'development_selection': ['dev'], 'report': ['report']}
        M.freeze_manifest(self.root / 'manifest', {}, sources, partitions)
        for changed in ({**sources, 'worker.py': 'c' * 64},
                        {**sources, 'search-fitness-task-content': 'd' * 64}):
            with self.assertRaises(M.ManifestError):
                M.freeze_manifest(self.root / 'manifest', {}, changed, partitions)
        with self.assertRaises(M.ManifestError):
            M.freeze_manifest(self.root / 'manifest', {}, sources, {**partitions, 'search_fitness': ['other']})

    def test_router_resume_rejects_ordered_roster_drift(self):
        driver = H.Driver.__new__(H.Driver)
        driver.root = self.root
        driver.state = {'router_roster': ['first', 'second']}
        driver.cells = lambda: [{'id': 'second'}, {'id': 'first'}]
        status = dict(sealed=False, pending_router_waves=[], pending_mutation_waves=[],
                      router_expert_ids=['first', 'second'])
        with mock.patch.object(M, 'read_manifest_status', return_value=status):
            with self.assertRaisesRegex(ValueError, 'expert order'):
                driver.validate_router_state()
