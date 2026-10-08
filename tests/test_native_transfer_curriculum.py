"""Integrity tests use distinct fixtures, never score prospective development."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import native_transfer_curriculum as T


class TransferCurriculum(unittest.TestCase):
    def root(self):
        parent = Path(tempfile.gettempdir()).resolve()
        folder = parent / ('cellforge-transfer-test-' + uuid.uuid4().hex)
        folder.mkdir()
        def cleanup():
            if folder.resolve().parent != parent or folder.is_symlink():
                raise RuntimeError('cleanup escaped its verified temporary parent')
            shutil.rmtree(folder)
        self.addCleanup(cleanup)
        return folder / 'experiment'

    def fixture_panels(self):
        # Dimensions include 2, absent from ALL prospective production panels.
        search = [[0, 2, 0], [4, 0, 0]]
        unseen = [[0, 3, 0, 0], [7, 0, 0, 0]]
        panels = {name: [{'id': f'{name}-{i}', 'input': copy.deepcopy(search)} for i in range(n)]
                  for name, n in (('demonstrations', 6), ('search', 12))}
        dev = [{'id': f'color-{i}', 'stratum': 'unseen_color', 'input': unseen} for i in range(8)]
        for i in range(4):
            perm = [0] + [1 + c % 9 for c in range(1, 10)]
            dev += [{'id': f'perm-{i}-base', 'input': unseen, 'stratum': 'permutation', 'pair': i, 'role': 'base'},
                    {'id': f'perm-{i}-mapped', 'input': [[perm[v] for v in row] for row in unseen],
                     'stratum': 'permutation', 'pair': i, 'role': 'mapped', 'permutation': perm}]
        dev += [{'id': f'size-{i}', 'stratum': 'larger_rectangle', 'input': unseen} for i in range(8)]
        panels['development'] = dev
        return panels

    def prepare(self):
        root = self.root()
        sealed_inputs = {T.N.C.digest(row['input']) for row in self.fixture_panels()['development']}
        original = T.oracle
        def preparation_oracle(grid):
            self.assertNotIn(T.N.C.digest(grid), sealed_inputs, 'development labels materialized during preparation')
            return original(grid)
        with mock.patch.object(T, 'inputs', side_effect=self.fixture_panels), mock.patch.object(T, 'oracle', side_effect=preparation_oracle):
            T.prepare(root)
        return root

    def source(self, root):
        path = root / 'fixture-source.json'
        if not path.exists():
            path.write_bytes(T.P.compact({'author_task_name': 'unit-fixture', 'author_model': 'unit-fixture-not-LLM',
                'author_reasoning': 'unknown', 'route': 'unit fixture', 'context_sha256': None,
                'context_provenance': 'parent_transcription_exact_bytes_unknown',
                'response_provenance': 'synthetic fixture only', 'credit_usage': None}))
        return path

    def raw(self, params=None):
        return T.P.compact([{'task_id': T.TASK, 'params': params or T.reference_params(), 'rationale': 'unit fixture'}])

    def consume(self, root, slot, raw=None):
        path = root / (slot + '-fixture.txt')
        path.write_bytes(self.raw() if raw is None else raw)
        return T.consume(root, slot, path, self.source(root))

    def test_independent_manual_oracle_and_separate_reference_fixtures(self):
        with mock.patch.object(T.N.S, 'run', side_effect=AssertionError('oracle called predictor')):
            self.assertEqual(T.oracle([[0, 2, 0], [4, 0, 0]]), [[4, 2, 0, 0, 2, 4]] * 4)
        self.assertEqual(T.fixture_check()['fixtures'], 3)
        with self.assertRaisesRegex(ValueError, 'one marker'):
            T.oracle([[1], [2]])

    def test_prospective_generator_structure_only_no_labels_or_scoring(self):
        with (mock.patch.object(T, 'oracle', side_effect=AssertionError('prospective labels materialized')),
              mock.patch.object(T.N.S, 'run', side_effect=AssertionError('prospective prediction'))):
            panels = T.inputs()
            self.assertEqual(panels, T.inputs())
        self.assertEqual([len(panels[k]) for k in ('demonstrations', 'search', 'development')], [6, 12, 24])
        ordinary = panels['demonstrations'] + panels['search'] + [r for r in panels['development'] if r.get('role') != 'mapped']
        self.assertEqual(len({T.geometry(r['input']) for r in ordinary}), len(ordinary))
        all_rows = sum(panels.values(), [])
        self.assertEqual(len({T.N.C.digest(r['input']) for r in all_rows}), 42)
        self.assertTrue(all('output' not in row for row in all_rows))
        for row in panels['development']:
            grid = row['input']
            if row['stratum'] == 'unseen_color':
                self.assertLessEqual({v for line in grid for v in line}, {0, 1, 3, 6, 7, 8, 9})
            if row['stratum'] == 'larger_rectangle':
                self.assertTrue(6 <= len(grid) <= 9 and 6 <= len(grid[0]) <= 9 and len(grid) != len(grid[0]))

    def test_development_unread_before_selection_and_one_time_open_replays(self):
        root = self.prepare()
        original_read = Path.read_bytes
        def guarded_read(path):
            if path == root / T.SEALED and not (root / 'HOST_EVALUATION/FINAL_OPEN.json').exists():
                raise AssertionError('development read before final opening')
            return original_read(path)
        original_oracle = T.oracle
        def guarded_oracle(grid):
            self.assertTrue((root / 'HOST_EVALUATION/SEARCH_SELECTION.json').exists())
            self.assertTrue((root / 'HOST_EVALUATION/FINAL_OPEN.json').exists())
            return original_oracle(grid)
        with mock.patch.object(Path, 'read_bytes', guarded_read), mock.patch.object(T, 'oracle', side_effect=guarded_oracle):
            T.preflight(root)
            self.consume(root, 'founder_a')
            self.consume(root, 'founder_b')
            T.verify(root, 'founder_a')
            selected_before = T.selection(root)[1]
            self.assertFalse((root / 'HOST_EVALUATION/FINAL_OPEN.json').exists())
            result = T.finalize(root)
            self.assertEqual(result['charged_positions'], {'native': 334, 'random': 334})
            self.assertEqual(result['selected_on_search_only'], selected_before['selected'])
            self.assertEqual(set(result['development_counts']), {'founder_a:native', 'founder_a:random',
                'founder_b:native', 'founder_b:random', 'reference', 'zero'})
            self.assertEqual(T.finalize(root, replay=True)['result_hash'], result['result_hash'])
            with self.assertRaises(FileExistsError): T.finalize(root)
            with self.assertRaisesRegex(ValueError, 'closed'): self.consume(root, 'repair_a')

    def test_invalids_charge_without_repair_and_source_response_tampering_fails(self):
        root = self.prepare()
        first = self.consume(root, 'founder_a', b'[]')
        self.assertEqual(first['charged_positions'], {'native': 167, 'random': 167})
        self.assertEqual(first['validation']['status'], 'invalid_or_refused')
        self.consume(root, 'founder_b')
        with self.assertRaisesRegex(ValueError, 'invalid or search-perfect'): T.prepare_repair(root, 'repair_a')
        with self.assertRaises(FileExistsError): self.consume(root, 'founder_a', b'[]')
        T.verify(root, 'founder_a')
        (root / 'HOST_EVALUATION/founder_a/RESPONSE.bin').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'attempt changed'): T.verify(root, 'founder_a')

    def test_failed_founder_requires_one_search_only_replacement(self):
        root = self.prepare()
        params = T.reference_params()
        params['ray_cast']['src'] = [-9] * 10
        self.consume(root, 'founder_a', self.raw(params))
        self.consume(root, 'founder_b')
        with self.assertRaisesRegex(ValueError, 'eligible repairs'): T.finalize(root)
        self.assertFalse((root / 'HOST_EVALUATION/FINAL_OPEN.json').exists())
        receipt = T.prepare_repair(root, 'repair_a')
        packet = (root / 'AUTHORING/repair_a.txt').read_bytes()
        body = json.loads(packet.split(b'\n', 1)[1])
        self.assertEqual(body['search_failure']['id'], 'search-0')
        self.assertEqual(body['search_failure']['partition'], 'search')
        self.assertFalse(receipt['development_feedback'])
        with self.assertRaises(FileExistsError): T.prepare_repair(root, 'repair_a')
        self.consume(root, 'repair_a')
        T.verify(root, 'repair_a')
        result = T.finalize(root)
        self.assertEqual(result['charged_positions'], {'native': 501, 'random': 501})
        with self.assertRaisesRegex(ValueError, 'closed'): self.consume(root, 'repair_b')

    def test_both_winners_use_only_search_scores(self):
        root = self.root()
        # Both native founders search-perfect => no repair; higher development fields are decoys.
        def result(slot):
            a = slot == 'founder_a'
            return {'slot': slot, 'hash': slot, 'validation': {'status': 'valid'},
                'cells': {'native': {'id': 'a' if a else 'b'}, 'random': {'id': 'c' if a else 'd'}},
                'scores': {'native': {'correct': 12, 'errors': 0}, 'random': {'correct': 10 if a else 2, 'errors': 0}},
                'development': {'native': 0 if a else 24, 'random': 0 if a else 24}}
        with mock.patch.object(T, 'verify', side_effect=lambda root, slot: result(slot)):
            selected = T.selection(root)[1]
        self.assertEqual(selected['selected']['native']['slot'], 'founder_a')
        self.assertEqual(selected['selected']['random']['slot'], 'founder_a')
        self.assertFalse(selected['development_used'])

    def test_four_attempt_cap_and_invalid_replacements_are_charged(self):
        root = self.prepare()
        params = T.reference_params()
        params['ray_cast']['src'] = [-9] * 10
        for slot in ('founder_a', 'founder_b'):
            self.consume(root, slot, self.raw(params))
        for slot in ('repair_a', 'repair_b'):
            T.prepare_repair(root, slot)
            self.consume(root, slot, b'[]')
        result = T.finalize(root)
        self.assertEqual(result['charged_positions'], {'native': 668, 'random': 668})
        self.assertFalse(result['transfer_criterion_passed'])
        with self.assertRaisesRegex(ValueError, 'unknown slot'): self.consume(root, 'fifth_author')

    def test_acceptance_uses_all_random_scores_without_reselecting_either_winner(self):
        root = self.prepare()
        self.consume(root, 'founder_a')
        self.consume(root, 'founder_b')
        results, selected = T.selection(root)
        selected_random = selected['selected']['random']['slot']
        other = next(r for r in results if r['slot'] != selected_random)
        other_random_id = other['cells']['random']['id']
        original = T.evaluate
        def controlled(cell, rows, demos, root):
            score = original(cell, rows, demos, root)
            if len(rows) == 24 and cell['id'] == other_random_id:
                score['correct'] = 24
            return score
        with mock.patch.object(T, 'evaluate', side_effect=controlled):
            final = T.finalize(root)
        self.assertEqual(final['selected_on_search_only'], selected['selected'])
        self.assertFalse(final['transfer_criterion_passed'])
        self.assertEqual(final['development_counts'][other['slot'] + ':random'], 24)

    def test_sealed_input_tamper_fails_closed_before_prediction(self):
        root = self.prepare()
        self.consume(root, 'founder_a')
        self.consume(root, 'founder_b')
        (root / T.SEALED).write_bytes(b'[]')
        with mock.patch.object(T, 'oracle', side_effect=AssertionError('labels used before seal check')):
            with self.assertRaisesRegex(ValueError, 'sealed development input changed'): T.finalize(root)
        self.assertTrue((root / 'HOST_EVALUATION/FINAL_OPEN.json').exists())
        with self.assertRaisesRegex(ValueError, 'closed'): self.consume(root, 'repair_a')

    def test_fixed_mask_charge_and_stop_unchanged(self):
        rng = mock.Mock()
        rng.uniform.return_value = 0.5
        with mock.patch.object(T.P.random, 'Random', return_value=rng):
            params = T.P.random_params(T.reference_params(), 1)
        self.assertEqual(rng.uniform.call_count, 103)
        self.assertEqual(sum(T.N.S.positions(p) for p in T.N.spatial_cell(params)['parts']), 167)
        T.validate(self.raw(params))
        params['tile_orient']['orient'][0][0][7] = -8
        with self.assertRaisesRegex(ValueError, 'must equal -9'): T.validate(self.raw(params))
        root = self.prepare()
        (root / 'STOP').write_text('owner stop')
        with self.assertRaisesRegex(RuntimeError, 'STOP'): T.preflight(root)


if __name__ == '__main__':
    unittest.main()
