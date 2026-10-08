"""Separate fixtures only for actual predictions; prospective panels structural-only."""
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import native_composition_batch as B

PREVIOUS = Path(__file__).resolve().parents[1] / 'runs/weights-native-transfer-curriculum01-20261006'


class Composition(unittest.TestCase):
    def root(self):
        base = Path(tempfile.gettempdir()).resolve()
        parent = base / ('cellforge-composition-fixture-' + uuid.uuid4().hex)
        parent.mkdir()
        def cleanup():
            if parent.resolve().parent != base or parent.is_symlink(): raise RuntimeError('cleanup scope changed')
            shutil.rmtree(parent)
        self.addCleanup(cleanup)
        return parent / 'run'

    def panels(self):
        g = [[0, 2, 0], [4, 0, 0]]  # 2 rows never occur in prospective panels.
        panel = {name: [{'id': f'{name}-{i}', 'family': 'H', 'input': g} for i in range(n)] for name, n in (('h_demos', 6), ('h_search', 12))}
        def task(tid, family, grid):
            return {'id': tid, 'family': family, 'stratum': 'fixture', 'train_inputs': [grid] * 3, 'query_inputs': [grid] * 2}
        panel['mixed_search'] = [task(f'search-{i}', 'CH'[i % 2], g) for i in range(12)]
        dev = [task(f'color-{i}', 'CH'[i % 2], g) for i in range(8)]
        for i in range(4):
            permutation = [0] + [1 + c % 9 for c in range(1, 10)]
            mapped = [[permutation[v] for v in row] for row in g]
            dev += [task(f'perm-{i}-base', 'CH'[i % 2], g), dict(task(f'perm-{i}-mapped', 'CH'[i % 2], mapped), permutation=permutation)]
        dev += [task(f'size-{i}', 'CH'[i % 2], g) for i in range(8)]
        panel['development'] = dev
        return panel

    def prepare(self):
        root = self.root()
        with mock.patch.object(B, 'generate', side_effect=lambda excluded: self.panels()): B.prepare(root, PREVIOUS)
        return root

    def consume(self, root, slot, params=None, raw=None):
        source = root / 'SOURCE.json'
        if not source.exists():
            source.write_bytes(B.P.compact({'author_task_name': 'fixture-not-model', 'author_model': 'fixture', 'author_reasoning': 'unknown',
                'route': 'unit fixture', 'context_sha256': None, 'context_provenance': 'parent_transcription_exact_bytes_unknown',
                'response_provenance': 'synthetic test response', 'credit_usage': None}))
        path = root / (slot + '.txt')
        if raw is None:
            raw = B.P.compact([{'task_id': 'router' if B.is_router(slot) else B.H_TASK,
                               'params': params if params is not None else (B.IDENTITY if B.is_router(slot) else B.h_reference()), 'rationale': 'fixture'}])
        path.write_bytes(raw)
        return B.consume(root, slot, path, source)

    def test_oracle_and_bank_independent_fixture_and_stripped_queries(self):
        root = self.prepare()
        _, protocol, _, _, rows = B.context(root)
        cells = [protocol['column'], B.N.spatial_cell(B.h_reference())]
        original = B.S.predict_bank
        def checked(artifact, task):
            self.assertEqual(set(task), {'train', 'test'})
            self.assertTrue(all(set(q) == {'input'} for q in task['test']))
            return original(artifact, task)
        with mock.patch.object(B.S, 'predict_bank', side_effect=checked):
            score = B.bank_score(cells, B.IDENTITY, rows, root)
        self.assertEqual((score['correct'], score['errors']), (12, 0))
        self.assertEqual(B.bank_score(cells, B.ZERO, rows, root)['correct'], 6)
        self.assertEqual(B.bank_score(cells, B.ALWAYS_H, rows, root)['correct'], 6)

    def test_runtime_exception_is_never_a_feature_including_legacy_swallow(self):
        root = self.prepare()
        _, p, _, _, rows = B.context(root)
        cells = [p['column'], B.N.spatial_cell(B.h_reference())]
        original_run, original_demo = B.S.run, B.S.Scorer.demo_error
        phase = {'inside_legacy': False}
        def failing_run(*args, **kwargs):
            if phase['inside_legacy']: raise RuntimeError('fixture deliberate demo failure')
            return original_run(*args, **kwargs)
        def legacy(self, *args, **kwargs):
            phase['inside_legacy'] = True
            try: return original_demo(self, *args, **kwargs)
            finally: phase['inside_legacy'] = False
        with mock.patch.object(B.S, 'run', side_effect=failing_run), mock.patch.object(B.S.Scorer, 'demo_error', legacy):
            result = B.bank_score(cells, B.IDENTITY, rows[:1], root)
        self.assertEqual(result['errors'], 1)
        self.assertIsNone(result['rows'][0]['correct'])
        self.assertIsNone(result['rows'][0]['features'])
        self.assertIn('runtime failure', result['rows'][0]['error'])
        self.assertIs(B.S.run, original_run)

    def test_production_generator_structure_only_and_no_outputs(self):
        with (mock.patch.object(B, 'oracle', side_effect=AssertionError('prospective labels')),
              mock.patch.object(B.S, 'run', side_effect=AssertionError('prospective prediction'))):
            panels = B.generate(set())
            self.assertEqual(panels, B.generate(set()))
        self.assertEqual([len(panels[k]) for k in ('h_demos', 'h_search', 'mixed_search', 'development')], [6, 12, 12, 24])
        self.assertEqual(sum(r['family'] == 'C' for r in panels['development']), 12)
        grids = [r['input'] for k in ('h_demos', 'h_search') for r in panels[k]]
        grids += [g for k in ('mixed_search', 'development') for r in panels[k] if 'permutation' not in r for g in r['train_inputs'] + r['query_inputs']]
        self.assertEqual(len({B.T.geometry(g) for g in grids}), len(grids))
        for grid in grids:
            self.assertTrue(all(sum(v != 0 for v in row) <= 1 for row in grid))
            self.assertTrue(all(sum(row[c] != 0 for row in grid) <= 1 for c in range(len(grid[0]))))

    def test_two_stage_freeze_seal_search_selection_and_final_replay(self):
        root = self.prepare()
        original_read, original_oracle = Path.read_bytes, B.oracle
        def guarded_read(path):
            if path == root / B.SEALED:
                self.assertTrue((root / 'HOST_EVALUATION/SEARCH_SELECTION.json').exists())
                self.assertTrue((root / 'HOST_EVALUATION/FINAL_OPEN.json').exists())
            return original_read(path)
        def guarded_oracle(*args):
            self.assertTrue((root / 'HOST_EVALUATION/FINAL_OPEN.json').exists())
            return original_oracle(*args)
        with mock.patch.object(Path, 'read_bytes', guarded_read), mock.patch.object(B, 'oracle', side_effect=guarded_oracle):
            with self.assertRaises(FileNotFoundError): self.consume(root, 'router_founder')
            self.consume(root, 'specialist_founder')
            B.verify(root, 'specialist_founder')
            with self.assertRaisesRegex(ValueError, 'search-perfect'): B.prepare_repair(root, 'specialist_repair')
            B.freeze_roster(root)
            self.consume(root, 'router_founder')
            result = B.finalize(root)
            self.assertEqual(result['charged_positions'], {'native': 173, 'random': 173})
            self.assertEqual(result['counts']['fixed_identity'], 24)
            self.assertEqual(result['identity_margin'], 0)
            self.assertEqual(B.finalize(root, replay=True)['result_hash'], result['result_hash'])
            with self.assertRaises(FileExistsError): B.finalize(root)
            with self.assertRaisesRegex(ValueError, 'closed'): self.consume(root, 'router_repair')

    def test_valid_failures_can_replace_once_full346_positions(self):
        root = self.prepare()
        bad = B.h_reference()
        bad['ray_cast']['src'] = [-9] * 10
        self.consume(root, 'specialist_founder', bad)
        with self.assertRaisesRegex(ValueError, 'replacement must finish'): B.freeze_roster(root)
        B.prepare_repair(root, 'specialist_repair')
        self.consume(root, 'specialist_repair')
        B.freeze_roster(root)
        self.consume(root, 'router_founder', B.ZERO)
        with self.assertRaisesRegex(ValueError, 'replacement must finish'): B.finalize(root)
        B.prepare_repair(root, 'router_repair')
        self.consume(root, 'router_repair')
        repair = B.verify(root, 'router_repair')
        self.assertEqual(repair['genomes']['native']['parents'], repair['parent_genome_ids'])
        self.assertEqual(len(repair['parent_genome_ids']), 1)
        final = B.finalize(root)
        self.assertEqual(final['charged_positions'], {'native': 346, 'random': 346})
        with self.assertRaisesRegex(ValueError, 'unknown slot'): self.consume(root, 'fifth')

    def test_invalid_specialist_is_charged_then_rejects_without_development(self):
        root = self.prepare()
        result = self.consume(root, 'specialist_founder', raw=b'[]')
        self.assertEqual(result['charged_positions'], {'native': 167, 'random': 167})
        with self.assertRaisesRegex(ValueError, 'invalid'): B.prepare_repair(root, 'specialist_repair')
        rejection = B.freeze_roster(root)
        self.assertFalse(rejection['development_opened'])
        with self.assertRaisesRegex(ValueError, 'closed'): self.consume(root, 'router_founder')

    def test_matching_random_blocks_acceptance_and_stop_still_blocks(self):
        root = self.prepare()
        self.consume(root, 'specialist_founder')
        B.freeze_roster(root)
        self.consume(root, 'router_founder')
        control = B.verify(root, 'router_founder')['genomes']['random']['params']
        original = B.bank_score
        def score(cells, params, rows, root):
            value = original(cells, params, rows, root)
            if len(rows) == 24 and params == control: value['correct'] = 24
            return value
        with mock.patch.object(B, 'bank_score', side_effect=score):
            result = B.finalize(root)
        self.assertFalse(result['comparative_acceptance'])
        self.assertEqual(result['counts']['router_founder:random'], 24)
        self.assertEqual(result['counts']['router_founder:native'], 24)
        self.assertEqual(result['counts']['single_C'], 12)
        (root / 'STOP').write_text('stop')
        with self.assertRaisesRegex(RuntimeError, 'STOP'): B.preflight(root)


if __name__ == '__main__': unittest.main()
