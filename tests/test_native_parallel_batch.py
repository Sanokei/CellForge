"""Bounded synthetic checks; never read a live author response or call a model."""
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
import native_parallel_batch as P


class ParallelBatch(unittest.TestCase):
    def root(self):
        parent = Path(tempfile.gettempdir()).resolve()
        path = parent / ('cellforge-parallel-test-' + uuid.uuid4().hex)
        path.mkdir()
        def cleanup():
            if path.resolve().parent != parent or path.is_symlink():
                raise RuntimeError('test cleanup escaped temporary parent')
            shutil.rmtree(path)
        self.addCleanup(cleanup)
        return path

    def params(self):
        def zeros(shape):
            return [zeros(shape[1:]) for _ in range(shape[0])] if shape else 0.0
        params = {name: {key: zeros(shape) for key, shape in P.N.S.TEMPLATES[name]['shapes'].items()}
                  for name in P.N.SPATIAL_TEMPLATES}
        for row in params['tile_orient']['orient']:
            for vector in row:
                vector[4:] = [-9.0] * 4
        return params

    def raw(self, params):
        return P.compact([{'task_id': 'f5b8619d', 'params': params, 'rationale': 'synthetic fixture, not an author response'}])

    def origin(self):
        return {'author_task_name': 'unit-fixture', 'author_model': 'unit-fixture-not-an-LLM',
                'author_reasoning': 'unknown', 'route': 'unit fixture', 'context_sha256': None,
                'context_provenance': 'parent_transcription_exact_bytes_unknown',
                'response_provenance': 'synthetic test only', 'credit_usage': None}

    def fixture(self, root):
        parent = P.N.spatial_cell(self.params())
        protocol = {'hash': 'a' * 64, 'parent_cell': parent, 'base_protocol': {'reference_cell': parent},
                    'random_seeds': P.SEEDS, 'author_inputs': {slot: {'AUTHOR_INPUT.txt': {'sha256': 'b' * 64}} for slot in P.SLOTS}}
        row = {'id': 'f5b8619d', 'task': {'train': [{'input': [[1, 0], [0, 0]], 'output': [[1, 0], [0, 0]]}],
                                        'test': [{'input': [[1, 0], [0, 0]], 'output': [[1, 0], [0, 0]]}]}}
        return root, protocol, {'rows': {'search_fitness': [row], 'development_selection': [dict(row, id='development-fixture')]}}

    def test_167_total_103_free_64_fixed_and_shared_vocabulary(self):
        reference = self.params()
        rng = mock.Mock()
        rng.uniform.return_value = 0.25
        with mock.patch.object(P.random, 'Random', return_value=rng):
            params = P.random_params(reference, 13)
        self.assertEqual(rng.uniform.call_count, 103)
        cell = P.N.spatial_cell(params)
        self.assertEqual(sum(P.N.S.positions(leaf) for leaf in cell['parts']), 167)
        self.assertEqual(sum(len(vector[4:]) for row in params['tile_orient']['orient'] for vector in row), 64)
        P.validate(self.raw(params), reference)
        schema = P.response_schema()['items']['properties']['params']['properties']['tile_orient']['properties']['orient']
        self.assertEqual(schema['items']['items']['prefixItems'][4:], [{'type': 'number', 'const': -9}] * 4)
        self.assertFalse(schema['items']['items']['items'])

    def test_forbidden_orientation_is_rejected_not_silently_repaired(self):
        params = self.params()
        params['tile_orient']['orient'][3][3][7] = 0.0
        before = copy.deepcopy(params)
        with self.assertRaisesRegex(ValueError, 'literal -9'):
            P.validate(self.raw(params), self.params())
        self.assertEqual(params, before)

    def test_all_frozen_random_controls_run_on_rectangles_and_ties_choose_safe_index(self):
        grids = [[[1, 0, 0], [0, 0, 2]], [[1], [0], [2]], [[0, 1, 0, 2]]]
        for seed in P.SEEDS.values():
            params = P.random_params(self.params(), seed)
            P.validate(self.raw(params), self.params())
            cell = P.N.spatial_cell(params)
            for grid in grids:
                prediction = P.N.S.run(cell, grid, [])
                self.assertGreater(len(prediction), 0)
                self.assertTrue(all(len(row) == len(prediction[0]) for row in prediction))
        tied = self.params()
        tied['tile_orient']['orient'] = [[[-9.0] * 8 for _ in range(4)] for _ in range(4)]
        P.validate(self.raw(tied), self.params())
        P.N.S.run(P.N.spatial_cell(tied), grids[0], [])

    def test_invalid_and_valid_slots_fully_charge_and_replay_without_repeat(self):
        root = self.root()
        (root / 'HOST_EVALUATION').mkdir()
        source = root / 'source.json'
        source.write_bytes(P.compact(self.origin()))
        response = root / 'response.json'
        fixture = self.fixture(root)
        with mock.patch.object(P, 'context', return_value=fixture):
            for slot, raw in zip(P.SLOTS, (b'[]', self.raw(self.params()), b'no JSON', self.raw(self.params()))):
                response.write_bytes(raw)
                result = P.consume(root, slot, response, source)
                self.assertEqual(result['charged_positions'], {'native': 167, 'random': 167})
                self.assertEqual(P.verify(root, slot)['result_hash'], result['result_hash'])
                with self.assertRaises(FileExistsError):
                    P.consume(root, slot, response, source)
            batch = P.finalize(root)
            self.assertEqual(batch['charged_positions'], {'native': 668, 'random': 668})
            self.assertEqual(batch['cumulative_native_arc_positions'], {'native': 1206, 'random': 1206})
            self.assertEqual(batch['distinct_candidate_count'], 1)
            self.assertEqual(len(batch['valid_candidate_ids']), 2)
            self.assertFalse(batch['automatic_admission'])

    def test_changed_response_is_rejected_by_replay(self):
        root = self.root()
        (root / 'HOST_EVALUATION').mkdir()
        source, response = root / 'source.json', root / 'response.json'
        source.write_bytes(P.compact(self.origin()))
        response.write_bytes(self.raw(self.params()))
        with mock.patch.object(P, 'context', return_value=self.fixture(root)):
            P.consume(root, 'author01', response, source)
            (root / 'HOST_EVALUATION/author01/RESPONSE.bin').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError, 'capture/accounting changed'):
                P.verify(root, 'author01')

    def test_actual_context_is_never_falsely_rebound_to_slot_reference(self):
        source = self.origin()
        source['context_provenance'] = 'verbatim_frozen_payload_parent_attested'
        source['context_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'differs from frozen context'):
            P.parse_source(P.compact(source), 'b' * 64)
        source['context_provenance'] = 'parent_transcription_exact_bytes_unknown'
        source['context_sha256'] = None
        P.parse_source(P.compact(source), 'b' * 64)

    def test_stop_file_remains_effective(self):
        root = self.root()
        (root / 'STOP').write_text('stop')
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            P.N.stop(root)


if __name__ == '__main__':
    unittest.main()
