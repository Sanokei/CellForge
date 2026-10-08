"""Offline literal parsing and one-attempt accounting; no native author or API."""
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
import native_proposal_import as N

PARAMS = {'sim': [[0.0] * 10 for _ in range(10)], 'temp': [0.0], 'keep': [0.0]}


class NativeImport(unittest.TestCase):
    def raw(self, params=None):
        return json.dumps([{'task_id': '00000001', 'params': PARAMS if params is None else params,
                            'rationale': 'unit fixture, not a model proposal'}]).encode()

    def test_complete_literal_candidate(self):
        self.assertEqual(N.validate_response(self.raw(), '00000001', PARAMS)['params'], PARAMS)

    def test_nonliteral_out_of_range_and_shape_mismatch_rejected(self):
        for value in (True, '1', float('nan'), float('inf'), 10, [1, 2]):
            with self.subTest(value=value):
                params = copy.deepcopy(PARAMS)
                params['temp'] = [value]
                with self.assertRaises((ValueError, TypeError)):
                    N.validate_response(self.raw(params), '00000001', PARAMS)

    def test_duplicate_fields_and_wrong_task_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            N.validate_response(b'{"a":1,"a":2}', '00000001', PARAMS)
        with self.assertRaisesRegex(ValueError, 'bound search task'):
            N.validate_response(self.raw(), '00000002', PARAMS)

    def test_v2_changed_parent_or_feedback_evidence_refuses(self):
        protocol = {'bound_evidence': {'fixture.json': N.digest_bytes(b'original')}}
        with mock.patch.object(Path, 'read_bytes', return_value=b'original'):
            N.verify_bound_evidence(protocol)
        with mock.patch.object(Path, 'read_bytes', return_value=b'changed'):
            with self.assertRaisesRegex(ValueError, 'previous evidence changed'):
                N.verify_bound_evidence(protocol)

    def test_v2_replacement_retains_parent_and_cumulative_full_charges(self):
        parent = N.S.make_cell('assoc_induct', PARAMS)
        protocol = {'reference_cell': parent, 'random_seed': 23, 'task_id': '00000001',
                    'parent_hashes': [parent['id']], 'native_sequence_number': 2,
                    'operator': 'complete_tensor_replacement',
                    'cumulative_positions_after_attempt': {'native': 204, 'random': 204}}
        row = {'id': '00000001', 'task': {'train': [{'input': [[1]], 'output': [[2]]}],
                                          'test': [{'input': [[1]], 'output': [[2]]}]}}
        rows = {'search_fitness': [row], 'development_selection': [dict(row, id='00000002')]}
        with mock.patch.object(N, 'stop'):
            result = N.compute(Path('unused'), protocol, {'rows': rows}, b'[]')
        self.assertEqual(result['validation']['status'], 'invalid_or_refused')
        self.assertEqual(result['charged_positions'], {'native': 102, 'random': 102})
        self.assertEqual(result['cumulative_charged_positions'], {'native': 204, 'random': 204})
        self.assertEqual(result['parent_hashes'], [parent['id']])
        self.assertEqual(result['cells']['reference'], parent)
        self.assertNotIn('native', result['cells'])

    def spatial_params(self):
        def tensor(shape):
            return [tensor(shape[1:]) for _ in range(shape[0])] if shape else 0.0
        return {name: {key: tensor(shape) for key, shape in N.S.TEMPLATES[name]['shapes'].items()}
                for name in N.SPATIAL_TEMPLATES}

    def temporary_root(self):
        parent = Path(tempfile.gettempdir()).resolve()
        root = parent / ('cellforge-native-test-' + uuid.uuid4().hex)
        root.mkdir()
        def cleanup():
            if root.resolve().parent != parent or root.is_symlink():
                raise RuntimeError('temporary cleanup escaped its parent')
            shutil.rmtree(root)
        self.addCleanup(cleanup)
        return root

    def spatial_fixture(self):
        params = self.spatial_params()
        protocol = {'hash': 'b' * 64, 'reference_cell': N.spatial_cell(params), 'random_seed': 29,
            'task_id': '00000001', 'response_kind': N.SPATIAL_KIND, 'positions_per_arm': 167,
            'parent_hashes': [], 'operator': 'founder', 'native_sequence_number': 3,
            'cumulative_positions_after_attempt': {'native': 371, 'random': 371},
            'authoring_route': 'unit fixture; not a model call'}
        row = {'id': '00000001', 'task': {'train': [{'input': [[1, 0], [0, 0]], 'output': [[1, 0], [0, 0]]}],
                'test': [{'input': [[1, 0], [0, 0]], 'output': [[1, 0], [0, 0]]}]}}
        return params, protocol, {'rows': {'search_fitness': [row],
                                         'development_selection': [dict(row, id='00000002')]}}

    def test_spatial_complete_layout_and_fixed_composition(self):
        params = self.spatial_params()
        self.assertEqual(N.validate_spatial_response(self.raw(params), '00000001', params)['params'], params)
        cell = N.spatial_cell(params)
        self.assertEqual([p['template'] for p in cell['parts']], list(N.SPATIAL_TEMPLATES))
        self.assertEqual(sum(N.S.positions(p) for p in cell['parts']), 167)
        self.assertEqual(N.charged_positions({'response_kind': N.SPATIAL_KIND}), {'native': 167, 'random': 167})
        with self.assertRaises(ValueError):
            N.charged_positions({'response_kind': N.SPATIAL_KIND, 'positions_per_arm': 0})

    def test_spatial_rejects_partial_or_nonliteral_unused_slots(self):
        reference = self.spatial_params()
        changes = [lambda p: p.pop('ray_cast'),
                   lambda p: p.update(other={}),
                   lambda p: p['tile_orient']['orient'][3][3].pop(),
                   lambda p: p['tile_orient']['orient'][3][3].__setitem__(7, True),
                   lambda p: p['ray_cast']['src'].__setitem__(0, 10),
                   lambda p: p['ray_cast']['src'].__setitem__(0, float('nan'))]
        for change in changes:
            params = copy.deepcopy(reference)
            change(params)
            with self.assertRaises((ValueError, TypeError)):
                N.validate_spatial_response(self.raw(params), '00000001', reference)

    def test_spatial_author_packet_rejects_solution_advice_or_baseline_fields(self):
        demonstrations = [{'input': [[1]], 'output': [[2]]}]
        packet = N.expected_spatial_packet(demonstrations)
        schema = N.expected_spatial_schema()
        N.validate_spatial_packet(packet, schema, demonstrations)
        for field, value in [('instruction', packet['instruction'] + ' Set paint[8] positive.'),
                             ('baseline_params', self.spatial_params()),
                             ('demonstrations', demonstrations + [{'input': [[9]], 'output': [[8]]}])]:
            changed = copy.deepcopy(packet)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, 'exact search demonstrations'):
                N.validate_spatial_packet(changed, schema, demonstrations)

    def test_spatial_random_draws_all_167_positions_and_labels_baseline(self):
        params, protocol, inputs = self.spatial_fixture()
        draws = mock.Mock(return_value=0.25)
        with mock.patch.object(N.random, 'Random') as random_class, mock.patch.object(N, 'stop'):
            random_class.return_value.uniform = draws
            result = N.compute(Path('unused'), protocol, inputs, self.raw(params))
        self.assertEqual(draws.call_count, 167)
        self.assertIn('hand_coded', result['cells'])
        self.assertNotIn('reference', result['cells'])
        self.assertEqual(result['fixed_diagnostic_arms'], ['zero', 'hand_coded'])
        self.assertEqual(result['charged_positions'], {'native': 167, 'random': 167})
        self.assertFalse(result['scientific_acceptance'])

    def test_spatial_invalid_attempt_charged_once_and_saved_result_replays(self):
        for response in (b'[]', self.raw(self.spatial_params())):
            with self.subTest(valid=response != b'[]'):
                root = self.temporary_root()
                host = root / 'HOST_EVALUATION'
                host.mkdir()
                (root / 'NATIVE_PROPOSAL.json').write_bytes(response)
                _, protocol, inputs = self.spatial_fixture()
                identity = '00000000-0000-0000-0000-000000000001'
                with mock.patch.object(N, 'context', return_value=(root, host, protocol, inputs)):
                    result = N.consume(root, identity, identity, 'unit-fixture-not-an-LLM', 'unknown')
                    self.assertEqual(N.verify(root)['result_hash'], result['result_hash'])
                    with self.assertRaises(FileExistsError):
                        N.consume(root, identity, identity, 'unit-fixture-not-an-LLM', 'unknown')
                attempt = N.read_signed(host / 'ATTEMPT.json')
                self.assertEqual(attempt['charged_positions'], {'native': 167, 'random': 167})
                self.assertIsNone(attempt['authoring_credit_usage'])
                saved = N.read_signed(host / 'RESULT.json')
                self.assertEqual(saved['cumulative_charged_positions'], {'native': 371, 'random': 371})

    def test_predictor_gets_only_query_input_and_demos_and_crashes_stay_unknown(self):
        rows = [{'id': 'fixture', 'task': {'train': [{'input': [[1]], 'output': [[2]]}],
                                         'test': [{'input': [[3]], 'output': [[9]]}]}}]
        def crash(cell, grid, demos):
            self.assertEqual(grid, [[3]])
            self.assertEqual(demos, [{'input': [[1]], 'output': [[2]]}])
            raise RuntimeError('fixture failure')
        with mock.patch.object(N.S, 'run', side_effect=crash), mock.patch.object(N, 'stop'):
            result = N.score({}, rows, Path('unused'))
        self.assertIsNone(result['fixture']['correct'])
        self.assertEqual(result['fixture']['errors'][0]['type'], 'RuntimeError')

    def test_invalid_response_consumes_full_attempt_and_cannot_repeat(self):
        temp_root = Path(tempfile.gettempdir()).resolve()
        root = temp_root / ('cellforge-native-test-' + uuid.uuid4().hex)
        root.mkdir()
        def cleanup():
            if root.resolve().parent != temp_root or root.is_symlink():
                raise RuntimeError('temporary cleanup escaped its parent')
            shutil.rmtree(root)
        self.addCleanup(cleanup)
        host = root / 'HOST_EVALUATION'
        host.mkdir()
        (root / 'NATIVE_PROPOSAL.json').write_bytes(b'[]')
        reference = N.S.make_cell('assoc_induct', PARAMS)
        protocol = {'hash': 'a' * 64, 'reference_cell': reference, 'random_seed': 17,
                    'task_id': '00000001', 'authoring_route': 'unit fixture, not a model call'}
        row = {'id': '00000001', 'task': {'train': [{'input': [[1]], 'output': [[2]]}],
                                          'test': [{'input': [[1]], 'output': [[2]]}]}}
        rows = {'search_fitness': [row], 'development_selection': [dict(row, id='00000002')]}
        identity = '00000000-0000-0000-0000-000000000001'
        with mock.patch.object(N, 'context', return_value=(root, host, protocol, {'rows': rows})):
            result = N.consume(root, identity, identity, 'unit-fixture-not-an-LLM', 'unknown')
            with self.assertRaises(FileExistsError):
                N.consume(root, identity, identity, 'unit-fixture-not-an-LLM', 'unknown')
        self.assertEqual(result['validation']['status'], 'invalid_or_refused')
        attempt = N.read_signed(host / 'ATTEMPT.json')
        self.assertEqual(attempt['charged_positions'], {'native': 102, 'random': 102})
        self.assertIsNone(attempt['authoring_credit_usage'])
        self.assertNotIn('native', N.read_signed(host / 'RESULT.json')['cells'])


if __name__ == '__main__':
    unittest.main()
