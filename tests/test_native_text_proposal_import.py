"""Offline text-route ledger checks; fixture tensors are not model proposals."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import native_text_proposal_import as T


class TextImport(unittest.TestCase):
    def root(self):
        parent = Path(tempfile.gettempdir()).resolve()
        path = parent / ('cellforge-text-test-' + uuid.uuid4().hex)
        path.mkdir()
        def cleanup():
            if path.resolve().parent != parent or path.is_symlink():
                raise RuntimeError('test cleanup escaped temporary parent')
            shutil.rmtree(path)
        self.addCleanup(cleanup)
        return path

    def test_invalid_then_valid_fully_charged_and_replayed_without_repeat(self):
        def zeros(shape):
            return [zeros(shape[1:]) for _ in range(shape[0])] if shape else 0.0
        params = {name: {key: zeros(shape) for key, shape in T.N.S.TEMPLATES[name]['shapes'].items()}
                  for name in T.N.SPATIAL_TEMPLATES}
        entry = {'task_id': 'fixture', 'params': params, 'rationale': 'unit fixture, not native author'}
        raw = [json.dumps(entry).encode(), json.dumps([entry]).encode()]
        root = self.root()
        host = root / 'HOST_EVALUATION'
        host.mkdir()
        base = {'reference_cell': T.N.spatial_cell(params), 'response_kind': T.N.SPATIAL_KIND,
                'positions_per_arm': 167, 'task_id': 'fixture', 'parent_hashes': [], 'operator': 'founder'}
        protocol = {'hash': 'a' * 64, 'base_protocol': base, 'random_seeds': [31, 37],
                    'receipt_hashes': {'SOURCE.json': 'b' * 64}}
        origin = {'author_task_name': '/unit-fixture', 'author_model': 'unit-fixture-not-an-LLM',
                  'author_reasoning': 'unknown', 'route': 'unit fixture'}
        row = {'id': 'fixture', 'task': {'train': [{'input': [[1]], 'output': [[1]]}],
                                       'test': [{'input': [[1]], 'output': [[1]]}]}}
        inputs = {'rows': {'search_fitness': [row], 'development_selection': [dict(row, id='other')]}}
        context = (root, host, protocol, origin, raw, inputs)
        with mock.patch.object(T, 'context', return_value=context):
            result = T.consume(root)
            self.assertEqual([v['status'] for v in result['validation']], ['invalid_or_refused', 'valid'])
            self.assertEqual(result['charged_positions'], {'native': 334, 'random': 334})
            self.assertEqual(result['cumulative_native_arc_positions'], {'native': 538, 'random': 538})
            self.assertIsNone(result['actual_author_context_sha256'])
            self.assertEqual(T.verify(root)['hash'], result['hash'])
            with self.assertRaises(FileExistsError):
                T.consume(root)
            raw[1] = b'[]'
            with self.assertRaisesRegex(ValueError, 'attempt/source/accounting binding differs'):
                T.verify(root)
        first = T.N.read_signed(host / 'ATTEMPT1.json')
        second = T.N.read_signed(host / 'ATTEMPT2.json')
        self.assertEqual(second['previous_attempt_hash'], first['hash'])
        self.assertIsNone(first['author_cloud_uuid'])
        self.assertIsNone(second['authoring_credit_usage'])
        self.assertNotEqual(first['random_seed'], second['random_seed'])

    def test_reference_context_cannot_be_labeled_as_actual_author_context(self):
        root = self.root()
        receipts = root / 'TEXT_ROUTE_RECEIPTS'
        receipts.mkdir()
        source = {'author_task_name': '/root/author_spatial_weights_text', 'author_cloud_uuid': None,
                  'author_turn_uuid': None, 'actual_author_context_sha256': 'f' * 64,
                  'actual_author_context_bytes': None, 'reference_context_is_actual_author_context': False,
                  'authoring_credit_usage': None}
        (receipts / 'SOURCE.json').write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, 'unknown actual context/credits required'):
            T.source(root)


if __name__ == '__main__':
    unittest.main()
