"""Small offline initialization guards; synthetic fixtures are never run seeds."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest import mock
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fresh_inputs as F
import evolution_epochs as E


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreshInputs(unittest.TestCase):
    def setUp(self):
        # Normal inherited directory permissions work inside the Windows sandbox;
        # this bundled Python's TemporaryDirectory(0700) denies its own child I/O.
        temp_root = Path(tempfile.gettempdir()).resolve()
        self.root = temp_root / ('cellforge-fresh-test-' + uuid.uuid4().hex)
        self.root.mkdir()
        def cleanup():
            if self.root.resolve().parent != temp_root or self.root.is_symlink():
                raise RuntimeError('temporary cleanup escaped its declared parent')
            shutil.rmtree(self.root)
        self.addCleanup(cleanup)
        (self.root / 'training').mkdir()
        license_bytes = b'offline test fixture, not a downloaded dataset'
        (self.root / 'LICENSE').write_bytes(license_bytes)
        panels = {}
        for panel, tid in [('search_fitness', '00000001'), ('development_selection', '00000002')]:
            path = self.root / 'training' / (tid + '.json')
            digest = save(path, {'train': [{'input': [[1]], 'output': [[2]]}],
                                'test': [{'input': [[3]], 'output': [[4]]}]})
            payload = path.read_bytes()
            panels[panel] = [{'id': tid, 'sha256': digest, 'downloaded': True,
                'git_blob': hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()}]
        panels['report'] = [{'id': '00000003', 'downloaded': False, 'git_blob': '0' * 40}]
        self.source = {'schema': 'cellforge-public-training-inputs/1',
            'repository': 'https://github.com/arcprize/ARC-AGI-2', 'commit': '0' * 40,
            'license': 'Apache-2.0', 'license_sha256': hashlib.sha256(license_bytes).hexdigest(),
            'evaluation_downloaded': False, 'panels': panels}
        self.manifest = {'schema': 'cellforge-fresh-inputs/1', 'source': str(self.root / 'SOURCE.json'),
            'source_sha256': save(self.root / 'SOURCE.json', self.source),
            'baseline': F.make_baseline(['glyph_recolor'], 31)}
        self.refresh()

    def refresh(self):
        self.manifest['source_sha256'] = save(self.root / 'SOURCE.json', self.source)
        self.config = {'manifest': str(self.root / 'inputs.json'),
                       'sha256': save(self.root / 'inputs.json', self.manifest)}

    def test_reserved_report_needs_no_file_or_labels(self):
        loaded = F.load(self.config)
        self.assertEqual(loaded['report_ids'], ['00000003'])
        self.assertEqual(loaded['rows']['report'], [])
        self.assertFalse((self.root / 'training' / '00000003.json').exists())
        self.assertFalse(loaded['baseline']['llm_provenance'])

    def test_task_tamper_rejected_before_campaign_creation(self):
        (self.root / 'training' / '00000001.json').write_bytes(b'{}')
        campaign = self.root / 'weights-rejected'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            E.execute_phase(campaign, {'enabled': True, 'fresh_inputs': self.config}, 0)
        self.assertFalse(campaign.exists())

    def test_changed_weights_cannot_be_relabelled_as_declared_baseline(self):
        self.manifest['baseline']['cells'][0]['origin'] = 'LLM-written'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'random construction'):
            F.load(self.config)

    def test_partition_overlap_is_rejected(self):
        self.source['panels']['report'][0]['id'] = '00000001'
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            F.load(self.config)


class FreshResumeCLI(unittest.TestCase):
    def test_saved_fresh_inputs_block_historical_seed_without_cli_flag(self):
        import head_driver as D
        resumed = SimpleNamespace(fresh_inputs={'restored_from': 'FRESH_INPUTS.json'},
                                  phase_context=None, registry=[], worker_pool=mock.Mock(),
                                  run=mock.Mock(), save=mock.Mock())
        argv = ['head_driver.py', 'weights-existing-fresh', '--seed-from', 'historical-run']
        with (mock.patch.object(sys, 'argv', argv),
              mock.patch.object(D, 'Driver', return_value=resumed) as constructor,
              mock.patch.object(D.H, 'load', side_effect=AssertionError('historical seed must not be read')) as load):
            with self.assertRaisesRegex(RuntimeError, 'including on resume'):
                D.main()
        self.assertIsNone(constructor.call_args.kwargs['fresh_inputs'])
        load.assert_not_called()
        resumed.run.assert_not_called()
        resumed.save.assert_not_called()
        resumed.worker_pool.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
