"""Diagnostic mechanics only: these fixtures are not new research tasks."""
import sys
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from router_headroom_diagnostic import analyze, verify_search_replay


def packet(*rows):
    return {'expert_ids': ['C', 'H'], 'search_examples': [
        {'task_id': str(i), 'features': f, 'successful_expert_ids': s}
        for i, (f, s) in enumerate(rows)]}


class RouterHeadroomTests(unittest.TestCase):
    def test_identity_saturates_complementary_experts(self):
        r = analyze(packet(([0, -1], ['C']), ([-1, 0], ['H'])))
        self.assertEqual(r['single_expert_correct'], [1, 1])
        self.assertEqual(r['identity_correct'], 2)
        self.assertEqual(r['oracle_minus_identity'], 0)

    def test_conflicting_identical_features_bound_ceiling(self):
        r = analyze(packet(([0, 0], ['C']), ([0, 0], ['H'])))
        self.assertEqual(r['feature_deterministic_ceiling'], 1)
        self.assertEqual(len(r['contradictory_feature_groups']), 1)
        self.assertEqual(r['identity_misses_solved_by_expert'], ['1'])

    def test_overlapping_success_sets_are_not_contradictory(self):
        r = analyze(packet(([0, 0], ['C', 'H']), ([0, 0], ['H'])))
        self.assertEqual(r['feature_deterministic_ceiling'], 2)
        self.assertEqual(r['contradictory_feature_groups'], [])

    def test_unsolvable_rows_do_not_create_routing_conflict(self):
        r = analyze(packet(([0, 0], []), ([0, 0], ['C'])))
        self.assertEqual(r['oracle_correct'], 1)
        self.assertEqual(r['contradictory_feature_groups'], [])

    def test_signed_zero_is_same_feature(self):
        r = analyze(packet(([0., -1], ['C']), ([-0., -1], ['C'])))
        self.assertEqual(r['exact_feature_groups'], 1)

    def test_near_features_not_silently_rounded(self):
        r = analyze(packet(([0, -1], ['C']), ([1e-15, -1], ['H'])))
        self.assertEqual(r['exact_feature_groups'], 2)

    def test_replay_rejects_mismatched_runtime_before_import(self):
        def signed(body):
            return dict(body, hash=hashlib.sha256(json.dumps(body, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'HOST_EVALUATION').mkdir()
            (root / 'AUTHORING').mkdir()
            raw, search = b'prefix\n{}\n', b'[]\n'
            protocol = signed({'source_hashes': {'skillcore.py': 'wrong'},
                'file_hashes': {'MIXED_SEARCH.json': hashlib.sha256(search).hexdigest()}})
            roster = signed({'protocol_hash': protocol['hash'], 'development_used': False})
            receipt = signed({'protocol_hash': protocol['hash'], 'roster_hash': roster['hash'],
                              'payload_sha256': hashlib.sha256(raw).hexdigest()})
            for name, obj in [('HOST_EVALUATION/PROTOCOL.json', protocol),
                              ('HOST_EVALUATION/ROSTER.json', roster),
                              ('AUTHORING/router_founder.json', receipt)]:
                (root / name).write_text(json.dumps(obj))
            (root / 'AUTHORING/router_founder.txt').write_bytes(raw)
            (root / 'MIXED_SEARCH.json').write_bytes(search)
            runtime = root / 'skillcore.py'
            runtime.write_text('raise AssertionError("must never import")')
            with self.assertRaisesRegex(ValueError, 'frozen runtime hash mismatch'):
                verify_search_replay(root, runtime)

    def test_bad_rows_fail_closed(self):
        for f, s in (([float('nan'), 0], ['C']), ([0, None], ['C']),
                     ([False, 0], ['C']), ([0, 0], ['unknown']), ([0, 0], ['C', 'C'])):
            with self.subTest(f=f, s=s), self.assertRaises(ValueError):
                analyze(packet((f, s)))


if __name__ == '__main__':
    unittest.main()
