import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.construction.niche_weight_bridge import run_comparisons, numeric_proposal, task_seed


class BridgeTests(unittest.TestCase):
    def test_file_protocol_pairing_charges_and_resume(self):
        graph = Graph(defs={'cell': {'kind': 'cell', 'act': 'identity',
            'params': {'W1': 'W', 'b1': 'b'}}, 'root': {'kind': 'bundle', 'inputs': ['x'],
            'nodes': [{'id': 'c', 'use': 'cell', 'in': {'x': '$x'}}],
            'outputs': {'y': 'c.y'}}}, params={'W': np.ones((1, 1)), 'b': np.zeros(1)},
            entries={'predict': 'root'}, meta={'profile': 'flat/1', 'task': 'clf'})
        seeds = []
        def adapter(parent, shown, fresh):
            seeds.append((shown, fresh))
            return {'diagnostics': {'partition': 'development', 'evidence': 'train-pairs'},
                    'fixtures': [{'entry': 'root', 'inputs': {'x': [[1]]}, 'output_shapes': {'y': [1, 1]}}],
                    'evaluate': lambda g: {'score': float(g.params['W'][0, 0])},
                    'accept': lambda p, c: c['score'] > p['score']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'experiment'
            result = run_comparisons(root, {'gene': graph}, adapter, rounds=1)
            self.assertEqual(len(result['pending_inboxes']), 3)
            for file in result['pending_inboxes']:
                path = Path(file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'gene': {'actor': 'human-supervised', 'model': 'fixture',
                    'rationale': 'increase', 'edits': [{'param': 'W', 'index': [0, 0], 'value': 2}]}}))
            self.assertFalse(run_comparisons(root, {'gene': graph}, adapter, rounds=1)['pending_inboxes'])
            for seed in (11, 22, 33):
                row = json.loads((root / 'results' / str(seed) / 'round0.json').read_text())['gene']
                self.assertEqual(row['arms']['maker_repair']['status'], 'accepted')
                self.assertEqual(len(row['arms']), 3)
                self.assertTrue(all(r['kind'] == 'attempt' for r in row['arms'].values()))
            run_comparisons(root, {'gene': graph}, adapter, rounds=1)
            self.assertTrue(all(a != b for a, b in seeds))
            with self.assertRaises(ValueError):
                run_comparisons(root, {'gene': graph}, adapter, rounds=2)
            invalid = numeric_proposal(graph, {'edits': [{'param': 'W', 'index': [-1, 0], 'value': 2}]})
            self.assertIsNone(invalid['ops'])
        self.assertEqual(graph.params['W'][0, 0], 1)


if __name__ == '__main__':
    unittest.main()
