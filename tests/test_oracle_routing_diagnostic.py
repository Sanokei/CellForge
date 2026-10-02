import copy
import os
import sys
import unittest
from pathlib import Path

REPO = Path(os.environ.get('CELLFORGE_DIAGNOSTIC_REPO', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from oracle_routing_diagnostic import probe, score_outputs, summarize


class DiagnosticTests(unittest.TestCase):
    def test_score_boundary_and_decomposition(self):
        # Complementary parents, one routing miss, one lost inherited skill.
        records = [dict(scores={
            'left': {'actual': True, 'oracle': True},
            'right': {'actual': False, 'oracle': False},
            'child': {'actual': False, 'oracle': True}}),
            dict(scores={
            'left': {'actual': False, 'oracle': False},
            'right': {'actual': True, 'oracle': True},
            'child': {'actual': False, 'oracle': False}})]
        report = summarize(records, ['left', 'right', 'child'])
        self.assertEqual(report['complementarity_over_best_parent_expert_oracle'], 1)
        self.assertEqual(report['rows'][2]['routing_misses'], 1)
        self.assertEqual(report['rows'][2]['parent_union_capability_missing'], 1)
        observation = {'actual': [[[0]]], 'experts': [{'index': 0, 'output': [[1]]}]}
        before = copy.deepcopy(observation)
        self.assertTrue(score_outputs(observation, [[1]])['oracle'])
        self.assertFalse(score_outputs(observation, [[0]])['oracle'])
        self.assertEqual(observation, before)

    def test_target_fields_rejected_before_inference(self):
        with self.assertRaises(ValueError):
            probe(None, {'train': [], 'input': [[0]], 'output': [[1]]})
        with self.assertRaises(ValueError):
            probe(None, {'train': [], 'input': [[0]], 'test': []})

    def test_real_graph_parity_reset_and_target_independence(self):
        from cellforge.ir.graph import Graph
        from cellforge.runtime.grid import GridPredictor
        from measure_grid_expanded_capacity import family
        graph = Graph.load(REPO / 'runs/arc-grid-genes/expanded-breeding-v2/parent-crop-fill')
        original = graph.hash()
        tasks = family(3507000, 1)
        task = tasks[0]
        view = {'train': task['train'], 'input': task['test'][0]['input']}
        compiled = probe(graph, view)
        reference = probe(graph, view, backend='reference')
        self.assertEqual(compiled, reference)
        self.assertEqual(compiled['actual'], GridPredictor(graph)(view['train'], view['input']))
        other = tasks[-1]
        probe(graph, {'train': other['train'], 'input': other['test'][0]['input']})
        self.assertEqual(compiled, probe(graph, view))
        self.assertEqual(original, graph.hash())
        self.assertEqual(len(compiled['experts']), 15)
        self.assertTrue(score_outputs(compiled, task['test'][0]['output'])['oracle'])
        for index in (0, 8, 14):
            forced = graph.copy()
            for node in forced.defs[forced.entries['grid_step']]['nodes']:
                if node['use'] == 'dispatch':
                    branch = node['args']['branches'][index]
                    node['args']['branches'] = [branch] * len(node['args']['branches'])
            expected = compiled['experts'][index]['output']
            self.assertEqual(GridPredictor(forced)(view['train'], view['input']), [expected, expected])
        # Oracle target changes affect scoring only, never observed outputs.
        score_outputs(compiled, [[9]])
        self.assertEqual(compiled, probe(graph, view))


if __name__ == '__main__':
    unittest.main()
