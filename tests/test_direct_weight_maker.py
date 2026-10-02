import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
REPO = Path(os.environ.get('CELLFORGE_REPO', r'C:\Users\wkeif\Documents\GitHub\CellForge'))
sys.path.insert(0, str(REPO / 'src'))
from cellforge.ir.graph import Graph
from cellforge.runtime.interp import Interpreter
from cellforge.construction.direct_weight_maker import DirectWeightMaker, attach_affine_router_cells, parameter_scopes


def scalar_graph():
    return Graph(defs={
        'cell': {'kind': 'cell', 'act': 'identity', 'params': {'W1': 'W', 'b1': 'b'}},
        'root': {'kind': 'bundle', 'inputs': ['x'],
                 'nodes': [{'id': 'c', 'use': 'cell', 'in': {'x': '$x'}}],
                 'outputs': {'y': 'c.y'}}},
        params={'W': np.ones((1, 1)), 'b': np.zeros(1)}, entries={'predict': 'root'}, meta={'task': 'clf'})


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.graph = scalar_graph()
        self.fixtures = [{'entry': 'root', 'inputs': {'x': [[1]]}, 'output_shapes': {'y': [1, 1]}}]
        self.diagnostics = {'partition': 'development', 'evidence': 'train-pairs',
            'counterexamples': [{'input': [[1]], 'expected': [[2]], 'predicted': [[1]], 'cells_fired': ['cell']}]}
        self.proposal = {'base': self.graph.hash(), 'actor': 'unit-fixture', 'model': 'not-an-LLM-call',
            'scope': 'cell', 'rationale': 'explicit numeric fixture',
            'ops': [{'op': 'set_param', 'name': 'W', 'value': [[2]]}]}

    def maker(self, profile='flat/1', budget=8):
        return DirectWeightMaker(Path(self.temp.name) / 'ledger', budget=budget, profile=profile)

    def test_numeric_accept_reject_and_parent_integrity(self):
        maker = self.maker()
        before = self.graph.hash()
        result = maker.submit(self.graph, self.proposal, diagnostics=self.diagnostics, fixtures=self.fixtures,
            evaluate=lambda g: {'value': float(Interpreter(g).run_bundle('root', {'x': np.ones((1, 1))})['y'][0, 0])},
            accept=lambda metrics: metrics['value'] == 2)
        self.assertEqual(result['receipt']['status'], 'accepted')
        self.assertEqual(result['next_parent'].params['W'].tolist(), [[2]])
        self.assertEqual(self.graph.hash(), before)
        self.assertEqual(result['weight_changes'][0]['changed_scalars'], 1)
        rejected = maker.submit(self.graph, self.proposal, diagnostics=self.diagnostics, fixtures=self.fixtures,
            evaluate=lambda _: {'value': 2}, accept=lambda _: False)
        self.assertIs(rejected['next_parent'], self.graph)
        self.assertIsNotNone(rejected['candidate'])
        self.assertEqual(len(self.maker().operator.rows), 2)

    def test_invalid_slot_stale_neutral_and_scope_are_charged(self):
        maker = self.maker()
        proposals = [dict(self.proposal, base='stale'),
            dict(self.proposal, ops=[{'op': 'set_slot_gene', 'slot': 8, 'gene': 'x'}]),
            dict(self.proposal, ops=[{'op': 'set_param', 'name': 'W', 'value': [[1]]}]),
            dict(self.proposal, scope='router'),
            dict(self.proposal, ops=[{'op': 'set_param', 'name': 'W', 'value': [[1, 2]]}])]
        for proposal in proposals:
            result = maker.submit(self.graph, proposal, diagnostics=self.diagnostics, fixtures=self.fixtures)
            self.assertEqual(result['receipt']['status'], 'invalid')
            self.assertIs(result['next_parent'], self.graph)
        self.assertEqual(len(maker.operator.rows), 5)

    def test_reserved_report_evidence_rejected(self):
        result = self.maker().submit(self.graph, self.proposal,
            diagnostics=dict(self.diagnostics, heldout_answers=[]), fixtures=self.fixtures)
        self.assertEqual(result['receipt']['status'], 'invalid')

    def test_existing_language_router_identity_and_joint_weight_edits(self):
        from cellforge.runtime.lm import score_texts
        from cellforge.ir.profiles import check_profile
        graph = Graph.load(REPO / 'runs/unified-byte/artifact')
        original = graph.hash()
        calibrated = attach_affine_router_cells(graph)
        check_profile(calibrated, 'ladder-micro/1')
        a, _ = score_texts(graph, ['a'])
        b, _ = score_texts(calibrated, ['a'])
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(graph.hash(), original)
        scopes = parameter_scopes(calibrated)
        router = next(name for name, scope in scopes.items() if scope == {'router'})
        expert = next(name for name, scope in scopes.items() if scope == {'cell'})
        K, V, S = calibrated.meta['ctx_len'], len(calibrated.meta['vocab']), calibrated.meta['state_dim']
        inputs = {'ctx': np.zeros((1, K, V)).tolist(), 'prev': np.zeros((1, V)).tolist(),
                  'step': [[0]], 'state': np.zeros((1, S)).tolist()}
        fixtures = [{'entry': calibrated.entries['step'], 'inputs': inputs,
                     'output_shapes': {'logits': [1, V], 'state': [1, S]}}]
        ops = []
        for name in (router, expert):
            value = calibrated.params[name].copy()
            value.flat[0] += .01
            ops.append({'op': 'set_param', 'name': name, 'value': value.tolist()})
        proposal = dict(self.proposal, base=calibrated.hash(), scope='joint', ops=ops)
        result = self.maker(profile='ladder-micro/1').submit(calibrated, proposal,
            diagnostics=self.diagnostics, fixtures=fixtures)
        self.assertEqual(result['receipt']['status'], 'validated-unscored')
        self.assertEqual(len(result['weight_changes']), 2)
        self.assertIs(result['next_parent'], calibrated)
        self.assertNotEqual(result['candidate'].hash(), calibrated.hash())

    def test_assembly_requires_explicit_readiness_and_fail_closed(self):
        proposal = dict(self.proposal, readiness_evidence={'development_receipt': 'fixture-only'},
                        ops=[{'op': 'set_param', 'name': 'W', 'value': [[3]]}])
        maker = self.maker()
        result = maker.assemble(self.graph, proposal, diagnostics=self.diagnostics, fixtures=self.fixtures)
        self.assertEqual(result['receipt']['status'], 'validated-unscored')
        self.assertIs(result['next_parent'], self.graph)
        proposal.pop('readiness_evidence')
        result = maker.assemble(self.graph, proposal, diagnostics=self.diagnostics, fixtures=self.fixtures)
        self.assertEqual(result['receipt']['status'], 'invalid')

    def test_paired_competition_round_admission(self):
        import copy
        parent = {key: {'partition': 'development', 'manifest': key,
            'source': 'public-training-real' if key == 'anchor' else 'authored-generator',
            'outcomes': {'a': True, 'b': False}, 'false_positive_given_consistency': 0.}
            for key in ('fresh', 'archive', 'anchor')}
        calls = []
        def evaluate(graph):
            calls.append(graph.hash())
            metrics = copy.deepcopy(parent)
            metrics.update(caps_passed=True, fixtures_passed=True)
            metrics['fresh']['outcomes']['b'] = bool(graph.params['W'][0, 0] == 2)
            return metrics
        result = self.maker(budget=3).compare_repairs(self.graph,
            maker_proposal=self.proposal, random_ops=[{'op': 'set_param', 'name': 'W', 'value': [[3]]}],
            diagnostics=self.diagnostics, fixtures=self.fixtures, evaluate=evaluate,
            admission_parent=parent)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['maker_repair']['receipt']['status'], 'accepted')
        for arm in ('breeding_only', 'random_repair'):
            self.assertEqual(result[arm]['receipt']['status'], 'rejected')
            self.assertIs(result[arm]['next_parent'], self.graph)


if __name__ == '__main__':
    unittest.main()

