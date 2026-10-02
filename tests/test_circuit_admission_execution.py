"""Synthetic fault injection verifies execution gates; no scientific LLM evidence."""
import copy
import unittest
from cellforge.construction.circuit_population import CircuitArchive
from cellforge.construction.joint_weight_population import digest

CELL = {'inputs': ['x'], 'outputs': ['y'], 'layers': [{'W': [[1]], 'b': [0]}]}
CAT = {'a': CELL, 'b': CELL}
SPEC = {'format': 'typed-bit-adhesion/1', 'inputs': ['x'], 'parents': ['a', 'b'],
    'nodes': [
        {'id': 'a', 'kind': 'cell', 'cell': 'a', 'inputs': {'x': {'external': 'x'}}},
        {'id': 'b', 'kind': 'cell', 'cell': 'b', 'inputs': {'x': {'node': 'a', 'port': 'y'}}},
        {'id': 'g', 'kind': 'glue', 'input_ports': ['x'], 'output_ports': ['y'],
         'inputs': {'x': {'node': 'b', 'port': 'y'}}, 'params': {'W': [[1]], 'b': [0]}}],
    'outputs': {'y': {'node': 'g', 'port': 'y'}}, 'newly_authored': ['g.W', 'g.b']}
DEV = {'partition': 'development', 'source': 'synthetic-fault-fixture', 'manifest': 'unit-fixture',
    'cases': [{'id': str(i), 'request': {'bits': [i]},
               'target': {'bits': [i], 'output_ports': ['y']}} for i in (0, 1)]}

class TestCircuitAdmissionExecution(unittest.TestCase):
    def archive(self, dev=DEV):
        baseline = copy.deepcopy(SPEC)
        baseline['nodes'][-1]['params']['W'] = [[0]]
        return CircuitArchive(baseline, {'0': True, '1': False}, ['a', 'b'],
                              founder_weights=CAT, development=dev)

    def envelope(self, archive, dev=DEV):
        return {'development': copy.deepcopy(dev), 'random_seed': 1,
                'provenance': {'model': 'gpt-6.1-sol', 'reasoning': 'low',
                    'parent_hash': archive.live, 'prompt': {'scope': 'synthetic-test', 'parent_hash': archive.live,
                               'founder_hash': digest(CAT), 'development_manifest_hash': digest(dev)},
                    'raw_response': {'spec': copy.deepcopy(SPEC)}}}

    def test_caller_booleans_never_admit_and_attempt_is_charged(self):
        archive = self.archive()
        decision = archive.propose({'spec': SPEC, 'new_numeric_positions': 2},
            valid=True, contribution=True, dev_outcomes={'0': True, '1': True}, arm='llm')
        self.assertFalse(decision['accepted'])
        self.assertEqual(len(archive.charges), 1)
        self.assertEqual(len(archive.history), 1)
        self.assertEqual(archive.records[decision['child']]['outcomes'], {})

    def test_real_execution_receipt_binds_source_weights_and_input_manifest(self):
        archive = self.archive()
        receipt = archive._execute(SPEC, CAT, DEV, 'fixture')
        self.assertEqual(receipt['outcomes'], {'0': True, '1': True})
        self.assertEqual(receipt['manifest_hash'], digest(DEV))
        self.assertEqual(receipt['weights_hash'], digest({'spec': SPEC, 'founders': CAT}))
        self.assertEqual(len(receipt['source_hash']), 64)
        self.assertEqual(receipt['receipt_hash'], digest({k:v for k,v in receipt.items() if k!='receipt_hash'}))
        self.assertEqual(receipt['rows'][1]['execution']['cell_calls'], 2)
        self.assertEqual(receipt['rows'][1]['execution']['glue_calls'], 1)

    def test_synthetic_truth_table_cannot_earn_architecture_even_with_raw_spec(self):
        archive = self.archive()
        decision = archive.propose({'spec': SPEC, 'new_numeric_positions': 2},
            execution=self.envelope(archive), arm='llm')
        self.assertFalse(decision['accepted'])
        self.assertTrue(any('toy truth tables' in reason for reason in decision['reasons']))

    def test_proposal_cannot_replace_frozen_development_environment(self):
        archive = self.archive()
        execution = self.envelope(archive)
        execution['development']['source'] = 'public-training-real'
        decision = archive.propose({'spec': SPEC, 'new_numeric_positions': 2},
                                  execution=execution, arm='llm')
        self.assertFalse(decision['accepted'])
        self.assertTrue(any('constructor-frozen' in r for r in decision['reasons']))

    def test_control_and_ablation_receipts_are_executed_not_supplied(self):
        # Fault-inject the real-source marker to exercise measurements, never
        # propose or claim this synthetic fixture as admitted scientific evidence.
        dev = copy.deepcopy(DEV)
        dev['source'] = 'public-training-real'
        archive = self.archive(dev)
        receipts = archive._measure({'spec': SPEC, 'new_numeric_positions': 2},
                                   self.envelope(archive, dev), archive.live)
        self.assertEqual([r['label'] for r in receipts],
            ['candidate', 'parent', 'random', 'zero', 'ablate-cell:a', 'ablate-cell:b', 'ablate-glue:g'])
        self.assertEqual(sum(receipts[0]['outcomes'].values()), 2)
        for receipt in receipts[1:]:
            self.assertLess(sum(receipt['outcomes'].values()), 2)
            self.assertEqual(receipt['manifest_hash'], receipts[0]['manifest_hash'])
            self.assertEqual(receipt['source_hash'], receipts[0]['source_hash'])
        self.assertEqual(receipts[2]['random_seed'], 1)
        self.assertEqual(archive.history, [archive.live])

    def test_raw_response_mismatch_is_rejected_before_execution(self):
        dev = copy.deepcopy(DEV)
        dev['source'] = 'public-training-real'
        archive = self.archive(dev)
        execution = self.envelope(archive, dev)
        execution['provenance']['raw_response']['spec']['nodes'][-1]['params']['b'] = [1]
        with self.assertRaisesRegex(ValueError, 'exact raw numerical spec'):
            archive._measure({'spec': SPEC, 'new_numeric_positions': 2}, execution, archive.live)

    def test_prompt_must_bind_actual_founder_weights_and_manifest(self):
        dev = copy.deepcopy(DEV)
        dev['source'] = 'public-training-real'
        archive = self.archive(dev)
        execution = self.envelope(archive, dev)
        execution['provenance']['prompt']['founder_hash'] = 'arbitrary-label'
        with self.assertRaisesRegex(ValueError, 'exact raw numerical spec'):
            archive._measure({'spec': SPEC, 'new_numeric_positions': 2}, execution, archive.live)

    def test_target_never_enters_typed_predictor_request(self):
        archive = self.archive()
        dev = copy.deepcopy(DEV)
        dev['cases'][1]['request']['target'] = [1]
        receipt = archive._execute(SPEC, CAT, dev, 'fixture')
        self.assertFalse(receipt['outcomes']['1'])
        self.assertIn('error', receipt['rows'][1])

if __name__ == '__main__': unittest.main()
