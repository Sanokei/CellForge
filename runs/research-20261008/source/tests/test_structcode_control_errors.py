"""Fail-closed control qualification using fake execution and fake author responses."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import structcode as X
import struct_pilot as P

SOURCE = 'def run(params, grid, demos):\n    return grid\n'
PARAMS = {'a': [7]}
SHAPES = {'a': [1]}
TASK = {'train': [{'input': [[1]], 'output': [[2]]},
                  {'input': [[3]], 'output': [[4]]}],
        'test': [{'input': [[5]]}]}
GOOD = [([[2]], None), ([[4]], None)]
WRONG = [([[0]], None), ([[0]], None)]


class Box:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, source, params, grid, demos):
        self.calls.append(copy.deepcopy((source, params, grid, demos)))
        return next(self.responses)


class NoRead(dict):
    def __getitem__(self, key):
        raise AssertionError('feedback must not read task data')


class Controls(unittest.TestCase):
    def evaluate(self, arms=None, candidate=None):
        arms = [WRONG] * 4 if arms is None else arms
        responses = list(GOOD if candidate is None else candidate)
        for arm in arms:
            responses.extend(arm)
        box = Box(responses)
        result = X.evaluate(box, SOURCE, SHAPES, PARAMS, copy.deepcopy(TASK))
        return result, box

    def test_valid_wrong_controls_accept_with_exact_seeds_and_leave_one_out(self):
        with patch.object(X, 'random_like', wraps=X.random_like) as random_like:
            result, box = self.evaluate()
        self.assertTrue(result['controls_ok'])
        self.assertEqual(result['control_errors'], [])
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['control_results'], [
            {'arm': arm, 'seed': seed, 'fit': 0, 'errors': []}
            for arm, seed in [('zero', None), ('random', 11), ('random', 12), ('random', 13)]])
        self.assertEqual([call.args[1] for call in random_like.call_args_list], [11, 12, 13])
        expected_params = [PARAMS, X.zeros_like(PARAMS)] + [X.random_like(PARAMS, s) for s in (11, 12, 13)]
        self.assertEqual(len(box.calls), 10)
        for arm, params in enumerate(expected_params):
            for i in range(2):
                source, actual_params, grid, demos = box.calls[arm * 2 + i]
                self.assertEqual(source, SOURCE)
                self.assertEqual(actual_params, params)
                self.assertEqual(grid, TASK['train'][i]['input'])
                self.assertEqual(demos, [TASK['train'][1 - i]])
        self.assertIsNone(result['solved_test'])

    def test_each_arm_error_fails_closed_and_retains_all_attribution(self):
        errors = ['timeout or crash (worker killed)', 'worker exited',
                  'IndexError: bad index', 'ValueError: output values must be integers 0..9']
        for index, (arm, seed) in enumerate([('zero', None), ('random', 11), ('random', 12), ('random', 13)]):
            for error in errors:
                with self.subTest(arm=arm, seed=seed, error=error):
                    arms = [WRONG] * 4
                    arms[index] = [GOOD[0], (None, error)]
                    result, box = self.evaluate(arms)
                    self.assertFalse(result['controls_ok'])
                    self.assertEqual(result['control_errors'], [{'arm': arm, 'seed': seed, 'error': error}])
                    self.assertEqual(result['control_results'][index]['errors'], [error])
                    self.assertEqual(result['control_results'][index]['fit'], 1)
                    self.assertEqual(result['errors'], [])
                    self.assertEqual(len(box.calls), 10)

    def test_multiple_errors_are_not_truncated(self):
        arms = [[(None, 'first'), (None, 'second')]] * 4
        result, _ = self.evaluate(arms)
        self.assertFalse(result['controls_ok'])
        self.assertEqual(len(result['control_errors']), 8)
        for arm in result['control_results']:
            self.assertEqual(arm['errors'], ['first', 'second'])

    def test_any_perfect_control_rejects_without_runtime_error(self):
        for index in range(4):
            with self.subTest(index=index):
                arms = [WRONG] * 4
                arms[index] = GOOD
                result, _ = self.evaluate(arms)
                self.assertFalse(result['controls_ok'])
                self.assertEqual(result['control_errors'], [])
                self.assertEqual(result['control_results'][index]['fit'], 2)

    def test_candidate_failure_skips_every_control(self):
        for candidate in [WRONG, [(None, 'candidate crash'), GOOD[1]]]:
            with self.subTest(candidate=candidate), patch.object(X, 'random_like') as random_like:
                result, box = self.evaluate(candidate=candidate)
                self.assertFalse(result['controls_ok'])
                self.assertEqual(result['control_results'], [])
                self.assertEqual(result['control_errors'], [])
                self.assertIsNone(result['zero_fit'])
                self.assertIsNone(result['random_fit'])
                self.assertEqual(len(box.calls), 2)
                random_like.assert_not_called()

    def test_invalid_source_or_shapes_never_execute(self):
        for source, shapes in [('import os', SHAPES), (SOURCE, {'a': [2]})]:
            box = Box([])
            result = X.evaluate(box, source, shapes, PARAMS, TASK)
            self.assertFalse(result['valid'])
            self.assertFalse(result['controls_ok'])
            self.assertEqual(result['control_results'], [])
            self.assertEqual(box.calls, [])

    def test_feedback_retains_control_errors_without_reading_or_executing(self):
        result, _ = self.evaluate([[(None, 'control crash'), WRONG[1]]] + [WRONG] * 3)
        box = Box([])
        feedback = P.feedback_of(result, NoRead(), box, SOURCE, PARAMS)
        self.assertEqual(feedback, {'control_execution_errors': result['control_errors']})
        self.assertNotIn('tensors_do_not_matter', feedback)
        self.assertEqual(box.calls, [])

    def test_attempt_never_accepts_crashed_controls_with_fake_author_only(self):
        responses = GOOD + [(None, 'control crash'), WRONG[1]] + WRONG * 3
        box = Box(responses * 2)
        entry = [{'source': SOURCE, 'shapes': SHAPES, 'params': PARAMS}]
        with patch.object(P, 'sandbox', return_value=box), patch.object(P.H, 'ask_maker', return_value=entry) as fake_author:
            result = P.attempt(Path('/unused-fake-author'), 'test', 'fake-task', copy.deepcopy(TASK))
        self.assertFalse(result['accepted'])
        self.assertFalse(result['killed'])
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(fake_author.call_count, 2)
        self.assertIn('control_execution_errors', fake_author.call_args_list[1].args[0])
        self.assertNotIn('tensors_do_not_matter', fake_author.call_args_list[1].args[0])
        self.assertEqual(len(box.calls), 20)


class SandboxControls(unittest.TestCase):
    """New regression fixtures; legacy template and test bytes remain unchanged."""

    def evaluate_lut(self, source):
        task = {'train': [{'input': [[1, 3], [0, 1]], 'output': [[2, 3], [0, 2]]},
                          {'input': [[1, 1], [3, 0]], 'output': [[2, 2], [3, 0]]}],
                'test': [{'input': [[0, 1], [1, 3]], 'output': [[0, 2], [2, 3]]}]}
        box = X.Sandbox(timeout=3.0)
        try:
            return X.evaluate(box, source, {'lut': [10]},
                              {'lut': [0, 2, 2, 3, 4, 5, 6, 7, 8, 9]}, task)
        finally:
            box.close()

    def test_raw_lut_rejects_all_three_invalid_random_controls(self):
        source = "def run(params, grid, demos):\n    return np.round(params['lut']).astype(int)[grid]\n"
        result = self.evaluate_lut(source)
        self.assertTrue(result['valid'])
        self.assertEqual(result['demos_fit'], result['n_demos'])
        self.assertTrue(result['solved_test'])
        self.assertFalse(result['controls_ok'])
        self.assertEqual({e['seed'] for e in result['control_errors']}, {11, 12, 13})
        self.assertTrue(all(e['arm'] == 'random' for e in result['control_errors']))
        self.assertTrue(all('output values must be integers 0..9' in e['error']
                            for e in result['control_errors']))

    def test_total_domain_fixture_accepts_valid_wrong_controls(self):
        source = "def run(params, grid, demos):\n    return np.mod(np.round(params['lut']).astype(int), 10)[grid]\n"
        result = self.evaluate_lut(source)
        self.assertTrue(result['valid'])
        self.assertEqual(result['demos_fit'], result['n_demos'])
        self.assertTrue(result['solved_test'])
        self.assertTrue(result['controls_ok'])
        self.assertEqual(result['control_errors'], [])
        self.assertEqual(len(result['control_results']), 4)
        self.assertTrue(all(not c['errors'] and c['fit'] < result['n_demos']
                            for c in result['control_results']))


if __name__ == '__main__':
    unittest.main()
