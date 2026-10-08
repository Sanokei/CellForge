"""Training-only counterexample selection, without executing model-written code."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import struct_pilot as P


class TrainingOnly(dict):
    def __getitem__(self, key):
        if key != 'train':
            raise AssertionError('query/report access in revision feedback')
        return super().__getitem__(key)


class Box:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, source, params, grid, demos):
        self.calls.append(copy.deepcopy((grid, demos)))
        return next(self.responses)


class Feedback(unittest.TestCase):
    def task(self):
        return TrainingOnly(train=[{'input': [[1]], 'output': [[2]]}, {'input': [[3]], 'output': [[4]]}])

    def result(self):
        return dict(valid=True, demos_fit=1, n_demos=2, errors=[])

    def test_later_failure_and_leave_one_out_context(self):
        task = self.task()
        before = copy.deepcopy(task)
        box = Box([([[2]], None), ([[5]], None)])
        result = P.feedback_of(self.result(), task, box, '', {})
        failure = result['failing_demonstration']
        self.assertEqual(failure['index'], 1)
        self.assertEqual(failure['differing_pixels'], 1)
        self.assertEqual(failure['grid_differences'], [{'row': 0, 'column': 0, 'expected': 4, 'predicted': 5}])
        self.assertEqual(box.calls[0][1], [task['train'][1]])
        self.assertEqual(box.calls[1][1], [task['train'][0]])
        self.assertEqual(task, before)

    def test_runtime_error_is_preserved(self):
        box = Box([(None, 'worker timeout')])
        failure = P.feedback_of(self.result(), self.task(), box, '', {})['failing_demonstration']
        self.assertEqual(failure['error'], 'worker timeout')
        self.assertIsNone(failure['predicted'])
        self.assertNotIn('differing_pixels', failure)

    def test_shape_failure_has_no_fake_pixel_count(self):
        box = Box([([[2, 2]], None)])
        failure = P.feedback_of(self.result(), self.task(), box, '', {})['failing_demonstration']
        self.assertFalse(failure['shape_matches'])
        self.assertNotIn('grid_differences', failure)

    def test_difference_payload_is_bounded(self):
        task = TrainingOnly(train=[{'input': [[0] * 10] * 10, 'output': [[1] * 10] * 10}])
        box = Box([([[0] * 10] * 10, None)])
        failure = P.feedback_of(self.result(), task, box, '', {})['failing_demonstration']
        self.assertEqual(failure['differing_pixels'], 100)
        self.assertEqual(len(failure['grid_differences']), 64)
        self.assertTrue(failure['differences_truncated'])

    def test_nondeterministic_replay_does_not_invent_failure(self):
        box = Box([([[2]], None), ([[4]], None)])
        result = P.feedback_of(self.result(), self.task(), box, '', {})
        self.assertIn('replay_inconsistency', result)
        self.assertNotIn('failing_demonstration', result)

    def test_invalid_and_control_feedback_never_execute(self):
        box = Box([])
        rejected = dict(valid=False, errors=['bad source'])
        self.assertEqual(P.feedback_of(rejected, self.task(), box, '', {}), {'rejected': ['bad source']})
        result = dict(valid=True, demos_fit=2, n_demos=2, zero_fit=2, random_fit=1)
        self.assertIn('tensors_do_not_matter', P.feedback_of(result, self.task(), box, '', {}))
        self.assertEqual(box.calls, [])


if __name__ == '__main__':
    unittest.main()
