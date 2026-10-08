"""Independent synthetic fixtures for a training-only mathematical diagnostic."""
import unittest
from scripts.nca_equivariance_diagnostic import analyze, at_radius, signature


def task(tid, source, target):
    return {'task_id': tid, 'train': [{'input': source, 'output': target}]}


class LocalityDiagnosticTests(unittest.TestCase):
    def test_outside_is_not_black_and_border_geometry_is_visible(self):
        self.assertNotEqual(signature([[0]], 0, 0, 1), signature([[0]*3]*3, 1, 1, 1))

    def test_translation_equivalent_positions_cannot_have_distinct_outputs(self):
        t = task('00000001', [[0]*5], [[0,0,1,0,0]])
        result = at_radius([t], 1)
        self.assertEqual(result['conflict_classes'], 1)
        self.assertEqual(result['pixel_error_lower_bound'], 1)
        self.assertEqual(result['within_task']['00000001']['pixel_error_lower_bound'], 1)
        self.assertEqual(at_radius([t], 2)['conflict_classes'], 0)

    def test_shared_weights_cross_task_conflict_is_not_individual_conflict(self):
        panel = [task('00000001', [[1]], [[2]]), task('00000002', [[1]], [[3]])]
        r = at_radius(panel, 8)
        self.assertEqual(r['pixel_error_lower_bound'], 1)
        self.assertTrue(all(v['conflict_classes'] == 0 for v in r['within_task'].values()))
        self.assertIsNone(analyze(panel)['minimum_shared_radius_not_ruled_out'])

    def test_constant_labels_have_no_conflict_despite_repeated_signatures(self):
        r = at_radius([task('00000001', [[0]*5], [[1]*5])], 0)
        self.assertEqual(r['repeated_signature_classes'], 1)
        self.assertEqual(r['pixel_error_lower_bound'], 0)

    def test_shape_exclusions_stay_in_full_accounting(self):
        r = analyze([task('00000001', [[0]], [[0,0]]), task('00000002', [[0]], [[1]])])
        self.assertEqual((r['total_tasks'], r['total_demos'], r['compatible_demos']), (2,2,1))
        self.assertEqual(r['compatible_task_ids'], ['00000002'])

    def test_query_fields_rejected(self):
        t = task('00000001', [[0]], [[1]])
        t['test'] = []
        with self.assertRaises(ValueError):
            analyze([t])


if __name__ == '__main__':
    unittest.main()
