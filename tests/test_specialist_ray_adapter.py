"""Bounded adapter checks on new tiny fixtures, never the composition panels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import agent_contracts as C
import skillcore as G
from specialist_runtime import Binding, LocalDispatcher, _grid_preflight


BANK = SCRIPTS.parent / 'runs/weights-native-composition01-20261006/SELECTED_SKILL_BANK.json'
BANK_SHA256 = '1415099c2d3b6f44de91d5a47dcdc9341f71a9aa871c395caded2923dc869253'
PROVENANCE = C.Provenance('ray-adapter-operational-fixture', C.Origin.HOST,
                         ('new-2x3-fixture-not-composition-panel',))
RAY = {'kind': 'cell', 'template': 'ray_cast'}
TILE = {'kind': 'cell', 'template': 'tile_orient'}


def topology(*experts):
    return {'schema': 'cellforge-skill-bank/1', 'experts': list(experts)}


def chain(first, second):
    return {'kind': 'chain', 'parts': [first, second]}


def literal_zeros(value):
    if isinstance(value, dict):
        return {key: literal_zeros(item) for key, item in value.items()}
    if isinstance(value, list):
        return [literal_zeros(item) for item in value]
    return 0.0


class RayAdmission(unittest.TestCase):
    def test_geometric_ray_steps_are_bounded_including_singleton(self):
        # Count paths independently of tensors, source masks and early stops.
        # Even a dense grid with all eight directions enabled cannot exceed it.
        for height, width in ((1, 1), (1, 7), (3, 5), (30, 30)):
            steps = 0
            for row in range(height):
                for column in range(width):
                    for dr, dc in G.RAY_DIRS:
                        rr, cc = row + dr, column + dc
                        while 0 <= rr < height and 0 <= cc < width:
                            steps += 1
                            rr, cc = rr + dr, cc + dc
            result = _grid_preflight(topology(RAY), [], [(height, width)])
            self.assertEqual(result['ray_launches_bound'], 8 * height * width)
            self.assertEqual(result['ray_path_steps_bound'], 8 * height * width * (max(height, width) - 1))
            self.assertLessEqual(steps, result['ray_path_steps_bound'])
            self.assertEqual(result['work_units_bound'], result['grid_cell_visits_bound']
                             + result['ray_launches_bound'] + result['ray_path_steps_bound'])

    def test_recursive_staging_charges_demo_inputs_outputs_and_query(self):
        # Three LOO calls each visit main ray + two input/output demo pairs;
        # one query visits main ray + three pairs: 22 rays plus four tile calls.
        # Output demo grids (4x3) cost more ray work than input grids (2x3).
        result = _grid_preflight(topology(chain(RAY, TILE)), [((2, 3), (4, 3))] * 3, [(2, 3)])
        self.assertEqual(result['leaf_calls_bound'], 26)
        self.assertEqual(result['ray_launches_bound'], 1488)
        self.assertEqual(result['ray_path_steps_bound'], 3840)
        # All possible query experts are included, even though routing picks one.
        doubled = _grid_preflight(topology(chain(RAY, TILE), chain(RAY, TILE)),
                                  [((2, 3), (4, 3))] * 3, [(2, 3)])
        self.assertEqual(doubled['work_units_bound'], 2 * result['work_units_bound'])

    def test_ray_step_work_rejects_before_small_grid_visits_reach_cap(self):
        with self.assertRaises(C.ContractError) as caught:
            _grid_preflight(topology(RAY), [], [(30, 30)] * 5)
        self.assertEqual(caught.exception.code, 'resource_limit')
        # Only 9,000 grid visits, but rays would exceed a million total units.
        self.assertLess(5 * 2 * 30 * 30, 1000000)

    def test_unknown_templates_and_previous_shape_explosion_stay_rejected(self):
        for artifact, code in ((topology({'kind': 'cell', 'template': 'unbounded_ray'}), 'unsupported_input'),
                               (topology(chain(chain(TILE, TILE), chain(TILE, TILE))), 'resource_limit')):
            with self.assertRaises(C.ContractError) as caught:
                _grid_preflight(artifact, [((30, 30), (30, 30))], [(30, 30)])
            self.assertEqual(caught.exception.code, code)


class SelectedBankAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = BANK.read_bytes()
        if hashlib.sha256(cls.raw).hexdigest() != BANK_SHA256:
            raise AssertionError('selected artifact changed; inspect before updating fixture pin')
        cls.bank = json.loads(cls.raw)
        cls.pins = tuple((name, hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest())
                         for name in ('agent_contracts.py', 'specialist_runtime.py', 'skillcore.py', 'cell_contracts.py'))
        cls.binding = Binding('grid.predict', cls.raw, BANK_SHA256, cls.pins,
            C.Provenance('host-selected-composition-artifact', C.Origin.HOST,
                         ('runs/weights-native-composition01-20261006/SELECTED_SKILL_BANK.json',)))

    def fixture(self, horizontal=False):
        # Two-row grids are separate operational fixtures; no saved task or label is loaded.
        expected = ([[1] * 6, [2] * 6, [2] * 6, [1] * 6] if horizontal
                    else [[1, 0, 2, 2, 0, 1]] * 4)
        value = {'schema': 'cellforge-grid-task/1',
                 'train': [{'input': [[1, 0, 0], [0, 0, 2]], 'output': expected}],
                 'test': [{'input': [[3, 0, 0], [0, 0, 4]]}]}
        return C.Task('grid', 'operational', C.Capability.GRID, json.dumps(value), PROVENANCE)

    def runtime(self, task, units=100000):
        return LocalDispatcher(task, C.Budget(units, 0, 0, 0, 0, 1000),
                               (self.binding,), clock_ms=lambda: 0)

    def test_exact_selected_bank_routes_both_small_fixtures(self):
        for horizontal in (False, True):
            task = self.fixture(horizontal)
            runtime = self.runtime(task)
            result = runtime.dispatch('grid')
            self.assertEqual(result.status, C.Status.COMPLETED, result.detail)
            expected = ([[3] * 6, [4] * 6, [4] * 6, [3] * 6] if horizontal
                        else [[3, 0, 4, 4, 0, 3]] * 4)
            self.assertEqual(json.loads(result.output_json)['predictions'], [expected])
            receipt = runtime.receipt('grid')
            self.assertEqual(receipt['artifact_sha256'], BANK_SHA256)
            bound = receipt['grid_preflight']
            self.assertGreater(bound['ray_path_steps_bound'], 0)
            self.assertEqual(receipt['charged_work_units'], 1 + bound['work_units_bound'])
            self.assertEqual(receipt['artifact_hash'], self.bank['artifact_hash'])

    def test_zero_and_all_direction_overrides_keep_topology_and_bound(self):
        task = self.fixture()
        zero = literal_zeros(self.bank['tensors'])
        all_directions = deepcopy(zero)
        for index in range(2):
            all_directions[f'{index}:A.src'] = [1.0] * 10
            all_directions[f'{index}:A.dirs'] = [1.0] * 8
            all_directions[f'{index}:B.per'] = [0.0, 0.0, 0.0, 1.0]
            all_directions[f'{index}:B.pec'] = [0.0, 0.0, 0.0, 1.0]
        before = deepcopy(self.bank)
        saved = self.runtime(task)
        saved.dispatch('grid')
        for override in (zero, all_directions):
            unchanged = deepcopy(override)
            runtime = self.runtime(task)
            result = runtime.dispatch('grid', tensor_override=override)
            self.assertEqual(result.status, C.Status.COMPLETED, result.detail)
            direct = G.predict_bank(self.bank, C.grid_bank_input(task), tensors=override)
            self.assertEqual(json.loads(result.output_json)['predictions'], [row[0] for row in direct])
            self.assertEqual(runtime.receipt('grid')['grid_preflight'], saved.receipt('grid')['grid_preflight'])
            self.assertIsNotNone(runtime.receipt('grid')['tensor_override_sha256'])
            self.assertEqual(override, unchanged)
        self.assertEqual(self.bank, before)
        self.assertEqual(BANK.read_bytes(), self.raw)

    def test_budget_including_ray_steps_is_required_before_prediction(self):
        task = self.fixture()
        stats = _grid_preflight(self.bank, [((2, 3), (4, 6))], [(2, 3)])
        runtime = self.runtime(task, units=1 + stats['grid_cell_visits_bound'])
        with mock.patch.object(G, 'predict_bank', side_effect=AssertionError('must not execute')) as predictor:
            result = runtime.dispatch('grid')
        self.assertEqual(result.error_code, 'budget_exhausted')
        self.assertFalse(runtime.receipt('grid')['adapter_called'])
        predictor.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
