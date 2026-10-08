"""Engine-boundary and replay checks; no ARC data, NumPy or policy execution.

Load the production functions with mocked engine/numerical/worker dependencies.
These tests inject interrupted observations and tampered saved evidence; they do
not measure policy competence or validate the sandbox's numerical execution.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_module(name, filename, dependencies):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


def modules():
    numerical = Mock()
    numerical.array_equal.return_value = False
    worker = Mock()
    worker.frame_key.return_value = 'frame-key'
    env = load_module('integrity_env', 'arc3_env.py',
                      {'numpy': numerical, 'arc3cell': worker})
    pilot = load_module('integrity_pilot', 'arc3_shared_pilot.py',
                        {'arc3_env': env, 'arc3cell': worker, 'maker': Mock(),
                         'lab_manager': SimpleNamespace(CONTRACT='unused')})
    return env, pilot


RUNNING = SimpleNamespace(name='NOT_FINISHED')
GAME_OVER = SimpleNamespace(name='GAME_OVER')
WIN = SimpleNamespace(name='WIN')


def observation(state=RUNNING, levels=0):
    return SimpleNamespace(state=state, levels_completed=levels,
                           available_actions=[0, 1])


class ObservationIntegrity(unittest.TestCase):
    def setUp(self):
        self.env, self.pilot = modules()

    def play(self, initial, next_observation=None, budget=2):
        policy = Mock(error=None)
        policy.step.return_value = (1, 0, 0, None)
        engine = Mock(observation_space=initial)
        engine.step.return_value = next_observation
        arcade = Mock()
        arcade.close_scorecard.return_value = None
        actions = SimpleNamespace(RESET=0)
        states = SimpleNamespace(GAME_OVER=GAME_OVER, WIN=WIN)
        with patch.dict(sys.modules, {'arcengine': SimpleNamespace(
                GameAction=actions, GameState=states)}), \
             patch.object(self.env, 'new_env', return_value=(arcade, 'card', engine)), \
             patch.object(self.env, 'info_of', return_value=SimpleNamespace(baseline_actions=[1])), \
             patch.object(self.env.C, 'Policy', return_value=policy), \
             patch.object(self.env, 'last_frame', return_value='frame'), \
             patch.object(self.env, 'dump', return_value='frame'), \
             patch.object(self.env, '_do', return_value=next_observation):
            result = self.env.play('unused', {'W': [1]}, 'fixture', budget, keep_trace=True)
        policy.close.assert_called_once()
        arcade.close_scorecard.assert_called_once_with('card')
        return result, policy

    def test_missing_initial_observation_is_invalid_without_an_action(self):
        result, policy = self.play(None)
        self.assertIsNotNone(result['error'])
        self.assertEqual((result['used'], result['state']), (0, 'NONE'))
        policy.step.assert_not_called()

    def test_missing_action_observation_is_invalid_and_action_is_charged(self):
        result, _ = self.play(observation())
        self.assertIsNotNone(result['error'])
        self.assertEqual((result['used'], result['action_hist']), (1, {'1': 1}))

    def test_missing_reset_observation_is_invalid_and_reset_is_charged(self):
        result, policy = self.play(observation(GAME_OVER))
        self.assertIsNotNone(result['error'])
        self.assertEqual((result['used'], result['resets']), (1, 1))
        policy.step.assert_not_called()

    def test_valid_budget_exhaustion_remains_valid(self):
        result, _ = self.play(observation(), observation())
        self.assertIsNone(result['error'])
        self.assertEqual((result['used'], result['state']), (2, 'NOT_FINISHED'))

    def test_early_win_remains_valid(self):
        result, _ = self.play(observation(), observation(WIN, 1))
        self.assertIsNone(result['error'])
        self.assertEqual((result['used'], result['levels'], result['state']), (1, 1, 'WIN'))

    def test_interrupted_control_cannot_earn_shared_policy_credit(self):
        interrupted, _ = self.play(observation())
        valid, _ = self.play(observation(), observation())
        winner, _ = self.play(observation(), observation(WIN, 1))
        controls = dict(zero=interrupted, blind=valid, random=[valid] * 3)
        with patch.object(self.env, 'play', return_value=winner), \
             patch.object(self.env, 'controls', return_value=controls):
            measured = self.pilot.evaluate(dict(source_lines=['unused'], params={'W': [1]}))
        self.assertEqual(measured['raw_total'], 3)
        self.assertEqual(measured['credited_total'], 0)
        self.assertTrue(all(not row['controls_valid'] for row in measured['games']))


class ReplayIntegrity(unittest.TestCase):
    def setUp(self):
        _, self.pilot = modules()
        run = dict(game='fixture', seed=0, budget=2, n_levels=1, levels=0,
                   used=2, error=None, level_steps=[], state='NOT_FINISHED',
                   resets=0, noops=1, distinct=1, action_hist={'1': 2},
                   final_frame='frame', trace=[(1, 0, 0), (1, 0, 0)])
        self.record = dict(raw_total=0, credited_total=0, games=[dict(
            game='fixture', result=copy.deepcopy(run), credited=0,
            controls_valid=True, controls=dict(zero=copy.deepcopy(run),
            blind=copy.deepcopy(run), random=[copy.deepcopy(run) for _ in range(3)]))])

    def test_json_round_trip_preserves_replay_equality(self):
        saved = json.loads(json.dumps(self.record))
        self.assertEqual(self.pilot.metrics(self.record), self.pilot.metrics(saved))

    def test_changed_exploration_or_protocol_evidence_is_detected(self):
        replacements = dict(game='other', seed=1, budget=3, n_levels=2,
                            state='NONE', resets=1, noops=2, distinct=2,
                            action_hist={'2': 2}, final_frame='altered',
                            trace=[(2, 0, 0), (1, 0, 0)])
        for field, value in replacements.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(self.record)
                changed['games'][0]['result'][field] = value
                self.assertNotEqual(self.pilot.metrics(self.record), self.pilot.metrics(changed))

    def test_changed_control_evidence_is_detected(self):
        for arm in ('zero', 'blind', 'random'):
            with self.subTest(arm=arm):
                changed = copy.deepcopy(self.record)
                run = changed['games'][0]['controls'][arm]
                if arm == 'random':
                    run = run[2]
                run['state'] = 'NONE'
                self.assertNotEqual(self.pilot.metrics(self.record), self.pilot.metrics(changed))

    def test_changed_aggregate_totals_are_detected(self):
        for field in ('raw_total', 'credited_total'):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.record)
                changed[field] += 1
                self.assertNotEqual(self.pilot.metrics(self.record), self.pilot.metrics(changed))


if __name__ == '__main__':
    unittest.main()
