import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
import planning_skill as P
import receding_skill as R


class PlanningTests(unittest.TestCase):
    def test_full_lookahead_executes_a_legal_shortest_plan(self):
        q = {'initial': [0, 0, 0, 1], 'goal': 3, 'capacity': 2,
             'max_energy': 5, 'costs': [1, 1, 2], 'max_steps': 16}
        params = {'reward': [1, 9, 1, 0, 0, 0], 'action': [0]*4,
                  'horizon': [0]*15+[1], 'beam': [0]*5+[1]}
        output, plan = P.execute(params, q)
        state = tuple(q['initial'])
        for action in plan:
            state = P.transition(state, action, q)
            self.assertIsNotNone(state)
        self.assertEqual(state[2], q['goal'])
        self.assertEqual(output['steps'], len(P.reference(q)))
        params['horizon'] = [1]+[0]*15
        self.assertFalse(P.execute(params, q)[0]['complete'])

    def test_depth_three_replans_instead_of_returning_a_reference_answer(self):
        q = {'initial': [0, 0, 0, 3], 'goal': 1, 'capacity': 2,
             'max_energy': 3, 'costs': [1, 1, 1], 'max_steps': 6}
        params = {'reward': [9, 9, 1, 1, 2, 4], 'action': [0]*4,
                  'horizon': [0, 0, 1], 'beam': [0]*5+[1]}
        self.assertEqual(R.predict(params, q), {'complete': True, 'steps': 3})
        params['reward'] = [0]*6
        params['action'] = [9, 0, 0, 0]
        self.assertNotEqual(R.predict(params, q), {'complete': True, 'steps': 3})


if __name__ == '__main__':
    unittest.main()
