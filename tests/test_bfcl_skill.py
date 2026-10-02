import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
import bfcl_skill as B
import skillcore as S


class BFCLCellTests(unittest.TestCase):
    def test_shuffled_typed_arguments_and_tools(self):
        def tool(name, desc):
            return {'name': name, 'description': desc, 'parameters': {'required': ['city', 'days', 'active'],
                'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}, 'active': {'type': 'boolean'}}}}
        params = {'route': [1, 0, 0, 0], 'bind': [9, 0, 0, 0], 'type': [[3 if i == j else 0 for j in range(4)] for i in range(4)]}
        cell = S.make_cell('bfcl_call2', params)
        request = {'query': 'weather forecast; active=true; extra=999; days=4; city="Boston"',
                   'tools': [tool('travel', 'book journey'), tool('forecast', 'weather forecast')]}
        self.assertEqual(S.run(cell, request, []), {'name': 'forecast', 'arguments': {'city': 'Boston', 'days': 4, 'active': True}})
        params['route'][0] = -1
        self.assertEqual(S.run(S.make_cell('bfcl_call2', params), request, [])['name'], 'travel')

    def test_types_are_scored_strictly(self):
        self.assertFalse(B.strict_equal({'x': True}, {'x': 1}))
        self.assertFalse(B.strict_equal({'x': 2.0}, {'x': 2}))

    def test_no_candidate_fails_closed(self):
        self.assertEqual(B.candidates('nothing here'), [])


if __name__ == '__main__':
    unittest.main()
