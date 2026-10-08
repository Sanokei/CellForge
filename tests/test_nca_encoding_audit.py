"""Synthetic literal serialization fixtures, not new genome proposals."""
import copy
import unittest
from scripts.nca_encoding_audit import encode, decode, checked_layout


class EncodingAuditTests(unittest.TestCase):
    def setUp(self):
        self.layout = [{'name':'a','shape':[1,3],'positions':3},
                       {'name':'b','shape':[2],'positions':2}]
        self.tensors = {'a':[[-1,.25,0]], 'b':[.5,1]}

    def test_values_types_and_shapes_roundtrip_without_input_mutation(self):
        before = copy.deepcopy(self.tensors)
        payload = encode(self.tensors,self.layout)
        self.assertEqual(payload['values'],[-1,.25,0,.5,1])
        self.assertEqual(decode(payload,self.layout),self.tensors)
        self.assertEqual(self.tensors,before)

    def test_all_positions_required_and_no_extras(self):
        for values in [[0]*4,[0]*6]:
            p=encode(self.tensors,self.layout);p['values']=values
            with self.assertRaises(ValueError):decode(p,self.layout)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):encode({'a':[-1,.25,0],'b':[.5,1]},self.layout)

    def test_layout_drift_rejected(self):
        p=encode(self.tensors,self.layout)
        changed=copy.deepcopy(self.layout);changed[0]['shape']=[3,1]
        with self.assertRaises(ValueError):decode(p,changed)

    def test_boolean_nonfinite_string_and_out_of_vocab_rejected(self):
        for value in [True,float('nan'),float('inf'),'1',.1]:
            p=encode(self.tensors,self.layout);p['values'][0]=value
            with self.assertRaises(ValueError):decode(p,self.layout)

    def test_duplicate_unordered_layout_and_wrong_count_rejected(self):
        for layout in [self.layout[::-1],self.layout+[self.layout[0]],
                       [dict(self.layout[0],positions=99)]]:
            with self.assertRaises(ValueError):checked_layout(layout)

    def test_extra_envelope_fields_rejected(self):
        p=encode(self.tensors,self.layout);p['code']='ignored'
        with self.assertRaises(ValueError):decode(p,self.layout)


if __name__=='__main__':unittest.main()
