import unittest
from cellforge.construction.arithmetic_cell import predict_adder
class ArithmeticCellTests(unittest.TestCase):
    def setUp(self):
        # Arithmetic fixture only; genuine maker response retained separately.
        self.p={'sum.W':[[1,1,1]],'sum.b':[0],'carry.W':[[1]],'carry.b':[-9.5],
                'readout.W':[[1,-10],[0,1]],'readout.b':[0,0]}
    def test_execute_coefficients_and_carry_boundary(self):
        for a,b,c in [(0,0,0),(4,5,0),(4,5,1),(9,9,1)]:
            s=a+b+c;self.assertEqual(predict_adder(self.p,{'a':a,'b':b,'carry':c}),{'digit':s%10,'carry':s//10})
        self.p['sum.W'][0][1]=0
        self.assertNotEqual(predict_adder(self.p,{'a':2,'b':3,'carry':0}),{'digit':5,'carry':0})
    def test_labels_foreign_modality_and_bad_outputs_rejected(self):
        for r in ({'a':1,'b':2,'carry':0,'target':3},{'a':1.0,'b':2,'carry':0},{'a':10,'b':2,'carry':0},{'image':[]}):
            with self.assertRaises(ValueError):predict_adder(self.p,r)
        self.p['readout.W'][0][1]=0
        with self.assertRaises(ValueError):predict_adder(self.p,{'a':9,'b':9,'carry':1})

