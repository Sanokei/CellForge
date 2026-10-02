from pathlib import Path
import unittest
import numpy as np
from cellforge.construction.typed_population import parity_row_stencil,calibrate_caps,resource_pass
class NewGateTests(unittest.TestCase):
    def test_authored_stencil_semantics_and_input_guard(self):
        p={'stencil.W':[[0,1,0],[1,0,1]],'foreground.W':[[0,1,1,1,1,1,1,1,1,1]],'activation.b':[-.5]}
        x=[[0,9,0,0],[0,0,0,0],[0,0,0,0]]
        self.assertEqual(parity_row_stencil(p,x),[[0,9,0,0],[9,0,9,0],[0,9,0,0]])
        self.assertEqual(x,[[0,9,0,0],[0,0,0,0],[0,0,0,0]])
        self.assertEqual(parity_row_stencil(p,[[0,9],[1,0]]),[[0,9],[1,0]])
        with self.assertRaises(ValueError):parity_row_stencil(p,[[10]])
    def test_resource_caps_are_local_and_fail_closed(self):
        sample={'wall_seconds':.1,'traced_peak_bytes':1000}
        with self.assertRaises(ValueError):calibrate_caps([sample])
        c=calibrate_caps([sample]*3)
        self.assertTrue(resource_pass(sample,c))
        self.assertFalse(resource_pass(dict(sample,wall_seconds=1.),c))
if __name__=='__main__':unittest.main()

