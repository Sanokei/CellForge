import unittest,copy
from cellforge.construction.component_count import predict_component_count
class ComponentTests(unittest.TestCase):
 def params(self):
  return {'target.W':[[0,1,0,0,0,0,0,0,0,0]],'component.W':[[1,1,1,1]],'component.b':[-3.5],
          'readout.W':[[1]],'readout.b':[[-.5,-1.5,-2.5,-3.5,-4.5]],'palette':[0,1]}
 def test_geometry_color_and_component_connectivity(self):
  p=self.params()
  self.assertEqual(predict_component_count(p,[[1,1,0,2,2],[1,1,0,2,2],[0,0,0,0,0],[1,0,1,1,0],[0,0,1,1,0]]),[[1,1,0,0,0]])
  self.assertEqual(predict_component_count(p,[[1,1,1],[1,0,0]]),[[0,0,0,0,0]])
 def test_weights_actually_control_decision(self):
  p=self.params();p['component.W']=[[0,0,0,0]]
  self.assertEqual(predict_component_count(p,[[1,1],[1,1]]),[[0,0,0,0,0]])
 def test_invalid_inputs(self):
  for grid in ([[10]],[[-1]],[[1.5]],[]):
   with self.assertRaises(ValueError):predict_component_count(self.params(),grid)
  p=self.params();p['palette']=[0,.5]
  with self.assertRaises(ValueError):predict_component_count(p,[[1]])
if __name__=='__main__':unittest.main()

