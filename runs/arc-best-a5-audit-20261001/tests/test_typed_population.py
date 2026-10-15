import copy
from pathlib import Path
import unittest
import numpy as np
from cellforge.ir.graph import Graph, GraphError
from cellforge.construction.typed_population import PaletteExecutor,TypedArtifact,validate_repair,repaired_count

ROOT=Path(r'C:\Users\wkeif\Documents\GitHub\CellForge')

class TypedPopulationTests(unittest.TestCase):
    def test_colors_not_clipped_and_ten_color_identity(self):
        four=PaletteExecutor(ROOT/'runs/weights-pop1/pop1_core.py',4)
        with self.assertRaises(ValueError):four.predict('colormap',{'map.W':np.eye(4).tolist()},[[9]])
        ten=PaletteExecutor(ROOT/'runs/weights-pop1/pop1_core.py',10)
        self.assertEqual(ten.predict('colormap',{'map.W':np.eye(10).tolist()},[[0,4,9]]),[[0,4,9]])
        with self.assertRaises(ValueError):ten.predict('colormap',{'map.W':np.eye(4).tolist()},[[9]])

    def test_malformed_grid_and_nonfinite(self):
        ten=PaletteExecutor(ROOT/'runs/weights-pop1/pop1_core.py',10)
        for grid in ([[1.5]], [[-1]], [[10]], []):
            with self.assertRaises(ValueError):ten.predict('colormap',{'map.W':np.eye(10).tolist()},grid)
        m=np.eye(10);m[0,0]=np.nan
        with self.assertRaises(ValueError):ten.predict('colormap',{'map.W':m.tolist()},[[0]])

    def test_typed_actual_horn_and_cross_domain_rejection(self):
        horn=TypedArtifact(Graph.load(ROOT/'runs/parallel-deduction/artifact-v1'))
        ocr=TypedArtifact(Graph.load(ROOT/'runs/p1/exports/r1-iter-b'))
        self.assertTrue(horn.predict({'facts':['a'],'rules':[{'if':['a'],'then':'b'}],'query':'b'})['entailed'])
        with self.assertRaises(GraphError):horn.predict({'facts':[],'rules':[],'query':'a','target':True})
        with self.assertRaises(GraphError):ocr.predict({'facts':[],'rules':[],'query':'a'})
        plan=horn.transfer_plan(ocr)
        self.assertFalse(plan['accepted']);self.assertEqual(plan['copied_weights'],[])

    def test_directional_residual_fixture_and_corrupt_glue(self):
        # Arithmetic fixture is explicitly a test, not a founder/proposal.
        parent={'self.W':[[0,0]],'cnt.W':[[0,1]],'cnt.b':[0],'paint':[0,1]}
        donor={'conv.W':[[0,0],[0,2],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],'conv.b':[0]}
        child=dict(copy.deepcopy(parent),**{'residual.W':[[0,-1],[0,1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1]],'residual.center':[[0,0]]})
        validate_repair(parent,donor,child)
        out,pre=repaired_count(child,[[1,0],[0,0]])
        self.assertEqual(pre.tolist(),[[0,0],[2,0]])
        child['residual.W'][1][1]=0
        with self.assertRaises(ValueError):validate_repair(parent,donor,child)

if __name__=='__main__':unittest.main()
