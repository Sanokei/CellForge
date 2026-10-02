
import copy,json,unittest
from pathlib import Path
from cellforge.construction.count_growth_cell import count_patch,rectangle_filter,count_pre
from cellforge.construction.cell_population import CellPopulation
class CountGrowthTests(unittest.TestCase):
    def setUp(self):
        self.p={'self.W':[[1]+[-9]*9],'cnt.W':[[1]+[0]*9],'cnt.b':[-8.5],'paint':[0,1]+[0]*8,
        'patch.emit.W':[[1]*3 for _ in range(3)],'patch.emit.b':[-.5],'patch.guard.density':[.3]}
    def test_count_anchor_and_footprint_from_immutable_input(self):
        grid=[[2]*5 for _ in range(5)]
        for r in range(1,4):
            for c in range(1,4):grid[r][c]=0
        self.assertGreater(count_pre(self.p,grid)[2,2],0)
        self.assertLess(count_pre(self.p,grid)[1,1],0)
        out=count_patch(self.p,grid)
        self.assertEqual(out[1:4],[[2,1,1,1,2]]*3)
        self.assertEqual(grid[2][2],0)
    def test_global_filter_uses_authored_weights_and_preserves_shapes(self):
        p=dict(copy.deepcopy(self.p),**{'filter.body.W':[[0,1]+[0]*8],
        'filter.marker.W':[[0]*8+[1,0]],'filter.component.W':[[3,-1]],'filter.component.b':[-1.5]})
        grid=[[1,8,0,8,8],[1,1,0,1,1],[0,0,0,0,0]]
        out=rectangle_filter(p,grid)
        self.assertEqual(out,[[1,8,0,0,0],[1,1,0,0,0],[0,0,0,0,0]])
        p['filter.component.b']=[-.5]
        self.assertEqual(rectangle_filter(p,grid),grid)
    def test_snapshot_resume_preserves_retired_and_opponent_records(self):
        evidence={k:{'partition':'development','manifest':k,'source':'public-training-real',
        'outcomes':{'case':False},'false_positive_given_consistency':0.0} for k in ('fresh','archive','anchor')}
        p=CellPopulation();cid=p.seed('unit',{'W':[0]},{'raw_response':'fixture'},evidence)
        clone=CellPopulation.from_snapshot(p.snapshot())
        clone.cells[cid]['params']['W'][0]=1
        self.assertEqual(p.cells[cid]['params']['W'],[0])
        self.assertEqual(len(clone.histories['unit'].records),1)
