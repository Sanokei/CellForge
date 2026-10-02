import sys
from pathlib import Path
import pytest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_repeat_gene import build_repeat_gene
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_curriculum import generate_task
from cellforge.ir.profiles import check_profile

@pytest.mark.parametrize('backend',['reference','compiled'])
def test_whole_grid_tiling_and_overflow(backend):
 g=build_repeat_gene(2,'tile');check_profile(g,'ladder-micro/1');p=GridPredictor(g,backend=backend)
 for i in range(100):
  t=generate_task('tiling2x',3267000+i);assert p(t['train'],t['test'][0]['input'])[0]==t['test'][0]['output']
 for x in ([[1]*16]*2,[[1]]*16):assert p([{'input':[[1]],'output':[[1]]}],x)[0]==x
 x=[[1,2,3],[4,5,6]]
 assert p([{'input':[[1]],'output':[[1]]}],x)[0]==np.tile(x,(2,2)).tolist()
