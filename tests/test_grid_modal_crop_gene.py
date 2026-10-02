import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_modal_crop_gene import build_modal_crop
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.profiles import check_profile

@pytest.mark.parametrize('backend',['reference','compiled'])
def test_modal_background_crop(backend):
 g=build_modal_crop();check_profile(g,'ladder-micro/1');p=GridPredictor(g,backend=backend)
 rng=np.random.default_rng(3287000)
 for i in range(100):
  bg=i%10;colors=[c for c in range(10) if c!=bg];x=np.full((12,13),bg);obj=rng.choice(colors,size=(3,4));x[3:6,5:9]=obj
  assert p([{'input':x.tolist(),'output':obj.tolist()}],x.tolist())[0]==obj.tolist()
 x=[[9]*4]*3;assert p([{'input':x,'output':x}],x)[0]==x
