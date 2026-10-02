import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_repeat_gene import build_repeat_gene
from cellforge.runtime.grid import GridPredictor


@pytest.mark.parametrize('factor',[2,3])
def test_repetition_matches_scalar_on_rectangles_zero_colors_and_caps(factor):
    rng=np.random.default_rng(3)
    for backend in ('reference','compiled'):
        predictor=GridPredictor(build_repeat_gene(factor),backend=backend)
        for h,w in ((1,1),(2,3),(30//factor,30//factor),(30//factor+1,2)):
            x=rng.integers(0,10,size=(h,w)).tolist()
            expected=np.repeat(np.repeat(x,factor,axis=0),factor,axis=1).tolist() if max(h,w)*factor<=30 else x
            assert predictor([{'input':[[1]],'output':[[9]]}],x)==[expected]
    assert build_repeat_gene(factor).stats()['unique_param_storage']==31
