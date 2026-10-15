import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_scaling_selector import build_scaling_selector
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.profiles import check_profile


def repeat(x,f):return np.repeat(np.repeat(x,f,axis=0),f,axis=1).tolist()


@pytest.mark.parametrize('backend',['reference','compiled'])
def test_demo_selection_reset_order_and_no_update_control(backend):
    predictor=GridPredictor(build_scaling_selector(),backend=backend)
    control=GridPredictor(build_scaling_selector(False),backend=backend)
    x=[[0,2,1],[3,0,4]];query=[[7,0],[1,5],[4,2]]
    for f in (2,3,1,3,2):
        demos=[dict(input=x,output=repeat(x,f)),dict(input=[[8,0]],output=repeat([[8,0]],f))]
        expected=[repeat(query,f),query]
        assert predictor(demos,query)==expected
        assert predictor(list(reversed(demos)),query)==expected
        assert control(demos,query)==[query,query]
    oversized=[[i%10 for i in range(2)] for _ in range(16)]
    assert predictor([dict(input=x,output=repeat(x,2))],oversized)==[oversized,oversized]
    g=build_scaling_selector()
    assert check_profile(g,'ladder-micro/1')['dispatches']==1
    assert g.meta['state_dim']==3 and len(g.meta['import_lineage'])==2
    assert g.stats()['unique_param_storage']==64
