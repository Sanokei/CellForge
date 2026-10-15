import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_numeric_neighbor_gate import build_weight_gate,examples
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.profiles import check_profile

@pytest.mark.parametrize('backend',['reference','compiled'])
def test_authored_numeric_horizontal_gate(backend):
 g=build_weight_gate((0,0,1,1),-1.5);check_profile(g,'ladder-micro/1');p=GridPredictor(g,backend=backend)
 for t in examples(3357000):assert t['test'][0]['output'] in p(t['train'],t['test'][0]['input'])
