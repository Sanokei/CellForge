import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_periodic_marker_gene import build_periodic_marker_gene,examples
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.profiles import check_profile


@pytest.mark.parametrize('backend',['reference','compiled'])
def test_periodic_completion_across_variable_dimensions(backend):
    g=build_periodic_marker_gene();check_profile(g,'ladder-micro/1')
    p=GridPredictor(g,backend=backend)
    for task in examples(4757000,10):
        assert task['test'][0]['output'] in p(task['train'],task['test'][0]['input'])


def test_no_markers_and_separators_preserved():
    p=GridPredictor(build_periodic_marker_gene(),backend='compiled')
    for x in [[[0]],[[1]],[[0,1,0],[1,0,1]],[[3,0,7]]]:
        assert p([dict(input=x,output=x)],x)==[x]
