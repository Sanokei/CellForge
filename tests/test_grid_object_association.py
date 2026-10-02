import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_object_association_child import build_child,family
from build_grid_associative import build_associative_grid
from cellforge.runtime.grid import GridPredictor
import pytest


@pytest.mark.parametrize('iterations',[16,64])
def test_imported_object_cell_and_demo_recolor_generalize_on_fixture(iterations):
    parent=build_associative_grid();g=build_child(parent,iterations)
    assert g.meta['parent_hashes'][0]==parent.hash()
    assert g.meta['state_dim']==1010
    tasks=family()['tasks'][:2]
    for backend in ('reference','compiled'):
        predict=GridPredictor(g,backend=backend)
        for task in tasks:
            assert task['test'][0]['output'] in predict(task['train'],task['test'][0]['input'])
