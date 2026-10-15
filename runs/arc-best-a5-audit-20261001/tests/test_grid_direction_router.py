import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_direction_router_gene import build_direction_router
from build_grid_context_memory_gene import generate_context_memory_family
from cellforge.runtime.grid import GridPredictor


def test_graph_selects_each_direction_and_preserves_boundary_colors():
    tasks=generate_context_memory_family(8,2517000)['tasks']
    for backend in ('reference','compiled'):
        predictor=GridPredictor(build_direction_router(),backend=backend)
        for task in tasks:
            result=predictor(task['train'],task['test'][0]['input'])
            assert result[0]==task['test'][0]['output']
            assert result[1]==task['test'][0]['input']
    assert build_direction_router().stats()['unique_param_storage']==2


def test_no_update_control_and_shape_mismatch_fall_back_to_identity():
    task=generate_context_memory_family(1,2527000)['tasks'][0]
    query=task['test'][0]['input']
    assert GridPredictor(build_direction_router(False))(task['train'],query)==[query,query]
    assert GridPredictor(build_direction_router())([{'input':[[1,2]],'output':[[1]]}],query)==[query,query]
