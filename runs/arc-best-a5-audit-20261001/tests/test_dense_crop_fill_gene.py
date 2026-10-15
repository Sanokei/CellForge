import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_dense_crop_fill_gene import build_dense_crop_fill_gene
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


@pytest.mark.parametrize('engine',[Interpreter,CompiledInterpreter])
def test_numerical_gate_selects_dense_crop_and_sparse_fill(engine):
    folder=Path(__file__).parents[1]/'runs/priority-a-v2/left'
    if not (folder/'graph.json').exists():pytest.skip('local inventory unavailable')
    g=build_dense_crop_fill_gene(Graph.load(folder));e=engine(g)
    for grid,rule in [([[0,0,0,0],[0,2,3,0],[0,4,5,0],[0,0,0,0]],'bbox.geometry'),
                      ([[0,0,0],[0,4,0],[0,0,0]],'bank_local_fill.gene')]:
        x,_,h,w=pack_grid(grid);ins=dict(input=x,input_height=h,input_width=w)
        actual=e.run_bundle('dense_crop_fill.gene',ins);expected=e.run_bundle(rule,ins)
        for k in actual:np.testing.assert_array_equal(actual[k],expected[k])
