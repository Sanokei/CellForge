import importlib.util
import sys
from pathlib import Path
import numpy as np
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

scripts=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('largest_builder',scripts/'build_grid_largest_component_gene.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_largest4_background_nonzero_ties_diagonal_policy_and_empty():
    graph=builder.build_largest_component_gene(64)
    cases=[([[0,1,1,0],[0,1,0,2],[0,0,0,2],[0,0,0,0]],[[1,1],[1,0]]),
           ([[9,0,0,9],[9,0,9,3],[9,9,9,3],[9,9,9,9]],[[0,0],[0,9]]),
           ([[0,3,3,0],[0,0,0,0],[0,1,1,0],[0,0,0,0]],[[3,3]]),
           ([[0,1,0,0],[0,0,1,0],[2,2,0,0],[0,0,0,0]],[[2,2]]),
           ([[5,5],[5,5]],[[5,5],[5,5]])]
    for backend in ('reference','compiled'):
        predictor=GridPredictor(graph,backend=backend)
        for x,y in cases:assert predictor([{'input':[[0]],'output':[[0]]}],x)==[y]
    for x,_ in cases:
        value,_,h,w=pack_grid(x);inputs={'input':value,'input_height':h,'input_width':w}
        a=Interpreter(graph).run_bundle('largest4.gene',inputs,'/gene')
        b=CompiledInterpreter(graph).run_bundle('largest4.gene',inputs,'/gene')
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert graph.stats()['unique_param_storage']==11


def test_snake_over64_steps_falls_back_identity_in_graph():
    snake=[[0]*30 for _ in range(30)]
    for i in range(10):
        snake[2*i]=[1]*30
        if i<9:snake[2*i+1][29 if i%2==0 else 0]=1
    graph=builder.build_largest_component_gene(64)
    for backend in ('reference','compiled'):
        assert GridPredictor(graph,backend=backend)([{'input':[[0]],'output':[[0]]}],snake)==[snake]
