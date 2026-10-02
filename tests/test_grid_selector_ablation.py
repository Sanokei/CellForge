import importlib.util
import sys
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('selector_builder',SCRIPTS/'build_grid_selector_ablation.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_selector_changes_spatial_routing_control_keeps_state_and_cells():
    parent=Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact')
    ranked=builder.ranked_two(parent);control=builder.fixed_score_control(ranked)
    assert ranked.meta['state_dim']==control.meta['state_dim']==808
    assert ranked.params.keys()==control.params.keys()
    for key in ranked.params:np.testing.assert_array_equal(ranked.params[key],control.params[key])
    for key,definition in ranked.defs.items():
        if definition['kind']=='cell':assert definition==control.defs[key]
    # Repeated colors prevent arbitrary color association from explaining a
    # spatial permutation; query dimensions also distinguish transpose.
    x=[[1,2,1],[3,1,2]]
    cases=[x,[[1,3],[2,1],[1,2]],[[1,2,1],[2,1,3]]]
    for y in cases:
        demos=[{'input':x,'output':y}]
        assert GridPredictor(ranked)(demos,x)[0]==y
        xv,xm,xh,xw=pack_grid(x);yv,ym,yh,yw=pack_grid(y)
        inputs={'input':xv,'input_mask':xm,'input_height':xh,'input_width':xw,
                'output':yv,'output_mask':ym,'output_height':yh,'output_width':yw,
                'phase':np.zeros((1,1)),'index':np.zeros((1,1)),'state':np.zeros((1,808))}
        a=Interpreter(ranked).run_bundle(ranked.entries['grid_step'],inputs,'/fixture')
        b=Interpreter(control).run_bundle(control.entries['grid_step'],inputs,'/fixture')
        np.testing.assert_array_equal(a['state'],b['state'])
    transpose=cases[1]
    assert transpose not in GridPredictor(control)([{'input':x,'output':transpose}],x)
    assert parent.meta['grid_attempts']==1


def test_constant_control_really_returns_fixed_scores():
    graph=builder.fixed_score_control(builder.ranked_two(Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact')))
    for state in (np.zeros((1,808)),np.arange(808,dtype=float)[None]):
        out=Interpreter(graph).run_bundle('router',{'state':state},'/router')
        np.testing.assert_allclose(out['scores'],-np.arange(8)[None],rtol=0,atol=1e-14)
