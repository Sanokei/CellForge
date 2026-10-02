import importlib.util
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

spec=importlib.util.spec_from_file_location('serial_builder',Path(__file__).parents[1]/'scripts/build_grid_serial_gene.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def check_case(base,first,second,x,expected):
    graph=builder.compose_into_slot9(base,first,second)
    values,_,h,w=pack_grid(x);inputs={'input':values,'input_height':h,'input_width':w}
    a=Interpreter(graph).run_bundle('slot9_geometry',inputs,'/serial')
    b=CompiledInterpreter(graph).run_bundle('slot9_geometry',inputs,'/serial')
    for name in a:np.testing.assert_array_equal(a[name],b[name])
    height=int(a['height'][0].argmax())+1;width=int(a['width'][0].argmax())+1
    assert a['colors'][0,:height,:width].argmax(-1).tolist()==expected
    assert not a['colors'][0,height:].any() and not a['colors'][0,:,width:].any()
    assert graph.params.keys()==base.params.keys() and graph.meta['state_dim']==base.meta['state_dim']==1010
    for name in base.params:np.testing.assert_array_equal(graph.params[name],base.params[name])
    assert all(graph.defs[name]==base.defs[name] for name in base.defs if name!='slot9_geometry')
    assert graph.stats()['unique_param_storage']==195


def test_crop_flip_order_identity_and_unequal_dimensions():
    base=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/two-slot-genome-v1/base')
    x=[[0,0,0,0],[0,1,2,0],[0,3,0,0],[0,0,0,0],[0,0,0,0]]
    # Correct bbox crop and reflection commute on this fixture; no invented
    # order effect. Unequal dimensions catch width/height/mask mistakes.
    check_case(base,'bbox.geometry','geometry_2',x,[[2,1],[0,3]])
    check_case(base,'geometry_2','bbox.geometry',x,[[2,1],[0,3]])
    check_case(base,'geometry_0','geometry_0',[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]])
    check_case(base,'geometry_4','geometry_0',[[1,2,3],[4,5,6]],[[1,4],[2,5],[3,6]])


def test_recolor_fill_order_and_intermediate_padding_cannot_act_as_color0():
    base=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/two-slot-genome-v1/base')
    check_case(base,'bank_adjacency_recolor.gene','bank_local_fill.gene',[[4,2,0]],[[4,3,3]])
    check_case(base,'bank_local_fill.gene','bank_adjacency_recolor.gene',[[4,2,0]],[[4,3,2]])
    # Marker0 exists in crop-padding logits from the old bbox gene but is
    # outside the new1x1dimensions, so it must not trigger the second stage.
    zero_marker=base.copy();zero_marker.params['bank_adjacency_recolor.marker.W']=np.eye(10)[0:1]
    check_case(zero_marker,'bbox.geometry','bank_adjacency_recolor.gene',[[0,0,0],[0,2,0],[0,0,0]],[[2]])
