import numpy as np
from cellforge.construction.nca_shallow import build_shallow_nca
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter
from cellforge.ir.profiles import check_profile


def test_parameter_budget_microcells_and_ladder_profile():
    graph=build_shallow_nca()
    assert graph.stats()['unique_param_storage']==414
    assert sum(d['kind']=='cell' for d in graph.defs.values())==22
    assert check_profile(graph,'ladder-micro/1')['profile']=='ladder-micro/1'
    assert all(n['use'].startswith('op:') or graph.defs[n['use']]['kind']=='cell' for n in graph.defs['nca_shallow_gene']['nodes'])


def test_residual_identity_all_colors_and_explicit_direct_lossy_channels():
    predictor=GridPredictor(build_shallow_nca())
    grid=[[0,1,2],[3,2,1]]
    assert predictor([{'input':grid,'output':grid}],grid)==[grid]
    assert predictor([{'input':[[9]],'output':[[9]]}],[[4,5,6,7,8,9]])==[[[4,5,6,7,8,9]]]
    direct=GridPredictor(build_shallow_nca(architecture='direct'))
    assert direct([{'input':[[9]],'output':[[9]]}],[[4,5,6,7,8,9]])==[[[0,0,0,0,0,0]]]
    swap=GridPredictor(build_shallow_nca('swap01'))
    assert swap([{'input':grid,'output':grid}],[[0,1,2,3]])==[[[1,0,2,3]]]


def test_compiled_reference_full_color_mask_and_reset_parity():
    for seed in ('identity','swap01'):
        graph=build_shallow_nca(seed)
        x,m,h,w=pack_grid((np.arange(900).reshape(30,30)%10).tolist())
        inputs={'input':x,'input_mask':m,'input_height':h,'input_width':w}
        a=Interpreter(graph).run_bundle('nca_shallow_gene',inputs)
        b=CompiledInterpreter(graph).run_bundle('nca_shallow_gene',inputs)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        x,m,h,w=pack_grid([[1]])
        result=Interpreter(graph).run_bundle('nca_shallow_gene',{'input':x,'input_mask':m,'input_height':h,'input_width':w})
        assert not result['colors'][0,1:].any() and not result['colors'][0,:,1:].any()
