import importlib.util
from pathlib import Path
import numpy as np
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

spec=importlib.util.spec_from_file_location('fill_builder',Path(__file__).parents[1]/'scripts/build_grid_local_fill_gene.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_independent_fill_fixtures_reference_compiled_edges_ties_simultaneous():
    graph=builder.build_local_fill_gene()
    fixtures=[([[0,2,0],[3,0,9],[0,1,0]],[[3,2,9],[3,9,9],[3,1,9]]),
              ([[5,0,0,0]],[[5,5,0,0]]),([[0],[7],[0]],[[7],[7],[7]]),
              ([[0]],[[0]]),([[4]],[[4]]),([[2,9],[8,1]],[[2,9],[8,1]])]
    for backend in ('reference','compiled'):
        predict=GridPredictor(graph,backend=backend)
        for x,expected in fixtures:
            assert predict([{'input':[[0]],'output':[[0]]}],x)==[expected]
    assert graph.stats()['unique_param_storage']==11


def test_typed_gene_full_tensor_parity_and_padding_mask():
    graph=builder.build_local_fill_gene();x,xm,xh,xw=pack_grid([[0,2,0],[9,0,3]])
    inputs={'input':x,'input_mask':xm,'input_height':xh,'input_width':xw}
    a=Interpreter(graph).run_bundle('local_fill.gene',inputs,'/gene')
    b=CompiledInterpreter(graph).run_bundle('local_fill.gene',inputs,'/gene')
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert not a['colors'][0,2:].any()
    assert not a['colors'][0,:,:][...,3:,:].any()
