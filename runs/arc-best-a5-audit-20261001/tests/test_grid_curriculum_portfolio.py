import importlib.util
import sys
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile
from cellforge.construction.transactions import apply_ops
from cellforge.runtime.grid import pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

scripts=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('portfolio',scripts/'build_grid_curriculum_portfolio.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_every_slot_rewire_preserves_capacity_and_is_atomic_typed():
    parent=builder.build_portfolio(Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact'))
    before=parent.hash()
    assert parent.meta['state_dim']==1010 and parent.meta['grid_attempts']==2
    assert parent.defs['slot9_geometry']==parent.defs['geometry_0']
    for gene in builder.GENES:
        transaction=builder.slot_transaction(parent,gene)
        child,pure=apply_ops(parent,transaction['ops'])
        assert not pure and transaction['base']==before
        child.validate(Limits(strict_microcells=True));check_profile(child,'ladder-micro/1')
        assert child.defs.keys()==parent.defs.keys() and child.params.keys()==parent.params.keys()
        assert child.meta['state_dim']==parent.meta['state_dim']
        assert [name for name in parent.defs if parent.defs[name]!=child.defs[name]]==['slot9_geometry']
        for name in parent.params:np.testing.assert_array_equal(child.params[name],parent.params[name])
    assert parent.hash()==before
    assert builder.random_control_transaction(parent,55)==builder.random_control_transaction(parent,55)


def test_imported_fill_flat_cluster_reference_compiled_and_donor_parity():
    donor=builder.build_local_fill_gene()
    graph=builder.build_portfolio(Graph.load(Path(__file__).parents[1]/'runs/grid-associative/artifact'),donor)
    x,xm,xh,xw=pack_grid([[0,2,0],[4,0,7]])
    inputs={'input':x,'input_height':xh,'input_width':xw}
    expected=Interpreter(donor).run_bundle('local_fill.gene',{**inputs,'input_mask':xm},'/donor')
    a=Interpreter(graph).run_bundle('bank_local_fill.gene',inputs,'/import')
    b=CompiledInterpreter(graph).run_bundle('bank_local_fill.gene',inputs,'/import')
    for name in expected:
        np.testing.assert_array_equal(a[name],expected[name]);np.testing.assert_array_equal(a[name],b[name])
    assert graph.meta['portfolio']['fill_donor_hash']==donor.hash()
    assert all(graph.defs[n['use']]['kind']=='cell' for n in graph.defs['bank_local_fill.gene']['nodes'] if not n['use'].startswith('op:'))
