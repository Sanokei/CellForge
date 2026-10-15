import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from grid_geometry_memo import GeometryMemoGridPredictor
from grid_geometry_memo import GeometryMemoInterpreter
from cellforge.runtime.compiled import CompiledInterpreter
from cellforge.runtime.grid import pack_grid
from cellforge.runtime.grid import GridPredictor
from cellforge.ir.graph import Graph


def test_geometry_memo_output_semantic_counters_and_step_reset():
    g=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/expanded-seven-slot-v1/expanded')
    reference=GridPredictor(g,backend='compiled');cached=GeometryMemoGridPredictor(g)
    for factor in (2,3,1,2):
        x=[[0,2,1],[4,5,6]];q=[[7,0],[3,8],[5,1]]
        y=np.repeat(np.repeat(x,factor,axis=0),factor,axis=1).tolist()
        demos=[dict(input=x,output=y)]
        assert cached(demos,q)==reference(demos,q)
        assert cached.last_exec==reference.last_exec
    assert cached._compiled.memo_hits>0
    assert all(set(cached.graph.defs[n]['inputs'])=={'input','input_height','input_width'} for n in cached._compiled._eligible)
    for output,*_ in cached._compiled._memo.values():assert all(not v.flags.writeable for v in output.values())


def test_duplicate_object_geometry_exact_state_and_logits_parity():
    from expanded_grid_genetics import instantiate
    base=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/expanded-seven-slot-v1/expanded')
    g=instantiate(base,['object_bank_largest4.gene']*7)
    plain=CompiledInterpreter(g);memo=GeometryMemoInterpreter(g)
    states=[np.zeros((1,1515)),np.zeros((1,1515))]
    x=[[0,0,0,0],[0,3,3,0],[0,3,3,0],[0,0,0,0]]
    for phase,y in [(0,[[3,3],[3,3]]),(0,[[7,7],[7,7]]),(1,[[0]])]:
        colors,_,height,width=pack_grid(x);out,_,oh,ow=pack_grid(y)
        values=[]
        for i,engine in enumerate((plain,memo)):
            inputs=dict(input=colors,input_height=height,input_width=width,output=out if not phase else np.zeros_like(out),
                        output_height=oh if not phase else np.zeros_like(oh),output_width=ow if not phase else np.zeros_like(ow),
                        phase=np.array([[float(phase)]]),state=states[i])
            result=engine.run_bundle(g.entries['grid_step'],inputs)
            states[i]=result['state'];values.append(result)
        for key in values[0]:np.testing.assert_array_equal(values[0][key],values[1][key])
        assert plain.stats.as_dict()==memo.stats.as_dict()
    assert memo.memo_hits>=18
