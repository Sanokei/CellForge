# SPDX-License-Identifier: MIT-0
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_context_backoff_gene import build_context_backoff_gene
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


def args(x,y,phase=0,state=None):
    xv,_,xh,xw=pack_grid(x);yv,_,yh,yw=pack_grid(y)
    return {'input':xv,'output':yv,'input_height':xh,'input_width':xw,'output_height':yh,'output_width':yw,
            'phase':np.array([[float(phase)]]),'state':np.zeros((1,3702)) if state is None else state}


def scalar_coarse(demos,x):
    table=np.zeros((10,10),dtype=int)
    for d in demos:
        for rowx,rowy in zip(d['input'],d['output']):
            for a,b in zip(rowx,rowy):table[a,b]+=1
    return [[int(np.argmax(table[v])) if table[v].sum()>0 and table[v].max()==table[v].sum() else v for v in row] for row in x]


def test_unseen_context_coarse_recolor_zero_and_padding_scalar_reference():
    g=build_context_backoff_gene()
    demos=[{'input':[[0,1]],'output':[[7,1]]}];query=[[0,0],[1,1]]
    expected=scalar_coarse(demos,query)
    for backend in ('reference','compiled'):
        assert GridPredictor(g,backend=backend)(demos,query)==[expected,query]
    assert g.stats()['unique_param_storage']==23 and g.stats()['cell_defs']==3


def test_coarse_conflict_identity_but_fine_unanimous_has_priority():
    g=build_context_backoff_gene()
    demos=[{'input':[[0,1]],'output':[[7,1]]},
           {'input':[[0],[2]],'output':[[8],[2]]}]
    pred=GridPredictor(g,backend='compiled')
    assert pred(demos,[[0,1]])[0]==[[7,1]]
    assert pred(demos,[[0,0]])[0]==[[0,0]]


def test_coarse_counts_accumulate_despite_fine_bank_overwrite():
    g=build_context_backoff_gene();rt=CompiledInterpreter(g);state=np.zeros((1,3702))
    demos=[{'input':[[0]],'output':[[7]]},{'input':[[1]],'output':[[2]]},{'input':[[2]],'output':[[3]]}]
    for d in demos:state=rt.run_bundle('context_memory_update',args(d['input'],d['output'],state=state),'/u')['state']
    table=state[0,3602:].reshape(10,10)
    expected=np.zeros((10,10));expected[0,7]=expected[1,2]=expected[2,3]=1
    np.testing.assert_array_equal(table,expected)
    assert GridPredictor(g)(demos,[[0,0]])[0]==[[7,7]]


def test_full_tensor_parity_and_query_labels_frozen_shape_fallback():
    g=build_context_backoff_gene();a=args([[0,1],[0,2]],[[7,1],[7,2]])
    ref=Interpreter(g).run_bundle('context_memory_root',a,'/r')
    comp=CompiledInterpreter(g).run_bundle('context_memory_root',a,'/r')
    for k in ref:np.testing.assert_array_equal(ref[k],comp[k])
    rt=CompiledInterpreter(g)
    q1=rt.run_bundle('context_memory_root',args([[0,0]],[[9,9]],1,comp['state']),'/r')
    q2=rt.run_bundle('context_memory_root',args([[0,0]],[[3]],1,comp['state']),'/r')
    for k in q1:np.testing.assert_array_equal(q1[k],q2[k])
    np.testing.assert_array_equal(q1['state'],comp['state'])
    assert GridPredictor(g)([{'input':[[0]],'output':[[7,7]]}],[[0]])==[[[0]],[[0]]]
