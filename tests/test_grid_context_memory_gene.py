# SPDX-License-Identifier: MIT-0
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_context_memory_gene import build_context_memory_gene,generate_context_memory_family
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


def args(x,y,phase=0,state=None):
    xv,_,xh,xw=pack_grid(x);yv,_,yh,yw=pack_grid(y)
    return {'input':xv,'output':yv,'input_height':xh,'input_width':xw,'output_height':yh,'output_width':yw,
            'phase':np.array([[float(phase)]]),'state':np.zeros((1,3602)) if state is None else state}


def keys(x):
    h,w=len(x),len(x[0]);out=np.zeros(900)
    for r in range(30):
        for c in range(30):
            key=0;digit=0
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr,cc=r-dr,c-dc
                    v=x[rr][cc]+1 if 0<=rr<h and 0<=cc<w else 0
                    key+=v*11**digit;digit+=1
            out[r*30+c]=key
    return out


def lookup(demos,query):
    table={}
    for d in demos[-2:]:
        kk=keys(d['input']);h,w=len(d['output']),len(d['output'][0])
        for r in range(h):
            for c in range(w):table.setdefault(int(kk[r*30+c]),set()).add(d['output'][r][c])
    kk=keys(query)
    return [[next(iter(table[int(kk[r*30+c])])) if int(kk[r*30+c]) in table and len(table[int(kk[r*30+c])])==1 else color for c,color in enumerate(row)] for r,row in enumerate(query)]


def test_keys_and_full_tensor_parity_include_zero_color_and_padding():
    g=build_context_memory_gene();x=[[0,9],[1,2]];a=args(x,[[3,4],[5,6]])
    u=Interpreter(g).run_bundle('context_memory_update',a,'/u')
    v=CompiledInterpreter(g).run_bundle('context_memory_update',a,'/u')
    for k in u:np.testing.assert_array_equal(u[k],v[k])
    root_a=Interpreter(g).run_bundle('context_memory_root',a,'/r')
    root_b=CompiledInterpreter(g).run_bundle('context_memory_root',a,'/r')
    for k in root_a:np.testing.assert_array_equal(root_a[k],root_b[k])
    np.testing.assert_array_equal(u['keys'][0],keys(x))
    assert u['state'][0,900]==4 and u['state'][0,3600]==1
    assert g.stats()['unique_param_storage']==23 and g.stats()['cell_defs']==3
    assert GridPredictor(g)([{'input':[[0]],'output':[[7]]}],[[0]])==[[[7]],[[0]]]
    assert GridPredictor(g)([{'input':[[0]],'output':[[7]]}],[[1]])==[[[1]],[[1]]]


def test_exact_lookup_conflicts_unknown_and_bounded_overwrite():
    g=build_context_memory_gene()
    demos=[{'input':[[0]],'output':[[7]]},{'input':[[0]],'output':[[8]]}]
    for backend in ('reference','compiled'):
        pred=GridPredictor(g,backend=backend)
        assert pred(demos,[[0]])==[[[0]],[[0]]]
        demos3=demos+[{'input':[[2]],'output':[[9]]}]
        assert pred(demos3,[[0]])==[[[8]],[[0]]]
        assert pred(demos3,[[2]])==[[[9]],[[2]]]
        assert pred(demos3,[[3]])==[[[3]],[[3]]]


def test_query_labels_cannot_modify_memory_and_shape_mismatch_fallback():
    g=build_context_memory_gene();runtime=CompiledInterpreter(g)
    u=runtime.run_bundle('context_memory_update',args([[0]],[[7]]),'/u')
    q1=runtime.run_bundle('context_memory_root',args([[0]],[[1]],1,u['state']),'/r')
    q2=runtime.run_bundle('context_memory_root',args([[0]],[[9,9]],1,u['state']),'/r')
    for k in q1:np.testing.assert_array_equal(q1[k],q2[k])
    np.testing.assert_array_equal(q1['state'],u['state'])
    assert GridPredictor(g)([{'input':[[0]],'output':[[7,7]]}],[[0]])==[[[0]],[[0]]]


def test_original_generator_and_independent_lookup_on_small_cases():
    data=generate_context_memory_family(4,33)
    assert data==generate_context_memory_family(4,33)
    for task in data['tasks']:
        dr,dc=task['direction']
        for pair in task['train']+task['test']:
            x,y=pair['input'],pair['output']
            for r,row in enumerate(x):
                for c,value in enumerate(row):
                    assert y[r][c]==(x[r+dr][c+dc] if 0<=r+dr<len(x) and 0<=c+dc<len(row) else value)
    demos=[{'input':[[0,1],[1,0]],'output':[[1,0],[0,1]]},
           {'input':[[0,1],[1,0]],'output':[[1,0],[0,1]]}]
    query=[[0,1],[1,0]]
    g=build_context_memory_gene()
    expected=lookup(demos,query)
    assert GridPredictor(g)(demos,query)[0]==expected
