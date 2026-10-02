# SPDX-License-Identifier: MIT-0
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_conditional_state_gene import build_conditional_state_gene,generate_conditional_state_family
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter


def inputs(x,y,phase=0,state=None):
    xv,xm,xh,xw=pack_grid(x);yv,ym,yh,yw=pack_grid(y)
    return {'input':xv,'output':yv,'input_height':xh,'input_width':xw,
            'output_height':yh,'output_width':yw,'phase':np.array([[float(phase)]]),
            'state':np.zeros((1,1002)) if state is None else state}


def transform(x,s,m,t):
    return [[t if v==s and any(0<=r+dr<len(x) and 0<=c+dc<len(row) and x[r+dr][c+dc]==m for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))) else v for c,v in enumerate(row)] for r,row in enumerate(x)]


def test_all_rules_reconstruction_state_matches_scalar_reference_and_parity():
    g=build_conditional_state_gene();x=[[0,4,2],[2,9,4]];y=[[3,4,3],[2,9,4]]
    args=inputs(x,y)
    a=Interpreter(g).run_bundle('conditional_update',args,'/u')
    b=CompiledInterpreter(g).run_bundle('conditional_update',args,'/u')
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    expected=[]
    for s in range(10):
        for m in range(10):
            for t in range(10):
                z=transform(x,s,m,t);expected.append(sum(z[r][c]!=y[r][c] for r in range(len(x)) for c in range(len(x[0]))))
    np.testing.assert_array_equal(a['state'][0,:1000],expected)
    assert min(expected)>=0 and g.stats()['unique_param_storage']==2
    assert g.stats()['cell_defs']==1
    assert a['state'][0,1000]==0
    assert a['state'][0,1001]==2


def test_conditional_query_generalizes_colors_and_padding_and_backend_parity():
    g=build_conditional_state_gene()
    for source in range(10):
        marker=(source+1)%10;target=(source+2)%10;bg=(source+3)%10
        x=[[source,marker,bg,bg],[bg,bg,bg,source]]
        y=transform(x,source,marker,target)
        demo=[{'input':x,'output':y}]
        query=[[marker,source,source],[bg,bg,bg]];expected=transform(query,source,marker,target)
        a=GridPredictor(g,backend='reference')(demo,query)
        b=GridPredictor(g,backend='compiled')(demo,query)
        assert a==b and expected in a
    # A zero marker cannot come from padding.
    assert [[2]] in GridPredictor(g)([{'input':[[2,0,9],[9,2,9]],'output':[[3,0,9],[9,3,9]]}],[[2]])


def test_query_labels_do_not_change_state_and_mismatch_falls_back_identity():
    g=build_conditional_state_gene();rt=CompiledInterpreter(g)
    a=rt.run_bundle('conditional_root',inputs([[2,4]],[[3,4]]),'/r')
    q1=rt.run_bundle('conditional_root',inputs([[4,2]],[[8,8]],1,a['state']),'/r')
    q2=rt.run_bundle('conditional_root',inputs([[4,2]],[[1]],1,a['state']),'/r')
    np.testing.assert_array_equal(q1['state'],a['state'])
    for k in q1:np.testing.assert_array_equal(q1[k],q2[k])
    assert GridPredictor(g)([{'input':[[2,4]],'output':[[3]]}],[[2,4]])==[[[2,4]],[[2,4]]]


def test_original_generator_positive_negative_cases_and_changed_task_colors():
    data=generate_conditional_state_family(30,93)
    assert data==generate_conditional_state_family(30,93)
    assert len({tuple(t['colors'].values()) for t in data['tasks']})>20
    for task in data['tasks']:
        s,m,t,b=task['colors'].values()
        for pair in task['train']+task['test']:
            assert pair['output']==transform(pair['input'],s,m,t)
            assert pair['input'][0][0]==s and pair['output'][0][0]==t
            assert pair['input'][-1][-1]==pair['output'][-1][-1]==s


def test_authored_rank_microcell_is_executed_and_its_weight_is_evolvable():
    g=build_conditional_state_gene()
    errors=np.arange(1000,dtype=float).reshape(1,1000,1)
    for runtime in (Interpreter,CompiledInterpreter):
        got=runtime(g).run_bundle('conditional_rank_cluster',{'errors':errors},'/rank')
        np.testing.assert_array_equal(got['scores'],-errors)
    g.params['conditional_rank.W'][0,0]=-2.
    got=Interpreter(g).run_bundle('conditional_rank_cluster',{'errors':errors},'/rank')
    np.testing.assert_array_equal(got['scores'],-2*errors)
