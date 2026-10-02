# SPDX-License-Identifier: MIT-0
"""Graph-owned demonstration reconstruction of conditional adjacency rules.

No gradients, host rule enumeration, corpus fitting or benchmark reads.
The 1,000 reconstruction errors are task-fitted counting state, not weights.
"""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph, Limits
from cellforge.ir.profiles import check_profile
from build_grid_associative import Nodes


def build_conditional_state_gene():
    g=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':2,
        'state_dim':1002,'profile':'ladder-micro/1',
        'construction':'authored conditional adjacency state/2; no gradients',
        'provenance':'task-fitted counting/rule selection from demonstrations only',
        'gene_contract':{'update_cluster':'conditional_update',
            'apply_expert':'conditional_expert0','state_layout':'1000 lexicographic source/marker/target errors, shared shape penalty, identity pixel error',
            'rule':'source becomes target iff a valid orthogonal neighbor is marker; simultaneous updates',
            'query_state_frozen':True,'dimensions':'preserved; any dimension-mismatched demonstration forces identity'},
        'limitations':'one simultaneous orthogonal rule; no objects or size changes; ties lexicographic'})
    # A genuinely used, evolvable two-slot microcell scores reconstruction
    # errors. It replaces the historical generic negation, preserving the
    # authored initial rule selection exactly.
    g.params['conditional_rank.W']=np.array([[-1.]])
    g.params['conditional_rank.b']=np.array([0.])
    g.defs['conditional_rank']={'kind':'cell','act':'identity',
        'params':{'W1':'conditional_rank.W','b1':'conditional_rank.b'}}
    rank_nodes=Nodes()
    rank_output=rank_nodes.add('conditional_rank',{'x':'$errors'})
    g.defs['conditional_rank_cluster']=rank_nodes.bundle('cluster',['errors'],{'scores':rank_output})
    n=Nodes()
    shifts=[n.add('op:shift',{'x':'$input'},axis=a,k=k) for a,k in ((0,-1),(0,1),(1,-1),(1,1))]
    total=shifts[0]
    for v in shifts[1:]:total=n.op('add',a=total,b=v)
    neighbor=n.add('op:step',{'x':total})
    x=n.reshape('$input',[900,10]);y=n.reshape('$output',[900,10]);nb=n.reshape(neighbor,[900,10])
    f=n.op('mul',a=n.reshape(x,[900,10,1]),b=n.reshape(nb,[900,1,10]))
    fm=n.reshape(f,[900,100])
    ft=n.add('op:transpose',{'x':fm},perm=[1,0])
    c=n.reshape(n.op('matmul',a=ft,b=y),[10,10,10])
    own=n.add('op:sum',{'x':n.op('mul',a=x,b=y)},axis=1,keepdims=True)
    valid=n.add('op:sum',{'x':x},axis=1,keepdims=True)
    baseline=n.add('op:sum',{'x':n.op('sub',a=valid,b=own)},axis=0,keepdims=True)
    fy=n.op('mul',a=f,b=n.reshape(y,[900,10,1]))
    owncounts=n.add('op:sum',{'x':fy},axis=0,keepdims=False)
    delta=n.op('sub',a=n.reshape(owncounts,[10,10,1]),b=c)
    errors=n.reshape(n.op('add',a=delta,b=baseline),[1000])
    dimerrs=[]
    for p in ('height','width'):
        diff=n.op('abs',x=n.op('sub',a='$input_'+p,b='$output_'+p))
        dimerrs.append(n.add('op:sum',{'x':diff},axis=0,keepdims=True))
    mismatch=n.add('op:step',{'x':n.op('add',a=dimerrs[0],b=dimerrs[1])})
    penalty=n.add('op:scale',{'x':mismatch},c=1000000)
    errors=n.op('add',a=errors,b=penalty)
    update=n.add('op:concat',{'x0':errors,'x1':n.reshape(penalty,[1]),'x2':n.reshape(baseline,[1])},axis=0)
    demo=n.add('op:bias',{'x':n.op('neg',x='$phase')},c=1)
    state=n.op('add',a='$state',b=n.op('mul',a=update,b=demo))
    g.defs['conditional_update']=n.bundle('cluster',['input','output','input_height','input_width','output_height','output_width','phase','state'],{'state':state,'neighbor':neighbor})
    n=Nodes();errors=n.reshape(n.slice('$state',0,1000),[1000,1])
    scored=n.add('conditional_rank_cluster',{'errors':errors})
    scores=n.reshape(scored+'.scores',[1000])
    tie=n.add('op:coords',n=1000,start=0.,step=-1e-7)
    scores=n.op('add',a=scores,b=tie)
    picks=[]
    for rank in range(2):
        pick=n.add('op:onehot_argmax',{'x':scores},axis=0);picks.append(pick)
        scores=n.op('sub',a=scores,b=n.add('op:scale',{'x':pick},c=1e12))
    outs=[]
    for pick in picks:
        cube=n.reshape(pick,[10,10,10])
        source=n.add('op:sum',{'x':cube},axis=2,keepdims=False);source=n.add('op:sum',{'x':source},axis=1,keepdims=False)
        marker=n.add('op:sum',{'x':cube},axis=2,keepdims=False);marker=n.add('op:sum',{'x':marker},axis=0,keepdims=False)
        target=n.add('op:sum',{'x':cube},axis=0,keepdims=False);target=n.add('op:sum',{'x':target},axis=0,keepdims=False)
        sm=n.add('op:sum',{'x':n.op('mul',a='$input',b=source)},axis=2,keepdims=True)
        nm=n.add('op:sum',{'x':n.op('mul',a='$neighbor',b=marker)},axis=2,keepdims=True)
        gate=n.op('mul',a=sm,b=nm)
        okay=n.add('op:bias',{'x':n.op('neg',x=n.add('op:step',{'x':n.slice('$state',1000,1001)}))},c=1)
        gate=n.op('mul',a=gate,b=okay)
        inv=n.add('op:bias',{'x':n.op('neg',x=gate)},c=1)
        colors=n.op('add',a=n.op('mul',a='$input',b=inv),b=n.op('mul',a=gate,b=target))
        outs.append(n.reshape(colors,[1,30,30,10]))
    logits=n.add('op:concat',{'x0':outs[0],'x1':outs[1]},axis=0)
    hs=n.add('op:concat',{'x0':'$input_height','x1':'$input_height'},axis=0)
    ws=n.add('op:concat',{'x0':'$input_width','x1':'$input_width'},axis=0)
    outputs={'color_logits':logits,'height_logits':n.reshape(hs,[2,30]),'width_logits':n.reshape(ws,[2,30])}
    g.defs['conditional_expert0']=n.bundle('expert',['input','input_height','input_width','neighbor','state'],outputs)
    g.defs['conditional_expert1']=copy.deepcopy(g.defs['conditional_expert0'])
    n=Nodes();scores=n.add('op:coords',n=2,start=0.,step=0.)
    g.defs['conditional_router']=n.bundle('router',[],{'scores':scores})
    ports=['input','output','input_height','input_width','output_height','output_width','phase','state']
    n=Nodes();u=n.add('conditional_update',{p:'$'+p for p in ports});r=n.add('conditional_router')
    d=n.add('dispatch',{'input':'$input','input_height':'$input_height','input_width':'$input_width','state':u+'.state','neighbor':u+'.neighbor','scores':r+'.scores'},branches=['conditional_expert0','conditional_expert1'],k=1)
    g.defs['conditional_root']=n.bundle('root',ports,{'state':u+'.state',**{p:d+'.'+p for p in outputs}})
    g.entries={'grid_step':'conditional_root'}
    g.validate(Limits(strict_microcells=True));check_profile(g,'ladder-micro/1');return g


def generate_conditional_state_family(tasks=100,seed=1729):
    """Original generators only: each task changes its four distinct colors.

    Positive and negative source examples appear in every demonstration and
    query. Generator reference is never called from the model runtime.
    """
    rng=np.random.default_rng(seed);items=[]
    for i in range(tasks):
        source,marker,target,bg=map(int,rng.choice(10,4,replace=False));pairs=[]
        for j in range(3):
            h,w=map(int,rng.integers(4,10,size=2));x=np.full((h,w),bg,dtype=int)
            # Random marker/source pairs in the top band; isolated negative
            # source at bottom right (markers never enter its neighborhood).
            x[0,0]=source;x[0,1]=marker;x[-1,-1]=source
            for _ in range(int(rng.integers(1,w))):
                r=int(rng.integers(0,h-2));c=int(rng.integers(0,w));x[r,c]=int(rng.choice([source,marker,bg]))
            x[0,0]=source;x[0,1]=marker
            y=x.copy()
            for r in range(h):
                for c in range(w):
                    if x[r,c]==source and any(0<=r+dr<h and 0<=c+dc<w and x[r+dr,c+dc]==marker for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))):y[r,c]=target
            pairs.append({'input':x.tolist(),'output':y.tolist()})
        items.append({'id':f'conditional-state-{seed}-{i}','family':'conditional_adjacency_state',
                      'colors':{'source':source,'marker':marker,'target':target,'background':bg},'train':pairs[:2],'test':pairs[2:]})
    return {'protocol':'original-conditional-state-family/1','seed':seed,'tasks':items,
            'provenance':'original authored generator; no benchmark data; freeze splits before measurement'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('preserve frozen artifact')
    g=build_conditional_state_gene();digest=g.save(out)
    receipt={'model_hash':digest,'stats':g.stats(),'state_dim':1002,
             'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'benchmark_tasks_read':0,'curriculum_scored':False,
             'provenance':g.meta['provenance'],'limitations':g.meta['limitations'],
             'gene_contract':g.meta['gene_contract']}
    (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
