# SPDX-License-Identifier: MIT-0
"""Authored exact 3x3 context memory; task-fitted state, no gradients.

Two bounded demonstration banks. Padding token 0 differs from color 0 token
1. Unknown or conflicting contexts preserve input. No benchmark reads.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile
from build_grid_associative import Nodes


def build_context_memory_gene():
    g=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':2,'state_dim':3602,
        'profile':'ladder-micro/1','construction':'authored exact-context memory/1',
        'provenance':'task-fitted exact-context memory from two most recent demonstrations; counting, no gradients',
        'limitations':'exact 3x3 matches only; two bounded banks; conflicting/unknown contexts identity; any demonstration dimension mismatch forces identity'})
    for name,W,b in [('token',np.arange(1,11,dtype=float)[None],np.zeros(1)),
                     ('key',(11.**np.arange(9))[None],np.zeros(1)),
                     ('toggle',np.array([[-1.]]),np.ones(1))]:
        g.params[name+'.W']=W;g.params[name+'.b']=b
        g.defs[name]={'kind':'cell','act':'identity','params':{'W1':name+'.W','b1':name+'.b'}}
    n=Nodes();tokens=n.add('token',{'x':'$input'});values=n.add('token',{'x':'$output'})
    neighbors=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            v=tokens
            if dr:v=n.add('op:shift',{'x':v},axis=0,k=dr)
            if dc:v=n.add('op:shift',{'x':v},axis=1,k=dc)
            neighbors.append(v)
    digits=n.add('op:concat',{f'x{i}':v for i,v in enumerate(neighbors)},axis=2)
    keys=n.reshape(n.add('key',{'x':digits}),[900])
    vals=n.reshape(values,[900]);toggle=n.slice('$state',3600,3601)
    newtoggle=n.reshape(n.add('toggle',{'x':n.reshape(toggle,[1,1])}),[1])
    shape=[]
    for d in ('height','width'):
        diff=n.op('abs',x=n.op('sub',a='$input_'+d,b='$output_'+d))
        shape.append(n.add('op:sum',{'x':diff},axis=0,keepdims=True))
    mismatch=n.reshape(n.add('op:step',{'x':n.op('add',a=shape[0],b=shape[1])}),[1])
    demo=n.reshape(n.add('op:bias',{'x':n.op('neg',x='$phase')},c=1),[1])
    compatible=n.add('op:bias',{'x':n.op('neg',x=mismatch)},c=1)
    write=n.op('mul',a=demo,b=compatible)
    gates=[n.op('mul',a=write,b=n.add('op:bias',{'x':n.op('neg',x=toggle)},c=1)),n.op('mul',a=write,b=toggle)]
    banks=[]
    for i,gate in enumerate(gates):
        old=n.slice('$state',1800*i,1800*(i+1))
        data=n.add('op:concat',{'x0':keys,'x1':vals},axis=0)
        inverse=n.add('op:bias',{'x':n.op('neg',x=gate)},c=1)
        banks.append(n.op('add',a=n.op('mul',a=old,b=inverse),b=n.op('mul',a=data,b=gate)))
    flip=n.op('mul',a=n.op('sub',a=newtoggle,b=toggle),b=demo)
    toggle=n.op('add',a=toggle,b=flip)
    bad=n.add('op:step',{'x':n.op('add',a=n.slice('$state',3601,3602),b=n.op('mul',a=mismatch,b=demo))})
    state=n.add('op:concat',{'x0':banks[0],'x1':banks[1],'x2':toggle,'x3':bad},axis=0)
    g.defs['context_memory_update']=n.bundle('cluster',['input','output','input_height','input_width','output_height','output_width','phase','state'],{'state':state,'keys':keys})
    n=Nodes()
    bankkeys=n.add('op:concat',{'x0':n.slice('$state',0,900),'x1':n.slice('$state',1800,2700)},axis=0)
    bankvals=n.add('op:concat',{'x0':n.slice('$state',900,1800),'x1':n.slice('$state',2700,3600)},axis=0)
    def equal(a,b):
        dist=n.op('abs',x=n.op('sub',a=a,b=b))
        return n.add('op:bias',{'x':n.op('neg',x=n.add('op:step',{'x':dist}))},c=1)
    match=equal(n.reshape('$keys',[900,1]),n.reshape(bankkeys,[1,1800]))
    valid=n.reshape(n.add('op:step',{'x':bankvals}),[1,1800])
    match=n.op('mul',a=match,b=valid)
    colors=n.add('op:coords',n=10,start=1.,step=1.)
    valhot=equal(n.reshape(bankvals,[1800,1]),n.reshape(colors,[1,10]))
    counts=n.op('matmul',a=match,b=valhot)
    total=n.add('op:sum',{'x':counts},axis=1,keepdims=True)
    largest=n.add('op:max',{'x':counts},axis=1,keepdims=True)
    consistent=equal(total,largest)
    gate=n.op('mul',a=consistent,b=n.add('op:step',{'x':total}))
    okay=n.add('op:bias',{'x':n.op('neg',x=n.slice('$state',3601,3602))},c=1)
    gate=n.op('mul',a=gate,b=okay)
    gate=n.reshape(gate,[30,30,1])
    validinput=n.add('op:sum',{'x':'$input'},axis=2,keepdims=True)
    gate=n.op('mul',a=gate,b=validinput)
    chosen=n.reshape(n.add('op:onehot_argmax',{'x':counts},axis=1),[30,30,10])
    inverse=n.add('op:bias',{'x':n.op('neg',x=gate)},c=1)
    prediction=n.op('add',a=n.op('mul',a=chosen,b=gate),b=n.op('mul',a='$input',b=inverse))
    logits=n.add('op:concat',{'x0':n.reshape(prediction,[1,30,30,10]),'x1':n.reshape('$input',[1,30,30,10])},axis=0)
    hs=n.reshape(n.add('op:concat',{'x0':'$input_height','x1':'$input_height'},axis=0),[2,30])
    ws=n.reshape(n.add('op:concat',{'x0':'$input_width','x1':'$input_width'},axis=0),[2,30])
    out={'color_logits':logits,'height_logits':hs,'width_logits':ws}
    g.defs['context_memory_expert0']=n.bundle('expert',['input','input_height','input_width','keys','state'],out)
    g.defs['context_memory_expert1']=copy.deepcopy(g.defs['context_memory_expert0'])
    n=Nodes();scores=n.add('op:coords',n=2,start=0.,step=0.)
    g.defs['context_memory_router']=n.bundle('router',[],{'scores':scores})
    ports=['input','output','input_height','input_width','output_height','output_width','phase','state']
    n=Nodes();u=n.add('context_memory_update',{p:'$'+p for p in ports});r=n.add('context_memory_router')
    dispatch=n.add('dispatch',{'input':'$input','input_height':'$input_height','input_width':'$input_width','keys':u+'.keys','state':u+'.state','scores':r+'.scores'},branches=['context_memory_expert0','context_memory_expert1'],k=1)
    g.defs['context_memory_root']=n.bundle('root',ports,{'state':u+'.state',**{p:dispatch+'.'+p for p in out}})
    g.entries={'grid_step':'context_memory_root'}
    g.validate(Limits(strict_microcells=True));check_profile(g,'ladder-micro/1');return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve frozen artifact')
    g=build_context_memory_gene();h=g.save(out)
    card={'model_hash':h,'stats':g.stats(),'state_dim':3602,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'benchmark_tasks_read':0,'curriculum_scored':False,'provenance':g.meta['provenance'],'limitations':g.meta['limitations']}
    (out/'MODEL_CARD.json').write_text(json.dumps(card,indent=2)+'\n');print(json.dumps(card))


def generate_context_memory_family(tasks=100,seed=8129):
    """Original binary-color dense samples of fixed directional-copy rules.

    Exact-context coverage is not guaranteed and must be measured separately.
    Direction changes between tasks. Outside-grid pixels preserve the center.
    """
    rng=np.random.default_rng(seed);items=[]
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    for i in range(tasks):
        palette=list(map(int,rng.choice(10,2,replace=False)));dr,dc=directions[i%4];pairs=[]
        for j in range(3):
            h=w=12 if j<2 else 9
            x=rng.choice(palette,size=(h,w));y=x.copy()
            for r in range(h):
                for c in range(w):
                    if 0<=r+dr<h and 0<=c+dc<w:y[r,c]=x[r+dr,c+dc]
            pairs.append({'input':x.tolist(),'output':y.tolist()})
        items.append({'id':f'exact-context-{seed}-{i}','family':'directional_copy',
                      'direction':[dr,dc],'palette':palette,'train':pairs[:2],'test':pairs[2:]})
    return {'protocol':'original-exact-context-family/1','seed':seed,'tasks':items,
            'provenance':'original authored synthetic data; no benchmarks; context coverage unproven'}


if __name__=='__main__':main()
