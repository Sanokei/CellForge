# SPDX-License-Identifier: MIT-0
"""Authored directional-copy cells, graph-owned demo error selection, no gradients."""
import argparse,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from build_grid_associative import Nodes


def build_direction_router(adaptation=True):
    g=Graph(meta=dict(task='grid',grid_interface='grid-graph/1',grid_attempts=2,state_dim=6,
                     construction='authored directional router/1',gradients=False,
                     adaptation=adaptation,provenance='original authored primitive bank; demo squared-error state',
                     limitations='identity and four one-pixel neighbor copies only; second attempt identity; shape mismatch identity'))
    g.params['rank.W']=np.array([[-1.]])
    g.params['rank.b']=np.zeros(1)
    g.defs['rank']={'kind':'cell','act':'identity','params':{'W1':'rank.W','b1':'rank.b'}}
    directions=[(0,0),(-1,0),(1,0),(0,-1),(0,1)]
    for i,(dr,dc) in enumerate(directions):
        n=Nodes();x='$input'
        if dr:x=n.add('op:shift',{'x':x},axis=0,k=-dr)
        if dc:x=n.add('op:shift',{'x':x},axis=1,k=-dc)
        valid=n.add('op:sum',{'x':x},axis=2,keepdims=True)
        missing=n.add('op:bias',{'x':n.op('neg',x=valid)},c=1)
        x=n.op('add',a=x,b=n.op('mul',a='$input',b=missing))
        mask=n.add('op:sum',{'x':'$input'},axis=2,keepdims=True)
        x=n.op('mul',a=x,b=mask)
        g.defs[f'direction_{i}']=n.bundle('cluster',['input'],{'colors':x})
        n=Nodes();x=n.add(f'direction_{i}',{'input':'$input'})+'.colors'
        colors=n.add('op:concat',{'x0':n.reshape(x,[1,30,30,10]),'x1':n.reshape('$input',[1,30,30,10])},axis=0)
        dims={}
        for axis in ('height','width'):
            dims[axis+'_logits']=n.reshape(n.add('op:concat',{'x0':'$input_'+axis,'x1':'$input_'+axis},axis=0),[2,30])
        g.defs[f'expert_{i}']=n.bundle('expert',['input','input_height','input_width'],dict(color_logits=colors,**dims))
    n=Nodes();gate=n.reshape(n.add('op:bias',{'x':n.op('neg',x='$phase')},c=1),[1])
    if not adaptation:gate=n.op('mul',a=gate,b=n.add('op:coords',n=1,start=0.,step=0.))
    errors=[]
    for i in range(5):
        x=n.add(f'direction_{i}',{'input':'$input'})+'.colors'
        loss=n.sum(n.reshape(n.op('square',x=n.op('sub',a=x,b='$output')),[9000]))
        errors.append(n.op('add',a=n.slice('$state',i,i+1),b=n.op('mul',a=gate,b=loss)))
    diffs=[]
    for axis in ('height','width'):
        diffs.append(n.sum(n.op('abs',x=n.op('sub',a='$input_'+axis,b='$output_'+axis))))
    bad=n.add('op:step',{'x':n.op('add',a=n.slice('$state',5,6),b=n.op('mul',a=gate,b=n.op('add',a=diffs[0],b=diffs[1])))})
    state=n.add('op:concat',{**{f'x{i}':x for i,x in enumerate(errors)},'x5':bad},axis=0)
    penalties=n.add('op:coords',n=5,start=0.,step=1.)
    penalties=n.add('op:step',{'x':penalties})
    penalties=n.add('op:scale',{'x':penalties},c=1000000.)
    scores=n.op('add',a=n.add('op:concat',{f'x{i}':x for i,x in enumerate(errors)},axis=0),b=n.op('mul',a=penalties,b=bad))
    scores=n.reshape(n.add('rank',{'x':n.reshape(scores,[5,1])}),[5])
    d=n.add('dispatch',{'input':'$input','input_height':'$input_height','input_width':'$input_width','scores':scores},branches=[f'expert_{i}' for i in range(5)],k=1)
    g.defs['root']=n.bundle('root',['input','output','input_height','input_width','output_height','output_width','phase','state'],{'state':state,**{k:d+'.'+k for k in ('color_logits','height_logits','width_logits')}})
    g.entries={'grid_step':'root'};g.validate();return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh output required')
    g=build_direction_router();h=g.save(out)
    print(json.dumps(dict(hash=h,stats=g.stats())))


if __name__=='__main__':main()
