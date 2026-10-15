# SPDX-License-Identifier: MIT-0
"""Authored nearest-neighbor pixel enlargement; typed graph gene, no fitting."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from build_grid_associative import Nodes


def build_repeat_gene(factor):
    if factor not in (2,3):raise ValueError('declared factor2or3 required')
    g=Graph(meta=dict(task='grid',grid_interface='grid-graph/1',grid_attempts=1,state_dim=0,
                     construction='authored pixel-repetition gene/1',factor=factor,gradients=False,
                     provenance='expanded public-training demo-only census; no query-driven authoring',
                     gene_contract=dict(cluster='repeat.gene',inputs=['input','input_height','input_width'],outputs=['colors','height','width']),
                     limitations='fixed factor, no task adaptation; output dimension beyond30 returns identity'))
    g.params['dimension.W']=np.arange(1,31,dtype=float)[None];g.params['dimension.b']=np.zeros(1)
    g.defs['dimension']={'kind':'cell','act':'identity','params':{'W1':'dimension.W','b1':'dimension.b'}}
    n=Nodes();coords=n.add('op:coords',n=30)
    source=n.reshape(coords,[1,30]);target=n.reshape(coords,[30,1])
    scaled=n.add('op:scale',{'x':source},c=factor)
    matches=[]
    for offset in range(factor):
        distance=n.op('abs',x=n.op('sub',a=target,b=n.add('op:bias',{'x':scaled},c=offset)))
        matches.append(n.op('relu',x=n.add('op:bias',{'x':n.op('neg',x=distance)},c=1)))
    matrix=matches[0]
    for m in matches[1:]:matrix=n.op('add',a=matrix,b=m)
    x=n.reshape(n.op('matmul',a=matrix,b=n.reshape('$input',[30,300])),[30,30,10])
    x=n.add('op:transpose',{'x':x},perm=[1,0,2])
    x=n.reshape(n.op('matmul',a=matrix,b=n.reshape(x,[30,300])),[30,30,10])
    x=n.add('op:transpose',{'x':x},perm=[1,0,2])
    sizes=[];hots=[];excess=[]
    for axis in ('height','width'):
        size=n.add('dimension',{'x':'$input_'+axis})
        size=n.add('op:scale',{'x':size},c=factor);sizes.append(size)
        excess.append(n.op('relu',x=n.add('op:bias',{'x':size},c=-30)))
        distance=n.op('abs',x=n.op('sub',a=n.add('op:bias',{'x':coords},c=1),b=size))
        hots.append(n.op('relu',x=n.add('op:bias',{'x':n.op('neg',x=distance)},c=1)))
    bad=n.add('op:step',{'x':n.op('add',a=excess[0],b=excess[1])})
    good=n.add('op:bias',{'x':n.op('neg',x=bad)},c=1)
    colors=n.op('add',a=n.op('mul',a=x,b=good),b=n.op('mul',a='$input',b=bad))
    dims=[n.op('add',a=n.op('mul',a=hot,b=good),b=n.op('mul',a='$input_'+axis,b=bad)) for hot,axis in zip(hots,('height','width'))]
    g.defs['repeat.gene']=n.bundle('cluster',['input','input_height','input_width'],dict(colors=colors,height=dims[0],width=dims[1]))
    n=Nodes();call=n.add('repeat.gene',{p:'$'+p for p in ('input','input_height','input_width')})
    outputs={'state':'$state'}
    for port,source,shape in [('color_logits','colors',[1,30,30,10]),('height_logits','height',[1,30]),('width_logits','width',[1,30])]:
        outputs[port]=n.reshape(call+'.'+source,shape)
    ports=['input','input_height','input_width','state']
    g.defs['expert0']=n.bundle('expert',ports,outputs);g.defs['expert1']=copy.deepcopy(g.defs['expert0'])
    n=Nodes();scores=n.add('op:coords',n=2,start=0.,step=0.)
    g.defs['router']=n.bundle('router',[],{'scores':scores})
    n=Nodes();r=n.add('router');d=n.add('dispatch',{**{p:'$'+p for p in ports},'scores':r+'.scores'},branches=['expert0','expert1'],k=1)
    g.defs['root']=n.bundle('root',ports,{p:d+'.'+p for p in outputs});g.entries={'grid_step':'root'}
    g.validate();return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--factor',type=int,choices=(2,3),required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh gene required')
    g=build_repeat_gene(a.factor);print(json.dumps(dict(hash=g.save(out),stats=g.stats())))


if __name__=='__main__':main()
