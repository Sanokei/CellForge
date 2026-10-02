# SPDX-License-Identifier: MIT-0
"""Import authored scaling genes; graph-owned cumulative demo-error routing."""
import argparse,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.construction.transactions import import_def
from build_grid_associative import Nodes
from build_grid_repeat_gene import build_repeat_gene


def build_scaling_selector(adaptation=True):
    g=Graph(meta=dict(task='grid',grid_interface='grid-graph/1',grid_attempts=2,state_dim=3,
                     construction='authored scaling selector/1',gradients=False,adaptation=adaptation,
                     provenance='authored repeat genes from expanded train-demo census; imported with explicit lineage',
                     limitations='identity/repeat2/repeat3 only; cumulative squared demo error; second attempt identity; no evolution claim'))
    g.params['rank.W']=np.array([[-1.]])
    g.params['rank.b']=np.zeros(1)
    g.defs['rank']={'kind':'cell','act':'identity','params':{'W1':'rank.W','b1':'rank.b'}}
    n=Nodes();rank=n.add('rank',{'x':n.reshape('$errors',[3,1])})
    g.defs['rank.cluster']=n.bundle('cluster',['errors'],{'scores':n.reshape(rank,[3])})
    n=Nodes();rank=n.add('rank.cluster',{'errors':'$errors'})
    g.defs['router']=n.bundle('router',['errors'],{'scores':rank+'.scores'})
    n=Nodes();g.defs['identity']=n.bundle('cluster',['input','input_height','input_width'],
                                         dict(colors='$input',height='$input_height',width='$input_width'))
    genes=['identity'];lineage=[]
    for factor in (2,3):
        donor=build_repeat_gene(factor)
        lineage.append(dict(factor=factor,donor_hash=donor.hash(),transaction=import_def(g,donor,'repeat.gene',prefix=f'scale{factor}_')))
        genes.append(f'scale{factor}_repeat.gene')
    g.meta['import_lineage']=lineage
    for i,gene in enumerate(genes):
        n=Nodes();call=n.add(gene,{p:'$'+p for p in ('input','input_height','input_width')})
        colors=n.add('op:concat',{'x0':n.reshape(call+'.colors',[1,30,30,10]),'x1':n.reshape('$input',[1,30,30,10])},axis=0)
        dims={}
        for axis in ('height','width'):
            dims[axis+'_logits']=n.add('op:concat',{'x0':n.reshape(call+'.'+axis,[1,30]),'x1':n.reshape('$input_'+axis,[1,30])},axis=0)
        g.defs[f'expert_{i}']=n.bundle('expert',['input','input_height','input_width'],dict(color_logits=colors,**dims))
    n=Nodes();gate=n.reshape(n.add('op:bias',{'x':n.op('neg',x='$phase')},c=1),[1])
    if not adaptation:gate=n.add('op:scale',{'x':gate},c=0.)
    flat={p:n.reshape('$'+p,shape) for p,shape in [('input',[30,30,10]),('input_height',[30]),('input_width',[30])]}
    errors=[]
    for i,gene in enumerate(genes):
        call=n.add(gene,flat)
        loss=n.sum(n.reshape(n.op('square',x=n.op('sub',a=call+'.colors',b=n.reshape('$output',[30,30,10]))),[9000]))
        for axis in ('height','width'):
            diff=n.sum(n.op('abs',x=n.op('sub',a=call+'.'+axis,b=n.reshape('$output_'+axis,[30]))))
            loss=n.op('add',a=loss,b=n.add('op:scale',{'x':diff},c=9001.))
        errors.append(n.op('add',a=n.slice('$state',i,i+1),b=n.op('mul',a=gate,b=loss)))
    state=n.add('op:concat',{f'x{i}':x for i,x in enumerate(errors)},axis=0)
    scores=n.add('router',{'errors':state})+'.scores'
    d=n.add('dispatch',{**flat,'scores':scores},branches=[f'expert_{i}' for i in range(3)],k=1)
    outputs={'state':state}
    for name,shape in [('color_logits',[2,30,30,10]),('height_logits',[2,30]),('width_logits',[2,30])]:
        outputs[name]=n.reshape(d+'.'+name,shape)
    g.defs['root']=n.bundle('root',['input','output','input_height','input_width','output_height','output_width','phase','state'],outputs)
    g.entries={'grid_step':'root'};g.validate();check_profile(g,'ladder-micro/1');return g


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--no-adaptation',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    g=build_scaling_selector(not a.no_adaptation);print(json.dumps(dict(hash=g.save(a.out),stats=g.stats())))
