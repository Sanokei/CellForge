# SPDX-License-Identifier: MIT-0
"""Exact-context graph memory with unambiguous coarse color backoff.

Task-fitted counting state only; no gradients or benchmark data. Fine banks
overwrite, while the 10x10 color table accumulates all compatible demos.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from build_grid_context_memory_gene import build_context_memory_gene
from build_grid_associative import Nodes
from cellforge.ir.graph import Limits
from cellforge.ir.profiles import check_profile


def build_context_backoff_gene():
    g=build_context_memory_gene()
    g.meta.update(state_dim=3702,construction='authored exact-context plus coarse backoff/1',
        provenance='task-fitted counting: two exact-context banks plus accumulating 10x10 source/target counts; no gradients',
        limitations='fine known and unanimous first; else observed unanimous coarse color row; else identity; shape mismatch forces identity')
    update=g.defs['context_memory_update'];n=Nodes();n.nodes=copy.deepcopy(update['nodes'])
    x=n.reshape('$input',[900,10]);y=n.reshape('$output',[900,10])
    xt=n.add('op:transpose',{'x':x},perm=[1,0]);counts=n.reshape(n.op('matmul',a=xt,b=y),[100])
    shape=[]
    for dim in ('height','width'):
        diff=n.op('abs',x=n.op('sub',a='$input_'+dim,b='$output_'+dim))
        shape.append(n.add('op:sum',{'x':diff},axis=0,keepdims=True))
    mismatch=n.add('op:step',{'x':n.op('add',a=shape[0],b=shape[1])})
    compatible=n.add('op:bias',{'x':n.op('neg',x=mismatch)},c=1)
    demo=n.add('op:bias',{'x':n.op('neg',x='$phase')},c=1)
    gate=n.op('mul',a=compatible,b=demo)
    coarse=n.op('add',a=n.slice('$state',3602,3702),b=n.op('mul',a=counts,b=gate))
    state=n.add('op:concat',{'x0':update['outputs']['state'],'x1':coarse},axis=0)
    update['nodes']=n.nodes;update['outputs']['state']=state
    expert=g.defs['context_memory_expert0'];nodes=expert['nodes'];byid={v['id']:v for v in nodes}
    # Find the actual identity fallback multiply through the output dataflow;
    # replace its input with coarse colors, leaving fine lookup and gating intact.
    concat=byid[expert['outputs']['color_logits']]
    reshaped=byid[concat['in']['x0']];prediction=byid[reshaped['in']['x']]
    preserved=byid[prediction['in']['b']]
    if preserved['use']!='op:mul' or preserved['in']['a']!='$input':raise ValueError('fine-only fallback layout changed')
    n=Nodes();table=n.reshape(n.slice('$state',3602,3702),[10,10])
    total=n.add('op:sum',{'x':table},axis=1,keepdims=True)
    largest=n.add('op:max',{'x':table},axis=1,keepdims=True)
    conflict=n.add('op:step',{'x':n.op('abs',x=n.op('sub',a=total,b=largest))})
    unanimous=n.add('op:bias',{'x':n.op('neg',x=conflict)},c=1)
    rowgate=n.op('mul',a=unanimous,b=n.add('op:step',{'x':total}))
    rowgate=n.op('mul',a=rowgate,b=n.add('op:bias',{'x':n.op('neg',x=n.slice('$state',3601,3602))},c=1))
    mapping=n.add('op:onehot_argmax',{'x':table},axis=1)
    inputflat=n.reshape('$input',[900,10]);targets=n.reshape(n.op('matmul',a=inputflat,b=mapping),[30,30,10])
    pixelgate=n.reshape(n.op('matmul',a=inputflat,b=rowgate),[30,30,1])
    inverse=n.add('op:bias',{'x':n.op('neg',x=pixelgate)},c=1)
    fallback=n.op('add',a=n.op('mul',a=targets,b=pixelgate),b=n.op('mul',a='$input',b=inverse))
    rename={node['id']:'backoff_'+node['id'] for node in n.nodes}
    for node in n.nodes:
        node['id']=rename[node['id']]
        if 'in' in node:node['in']={p:rename.get(ref,ref) for p,ref in node['in'].items()}
    insertion=next(i for i,v in enumerate(nodes) if v['id']==preserved['id'])
    preserved['in']['a']=rename[fallback]
    expert['nodes']=nodes[:insertion]+n.nodes+nodes[insertion:]
    g.defs['context_memory_expert1']=copy.deepcopy(expert)
    g.validate(Limits(strict_microcells=True));check_profile(g,'ladder-micro/1');return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve frozen artifact')
    g=build_context_backoff_gene();h=g.save(out)
    card={'model_hash':h,'stats':g.stats(),'state_dim':3702,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'base_builder_sha256':hashlib.sha256(Path(__file__).with_name('build_grid_context_memory_gene.py').read_bytes()).hexdigest(),
          'benchmark_tasks_read':0,'curriculum_scored':False,'provenance':g.meta['provenance'],'limitations':g.meta['limitations']}
    (out/'MODEL_CARD.json').write_text(json.dumps(card,indent=2)+'\n');print(json.dumps(card))


if __name__=='__main__':main()
