"""Authored simultaneous conditional recoloring; generic cells/IR only.

Recolor source to target iff a valid orthogonal neighbor is marker. Other
pixels persist. Zero is a valid color, distinguished from all-zero padding.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile
from build_grid_local_fill_gene import Nodes


def add_adjacency_recolor_cluster(graph,source=2,target=3,marker=4,prefix='adjacency_recolor'):
    if any(type(c)!=int or not 0<=c<10 for c in (source,target,marker)):raise ValueError('colors must be integers0..9')
    if prefix+'.gene' in graph.defs:raise ValueError('gene prefix collision')
    for role,color in [('source',source),('marker',marker)]:
        graph.params[prefix+'.'+role+'.W']=np.eye(10)[color:color+1]
        graph.params[prefix+'.'+role+'.b']=np.zeros(1)
        graph.defs[prefix+'.'+role]={'kind':'cell','act':'identity','params':{'W1':prefix+'.'+role+'.W','b1':prefix+'.'+role+'.b'}}
    graph.params[prefix+'.target']=np.eye(10)[target]
    n=Nodes();source_mask=n.add(prefix+'.source',{'x':'$input'});marker_mask=n.add(prefix+'.marker',{'x':'$input'})
    neighbors=[n.add('op:shift',{'x':marker_mask},axis=axis,k=k) for axis,k in ((0,-1),(0,1),(1,-1),(1,1))]
    joined=n.add('op:concat',{f'x{i}':v for i,v in enumerate(neighbors)},axis=2)
    total=n.add('op:sum',{'x':joined},axis=2,keepdims=True)
    adjacent=n.add('op:step',{'x':total})
    gate=n.add('op:mul',{'a':source_mask,'b':adjacent})
    inverse=n.add('op:neg',{'x':gate});inverse=n.add('op:bias',{'x':inverse},c=1)
    preserved=n.add('op:mul',{'a':'$input','b':inverse})
    target_ref=n.add('param:'+prefix+'.target');target_ref=n.add('op:reshape',{'x':target_ref},shape=[1,1,10])
    replacement=n.add('op:mul',{'a':target_ref,'b':gate})
    colors=n.add('op:add',{'a':preserved,'b':replacement})
    graph.defs[prefix+'.gene']=n.bundle('cluster',['input','input_height','input_width'],{'colors':colors,'height':'$input_height','width':'$input_width'})
    return prefix+'.gene'


def build_adjacency_recolor_gene(source=2,target=3,marker=4):
    graph=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':1,'state_dim':0,'profile':'ladder-micro/1',
        'construction':'authored conditional adjacency recolor/1; no gradients/fitted labels',
        'gene_contract':{'cluster':'adjacency_recolor.gene','inputs':['input','input_height','input_width'],'outputs':['colors','height','width'],
        'source':source,'target':target,'marker':marker,'rule':'simultaneously recolor source iff orthogonal valid neighbor is marker; all other pixels persist'},
        'limitations':'fixed authored colors and one-step adjacency only; no object semantics or demonstration adaptation; duplicated expert wrappers solely for profile compatibility'})
    gene=add_adjacency_recolor_cluster(graph,source,target,marker)
    ports=['input','input_mask','input_height','input_width','output','output_mask','output_height','output_width','phase','index','state']
    n=Nodes();call=n.add(gene,{p:'$'+p for p in ('input','input_height','input_width')});outputs={'state':'$state'}
    for port,name,shape in [('color_logits','colors',[1,30,30,10]),('height_logits','height',[1,30]),('width_logits','width',[1,30])]:
        outputs[port]=n.add('op:reshape',{'x':call+'.'+name},shape=shape)
    graph.defs['recolor_expert0']=n.bundle('expert',ports,outputs);graph.defs['recolor_expert1']=copy.deepcopy(graph.defs['recolor_expert0'])
    n=Nodes();scores=n.add('op:coords',n=2,start=0.,step=0.)
    graph.defs['recolor_router']=n.bundle('router',[],{'scores':scores})
    n=Nodes();router=n.add('recolor_router');dispatch=n.add('dispatch',{**{p:'$'+p for p in ports},'scores':router+'.scores'},branches=['recolor_expert0','recolor_expert1'],k=1)
    graph.defs['recolor_root']=n.bundle('root',ports,{p:dispatch+'.'+p for p in outputs});graph.entries={'grid_step':'recolor_root'}
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1');return graph


def generate_adjacency_family(tasks=300,seed=841):
    """Author separate train/test fixtures; never invoked by inference."""
    if tasks<1:raise ValueError('positive task count required')
    rng=np.random.default_rng(seed);items=[]
    for i in range(tasks):
        pairs=[]
        for _ in range(3):
            h,w=map(int,rng.integers(3,10,size=2));bg=int(rng.choice([0,1,5,6,7,8,9]))
            x=[[bg for _ in range(w)] for _ in range(h)]
            for _ in range(int(rng.integers(1,h*w//3+1))):x[int(rng.integers(h))][int(rng.integers(w))]=2
            x[0][0]=2;x[0][1]=4;x[h-1][w-1]=2
            y=[row.copy() for row in x]
            for r in range(h):
                for c in range(w):
                    if x[r][c]==2 and any(0<=r+dy<h and 0<=c+dx<w and x[r+dy][c+dx]==4 for dy,dx in ((-1,0),(1,0),(0,-1),(0,1))):y[r][c]=3
            pairs.append({'input':x,'output':y})
        items.append({'id':f'conditional_adjacency_recolor-{seed}-{i:04d}','family':'conditional_adjacency_recolor','train':pairs[:2],'test':pairs[2:]})
    return {'protocol':'authored-adjacency-family/1','family':'conditional_adjacency_recolor','seed':seed,'tasks':items,
        'colors':{'source':2,'target':3,'marker':4},'split':'operator must freeze separate selection/validation before scoring',
        'scope':'separate generated family with guaranteed positive and negative source cases in every pair; no ARC data'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--source',type=int,default=2);p.add_argument('--target',type=int,default=3);p.add_argument('--marker',type=int,default=4)
    p.add_argument('--generate-curriculum');p.add_argument('--tasks',type=int,default=300);p.add_argument('--seed',type=int,default=841);args=p.parse_args()
    if not args.out and not args.generate_curriculum:p.error('output required')
    if args.out:
        out=Path(args.out)
        if out.exists():raise ValueError('preserve artifact')
        graph=build_adjacency_recolor_gene(args.source,args.target,args.marker);digest=graph.save(out)
        receipt={'hash':digest,'stats':graph.stats(),'profile':check_profile(graph,'ladder-micro/1'),'gene_contract':graph.meta['gene_contract'],
                 'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_tasks_read':0,'curriculum_scored':False}
        (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    if args.generate_curriculum:
        path=Path(args.generate_curriculum)
        if path.exists():raise ValueError('preserve generated fixtures')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(generate_adjacency_family(args.tasks,args.seed),indent=2)+'\n')


if __name__=='__main__':main()
