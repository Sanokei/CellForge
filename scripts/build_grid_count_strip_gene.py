"""Authored nonbackground-count to horizontal strip gene; no fitted data.

Background is the most frequent valid input color (lowest color wins ties).
Count all remaining pixels; emit1x min(count,30) of dominant foreground color.
Empty foreground emits1x1 background. Graph performs every inference choice.
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


def add_count_strip_cluster(graph,prefix='count_strip'):
    if prefix+'.gene' in graph.defs:raise ValueError('prefix already exists')
    graph.params[prefix+'.count.W']=np.ones((1,10))
    graph.params[prefix+'.count.b']=np.zeros(1)
    graph.defs[prefix+'.count']={'kind':'cell','act':'identity','params':{'W1':prefix+'.count.W','b1':prefix+'.count.b'}}
    n=Nodes()
    histogram=n.add('op:sum',{'x':'$input'},axis=0)
    histogram=n.add('op:sum',{'x':histogram},axis=0)
    background=n.add('op:onehot_argmax',{'x':histogram},axis=0)
    inverse=n.add('op:neg',{'x':background});inverse=n.add('op:bias',{'x':inverse},c=1)
    foreground=n.add('op:mul',{'a':histogram,'b':inverse})
    count=n.add(prefix+'.count',{'x':foreground})
    nonempty=n.add('op:step',{'x':count})
    empty=n.add('op:neg',{'x':nonempty});empty=n.add('op:bias',{'x':empty},c=1)
    dominant=n.add('op:onehot_argmax',{'x':foreground},axis=0)
    dominant=n.add('op:mul',{'a':dominant,'b':nonempty})
    fallback=n.add('op:mul',{'a':background,'b':empty})
    chosen=n.add('op:add',{'a':dominant,'b':fallback})
    excess=n.add('op:bias',{'x':count},c=-30);excess=n.add('op:relu',{'x':excess})
    width=n.add('op:sub',{'a':count,'b':excess});width=n.add('op:add',{'a':width,'b':empty})
    coords=n.add('op:coords',n=30)
    positions=n.add('op:bias',{'x':coords},c=1)
    distance=n.add('op:sub',{'a':positions,'b':width});distance=n.add('op:abs',{'x':distance})
    distance=n.add('op:neg',{'x':distance});distance=n.add('op:bias',{'x':distance},c=1)
    width_hot=n.add('op:relu',{'x':distance})
    row=n.add('op:neg',{'x':coords});row=n.add('op:bias',{'x':row},c=1);row=n.add('op:relu',{'x':row})
    columns=n.add('op:sub',{'a':width,'b':coords});columns=n.add('op:bias',{'x':columns},c=-.5)
    columns=n.add('op:step',{'x':columns})
    row3=n.add('op:reshape',{'x':row},shape=[30,1,1])
    col3=n.add('op:reshape',{'x':columns},shape=[1,30,1])
    color3=n.add('op:reshape',{'x':chosen},shape=[1,1,10])
    mask=n.add('op:mul',{'a':row3,'b':col3})
    colors=n.add('op:mul',{'a':mask,'b':color3})
    graph.defs[prefix+'.gene']=n.bundle('cluster',['input','input_height','input_width'],{'colors':colors,'height':row,'width':width_hot})
    return prefix+'.gene'


def build_count_strip_gene():
    graph=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':1,'state_dim':0,'profile':'ladder-micro/1',
        'construction':'authored count_nonbg_to_strip gene/1; no gradients/fitted data',
        'gene_contract':{'cluster':'count_strip.gene','inputs':['input','input_height','input_width'],'outputs':['colors','height','width'],
        'rule':'most-frequent valid color is background; lowest-index tie; count all nonbackground pixels; dominant foreground color lowest-index tie; horizontal strip cap30; empty1x1background'},
        'limitations':'fixed counting rule, no object counting, no demonstration adaptation; duplicate profile expert wrappers are identical'})
    gene=add_count_strip_cluster(graph)
    ports=['input','input_mask','input_height','input_width','output','output_mask','output_height','output_width','phase','index','state']
    n=Nodes();call=n.add(gene,{p:'$'+p for p in ('input','input_height','input_width')})
    outputs={'state':'$state'}
    for port,source,shape in [('color_logits','colors',[1,30,30,10]),('height_logits','height',[1,30]),('width_logits','width',[1,30])]:
        outputs[port]=n.add('op:reshape',{'x':call+'.'+source},shape=shape)
    graph.defs['count_expert0']=n.bundle('expert',ports,outputs);graph.defs['count_expert1']=copy.deepcopy(graph.defs['count_expert0'])
    n=Nodes();scores=n.add('op:coords',n=2,start=0.,step=0.)
    graph.defs['count_router']=n.bundle('router',[],{'scores':scores})
    n=Nodes();router=n.add('count_router');dispatch=n.add('dispatch',{**{p:'$'+p for p in ports},'scores':router+'.scores'},branches=['count_expert0','count_expert1'],k=1)
    graph.defs['count_root']=n.bundle('root',ports,{p:dispatch+'.'+p for p in outputs});graph.entries={'grid_step':'count_root'}
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1');return graph


def generate_count_family(tasks=300,seed=731):
    """Author synthetic tasks, not model inference; ground truth is count N.

    Each task has two demonstrations plus a novel query count. Background
    strictly outnumbers foreground; dimensions3..9, N1..20, random placements.
    This new family must receive its own operator-side split before scoring.
    """
    if tasks<1:raise ValueError('positive tasks required')
    rng=np.random.default_rng(seed);items=[]
    for i in range(tasks):
        bg,fg=map(int,rng.choice(10,size=2,replace=False))
        counts=list(map(int,rng.choice(np.arange(1,21),size=3,replace=False)))
        pairs=[]
        for count in counts:
            dimensions=[(h,w) for h in range(3,10) for w in range(3,10) if h*w>2*count]
            h,w=dimensions[int(rng.integers(len(dimensions)))];flat=[bg]*(h*w)
            for position in rng.choice(h*w,size=count,replace=False):flat[int(position)]=fg
            pairs.append({'input':[flat[r*w:(r+1)*w] for r in range(h)],'output':[[fg]*count]})
        items.append({'id':f'count_nonbg_to_strip-{seed}-{i:04d}','family':'count_nonbg_to_strip','train':pairs[:2],'test':pairs[2:]})
    return {'protocol':'authored-count-family/1','family':'count_nonbg_to_strip','seed':seed,'tasks':items,
            'split':'operator must freeze selection/validation split before evaluation; no private ARC data',
            'construction':'N sampled independently, majority background, all foreground same color, query count differs from both demonstrations'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--generate-curriculum');p.add_argument('--tasks',type=int,default=300);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args()
    if not args.out and not args.generate_curriculum:p.error('an output required')
    if args.out:
        out=Path(args.out)
        if out.exists():raise ValueError('preserve artifact')
        graph=build_count_strip_gene();digest=graph.save(out)
        receipt={'hash':digest,'stats':graph.stats(),'profile':check_profile(graph,'ladder-micro/1'),'gene_contract':graph.meta['gene_contract'],
                 'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_tasks_read':0,'curriculum_scored':False}
        (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    if args.generate_curriculum:
        path=Path(args.generate_curriculum)
        if path.exists():raise ValueError('preserve family fixture')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(generate_count_family(args.tasks,args.seed),indent=2)+'\n')


if __name__=='__main__':main()
