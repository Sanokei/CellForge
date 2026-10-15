"""Graph-only bounded same-color4-connected largest component and bbox crop.

Semantic inspiration read-only research/claude_arc/cells.py obj_largest;
this prototype uses4 neighbors instead of Claude's default8, and row-major
label ties instead of color-first component ordering. No source is executed.
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


def add_largest_component_cluster(graph,iterations=64,prefix='largest4'):
    if iterations not in (16,64):raise ValueError('predeclared16or64iterations required')
    if prefix+'.gene' in graph.defs:raise ValueError('prefix collision')
    for name,weights in [('color',np.arange(10,dtype=float)[None])]:
        graph.params[prefix+'.'+name+'.W']=weights;graph.params[prefix+'.'+name+'.b']=np.zeros(1)
        graph.defs[prefix+'.'+name]={'kind':'cell','act':'identity','params':{'W1':prefix+'.'+name+'.W','b1':prefix+'.'+name+'.b'}}
    n=Nodes()
    def op(name,**inputs):return n.add('op:'+name,inputs)
    def scale(x,c):return n.add('op:scale',{'x':x},c=c)
    def bias(x,c):return n.add('op:bias',{'x':x},c=c)
    def reshape(x,shape):return n.add('op:reshape',{'x':x},shape=shape)
    def inverse(x):return bias(op('neg',x=x),1)
    def equal(a,b):return op('relu',x=bias(op('neg',x=op('abs',x=op('sub',a=a,b=b))),1))
    histogram=n.add('op:sum',{'x':'$input'},axis=0);histogram=n.add('op:sum',{'x':histogram},axis=0)
    bg=n.add('op:onehot_argmax',{'x':histogram},axis=0)
    bgcode=n.add(prefix+'.color',{'x':bg});code=n.add(prefix+'.color',{'x':'$input'})
    valid=n.add('op:sum',{'x':'$input'},axis=2,keepdims=True)
    foreground=op('mul',a=inverse(equal(code,reshape(bgcode,[1,1,1]))),b=valid)
    labelcoords=n.add('op:coords',n=900,start=1.)
    initial=reshape(labelcoords,[30,30,1])
    labels=op('add',a=op('mul',a=initial,b=foreground),b=scale(inverse(foreground),901))
    agreements=[]
    for axis,k in ((0,-1),(0,1),(1,-1),(1,1)):
        neighbor=n.add('op:shift',{'x':code},axis=axis,k=k)
        neighborfg=n.add('op:shift',{'x':foreground},axis=axis,k=k)
        same=op('mul',a=equal(code,neighbor),b=op('mul',a=foreground,b=neighborfg))
        agreements.append((axis,k,same,inverse(same)))
    def propagate(current):
        candidates=[current]
        for axis,k,same,different in agreements:
            shifted=n.add('op:shift',{'x':current},axis=axis,k=k,value=901.)
            candidates.append(op('add',a=op('mul',a=shifted,b=same),b=scale(different,901)))
        joined=n.add('op:concat',{f'x{i}':v for i,v in enumerate(candidates)},axis=2)
        return n.add('op:min',{'x':joined},axis=2,keepdims=True)
    for _ in range(iterations):labels=propagate(labels)
    check=propagate(labels)
    change=op('abs',x=op('sub',a=labels,b=check))
    change=reshape(change,[900]);change=n.add('op:sum',{'x':change},axis=0,keepdims=True)
    settled=inverse(op('step',x=change))
    membership=equal(reshape(labels,[900,1]),reshape(labelcoords,[1,900]))
    areas=n.add('op:sum',{'x':membership},axis=0)
    winner=n.add('op:onehot_argmax',{'x':areas},axis=0)
    selected=op('matmul',a=membership,b=reshape(winner,[900,1]));selected=reshape(selected,[30,30,1])
    total=n.add('op:sum',{'x':reshape(selected,[900])},axis=0,keepdims=True)
    usable=op('mul',a=settled,b=op('step',x=total))
    cleaned=op('add',a=op('mul',a='$input',b=selected),b=op('mul',a=reshape(bg,[1,1,10]),b=inverse(selected)))
    coords=n.add('op:coords',n=30);positions=bias(coords,1)
    minimums=[];dimensions=[]
    for axis in (1,0):
        totals=n.add('op:sum',{'x':selected},axis=axis)
        active=op('step',x=reshape(totals,[30]))
        low=n.add('op:min',{'x':op('add',a=coords,b=scale(inverse(active),30))},axis=0,keepdims=True)
        high=n.add('op:max',{'x':op('mul',a=coords,b=active)},axis=0,keepdims=True)
        extent=bias(op('sub',a=high,b=low),1)
        minimums.append(low);dimensions.append(equal(positions,extent))
    cropped=cleaned
    for axis,minimum in enumerate(minimums):
        if axis:cropped=n.add('op:transpose',{'x':cropped},perm=[1,0,2])
        target=op('add',a=reshape(coords,[30,1]),b=reshape(minimum,[1,1]))
        matrix=equal(target,reshape(coords,[1,30]))
        cropped=reshape(op('matmul',a=matrix,b=reshape(cropped,[30,300])),[30,30,10])
        if axis:cropped=n.add('op:transpose',{'x':cropped},perm=[1,0,2])
    fallback=inverse(usable)
    colors=op('add',a=op('mul',a=cropped,b=reshape(usable,[1,1,1])),b=op('mul',a='$input',b=reshape(fallback,[1,1,1])))
    h=op('add',a=op('mul',a=dimensions[0],b=usable),b=op('mul',a='$input_height',b=fallback))
    w=op('add',a=op('mul',a=dimensions[1],b=usable),b=op('mul',a='$input_width',b=fallback))
    graph.defs[prefix+'.gene']=n.bundle('cluster',['input','input_height','input_width'],{'colors':colors,'height':h,'width':w})
    return prefix+'.gene'


def build_largest_component_gene(iterations=64):
    source=Path(__file__).resolve().parents[1]/'research/claude_arc/cells.py'
    graph=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':1,'state_dim':0,'profile':'ladder-micro/1',
        'construction':'authored bounded largest same-color4-component gene/1; no gradients/fitting',
        'gene_contract':{'cluster':'largest4.gene','iterations':iterations,'inputs':['input','input_height','input_width'],'outputs':['colors','height','width'],
        'background':'most frequent valid input color, lowest-color tie','component':'same-color4-neighbor; labels row-major1..900',
        'selection':'largest area, lowest row-major label tie; remove others to inferredbackground and crop selected bbox',
        'fallback':'identity if any component labels still change on stepT+1, or no foreground'},
        'semantic_source':{'path':'research/claude_arc/cells.py','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'read_only':True,'difference':'4-connectivity vs source default8; row-major vs source color-first tie'},
        'limitations':'bounded convergence; long/snake components may identityfallback; all900labelmembership tensor, full unrolled graph not tiny solely by parameter count; no demo adaptation'})
    gene=add_largest_component_cluster(graph,iterations)
    ports=['input','input_mask','input_height','input_width','output','output_mask','output_height','output_width','phase','index','state']
    n=Nodes();call=n.add(gene,{p:'$'+p for p in ('input','input_height','input_width')});outputs={'state':'$state'}
    for port,source,shape in [('color_logits','colors',[1,30,30,10]),('height_logits','height',[1,30]),('width_logits','width',[1,30])]:
        outputs[port]=n.add('op:reshape',{'x':call+'.'+source},shape=shape)
    graph.defs['largest_expert0']=n.bundle('expert',ports,outputs);graph.defs['largest_expert1']=copy.deepcopy(graph.defs['largest_expert0'])
    n=Nodes();scores=n.add('op:coords',n=2,start=0.,step=0.)
    graph.defs['largest_router']=n.bundle('router',[],{'scores':scores})
    n=Nodes();router=n.add('largest_router');dispatch=n.add('dispatch',{**{p:'$'+p for p in ports},'scores':router+'.scores'},branches=['largest_expert0','largest_expert1'],k=1)
    graph.defs['largest_root']=n.bundle('root',ports,{p:dispatch+'.'+p for p in outputs});graph.entries={'grid_step':'largest_root'}
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1');return graph


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--iterations',type=int,choices=(16,64),default=64);args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve prototype')
    graph=build_largest_component_gene(args.iterations);digest=graph.save(out)
    def expanded_nodes(name):
        definition=graph.defs[name]
        if definition['kind']=='cell':return 1
        total=len(definition['nodes'])
        for node in definition['nodes']:
            if node['use']=='dispatch':total+=sum(expanded_nodes(branch) for branch in node['args']['branches'])
            elif node['use'] in graph.defs:total+=expanded_nodes(node['use'])
        return total
    receipt={'hash':digest,'stats':graph.stats(),'gene_node_count':len(graph.defs['largest4.gene']['nodes']),
        'expanded_ir_nodes_all_dispatch_branches':expanded_nodes(graph.entries['grid_step']),
        'largest_membership_tensor_elements':810000,
        'profile':check_profile(graph,'ladder-micro/1'),'gene_contract':graph.meta['gene_contract'],'semantic_source':graph.meta['semantic_source'],
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_tasks_read':0,'prototype_only':True}
    (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
