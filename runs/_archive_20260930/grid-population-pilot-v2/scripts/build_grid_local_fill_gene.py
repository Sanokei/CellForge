"""Authored one-step four-neighbor fill, entirely in graph cells and wiring.

Nonzero pixels persist. A zero pixel takes its highest-color orthogonal
neighbor from the original input; updates are simultaneous, not flood fill.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph,Limits
from cellforge.ir.profiles import check_profile


class Nodes:
    def __init__(self):self.nodes=[]
    def add(self,use,inputs=None,**args):
        name='n'+str(len(self.nodes));item={'id':name,'use':use}
        if inputs:item['in']=inputs
        if args:item['args']=args
        self.nodes.append(item);return name
    def bundle(self,role,inputs,outputs):return {'kind':'bundle','role':role,'inputs':inputs,'nodes':self.nodes,'outputs':outputs}


def add_local_fill_cluster(graph,prefix='local_fill'):
    """Install reusable typed gene; caller may import_def its flat cluster."""
    if prefix+'.gene' in graph.defs:raise ValueError('gene prefix already present')
    graph.params[prefix+'.color.W']=np.arange(10,dtype=float)[None]
    graph.params[prefix+'.color.b']=np.zeros(1)
    graph.defs[prefix+'.color']={'kind':'cell','act':'identity','params':{'W1':prefix+'.color.W','b1':prefix+'.color.b'}}
    n=Nodes()
    masked=n.add('op:mul',{'a':'$input','b':'$input_mask'})
    color=n.add(prefix+'.color',{'x':masked})
    neighbors=[n.add('op:shift',{'x':color},axis=axis,k=k) for axis,k in ((0,-1),(0,1),(1,-1),(1,1))]
    joined=n.add('op:concat',{f'x{i}':ref for i,ref in enumerate(neighbors)},axis=2)
    highest=n.add('op:max',{'x':joined},axis=2,keepdims=True)
    empty=n.add('op:slice',{'x':masked},axis=2,start=0,stop=1)
    fill=n.add('op:mul',{'a':highest,'b':empty})
    chosen=n.add('op:add',{'a':color,'b':fill})
    colors=n.add('op:coords',n=10)
    colors=n.add('op:reshape',{'x':colors},shape=[1,1,10])
    distance=n.add('op:sub',{'a':chosen,'b':colors})
    distance=n.add('op:abs',{'x':distance})
    negative=n.add('op:neg',{'x':distance})
    threshold=n.add('op:bias',{'x':negative},c=1)
    hot=n.add('op:relu',{'x':threshold})
    hot=n.add('op:mul',{'a':hot,'b':'$input_mask'})
    graph.defs[prefix+'.gene']=n.bundle('cluster',['input','input_mask','input_height','input_width'],
        {'colors':hot,'height':'$input_height','width':'$input_width'})
    return prefix+'.gene'


def build_local_fill_gene():
    graph=Graph(meta={'task':'grid','grid_interface':'grid-graph/1','grid_attempts':1,'state_dim':0,
        'profile':'ladder-micro/1','construction':'authored local_fill gene/1; no fitted data',
        'gene_contract':{'cluster':'local_fill.gene','inputs':['input','input_mask','input_height','input_width'],
                         'outputs':['colors','height','width'],'rule':'one simultaneous orthogonal dilation into zero; highest neighbor color wins; original nonzero persists'},
        'limitations':'same dimensions, fixed color priority and one step; no demonstration adaptation; duplicate expert wrappers solely for profile compatibility'})
    name=add_local_fill_cluster(graph)
    ports=['input','input_mask','input_height','input_width','output','output_mask','output_height','output_width','phase','index','state']
    n=Nodes();gene=n.add(name,{p:'$'+p for p in ('input','input_mask','input_height','input_width')})
    outputs={'state':'$state'}
    for port,source,shape in [('color_logits','colors',[1,30,30,10]),('height_logits','height',[1,30]),('width_logits','width',[1,30])]:
        outputs[port]=n.add('op:reshape',{'x':gene+'.'+source},shape=shape)
    graph.defs['fill_expert0']=n.bundle('expert',ports,outputs)
    graph.defs['fill_expert1']=copy.deepcopy(graph.defs['fill_expert0'])
    n=Nodes();score=n.add('op:coords',n=2,start=0.,step=0.)
    graph.defs['fill_router']=n.bundle('router',[],{'scores':score})
    n=Nodes();router=n.add('fill_router');dispatch=n.add('dispatch',{**{p:'$'+p for p in ports},'scores':router+'.scores'},branches=['fill_expert0','fill_expert1'],k=1)
    graph.defs['fill_root']=n.bundle('root',ports,{p:dispatch+'.'+p for p in outputs})
    graph.entries={'grid_step':'fill_root'}
    graph.validate(Limits(strict_microcells=True));check_profile(graph,'ladder-micro/1')
    return graph


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('preserve frozen gene artifact')
    graph=build_local_fill_gene();digest=graph.save(out)
    receipt={'hash':digest,'stats':graph.stats(),'profile':check_profile(graph,'ladder-micro/1'),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_tasks_read':0,
        'curriculum_scored':False,'gene_contract':graph.meta['gene_contract']}
    (out/'MODEL_CARD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
