"""Append an authored foreground bounding-box crop gene to the grid pool.

Bounding coordinates, dynamic selection matrices, colors and dimensions are
computed by small cells plus existing generic IR ops. No Python grid solving.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
import numpy as np

spec=importlib.util.spec_from_file_location('gene_pool_v1',Path(__file__).with_name('build_grid_gene_pool.py'))
pool=importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)


class Nodes:
    def __init__(self):self.nodes=[]
    def add(self,use,inputs=None,**args):
        name='n'+str(len(self.nodes));node={'id':name,'use':use,'in':inputs or {}}
        if args:node['args']=args
        self.nodes.append(node);return name
    def op(self,use,**inputs):return self.add('op:'+use,inputs)
    def reshape(self,ref,shape):return self.add('op:reshape',{'x':ref},shape=shape)
    def slice(self,ref,start,stop):return self.add('op:slice',{'x':ref},axis=0,start=start,stop=stop)
    def sum(self,ref):return self.add('op:sum',{'x':ref},axis=0,keepdims=True)
    def bundle(self,role,inputs,outputs):return {'kind':'bundle','role':role,'inputs':inputs,'nodes':self.nodes,'outputs':outputs}


def bbox_cluster(graph, background='zero'):
    if background not in ('zero','modal'):raise ValueError('declared background rule required')
    graph.params['bbox.foreground.W']=np.array([[0]+[1]*9],dtype=float)
    graph.params['bbox.foreground.b']=np.zeros(1)
    graph.defs['bbox.foreground']={'kind':'cell','act':'identity','params':{'W1':'bbox.foreground.W','b1':'bbox.foreground.b'}}
    n=Nodes()
    if background=='zero':
        foreground=n.add('bbox.foreground',{'x':'$input'})
    else:
        counts=n.add('op:sum',{'x':'$input'},axis=0)
        counts=n.add('op:sum',{'x':counts},axis=0)
        # Integer pixel counts: a tiny fixed index bias resolves ties toward
        # the lowest color without changing unequal-count ordering.
        bias=n.add('op:coords',n=10,start=0.,step=-0.001)
        counts=n.op('add',a=counts,b=bias)
        maximum=n.add('op:max',{'x':counts},axis=0,keepdims=True)
        chosen=n.op('step',x=n.add('op:bias',{'x':n.op('sub',a=counts,b=maximum)},c=0.0005))
        bg=n.add('op:sum',{'x':n.op('mul',a='$input',b=chosen)},axis=2,keepdims=True)
        valid=n.add('op:sum',{'x':'$input'},axis=2,keepdims=True)
        foreground=n.op('sub',a=valid,b=bg)
    coords=n.add('op:coords',n=30);coordinates=n.add('op:bias',{'x':coords},c=1)
    minimums=[];dimensions=[]
    for axis,dimension in ((1,'$input_height'),(0,'$input_width')):
        total=n.add('op:sum',{'x':foreground},axis=axis)
        active=n.op('step',x=n.reshape(total,[30]))
        gate=n.op('step',x=n.sum(active))
        inverse=n.add('op:bias',{'x':n.op('neg',x=active)},c=1)
        sentinel=n.add('op:scale',{'x':inverse},c=30)
        lower=n.add('op:min',{'x':n.op('add',a=coords,b=sentinel)},axis=0,keepdims=True)
        lower=n.op('mul',a=lower,b=gate)
        upper=n.add('op:max',{'x':n.op('mul',a=coords,b=active)},axis=0,keepdims=True)
        extent=n.add('op:bias',{'x':n.op('sub',a=upper,b=lower)},c=1)
        original=n.add('dimension',{'x':dimension})
        reset=n.add('op:bias',{'x':n.op('neg',x=gate)},c=1)
        size=n.op('add',a=n.op('mul',a=extent,b=gate),b=n.op('mul',a=original,b=reset))
        distance=n.op('abs',x=n.op('sub',a=coordinates,b=size))
        hot=n.op('relu',x=n.add('op:bias',{'x':n.op('neg',x=distance)},c=1))
        minimums.append(lower);dimensions.append(hot)
    x='$input'
    for axis,minimum in enumerate(minimums):
        if axis:x=n.add('op:transpose',{'x':x},perm=[1,0,2])
        target=n.op('add',a=n.reshape(coords,[30,1]),b=n.reshape(minimum,[1,1]))
        source=n.reshape(coords,[1,30])
        difference=n.op('abs',x=n.op('sub',a=target,b=source))
        matrix=n.op('relu',x=n.add('op:bias',{'x':n.op('neg',x=difference)},c=1))
        x=n.reshape(n.op('matmul',a=matrix,b=n.reshape(x,[30,300])),[30,30,10])
        if axis:x=n.add('op:transpose',{'x':x},perm=[1,0,2])
    graph.defs['bbox.geometry']=n.bundle('cluster',['input','input_height','input_width'],{'colors':x,'height':dimensions[0],'width':dimensions[1]})


def append_bbox_gene(parent):
    g=parent.copy();bbox_cluster(g);count=9
    g.defs['mapping_8']=copy.deepcopy(g.defs['mapping_0'])
    first=g.defs['mapping_8']['nodes'][0];first['args'].update(start=808,stop=908)
    g.defs['expert_8']=copy.deepcopy(g.defs['expert_0'])
    for node in g.defs['expert_8']['nodes']:
        if node['use']=='geometry_0':node['use']='bbox.geometry'
        if node['use']=='mapping_0':node['use']='mapping_8'
    n=Nodes();scores=[]
    for i in range(count):
        state=n.slice('$state',i*101,(i+1)*101);counts=n.reshape(n.slice(state,0,100),[10,10])
        total=n.sum(n.add('op:sum',{'x':counts},axis=1));majority=n.sum(n.add('op:max',{'x':counts},axis=1))
        conflicts=n.op('sub',a=total,b=majority);shape_error=n.add('op:scale',{'x':n.slice(state,100,101)},c=9001)
        scores.append(n.op('neg',x=n.op('add',a=conflicts,b=shape_error)))
    joined=n.add('op:concat',{'x'+str(i):ref for i,ref in enumerate(scores)},axis=0)
    g.defs['router']=n.bundle('router',['state'],{'scores':joined})
    n=Nodes();updates=[]
    for i in range(count):
        name='bbox.geometry' if i==8 else 'geometry_'+str(i)
        geometry=n.add(name,{'input':'$input','input_height':'$input_height','input_width':'$input_width'})
        updates.append(n.add('association_update',{'colors':geometry+'.colors','height':geometry+'.height','width':geometry+'.width',
            'output':'$output','output_height':'$output_height','output_width':'$output_width','phase':'$phase','state':n.slice('$state',i*101,(i+1)*101)}))
    state=n.add('op:concat',{'x'+str(i):ref for i,ref in enumerate(updates)},axis=0)
    router=n.add('router',{'state':state})
    dispatch=n.add('dispatch',{'scores':router+'.scores','state':state,'input':'$input','input_height':'$input_height','input_width':'$input_width'},branches=['expert_'+str(i) for i in range(count)])
    g.defs['root']=n.bundle('root',['input','input_height','input_width','output','output_height','output_width','phase','state'],
                           {'state':state,**{p:dispatch+'.'+p for p in ('height_logits','width_logits','color_logits')}})
    g.meta.update({'state_dim':909,'construction':'associative pool plus graph bbox crop gene/1','new_gene':{'index':8,'name':'bbox.geometry',
                   'parent_hash':parent.hash(),'background':0,'empty_policy':'identity','scope':'foreground bounding box only; no object identity/segmentation'}})
    g.validate();check_profile(g,'ladder-micro/1');return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--single',action='store_true');args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('preserve previous gene artifacts')
    repo=Path(__file__).resolve().parents[1];g=append_bbox_gene(Graph.load(repo/'runs/grid-associative/artifact'))
    g=pool.fixed_single_gene(g,8) if args.single else pool.ranked_two(g)
    digest=g.save(out);print(json.dumps({'hash':digest,'stats':g.stats(),'gene':g.meta.get('new_gene')}))


if __name__=='__main__':main()
