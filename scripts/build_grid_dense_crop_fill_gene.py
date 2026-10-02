# SPDX-License-Identifier: MIT-0
"""Authored symbiotic crop/fill cell with a small numerical density gate."""
import copy
import numpy as np
from build_grid_local_fill_gene import Nodes
from cellforge.ir.graph import Graph


def build_dense_crop_fill_gene(base,density_weights=(1,-1),density_bias=.5,minimum=4):
    g=base.copy();n=Nodes();ports=['input','input_height','input_width']
    def inline(name,prefix):
        definition=base.defs[name]
        ids={node['id']:prefix+node['id'] for node in definition['nodes']}
        def ref(value):
            if value.startswith('$'):return value
            node,sep,port=value.partition('.')
            return ids[node]+(sep+port if sep else '')
        for node in definition['nodes']:
            item=copy.deepcopy(node);item['id']=ids[node['id']]
            item['in']={k:ref(v) for k,v in node.get('in',{}).items()};n.nodes.append(item)
        return {k:ref(v) for k,v in definition['outputs'].items()}
    crop=inline('bbox.geometry','crop_');fill=inline('bank_local_fill.gene','fill_')
    fg=n.add('bbox.foreground',dict(x='$input'))
    count=n.add('op:sum',dict(x=fg),axis=0)
    count=n.add('op:sum',dict(x=count),axis=0);count=n.add('op:reshape',dict(x=count),shape=[1])
    coords=n.add('op:coords',n=30,start=1.,step=1.)
    dims=[]
    for key in ('height','width'):
        values=n.add('op:mul',dict(a=crop[key],b=coords));values=n.add('op:sum',dict(x=values),axis=0)
        dims.append(n.add('op:reshape',dict(x=values),shape=[1]))
    area=n.add('op:mul',dict(a=dims[0],b=dims[1]))
    features=n.add('op:concat',dict(x0=count,x1=area),axis=0)
    for name,w,b in [('density',density_weights,density_bias),('size',[1],.5-minimum)]:
        g.params['dense_crop_fill.'+name+'.W']=np.asarray([w],dtype=float)
        g.params['dense_crop_fill.'+name+'.b']=np.asarray([b],dtype=float)
        g.defs['dense_crop_fill.'+name]=dict(kind='cell',act='identity',params=dict(W1='dense_crop_fill.'+name+'.W',b1='dense_crop_fill.'+name+'.b'))
    density=n.add('dense_crop_fill.density',dict(x=features));density=n.add('op:step',dict(x=density))
    size=n.add('dense_crop_fill.size',dict(x=count));size=n.add('op:step',dict(x=size))
    gate=n.add('op:mul',dict(a=density,b=size));inverse=n.add('op:neg',dict(x=gate));inverse=n.add('op:bias',dict(x=inverse),c=1)
    colors_gate=n.add('op:reshape',dict(x=gate),shape=[1,1,1]);colors_inverse=n.add('op:reshape',dict(x=inverse),shape=[1,1,1])
    outputs={}
    for key in ('colors','height','width'):
        a=n.add('op:mul',dict(a=crop[key],b=colors_gate if key=='colors' else gate))
        b=n.add('op:mul',dict(a=fill[key],b=colors_inverse if key=='colors' else inverse))
        outputs[key]=n.add('op:add',dict(a=a,b=b))
    g.defs['dense_crop_fill.gene']=n.bundle('cluster',ports,outputs)
    g.meta.update(gradients=False,construction='authored dense-rectangle crop else local-fill symbiotic gene/1',
        limitations='fixed zero background and numerical density/size thresholds; can misclassify dense local-fill inputs',
        gene_contract=dict(cluster='dense_crop_fill.gene',inputs=ports,outputs=list(outputs)))
    g.validate();return g
