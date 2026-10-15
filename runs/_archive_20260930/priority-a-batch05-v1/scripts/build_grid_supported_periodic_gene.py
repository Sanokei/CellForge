# SPDX-License-Identifier: MIT-0
"""Literal recurrence-support weights for per-line marker completion."""
import copy
import numpy as np
from build_grid_periodic_marker_gene import build_periodic_marker_gene
from build_grid_local_fill_gene import Nodes
from cellforge.construction.transactions import import_def

def build_supported_periodic_gene(period=3,axis=2,minimum=2,weights=None):
    if axis not in (0,1,2) or minimum not in (1,2,3,4):raise ValueError('bounded line/support vocabulary required')
    g=build_periodic_marker_gene(period);n=Nodes()
    g.params['supported.count.W']=np.eye(2) if weights is None else np.asarray(weights,dtype=float)
    g.params['supported.count.b']=np.full(2,.5-minimum)
    if g.params['supported.count.W'].shape!=(2,2):raise ValueError('two-marker count gate required')
    g.defs['supported.count']=dict(kind='cell',act='identity',params=dict(W1='supported.count.W',b1='supported.count.b'))
    source=n.add('adjacency_recolor.source',dict(x='$input'))
    m3=n.add('periodic.marker3',dict(x='$input'));m7=n.add('adjacency_recolor.marker',dict(x='$input'))
    markers=n.add('op:concat',dict(x0=m3,x1=m7),axis=2);supports=[]
    for line_axis in ((0,1) if axis==2 else (axis,)):
        counts=markers
        for distance in range(period,30,period):
            for sign in (-1,1):
                shifted=n.add('op:shift',dict(x=markers),axis=line_axis,k=sign*distance)
                counts=n.add('op:add',dict(a=counts,b=shifted))
        support=n.add('supported.count',dict(x=counts));supports.append(n.add('op:step',dict(x=support)))
    support=supports[0] if len(supports)==1 else n.add('op:maximum',dict(a=supports[0],b=supports[1]))
    seen3=n.add('op:slice',dict(x=support),axis=2,start=0,stop=1)
    seen7=n.add('op:slice',dict(x=support),axis=2,start=1,stop=2)
    inverse7=n.add('op:neg',dict(x=seen7));inverse7=n.add('op:bias',dict(x=inverse7),c=1)
    gate3=n.add('op:mul',dict(a=seen3,b=inverse7));gate3=n.add('op:mul',dict(a=gate3,b=source))
    gate7=n.add('op:mul',dict(a=seen7,b=source))
    active=n.add('op:add',dict(a=gate3,b=gate7));inverse=n.add('op:neg',dict(x=active));inverse=n.add('op:bias',dict(x=inverse),c=1)
    colors=n.add('op:mul',dict(a='$input',b=inverse))
    for gate,param in ((gate3,'adjacency_recolor.target'),(gate7,'periodic.target7')):
        target=n.add('param:'+param);target=n.add('op:reshape',dict(x=target),shape=[1,1,10])
        replacement=n.add('op:mul',dict(a=target,b=gate));colors=n.add('op:add',dict(a=colors,b=replacement))
    definition=n.bundle('cluster',['input','input_height','input_width'],dict(colors=colors,height='$input_height',width='$input_width'))
    g.defs['supported_periodic.gene']=definition
    g.meta.update(construction='authored same-residue line recurrence support/1',
        exposure='cbded52d train-only counterexample in priority-a-native-diag-v1',
        gene_contract=dict(cluster='supported_periodic.gene',inputs=['input','input_height','input_width'],outputs=['colors','height','width']),
        limitations='fixed colors1/3/7 and period; repeated evidence required per residue/line; marker7 wins conflicts')
    g.validate();return g

def build_joint_supported_periodic_gene(period=3,axis=2,minimum=2,weights=None,selector_weight=1.,selector_bias=-.5):
    """Supported line completion when evidenced, global sparse-template fallback."""
    g=build_supported_periodic_gene(period,axis,minimum,weights)
    n=Nodes();n.nodes=copy.deepcopy(g.defs['supported_periodic.gene']['nodes'])
    outputs=g.defs['supported_periodic.gene']['outputs']
    support=next(node['in']['x'] for node in n.nodes if node['use']=='op:slice' and node['args'].get('axis')==2 and node['args'].get('start')==0)
    total=support
    for dimension in (2,0,0):total=n.add('op:sum',dict(x=total),axis=dimension)
    total=n.add('op:reshape',dict(x=total),shape=[1])
    g.params['supported.selector.W']=np.asarray([[selector_weight]],dtype=float)
    g.params['supported.selector.b']=np.asarray([selector_bias],dtype=float)
    g.defs['supported.selector']=dict(kind='cell',act='identity',params=dict(W1='supported.selector.W',b1='supported.selector.b'))
    gate=n.add('supported.selector',dict(x=total));gate=n.add('op:step',dict(x=gate))
    inverse=n.add('op:neg',dict(x=gate));inverse=n.add('op:bias',dict(x=inverse),c=1)
    gate=n.add('op:reshape',dict(x=gate),shape=[1,1,1]);inverse=n.add('op:reshape',dict(x=inverse),shape=[1,1,1])
    original=g.defs['periodic.gene'];ids={node['id']:'global_'+node['id'] for node in original['nodes']}
    def ref(v):
        if v.startswith('$'):return v
        name,separator,port=v.partition('.');return ids[name]+(separator+port if separator else '')
    for node in original['nodes']:
        item=copy.deepcopy(node);item['id']=ids[node['id']]
        item['in']={key:ref(value) for key,value in item.get('in',{}).items()};n.nodes.append(item)
    selected=n.add('op:mul',dict(a=outputs['colors'],b=gate))
    fallback=n.add('op:mul',dict(a=ref(original['outputs']['colors']),b=inverse))
    colors=n.add('op:add',dict(a=selected,b=fallback))
    g.defs['joint_supported_periodic.gene']=n.bundle('cluster',['input','input_height','input_width'],
        dict(colors=colors,height='$input_height',width='$input_width'))
    g.meta['gene_contract']['cluster']='joint_supported_periodic.gene'
    g.meta['construction']='authored recurrence-supported completion with sparse-template fallback/1'
    g.validate();return g

def supported_periodic_import_ops(parent,donor,slot,prefix):
    """Import the public three-port donor and explicitly lift its internal mode port."""
    name='joint_supported_periodic.gene';scratch=parent.copy()
    import_def(scratch,donor,name,prefix=prefix)
    definition=copy.deepcopy(scratch.defs[prefix+name])
    if 'mode' in parent.defs[f'expanded_slot_{slot}']['inputs']:definition['inputs'].append('mode')
    return [dict(op='import_slot_gene',**{'from':donor.hash()},name=name,prefix=prefix,slot=slot),
        dict(op='put_def',name=prefix+name,**{'def':definition}),
        dict(op='put_def',name=f'expanded_slot_{slot}',**{'def':copy.deepcopy(definition)})]
