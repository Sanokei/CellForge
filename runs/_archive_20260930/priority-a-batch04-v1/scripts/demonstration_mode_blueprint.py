# SPDX-License-Identifier: MIT-0
"""Explicit IR rewiring for demonstration-conditioned crop/fill selection."""
import copy
from build_grid_local_fill_gene import Nodes
from cellforge.construction.transactions import apply_ops


def controller_with_mode(definition,ported):
    d=copy.deepcopy(definition);n=Nodes()
    query=n.add('op:slice',dict(x='$state'),axis=0,start=100,stop=101)
    query=n.add('op:step',dict(x=query))
    if d['role']=='root':
        changes=[]
        for key in ('height','width'):
            x=n.add('op:sub',dict(a='$input_'+key,b='$output_'+key))
            x=n.add('op:abs',dict(x=x));x=n.add('op:sum',dict(x=x),axis=0)
            changes.append(n.add('op:reshape',dict(x=x),shape=[1]))
        demo=n.add('op:add',dict(a=changes[0],b=changes[1]));demo=n.add('op:step',dict(x=demo))
        inv=n.add('op:neg',dict(x='$phase'));inv=n.add('op:bias',dict(x=inv),c=1)
        demo=n.add('op:mul',dict(a=demo,b=inv));query=n.add('op:mul',dict(a=query,b='$phase'))
        query=n.add('op:add',dict(a=demo,b=query))
    ids={node['id']:'context_'+node['id'] for node in n.nodes}
    for node in n.nodes:
        node['id']=ids[node['id']]
        node['in']={k:ids.get(v,v) for k,v in node.get('in',{}).items()}
    mode=ids[query]
    for node in d['nodes']:
        if node['use'] in ported:node.setdefault('in',{})['mode']=mode
    d['nodes']=n.nodes+d['nodes'];return d


def mode_blueprint_ops(graph,target,weight=1.,bias=-.5):
    """Return literal numerical/cell/wiring edits; caller charges evaluation."""
    if 'context_mode_gate' in graph.defs:raise ValueError('mode blueprint already present')
    record=graph.meta['expanded_genome']
    ported=set(record['available_genes'])|{f'expanded_slot_{i}' for i in record['slots']}
    if target not in ported:raise ValueError('declared compound gene required')
    original=graph.defs[target];d=copy.deepcopy(original)
    # Find the product of the existing density and size decisions exactly.
    by_id={x['id']:x for x in d['nodes']};products=[]
    for node in d['nodes']:
        if node['use']!='op:mul':continue
        sources=[by_id.get(v,{}) for v in node.get('in',{}).values()]
        if len(sources)!=2 or any(x.get('use')!='op:step' for x in sources):continue
        cells=[by_id.get(x.get('in',{}).get('x'),{}).get('use','') for x in sources]
        if any(x.endswith('dense_crop_fill.density') for x in cells) and any(x.endswith('dense_crop_fill.size') for x in cells):products.append(node)
    if len(products)!=1:raise ValueError('unambiguous compound gate required')
    gate=products[0];position=d['nodes'].index(gate)
    d['nodes'].insert(position,dict(id='context_linear',use='context_mode_gate',**{'in':dict(x='$mode')}))
    gate['use']='op:step';gate['in']=dict(x='context_linear')
    ops=[dict(op='set_param',name='context_mode.W',value=[[float(weight)]]),
        dict(op='set_param',name='context_mode.b',value=[float(bias)]),
        dict(op='put_def',name='context_mode_gate',**{'def':dict(kind='cell',act='identity',params=dict(W1='context_mode.W',b1='context_mode.b'))})]
    for name in sorted(ported):
        definition=copy.deepcopy(d if name==target or graph.defs[name]==original else graph.defs[name])
        if 'mode' not in definition['inputs']:definition['inputs'].append('mode')
        ops.append(dict(op='put_def',name=name,**{'def':definition}))
    for name,definition in graph.defs.items():
        if name==graph.entries['grid_step'] or name.startswith('expert_'):
            ops.append(dict(op='put_def',name=name,**{'def':controller_with_mode(definition,ported)}))
    return ops


def materialize_mode_blueprint(graph,target,weight=1.,bias=-.5):
    return apply_ops(graph,mode_blueprint_ops(graph,target,weight,bias))[0]
