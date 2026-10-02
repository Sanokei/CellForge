# SPDX-License-Identifier: MIT-0
"""Authored dynamic stacked-plane XOR using literal microcell weights."""
import numpy as np
from build_grid_repeat_gene import build_repeat_gene
from build_grid_local_fill_gene import Nodes

def build_stacked_xor_gene(weights=(1.,1.),bias=-1.,target=3):
    if target not in range(10):raise ValueError('grid color required')
    g=build_repeat_gene(2);n=Nodes()
    g.params['split.foreground.W']=np.asarray([[0.]+[1.]*9]);g.params['split.foreground.b']=np.zeros(1)
    g.params['split.xor.W']=np.asarray([weights]);g.params['split.xor.b']=np.asarray([bias])
    g.params['split.target']=np.eye(10)[target]
    for name in ('foreground','xor'):
        g.defs['split.'+name]=dict(kind='cell',act='identity',params=dict(W1='split.'+name+'.W',b1='split.'+name+'.b'))
    height=n.add('dimension',dict(x='$input_height'));half=n.add('op:scale',dict(x=height),c=.5)
    half=n.add('op:bias',dict(x=half),c=-.5)
    offset=n.add('op:bias',dict(x=half),c=1)
    coords=n.add('op:coords',n=30);rows=n.add('op:reshape',dict(x=coords),shape=[30,1])
    columns=n.add('op:reshape',dict(x=coords),shape=[1,30])
    shifted=n.add('op:add',dict(a=rows,b=offset));distance=n.add('op:sub',dict(a=shifted,b=columns))
    distance=n.add('op:abs',dict(x=distance));projection=n.add('op:neg',dict(x=distance))
    projection=n.add('op:bias',dict(x=projection),c=1);projection=n.add('op:relu',dict(x=projection))
    upper=n.add('split.foreground',dict(x='$input'))
    lower=n.add('op:reshape',dict(x=upper),shape=[30,30]);lower=n.add('op:matmul',dict(a=projection,b=lower))
    lower=n.add('op:reshape',dict(x=lower),shape=[30,30,1])
    both=n.add('op:concat',dict(x0=upper,x1=lower),axis=2)
    active=n.add('split.xor',dict(x=both));active=n.add('op:abs',dict(x=active));active=n.add('op:neg',dict(x=active))
    active=n.add('op:bias',dict(x=active),c=1);active=n.add('op:relu',dict(x=active))
    inverse=n.add('op:neg',dict(x=active));inverse=n.add('op:bias',dict(x=inverse),c=1)
    target_ref=n.add('param:split.target');target_ref=n.add('op:reshape',dict(x=target_ref),shape=[1,1,10])
    colors=n.add('op:mul',dict(a=active,b=target_ref))
    background=n.add('op:coords',n=10);background=n.add('op:neg',dict(x=background))
    background=n.add('op:bias',dict(x=background),c=1);background=n.add('op:relu',dict(x=background))
    background=n.add('op:reshape',dict(x=background),shape=[1,1,10])
    background=n.add('op:mul',dict(a=inverse,b=background));colors=n.add('op:add',dict(a=colors,b=background))
    for size,shape in ((half,[30,1,1]),(n.add('dimension',dict(x='$input_width')),[1,30,1])):
        mask=n.add('op:sub',dict(a=size,b=coords));mask=n.add('op:step',dict(x=mask))
        mask=n.add('op:reshape',dict(x=mask),shape=shape);colors=n.add('op:mul',dict(a=colors,b=mask))
    ones=n.add('op:bias',dict(x=coords),c=1);hot=n.add('op:sub',dict(a=half,b=ones))
    hot=n.add('op:abs',dict(x=hot));hot=n.add('op:neg',dict(x=hot));hot=n.add('op:bias',dict(x=hot),c=1);hot=n.add('op:relu',dict(x=hot))
    g.defs['stacked_xor.gene']=n.bundle('cluster',['input','input_height','input_width'],dict(colors=colors,height=hot,width='$input_width'))
    for definition in g.defs.values():
        if definition.get('role')=='expert':
            for node in definition['nodes']:
                if node['use']=='repeat.gene':node['use']='stacked_xor.gene'
    g.meta.update(gradients=False,construction='authored two stacked foreground planes XOR/1',
        exposure='3428a4f5 train-only counterexample in priority-a-native-diag-v1',
        gene_contract=dict(cluster='stacked_xor.gene',inputs=['input','input_height','input_width'],outputs=['colors','height','width']),
        limitations='odd input height and equal planes separated by one row; zero background, fixed output color; separator not inferred')
    g.validate();return g

def stacked_xor_import_ops(parent,donor,slot,prefix):
    import copy
    from cellforge.construction.transactions import import_def
    name='stacked_xor.gene';scratch=parent.copy()
    import_def(scratch,donor,name,prefix=prefix)
    definition=copy.deepcopy(scratch.defs[prefix+name])
    if 'mode' in parent.defs[f'expanded_slot_{slot}']['inputs']:definition['inputs'].append('mode')
    return [dict(op='import_slot_gene',**{'from':donor.hash()},name=name,prefix=prefix,slot=slot),
        dict(op='put_def',name=prefix+name,**{'def':definition}),
        dict(op='put_def',name=f'expanded_slot_{slot}',**{'def':copy.deepcopy(definition)})]
