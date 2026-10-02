# SPDX-License-Identifier: MIT-0
"""Authored period-three marker completion from exposed train diagnostics."""
import numpy as np
from build_grid_adjacency_recolor_gene import build_adjacency_recolor_gene
from build_grid_local_fill_gene import Nodes


def build_periodic_marker_gene(period=3):
    if type(period) is not int or not 1 <= period <= 10:
        raise ValueError('bounded positive period required')
    g=build_adjacency_recolor_gene(source=1,target=3,marker=7)
    g.params['periodic.marker3.W']=np.eye(10)[3:4]
    g.params['periodic.marker3.b']=np.zeros(1)
    g.params['periodic.target7']=np.eye(10)[7]
    g.defs['periodic.marker3']=dict(kind='cell',act='identity',params=dict(W1='periodic.marker3.W',b1='periodic.marker3.b'))
    n=Nodes();source=n.add('adjacency_recolor.source',dict(x='$input'))
    m3=n.add('periodic.marker3',dict(x='$input'));m7=n.add('adjacency_recolor.marker',dict(x='$input'))
    markers=n.add('op:concat',dict(x0=m3,x1=m7),axis=2)
    # Binary dilation reaches every same-residue location on a <=30 grid.
    # Each stage reads the same previous tensor in both directions.
    for axis in (0,1):
        distance=period
        while distance < 30:
            previous=markers
            left=n.add('op:shift',dict(x=previous),axis=axis,k=-distance)
            right=n.add('op:shift',dict(x=previous),axis=axis,k=distance)
            markers=n.add('op:maximum',dict(a=previous,b=left))
            markers=n.add('op:maximum',dict(a=markers,b=right))
            distance*=2
    seen3=n.add('op:slice',dict(x=markers),axis=2,start=0,stop=1)
    seen7=n.add('op:slice',dict(x=markers),axis=2,start=1,stop=2)
    inverse7=n.add('op:neg',dict(x=seen7));inverse7=n.add('op:bias',dict(x=inverse7),c=1)
    gate3=n.add('op:mul',dict(a=seen3,b=inverse7));gate3=n.add('op:mul',dict(a=gate3,b=source))
    gate7=n.add('op:mul',dict(a=seen7,b=source))
    active=n.add('op:add',dict(a=gate3,b=gate7));inverse=n.add('op:neg',dict(x=active));inverse=n.add('op:bias',dict(x=inverse),c=1)
    colors=n.add('op:mul',dict(a='$input',b=inverse))
    for gate,param in [(gate3,'adjacency_recolor.target'),(gate7,'periodic.target7')]:
        target=n.add('param:'+param);target=n.add('op:reshape',dict(x=target),shape=[1,1,10])
        replacement=n.add('op:mul',dict(a=target,b=gate));colors=n.add('op:add',dict(a=colors,b=replacement))
    definition=n.bundle('cluster',['input','input_height','input_width'],dict(colors=colors,height='$input_height',width='$input_width'))
    g.defs['adjacency_recolor.gene']=definition
    g.defs['periodic.gene']=definition.copy()
    g.meta.update(gradients=False,construction='authored period-three marker completion/1',
        exposure='cbded52d saved training-only failure in arc-maker-failures-v3; query labels not read',
        gene_contract=dict(cluster='periodic.gene',inputs=['input','input_height','input_width'],outputs=['colors','height','width']),
        limitations='fixed colors1/3/7 and fixed period; marker7 wins conflicting same-phase evidence; no general pattern inference')
    g.validate();return g


def examples(seed,count=30):
    rng=np.random.default_rng(seed);tasks=[]
    for i in range(count):
        pairs=[]
        for _ in range(3):
            nr,nc=map(int,rng.integers(2,11,size=2));h,w=nr*3-1,nc*3-1
            template=np.ones((2,2),dtype=int);positions=rng.permutation(4)[:2]
            for color,pos in zip((3,7),positions):template[pos//2,pos%2]=color
            y=np.zeros((h,w),dtype=int)
            for r in range(nr):
                for c in range(nc):y[r*3:r*3+2,c*3:c*3+2]=template
            x=y.copy();x[(x==3)|(x==7)]=1
            for color,pos in zip((3,7),positions):
                r,c=int(rng.integers(nr)),int(rng.integers(nc));x[r*3+pos//2,c*3+pos%2]=color
            pairs.append(dict(input=x.tolist(),output=y.tolist()))
        tasks.append(dict(train=pairs[:2],test=pairs[2:]))
    return tasks
