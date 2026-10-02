# SPDX-License-Identifier: MIT-0
"""Authored numeric neighbor gate for offline weight-repair experiments."""
import numpy as np
from build_grid_adjacency_recolor_gene import build_adjacency_recolor_gene

def build_weight_gate(weights=(1,1,1,1),bias=-0.5):
 g=build_adjacency_recolor_gene();g.params['neighbor_gate.W']=np.asarray([weights],dtype=float);g.params['neighbor_gate.b']=np.asarray([bias],dtype=float)
 g.defs['neighbor_gate']={'kind':'cell','act':'identity','params':{'W1':'neighbor_gate.W','b1':'neighbor_gate.b'}}
 gene=g.defs['adjacency_recolor.gene'];nodes=[n for n in gene['nodes'] if n['use']=='op:sum' and n.get('args',{}).get('axis')==2]
 assert len(nodes)==1;n=nodes[0];n['use']='neighbor_gate';n['in']={'x':n['in']['x']};n.pop('args',None)
 g.meta.update(gradients=False,construction='authored four-neighbor numeric gate/1',weight_scope='four direction weights and scalar threshold bias; no gradients or fitted labels')
 g.validate();return g

def examples(seed):
 rng=np.random.default_rng(seed);tasks=[]
 for mask in range(16):
  pairs=[]
  for j in range(3):
   x=rng.choice([0,1,5,6,7,8,9],size=(5,5));x[2,2]=2
   for bit,(r,c) in enumerate(((1,2),(3,2),(2,1),(2,3))):x[r,c]=4 if mask&(1<<bit) else 0
   y=x.copy()
   if mask&12==12:y[2,2]=3
   pairs.append(dict(input=x.tolist(),output=y.tolist()))
  tasks.append(dict(family='horizontal_two_marker_gate',task_id=str(mask),train=pairs[:2],test=pairs[2:]))
 return tasks
