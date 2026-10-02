# SPDX-License-Identifier: MIT-0
"""Authored marker-edge mirror hypothesis from task 2bcee788 train pairs."""
import numpy as np,json,argparse
from build_grid_repeat_gene import build_repeat_gene
from build_grid_associative import Nodes

def build_marker_mirror():
 g=build_repeat_gene(2);g.params['marker.W']=np.eye(10)[2:3];g.params['marker.b']=np.zeros(1);g.params['color.W']=np.arange(10,dtype=float)[None];g.params['color.b']=np.zeros(1)
 for name in ('marker','color'):g.defs[name]={'kind':'cell','act':'identity','params':{'W1':name+'.W','b1':name+'.b'}}
 n=Nodes();marker=n.add('marker',{'x':'$input'});valid=n.add('op:sum',{'x':'$input'},axis=2,keepdims=True);bg=n.add('op:slice',{'x':'$input'},axis=2,start=0,stop=1);obj=n.op('sub',a=n.op('sub',a=valid,b=bg),b=marker)
 def scalar(x):
  for axis in (0,1,2):x=n.add('op:sum',{'x':x},axis=axis,keepdims=True)
  return x
 coords=n.add('op:coords',n=30);pivots=[]
 for axis in (0,1):
  c=n.reshape(coords,[30,1,1] if axis==0 else [1,30,1]);v=n.op('mul',a=c,b=marker)
  for dim in (0,1,2):v=n.add('op:max',{'x':v},axis=dim,keepdims=True)
  pivots.append(v)
 color=n.add('color',{'x':'$input'});objectcolor=n.op('mul',a=color,b=obj)
 for axis in (0,1,2):objectcolor=n.add('op:max',{'x':objectcolor},axis=axis,keepdims=True)
 choices=[];fields=[]
 for i,(axis,direction) in enumerate(((0,-1),(0,1),(1,-1),(1,1))):
  neighbor=n.add('op:shift',{'x':obj},axis=axis,k=-direction);score=scalar(n.op('mul',a=marker,b=neighbor));choices.append(n.add('op:bias',{'x':score},c=-i*.001))
  pivot=n.add('op:bias',{'x':n.add('op:scale',{'x':pivots[axis]},c=2)},c=direction)
  target=n.reshape(coords,[30,1]);source=n.reshape(coords,[1,30]);wanted=n.op('sub',a=n.reshape(pivot,[1,1]),b=target);delta=n.op('abs',x=n.op('sub',a=wanted,b=source));matrix=n.op('relu',x=n.add('op:bias',{'x':n.op('neg',x=delta)},c=1))
  field=obj if axis==0 else n.add('op:transpose',{'x':obj},perm=[1,0,2]);field=n.reshape(n.op('matmul',a=matrix,b=n.reshape(field,[30,30])),[30,30,1])
  if axis==1:field=n.add('op:transpose',{'x':field},perm=[1,0,2])
  fields.append(field)
 maximum=choices[0]
 for s in choices[1:]:maximum=n.op('maximum',a=maximum,b=s)
 reflected=None
 for score,field in zip(choices,fields):
  gate=n.op('step',x=n.add('op:bias',{'x':n.op('sub',a=score,b=maximum)},c=.0005));term=n.op('mul',a=field,b=gate);reflected=term if reflected is None else n.op('add',a=reflected,b=term)
 union=n.op('step',x=n.op('add',a=obj,b=reflected));chosen=n.op('mul',a=union,b=objectcolor);palette=n.reshape(n.add('op:coords',n=10),[1,1,10]);distance=n.op('abs',x=n.op('sub',a=chosen,b=palette));hot=n.op('relu',x=n.add('op:bias',{'x':n.op('neg',x=distance)},c=1));hot=n.op('mul',a=hot,b=valid)
 g.defs['repeat.gene']=n.bundle('cluster',['input','input_height','input_width'],dict(colors=hot,height='$input_height',width='$input_width'));g.meta.update(construction='authored marker-edge mirror hypothesis/1',provenance='2bcee788 public-training demonstrations only',limitations='fixed marker2/background0; one object color; side chosen by marker-adjacent object count; no-marker behavior unsupported',gradients=False);g.validate();return g
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();g=build_marker_mirror();print(json.dumps(dict(hash=g.save(a.out),stats=g.stats())))
