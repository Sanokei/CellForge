"""Translation of the explicitly authored component-count numerical contract."""
import numpy as np
SHAPES={'target.W':(1,10),'component.W':(1,4),'component.b':(1,),
        'readout.W':(1,1),'readout.b':(1,5),'palette':(2,)}
def predict_component_count(params,grid):
    a=np.asarray(grid)
    if a.ndim!=2 or not a.size or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>9):
        raise ValueError('rectangular integer palette0..9 required')
    if set(params)!=set(SHAPES) or any(np.asarray(params[k]).shape!=shape or not np.isfinite(params[k]).all() for k,shape in SHAPES.items()):
        raise ValueError('complete finite literal component-count tensors required')
    palette=np.asarray(params['palette'])
    if not np.isin(palette,np.arange(10)).all():raise ValueError('palette entries must be integer colors')
    seen=set();selected=0
    for r in range(a.shape[0]):
      for c in range(a.shape[1]):
        if not a[r,c] or (r,c) in seen:continue
        color=int(a[r,c]);todo=[(r,c)];seen.add((r,c));points=[]
        while todo:
          x,y=todo.pop();points.append((x,y))
          for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0<=xx<a.shape[0] and 0<=yy<a.shape[1] and (xx,yy) not in seen and a[xx,yy]==color:
              seen.add((xx,yy));todo.append((xx,yy))
        rr=[x for x,y in points];cc=[y for x,y in points]
        features=np.array([params['target.W'][0][color],max(rr)-min(rr)+1==2,
                           max(cc)-min(cc)+1==2,len(points)==4],float)
        selected+=bool(np.asarray(params['component.W'])[0]@features+params['component.b'][0]>0)
    logits=params['readout.W'][0][0]*selected+np.asarray(params['readout.b'])[0]
    return np.where(logits>0,int(palette[1]),int(palette[0]))[None,:].tolist()

