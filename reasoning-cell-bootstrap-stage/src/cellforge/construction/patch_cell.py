"""Literal authored structural repair of an existing convolution cell."""
import numpy as np
PATCH_SHAPES={'patch.zero.W':(1,10),'patch.detect.b':(1,),'patch.paint':(10,),
              'patch.emit.W':(3,3),'patch.emit.b':(1,),'patch.guard.density':(1,)}
def predict_patch(params,grid,fallback):
    a=np.asarray(grid)
    if a.ndim!=2 or not a.size or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>9):raise ValueError('typed ten-color grid required')
    for key,shape in PATCH_SHAPES.items():
        v=np.asarray(params[key],float)
        if v.shape!=shape or not np.isfinite(v).all() or np.max(np.abs(v))>9:raise ValueError('invalid authored tensor')
    density=float(params['patch.guard.density'][0])
    if not 0<=density<=1:raise ValueError('density parameter outside typed range')
    colors=set(a[a!=0].tolist())
    if not (np.any(a==0) and len(colors)==1 and 1 not in colors and np.count_nonzero(a)/a.size>density):
        return fallback({k:v for k,v in params.items() if k not in PATCH_SHAPES},grid)
    h,w=a.shape;anchors=np.zeros_like(a,dtype=float)
    zero=np.asarray(params['patch.zero.W'])[0]
    for r in range(1,h-1):
        for c in range(1,w-1):
            anchors[r,c]=float(np.sum(zero[a[r-1:r+2,c-1:c+2]])+params['patch.detect.b'][0]>0)
    out=a.copy();emit=np.asarray(params['patch.emit.W']);paint=int(np.argmax(params['patch.paint']))
    for r in range(h):
        for c in range(w):
            score=params['patch.emit.b'][0]
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr,cc=r-dr,c-dc
                    if 0<=rr<h and 0<=cc<w:score+=emit[dr+1,dc+1]*anchors[rr,cc]
            if score>0:out[r,c]=paint
    return out.tolist()

