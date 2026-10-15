"""New frame-relative corner exchange (the actual selection-task geometry)."""
import random
import numpy as np
import skillcore as S
from wall_skill import score

TEMPLATE,NICHENAME='frame_corner2','framecorner'
SHAPES={'bg':(10,),'frame':(10,),'flip':(2,),'offset':(3,)}
ORACLE_PARAMS={'bg':[1]+[0]*9,'frame':[0]*7+[1,0,0],'flip':[1,1],'offset':[1,0,0]}


def predict(p,grid,demos=None):
    a=np.asarray(grid);bg=int(np.argmax(p['bg']));fc=int(np.argmax(p['frame']));rs,cs=np.where(a==fc)
    if not len(rs):raise ValueError('no frame')
    r0,r1,c0,c1=int(rs.min()),int(rs.max()),int(cs.min()),int(cs.max())
    if not (np.all(a[r0,c0:c1+1]==fc) and np.all(a[r1,c0:c1+1]==fc) and np.all(a[r0:r1+1,c0]==fc) and np.all(a[r0:r1+1,c1]==fc)):
        raise ValueError('not a rectangular frame')
    out=a.copy();offset=int(np.argmax(p['offset']))+1;written=set()
    for r in range(r0+1,r1):
        for c in range(c0+1,c1):
            if a[r,c] in (bg,fc):continue
            bottom=r>(r0+r1)/2;right=c>(c0+c1)/2
            if p['flip'][0]>0:bottom=not bottom
            if p['flip'][1]>0:right=not right
            dest=(r1+offset if bottom else r0-offset,c1+offset if right else c0-offset)
            if dest in written or not (0<=dest[0]<a.shape[0] and 0<=dest[1]<a.shape[1]):raise ValueError('corner collision/bounds')
            written.add(dest);out[r,c]=bg;out[dest]=a[r,c]
    return out.tolist()


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':False,'doc':
    'Find rectangular single-colour frame argmax(frame), background argmax(bg). Interior non-frame/non-bg coloured pixels each identify '
    'a quadrant. Move them to outside frame corners, offset argmax(offset)+1. Positive flip[0] exchanges top/bottom quadrants, '
    'positive flip[1] exchanges left/right. Erase old pixels to bg, preserve frame and other pixels. Collisions/bounds invalid.'}


def schema_rows(source):return [{'id':'frame-%d'%i,'index':i} for i in range(120)]


def generated(row,seed,tier=1):
    rng=random.Random(seed*101+row['index']);h,w=rng.randint(9,14),rng.randint(9,14);a=np.zeros((h,w),int);r0,c0,r1,c1=2,2,h-3,w-3
    a[r0,c0:c1+1]=a[r1,c0:c1+1]=7;a[r0:r1+1,c0]=a[r0:r1+1,c1]=7;out=a.copy()
    for (r,c),(rr,cc),color in zip([(r0+1,c0+1),(r0+1,c1-1),(r1-1,c0+1),(r1-1,c1-1)],
                                 [(r1+1,c1+1),(r1+1,c0-1),(r0-1,c1+1),(r0-1,c0-1)],rng.sample([1,2,3,4,5,6,8,9],4)):
        a[r,c]=color;out[rr,cc]=color
    return {'input':a.tolist(),'output':out.tolist()}


def truth(grid):
    a=np.asarray(grid);out=a.copy();rs,cs=np.where(a==7);r0,r1,c0,c1=rs.min(),rs.max(),cs.min(),cs.max()
    for (r,c),(rr,cc) in zip([(r0+1,c0+1),(r0+1,c1-1),(r1-1,c0+1),(r1-1,c1-1)],
                            [(r1+1,c1+1),(r1+1,c0-1),(r0-1,c1+1),(r0-1,c0-1)]):
        out[rr,cc]=a[r,c];out[r,c]=0
    return out.tolist()
