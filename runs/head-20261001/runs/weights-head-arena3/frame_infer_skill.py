"""New structurally identified frame colours. frame_corner2 remains frozen."""
import random
import numpy as np
import skillcore as S
import frame_skill as F

TEMPLATE,NICHENAME='frame_corner3','frameinfer'
SHAPES=dict(F.SHAPES,identify=(2,))
ORACLE_PARAMS=dict(F.ORACLE_PARAMS,frame=[1]*10,identify=[1,0])
schema_rows=F.schema_rows
score=F.score


def predict(p,grid,demos=None):
    if int(np.argmax(p['identify']))==1:return F.predict(p,grid,demos)
    a=np.asarray(grid);bg=int(np.argmax(p['bg']));choices=[]
    for color in np.unique(a):
        if color==bg or p['frame'][color]<=0:continue
        rs,cs=np.where(a==color);r0,r1,c0,c1=rs.min(),rs.max(),cs.min(),cs.max()
        if r1-r0<2 or c1-c0<2:continue
        if np.all(a[r0,c0:c1+1]==color) and np.all(a[r1,c0:c1+1]==color) and np.all(a[r0:r1+1,c0]==color) and np.all(a[r0:r1+1,c1]==color):choices.append(int(color))
    if len(choices)!=1:raise ValueError('one permitted structural frame required')
    params=dict(p,frame=[int(i==choices[0]) for i in range(10)])
    return F.predict(params,grid,demos)


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':False,'doc':
    'identify argmax1 uses named frame=argmax(frame); identify0 uses new structural identification: frame[10]>0 permits frame colours. Require exactly one permitted non-bg colour forming a complete rectangular '
    'border (height,width>=3). Then move interior coloured pixels to exterior corners by their quadrant; flip[0]>0 exchanges top/bottom, '
    'flip[1]>0 exchanges left/right, offset=argmax(offset)+1. Erase originals to argmax(bg); keep frame and other cells.'}


def generated(row,seed,tier=1):
    pair=F.generated(row,seed,tier);perm=list(range(1,10));random.Random(seed+row['index']).shuffle(perm)
    for key in ('input','output'):
        pair[key]=[[perm[v-1] if v else 0 for v in line] for line in pair[key]]
    return pair


def truth(grid):
    a=np.asarray(grid);out=a.copy();coords=np.argwhere(a!=0);r0,c0=coords.min(0);r1,c1=coords.max(0)
    for (r,c),(rr,cc) in zip([(r0+1,c0+1),(r0+1,c1-1),(r1-1,c0+1),(r1-1,c1-1)],
                            [(r1+1,c1+1),(r1+1,c0-1),(r0-1,c1+1),(r0-1,c0-1)]):out[rr,cc]=a[r,c];out[r,c]=0
    return out.tolist()
