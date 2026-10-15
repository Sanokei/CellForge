"""New rigid rotation of source objects about aligned anchor pixels."""
import random
import numpy as np
import skillcore as S
from wall_skill import score

TEMPLATE,NICHENAME = 'anchor_spin2','anchorspin'
SHAPES = {'bg':(10,),'source':(10,),'anchor':(10,),'angle':(4,),'paint':(10,),'match':(2,)}
ORACLE_PARAMS = {'bg':[0]*8+[1,0],'source':[0]*6+[1,0,0,0],'anchor':[0,1]+[0]*8,
                 'angle':[0,1,0,0],'paint':[0]*7+[1,0,0],'match':[0,1]}


def predict(p,grid,demos=None):
    a=np.asarray(grid); bg=int(np.argmax(p['bg'])); ac=int(np.argmax(p['anchor'])); paint=int(np.argmax(p['paint']))
    anchors=[tuple(x) for x in np.argwhere(a==ac)]
    if not anchors: raise ValueError('no anchors')
    out=a.copy(); sources=[o for o in S.objects(a,bg,eight=True,same_color=True) if np.asarray(p['source'])[o['color']]>0 and o['color']!=ac]
    for o in sources:
        for r,c in o['cells']:out[r,c]=bg
    for o in sources:
        center=np.mean(o['cells'],axis=0)
        aligned=[q for q in anchors if any(r==q[0] or c==q[1] for r,c in o['cells'])]
        choices=aligned if int(np.argmax(p['match']))==1 else anchors
        if not choices:raise ValueError('no aligned anchor')
        pivot=min(choices,key=lambda q:(abs(q[0]-center[0])+abs(q[1]-center[1]),q))
        for r,c in o['cells']:
            dr,dc=r-pivot[0],c-pivot[1]
            for _ in range(int(np.argmax(p['angle']))):dr,dc=-dc,dr
            rr,cc=pivot[0]+dr,pivot[1]+dc
            if not (0<=rr<a.shape[0] and 0<=cc<a.shape[1]):raise ValueError('rotation out of bounds')
            if a[rr,cc]==ac:raise ValueError('anchor collision')
            out[rr,cc]=paint
    return out.tolist()


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':False,'doc':
    'Rigid object rotation. bg=argmax(bg); anchor pixels have argmax(anchor). Source colours source>0 form same-colour 8-connected objects. '
    'Assign each to nearest anchor by centroid Manhattan distance, restricted to anchors aligned with ANY object pixel if match argmax=1; '
    'match0 uses all anchors. Clear original source cells to bg. Rotate offsets (dr,dc)->(-dc,dr), argmax(angle) times, around the anchor; '
    'paint transformed source pixels argmax(paint), keep anchor/other pixels. Out-of-bounds or anchor collisions invalid.'}


def schema_rows(source):return [{'id':'anchor-%d'%i,'index':i} for i in range(120)]


def generated(row,seed,tier=1):
    rng=random.Random(seed*101+row['index']);a=np.full((17,17),8,int);out=a.copy()
    for r,c in [(4,4),(4,12),(12,4),(12,12)]:
        dr,dc=rng.choice([(-1,0),(0,1),(1,0),(0,-1)]);n=rng.randint(1,3)
        a[r,c]=out[r,c]=1;a[r+dr*n,c+dc*n]=6;out[r-dc*n,c+dr*n]=7
    return {'input':a.tolist(),'output':out.tolist()}


def truth(grid):
    a=np.asarray(grid);out=a.copy();out[a==6]=8
    anchors=[tuple(x) for x in np.argwhere(a==1)]
    for r,c in np.argwhere(a==6):
        choices=[q for q in anchors if q[0]==r or q[1]==c]
        pr,pc=min(choices,key=lambda q:abs(q[0]-r)+abs(q[1]-c))
        out[pr-(c-pc),pc+(r-pr)]=7
    return out.tolist()
