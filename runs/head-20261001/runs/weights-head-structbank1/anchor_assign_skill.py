"""New one-to-one anchor assignment; greedy rotation versions remain frozen."""
import random
import numpy as np
import skillcore as S
import anchor_clip_skill as A

TEMPLATE,NICHENAME='anchor_assign4','anchorassign'
SHAPES=dict(A.SHAPES,assignment=(2,))
ORACLE_PARAMS=dict(A.ORACLE_PARAMS,assignment=[0,1])
score=A.score


def predict(p,grid,demos=None):
    if int(np.argmax(p['assignment']))==0:return A.predict(p,grid,demos)
    a=np.asarray(grid);bg=int(np.argmax(p['bg']));ac=int(np.argmax(p['anchor']));anchors=[tuple(x) for x in np.argwhere(a==ac)]
    objs=[o for o in S.objects(a,bg,eight=True,same_color=True) if p['source'][o['color']]>0 and o['color']!=ac]
    if len(anchors)>14 or len(objs)>len(anchors):raise ValueError('assignment domain bound/insufficient anchors')
    states={0:(0.,())}
    for o in objs:
        center=np.mean(o['cells'],axis=0);nxt={}
        for mask,(cost,chosen) in states.items():
            for j,pivot in enumerate(anchors):
                if mask & (1<<j):continue
                if int(np.argmax(p['match']))==1 and not any(r==pivot[0] or c==pivot[1] for r,c in o['cells']):continue
                key=mask|(1<<j);value=(cost+abs(center[0]-pivot[0])+abs(center[1]-pivot[1]),chosen+(j,))
                if key not in nxt or value<nxt[key]:nxt[key]=value
        states=nxt
    if not states:raise ValueError('no complete one-to-one assignment')
    chosen=min(states.values())[1];out=a.copy()
    for o in objs:
        for cell in o['cells']:out[cell]=bg
    for o,j in zip(objs,chosen):
        pivot=anchors[j]
        for r,c in o['cells']:
            dr,dc=r-pivot[0],c-pivot[1]
            for _ in range(int(np.argmax(p['angle']))):dr,dc=-dc,dr
            rr,cc=pivot[0]+dr,pivot[1]+dc
            if not (0<=rr<a.shape[0] and 0<=cc<a.shape[1]):
                if int(np.argmax(p['bounds']))==1:continue
                raise ValueError('rotation bounds')
            if a[rr,cc]==ac:raise ValueError('anchor collision')
            out[rr,cc]=int(np.argmax(p['paint']))
    return out.tolist()


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':False,'doc':S.TEMPLATES[A.TEMPLATE]['doc']+
    ' New assignment[2]: argmax0 uses old independent greedy anchor assignment; argmax1 uses globally minimum Manhattan-distance one-to-one '
    'assignment of source objects to eligible anchors (max14 anchors), preventing two sources choosing the same anchor. Lexicographic ties. '
    'Clipping bounds semantics remain; learned roles, angle, paint, matching, clipping and assignment selectors determine behavior.'}


def schema_rows(source):return [{'id':'anchor-assign-%d'%i,'index':i} for i in range(120)]


def generated(row,seed,tier=1):
    rng=random.Random(seed*101+row['index']);r,c,d=rng.randint(5,12),rng.randint(5,9),rng.randint(2,3)
    a=np.full((18,18),8,int);out=a.copy();a[r,c]=a[r,c+2*d]=out[r,c]=out[r,c+2*d]=1
    a[r-d,c]=a[r,c+d]=6;out[r,c-d]=out[r+d,c+2*d]=7
    return {'input':a.tolist(),'output':out.tolist()}


def truth(grid):
    a=np.asarray(grid);out=a.copy();out[a==6]=8;anchors=[tuple(x) for x in np.argwhere(a==1)];used=set()
    sources=[tuple(x) for x in np.argwhere(a==6)]
    for r,c in sorted(sources,key=lambda x:x[0]):
        choices=[q for q in anchors if q not in used and (q[0]==r or q[1]==c)]
        pivot=min(choices,key=lambda q:abs(q[0]-r)+abs(q[1]-c));used.add(pivot)
        out[pivot[0]-(c-pivot[1]),pivot[1]+(r-pivot[0])]=7
    return out.tolist()
