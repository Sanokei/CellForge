"""New context mapping from joint reference features to global source paint."""
import random
from collections import Counter
import numpy as np
import skillcore as S

TEMPLATE,NICHENAME='reference_paint3','referencepaint'
SHAPES={'bg':(10,),'reference':(10,),'source':(10,),'key':(3,),'paintmode':(2,),'remove':(2,)}
ORACLE_PARAMS={'bg':[1]+[0]*9,'reference':[0,1]+[0]*8,'source':[0]*8+[1,0],
               'key':[1,1,1],'paintmode':[1,0],'remove':[0,1]}


def reference_key(p,grid):
    a=np.asarray(grid);bg=int(np.argmax(p['bg']));ref=int(np.argmax(p['reference']));objs=S.objects(a,bg,eight=True,same_color=True)
    objects=[o for o in objs if o['color']==ref]
    if len(objects)!=1:raise ValueError('one reference required')
    o=objects[0];features=[o['color'],len(o['cells']),S.holes_of(o)]
    return tuple(v for v,w in zip(features,p['key']) if w>0)


def predict(p,grid,demos=None):
    table={};src=np.asarray(p['source'])>0
    for d in demos or []:
        a,b=np.asarray(d['input']),np.asarray(d['output'])
        if a.shape!=b.shape:continue
        outputs=b[src[a]]
        if outputs.size:
            table.setdefault(reference_key(p,d['input']),Counter()).update(int(x) for x in outputs)
    key=reference_key(p,grid)
    if key not in table:raise ValueError('unseen reference key')
    paint=table[key].most_common(1)[0][0];a=np.asarray(grid);out=a.copy()
    if int(np.argmax(p['paintmode']))==0:out[src[a]]=paint
    else:out[a==int(np.argmax(p['reference']))]=paint
    if int(np.argmax(p['remove']))==1:out[a==int(np.argmax(p['reference']))]=int(np.argmax(p['bg']))
    return out.tolist()


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':True,'doc':
    'Reference-conditioned in-context paint. Find one 8-connected reference object argmax(reference), bg argmax(bg). '
    'key[3]>0 selects joint features [reference colour, area, four-connected holes]. From each OTHER demonstration learn a histogram '
    'of output colours at input positions selected by source[10]>0, keyed by reference features. Query uses its key to choose modal paint. '
    'paintmode argmax0 repaints source pixels,1 repaints reference; remove argmax1 clears reference to bg. Unseen key invalid. '
    'Weights select roles and sufficient joint key; demonstration labels supply per-task associations. No query output is read.'}

PATTERNS=[[[1,1,1],[1,0,1],[0,1,0]],[[1,0,1],[0,1,0],[1,1,1]],[[0,1,0],[1,1,1],[0,1,0]]]
PAINTS=[7,3,2]


def schema_rows(source):return [{'id':'reference-paint-%d'%i,'index':i} for i in range(120)]


def example(kind,rng):
    a=np.zeros((10,12),int);mask=np.asarray(PATTERNS[kind]);a[6:9,2:5]=mask
    for r in range(1,4):
        for c in range(2,10):
            if rng.random()<.6:a[r,c]=8
    a[2,4]=8;out=a.copy();out[a==8]=PAINTS[kind];out[a==1]=0
    return {'input':a.tolist(),'output':out.tolist()}


def generated(row,seed,tier=1):
    rng=random.Random(seed*101+row['index']);pair=example(rng.randrange(3),rng)
    pair['context']=[example(k,rng) for k in range(3)]
    return pair


def score(cell,pairs):
    hits=[]
    for pair in pairs:
        try:hits.append(S.run(cell,pair['input'],pair.get('context',[]))==pair['output'])
        except Exception:hits.append(False)
    return sum(hits),hits


def truth(grid):
    a=np.asarray(grid);coords=np.argwhere(a==1);r0,c0=coords.min(0);r1,c1=coords.max(0);pattern=a[r0:r1+1,c0:c1+1].tolist()
    paint=PAINTS[PATTERNS.index(pattern)];out=a.copy();out[a==8]=paint;out[a==1]=0
    return out.tolist()
