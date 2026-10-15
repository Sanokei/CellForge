"""New reference-object property matching; selector behaviour remains tensors."""
import random
import numpy as np
import skillcore as S
from wall_skill import score

TEMPLATE,NICHENAME='reference_object2','referenceobject'
SHAPES={'bg':(10,),'reference':(10,),'source':(10,),'feature':(8,),'relation':(3,),'paint':(10,),'remove':(2,)}
ORACLE_PARAMS={'bg':[1]+[0]*9,'reference':[0,1]+[0]*8,'source':[0]*8+[1,0],
               'feature':[0,1]+[0]*6,'relation':[1,0,0],'paint':[0]*7+[1,0,0],'remove':[0,1]}


def predict(p,grid,demos=None):
    a=np.asarray(grid);bg=int(np.argmax(p['bg']));rc=int(np.argmax(p['reference']));objects=S.objects(a,bg,eight=True,same_color=True)
    refs=[o for o in objects if o['color']==rc]
    if len(refs)!=1:raise ValueError('one reference object required')
    mode=int(np.argmax(p['feature']));ref=S.obj_prop(mode,refs[0],objects,a.shape);relation=int(np.argmax(p['relation']));out=a.copy()
    for o in objects:
        if o is refs[0]:
            if int(np.argmax(p['remove']))==1:
                for cell in o['cells']:out[cell]=bg
            continue
        if p['source'][o['color']]<=0:continue
        feature=S.obj_prop(mode,o,objects,a.shape)
        match=(feature==ref) if relation==0 else (feature<ref) if relation==1 else (feature>ref)
        if match:
            for cell in o['cells']:out[cell]=int(np.argmax(p['paint']))
    return out.tolist()


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':False,'doc':
    'Find exactly one same-colour 8-connected reference object, colour argmax(reference), bg argmax(bg). '
    'Other objects of colours source>0 are candidates. feature argmax:0 area,1 four-connected holes,2 height,3 width,4 number of larger objects, '
    '5 size rank,6 border touch,7 (colour,holes). relation argmax0 equal,1 less,2 greater than reference property. '
    'Paint matching source objects argmax(paint); remove reference to bg iff remove argmax1. Preserve other cells.'}


def schema_rows(source):return [{'id':'reference-%d'%i,'index':i} for i in range(120)]


def generated(row,seed,tier=1):
    rng=random.Random(seed*101+row['index']);a=np.zeros((15,20),int);out=a.copy()
    for r,c in [(10,3),(10,4),(10,5),(11,3),(11,5),(12,4)]:a[r,c]=1
    for r,c,kind in [(1,2,'ring'),(1,10,'solid'),(6,10,'double')]:
        shape=np.ones((3,3 if kind=='ring' else 5),int)
        if kind=='ring':shape[1,1]=0
        elif kind=='double':shape[1,1]=shape[1,3]=0
        else:shape=shape[:2,:4]
        for dr,dc in np.argwhere(shape):
            a[r+dr,c+dc]=8;out[r+dr,c+dc]=7 if kind=='ring' else 8
    return {'input':a.tolist(),'output':out.tolist()}


def truth(grid):
    # Independent labels for this generated family: one-hole 3x3 ring, same-area
    # solid distractor, and two-hole distractor. No call to the cell predictor.
    a=np.asarray(grid);out=a.copy();out[a==1]=0
    ring=np.full((3,3),8);ring[1,1]=0
    for r in range(a.shape[0]-2):
        for c in range(a.shape[1]-2):
            surrounding=a[max(0,r-1):min(a.shape[0],r+4),max(0,c-1):min(a.shape[1],c+4)]
            if np.array_equal(a[r:r+3,c:c+3],ring) and int((surrounding==8).sum())==8:
                block=out[r:r+3,c:c+3];block[ring==8]=7
    return out.tolist()
