"""New corridor fill with learned replacement of the starting marker."""
import random
import numpy as np
import skillcore as S
from wall_skill import score

TEMPLATE,NICHENAME='corridor_marker2','corridormarker'
SHAPES={'marker':(10,),'through':(10,),'conn':(2,),'fill':(10,),'replace':(2,)}
ORACLE_PARAMS={'marker':[0]*9+[1],'through':[1]+[0]*9,'conn':[1,0],'fill':[0]*8+[1,0],'replace':[0,1]}


def predict(p,grid,demos=None):
    out=np.asarray(S.flood_through(p,grid));a=np.asarray(grid)
    if int(np.argmax(p['replace']))==1:
        color=int(np.argmax(p['fill'])) if max(p['fill'])>0 else int(np.argmax(p['marker']))
        out[a==int(np.argmax(p['marker']))]=color
    return out.tolist()


S.TEMPLATES[TEMPLATE]={'shapes':SHAPES,'fn':predict,'aware':False,'doc':
    'Flood connected corridors from every marker pixel argmax(marker), through colours through[10]>0; conn argmax0 four-neighbours,1 eight. '
    'Fill reached cells with argmax(fill) if max(fill)>0 else marker colour. New replace argmax1 ALSO replaces starting marker pixels '
    'with fill; replace0 preserves them. Preserve barriers and disconnected background.'}


def schema_rows(source):return [{'id':'corridor-%d'%i,'index':i} for i in range(120)]


def generated(row,seed,tier=1):
    rng=random.Random(seed*101+row['index']);h,w=rng.randint(8,16),rng.randint(10,18);c=rng.randint(2,w-5);d=rng.randint(2,3);mid=h//2
    path={(r,c) for r in range(mid+1)}|{(mid,j) for j in range(c,c+d+1)}|{(r,c+d) for r in range(mid,h)}
    a=np.zeros((h,w),int)
    for r,j in path:
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                q=(r+dr,j+dc)
                if q not in path and 0<=q[0]<h and 0<=q[1]<w:a[q]=7
    marker=rng.choice(sorted(path));a[marker]=9;out=a.copy()
    for q in path:out[q]=8
    return {'input':a.tolist(),'output':out.tolist()}


def truth(grid):
    a=np.asarray(grid);out=a.copy()
    for r in range(a.shape[0]):
        walls=np.where(a[r]==7)[0]
        if len(walls)>=2:
            for c in range(walls.min()+1,walls.max()):
                if a[r,c] in (0,9):out[r,c]=8
    return out.tolist()
