"""Executable literal-weight count-cell growth; no task labels or identities."""
import numpy as np

def count_pre(p,grid):
    a=np.asarray(grid);h,w=a.shape;f=np.asarray(p['self.W'])[0,a]+p['cnt.b'][0]
    for dr,dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        for r in range(h):
            for c in range(w):
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w:f[r,c]+=p['cnt.W'][0][a[rr,cc]]
    return f

def count_patch(p,grid):
    a=np.asarray(grid);colors=set(a[a!=0].tolist());density=p['patch.guard.density'][0]
    if not 0<=density<=1:raise ValueError('typed density range')
    if not (np.any(a==0) and len(colors)==1 and 1 not in colors and np.count_nonzero(a)/a.size>density):return a.tolist()
    anchors=count_pre(p,grid)>0;out=a.copy();h,w=a.shape
    for r in range(h):
        for c in range(w):
            f=p['patch.emit.b'][0]
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr,cc=r-dr,c-dc
                    if 0<=rr<h and 0<=cc<w:f+=p['patch.emit.W'][dr+1][dc+1]*anchors[rr,cc]
            if f>0:out[r,c]=int(np.argmax(p['paint']))
    return out.tolist()

def rectangle_filter(p,grid):
    a=np.asarray(grid);h,w=a.shape
    body=int(np.argmax(p['filter.body.W']));marker=int(np.argmax(p['filter.marker.W']))
    colors=set(a[a!=0].tolist())
    if body==marker or body==0 or marker==0 or not np.any(a==0) or colors!={body,marker}:return count_patch(p,grid)
    unseen={(r,c) for r in range(h) for c in range(w) if a[r,c] in (body,marker)}
    components=[]
    while unseen:
        start=min(unseen);unseen.remove(start);stack=[start];component=[start]
        while stack:
            r,c=stack.pop()
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                q=(r+dr,c+dc)
                if q in unseen:unseen.remove(q);stack.append(q);component.append(q)
        rs=[x[0] for x in component];cs=[x[1] for x in component]
        if (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)!=len(component):return count_patch(p,grid)
        components.append(component)
    out=np.zeros_like(a)
    for component in components:
        count=sum(p['filter.marker.W'][0][a[r,c]] for r,c in component)
        score=p['filter.component.W'][0][0]+p['filter.component.W'][0][1]*count+p['filter.component.b'][0]
        if score>0:
            for r,c in component:out[r,c]=a[r,c]
    return out.tolist()
