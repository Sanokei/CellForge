import glob,json,numpy as np,collections
from scipy import ndimage
from cells import bg_color
ROOT="C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data/training"
s={x['task'] for x in json.load(open('results_training_d2_v4.json')) if x['solved']}
s|={x.replace(chr(92),'/').split('/')[-1] for x in json.load(open('objsolved.json'))}
c=collections.Counter()
for f in sorted(glob.glob(ROOT+"/*.json")):
    n=f.replace(chr(92),'/').split('/')[-1]
    if n in s: continue
    d=json.load(open(f)); tr=[(np.array(e["input"]),np.array(e["output"])) for e in d["train"]]
    if not all(x.shape==y.shape for x,y in tr): continue
    flags=collections.Counter()
    for x,y in tr:
        b=bg_color(x); ch=x!=y
        if not ch.any(): continue
        add=ch&(x==b)
        if add.sum()==ch.sum():
            ys,xs=np.nonzero(x!=b); rows=set(ys); cols=set(xs)
            ay,ax=np.nonzero(add)
            flags['add_rowcol']+= all((a in rows) or (bb in cols) for a,bb in zip(ay,ax))
            fill=ndimage.binary_fill_holes(x!=b)
            flags['add_in_holes']+= bool((~(fill&(x==b))&add).sum()==0)
            flags['add_in_bbox']+= 1 if all(any(True for _ in [0]) for _ in [0]) and False else 0
            lab,k=ndimage.label(x!=b,structure=np.ones((3,3)))
            inb=np.zeros(x.shape,bool)
            for i in range(1,k+1):
                yy,xx=np.nonzero(lab==i); inb[yy.min():yy.max()+1,xx.min():xx.max()+1]=True
            flags['add_in_objbbox']+= bool((add&~inb).sum()==0)
            flags['add_colors_in_input']+= set(np.unique(y[add]))<=set(np.unique(x))
            flags['add_near_obj']+= bool((add&~ndimage.binary_dilation(x!=b,structure=np.ones((3,3)),iterations=2)).sum()==0)
    n_=len(tr)
    for k,v in flags.items():
        if v==n_: c[k]+=1
    c['total_add_tasks']+= 0
print(c)
