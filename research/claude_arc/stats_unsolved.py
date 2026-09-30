import glob,json,numpy as np,collections
from cells import bg_color
ROOT="C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data/training"
s={x['task'] for x in json.load(open('results_training_d2_v4.json')) if x['solved']}
s|={x.replace(chr(92),'/').split('/')[-1] for x in json.load(open('objsolved.json'))}
c=collections.Counter(); ex=collections.defaultdict(list)
for f in sorted(glob.glob(ROOT+"/*.json")):
    n=f.replace(chr(92),'/').split('/')[-1]
    if n in s: continue
    d=json.load(open(f)); tr=[(np.array(e["input"]),np.array(e["output"])) for e in d["train"]]
    if all(x.shape==y.shape for x,y in tr):
        add=rec=rem=0
        for x,y in tr:
            b=bg_color(x); ch=x!=y
            add+=(ch&(x==b)).sum(); rec+=(ch&(x!=b)&(y!=b)).sum(); rem+=(ch&(x!=b)&(y==b)).sum()
        k="same:"+("add" if add and not rec and not rem else "recolor" if rec and not add and not rem else "remove" if rem and not add and not rec else "mixed")
    elif all(y.size<x.size for x,y in tr): k="smaller"
    elif all(y.size>x.size for x,y in tr): k="bigger"
    else: k="other"
    c[k]+=1; ex[k].append(n)
print(c)
json.dump(ex,open('unsolved_cats.json','w'))
