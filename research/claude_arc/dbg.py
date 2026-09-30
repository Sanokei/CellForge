import sys,json,numpy as np
from objects import fit_objects
R="C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data/training/"
d=json.load(open(R+sys.argv[1]+".json"))
tr=[(np.array(e["input"]),np.array(e["output"])) for e in d["train"]]
te=[np.array(e["input"]) for e in d["test"]]; tg=[np.array(e["output"]) for e in d["test"]]
for k,p in fit_objects(tr,te):
    print(k)
    for a,b in zip(p,tg):
        if a.shape!=b.shape: print("shape",a.shape,b.shape)
        else:
            print("diff cells",(a!=b).sum()); 
            if (a!=b).sum()<40:
                print(a);print(b)
            print("test in colors",np.unique(te[0]))
