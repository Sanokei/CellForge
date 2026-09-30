"""Fast harness: run only object fitters on training; report demo-consistent / solved / false positives."""
import sys, glob, json, time, numpy as np
from multiprocessing import Pool
from objects import fit_objects
ROOT = "C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc2/data/training"
def run(f):
    d=json.load(open(f)); t0=time.time()
    tr=[(np.array(e["input"]),np.array(e["output"])) for e in d["train"]]
    te=[np.array(e["input"]) for e in d["test"]]; tg=[np.array(e["output"]) for e in d["test"]]
    sols=fit_objects(tr,te); att=[]
    for k,p in sols:
        if len(att)<2 and not any(all(np.array_equal(a,b) for a,b in zip(p,q)) for _,q in att): att.append((k,p))
    ok=[k for k,p in att if all(a.shape==b.shape and np.array_equal(a,b) for a,b in zip(p,tg))]
    return f.split('/')[-1], len(sols), ok[:1], [k for k,_ in att], time.time()-t0
if __name__=="__main__":
    fs=sorted(glob.glob(ROOT+"/*.json"))
    with Pool(3) as p: r=p.map(run,fs,chunksize=4)
    cons=[x for x in r if x[1]]; sol=[x for x in r if x[2]]
    print("consistent",len(cons),"solved",len(sol),"FP",len(cons)-len(sol),"sec/task",sum(x[4] for x in r)/len(r))
    from collections import Counter
    print(Counter(x[2][0].split(':')[0]+':'+x[2][0].split(':')[-1] for x in sol))
    print("FP:",[(x[0],x[3]) for x in cons if not x[2]][:40])
    json.dump([x[0] for x in sol],open("objsolved.json","w"))
