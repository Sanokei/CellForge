import sys, glob, os, time, zlib, json, numpy as np
from multiprocessing import Pool
import nca_evo as N
def setup(C,HID,T):
    N.C,N.HID,N.T=C,HID,T; N.NP=9*C*HID+HID+HID*C+C
def work(a):
    f,budget,sig,cfg=a
    os.environ["OMP_NUM_THREADS"]="1"; setup(*cfg)
    tr,te=N.load(f); name=os.path.basename(f); out={"name":name,"ident":int(N.exact(np.zeros(N.NP),te))}
    for arm,sel in (("es",True),("rand",False)):
        rng=np.random.default_rng(zlib.crc32(name.encode()))
        p,ev=N.es(tr,budget,rng,select=sel,sig0=sig)
        out[arm]={"train":int(N.exact(p,tr)),"test":int(N.exact(p,te)),"loss":round(float(N.loss_pairs(p,tr)),4)}
    return out
if __name__=="__main__":
    budget=int(sys.argv[1]); workers=int(sys.argv[2]); cfg=(10,4,3); sig=0.15
    setup(*cfg); files=[]
    for f in sorted(glob.glob("arc2/data/training/*.json")):
        tr,te=N.load(f)
        if N.ok(tr,te): files.append(f)
    print("tasks",len(files),"budget",budget,"workers",workers,flush=True); t0=time.time(); res=[]
    with Pool(workers) as pool:
        for r in pool.imap_unordered(work,[(f,budget,sig,cfg) for f in files]):
            res.append(r); 
            if len(res)%10==0: print(len(res),"done",f"{time.time()-t0:.0f}s","es_train",sum(x["es"]["train"] for x in res),"es_test",sum(x["es"]["test"] for x in res),"rand_train",sum(x["rand"]["train"] for x in res),flush=True)
    json.dump(res,open("nca_par_results.json","w"))
    s=lambda k,a:sum(x[k][a] for x in res)
    print("SUMMARY n=",len(res),"es train/test",s("es","train"),s("es","test"),"rand train/test",s("rand","train"),s("rand","test"),"ident",sum(x["ident"] for x in res),f"{time.time()-t0:.0f}s")
