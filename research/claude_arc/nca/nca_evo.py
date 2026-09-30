"""Gradient-free evolution of tiny neural cellular automata on ARC-AGI-2 public training tasks.
Seeded at identity (zero update). Compares (mu+lambda) selection vs random-perturbation search at equal eval budget."""
import json, sys, time, glob, os
import numpy as np
C,HID,T=16,8,10
NP=9*C*HID+HID+HID*C+C
def unpack(p):
    a=9*C*HID; W1=p[:a].reshape(9*C,HID); b1=p[a:a+HID]; o=a+HID
    W2=p[o:o+HID*C].reshape(HID,C); b2=p[o+HID*C:]; return W1,b1,W2,b2
def run(p,grid):
    W1,b1,W2,b2=unpack(p); H,W=grid.shape
    s=np.zeros((H,W,C)); s[np.arange(H)[:,None],np.arange(W)[None,:],grid]=1.0
    for _ in range(T):
        pad=np.pad(s,((1,1),(1,1),(0,0)))
        per=np.concatenate([pad[i:i+H,j:j+W] for i in range(3) for j in range(3)],axis=2)
        h=np.tanh(per@W1+b1); s=s+h@W2+b2
    return s[:,:,:10]
def loss_pairs(p,pairs):
    l=0
    for x,y in pairs:
        out=run(p,x); t=np.zeros_like(out); t[np.arange(y.shape[0])[:,None],np.arange(y.shape[1])[None,:],y]=1
        l+=np.mean((out-t)**2)
    return l/len(pairs)
def exact(p,pairs): return all(np.array_equal(run(p,x).argmax(2),y) for x,y in pairs)
def load(path):
    d=json.load(open(path)); f=lambda a:[(np.array(e["input"]),np.array(e["output"])) for e in a]
    return f(d["train"]),f(d["test"])
def ok(tr,te):
    return all(x.shape==y.shape and max(x.shape)<=14 for x,y in tr+te)
def es(pairs,budget,rng,select=True,lam=24,mu=6,sig0=0.05):
    seed=np.zeros(NP); pop=[(loss_pairs(seed,pairs),seed)]; sig=sig0; ev=1
    best=pop[0]
    while ev<budget:
        kids=[]
        for _ in range(lam):
            par=pop[rng.integers(len(pop))][1] if select else seed
            c=par+rng.normal(0,sig,NP); kids.append((loss_pairs(c,pairs),c)); ev+=1
        if select:
            allp=sorted(pop+kids,key=lambda t:t[0]); improved=allp[0][0]<pop[0][0]-1e-12
            pop=allp[:mu]; sig=min(sig*1.25,0.5) if improved else max(sig*0.8,0.003)
        else:
            pop=sorted(pop+kids,key=lambda t:t[0])[:mu]
        if pop[0][0]<best[0]: best=pop[0]
        if exact(best[1],pairs): break
    return best[1],ev
if __name__=="__main__":
    budget=int(sys.argv[1]); n=int(sys.argv[2])
    files=sorted(glob.glob("arc2/data/training/*.json"))
    tasks=[]
    for f in files:
        tr,te=load(f)
        if ok(tr,te): tasks.append((os.path.basename(f),tr,te))
    print("same-shape small tasks:",len(tasks),"of",len(files),flush=True)
    tasks=tasks[:n]; res={"es":[0,0],"rand":[0,0],"ident":0}
    t0=time.time()
    for name,tr,te in tasks:
        line=[name]
        seedp=np.zeros(NP)
        idt=exact(seedp,te); res["ident"]+=idt
        for arm,sel in (("es",True),("rand",False)):
            rng=np.random.default_rng(__import__("zlib").crc32(name.encode()))
            p,ev=es(tr,budget,rng,select=sel)
            trs=exact(p,tr); tes=exact(p,te); res[arm][0]+=trs; res[arm][1]+=tes
            line.append(f"{arm}:train={int(trs)} test={int(tes)} ev={ev}")
        print(*line,"ident",int(idt),f"{time.time()-t0:.0f}s",flush=True)
    print("SUMMARY n=",len(tasks),"budget",budget,res,f"{time.time()-t0:.0f}s")
