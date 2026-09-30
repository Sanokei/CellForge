"""Grokking-style test for gradient-free (mu+lambda) evolution of a tiny NCA. numpy only, selection+mutation only."""
import json, sys, time, os, zlib
import numpy as np
C,HID,T=10,4,3
NP=9*C*HID+HID+HID*C+C
GRID=6
CHECK=[1000,3000,10000,30000,100000,300000]

# ---- synthetic tasks: each returns (x,y) from rng ----
def _nb(x,dy,dx,fill=-1):
    H,W=x.shape; o=np.full((H,W),fill); 
    ys=slice(max(dy,0),H+min(dy,0)); yd=slice(max(-dy,0),H+min(-dy,0))
    xs=slice(max(dx,0),W+min(dx,0)); xd=slice(max(-dx,0),W+min(-dx,0))
    o[ys,xs]=x[yd,xd]; return o   # o[i,j]=x[i-dy,j-dx]
def rg(r,k,p=None): return r.choice(k,size=(GRID,GRID),p=p)
def t_swap(r):
    x=rg(r,3); y=x.copy(); y[x==1]=2; y[x==2]=1; return x,y
def t_cycle(r):
    x=rg(r,4); y=np.array([0,2,3,1])[x]; return x,y
def t_count(r):
    x=rg(r,2); n=sum((_nb(x,a,b)==x) for a,b in((1,0),(-1,0),(0,1),(0,-1)))
    y=x.copy(); y[(x==1)&(n>=2)]=2; return x,y
def t_border(r):
    x=rg(r,3); y=x.copy(); y[0,:]=5;y[-1,:]=5;y[:,0]=5;y[:,-1]=5; return x,y
def t_hole(r):
    x=rg(r,2,[.3,.7]); n=sum((_nb(x,a,b)==1) for a,b in((1,0),(-1,0),(0,1),(0,-1)))
    y=x.copy(); y[(x==0)&(n==4)]=2; return x,y
def t_shift(r):
    x=rg(r,4); y=_nb(x,0,1,0); return x,y
def t_life(r):
    x=rg(r,2); n=sum((_nb(x,a,b,0)==1) for a in(-1,0,1) for b in(-1,0,1) if (a,b)!=(0,0))
    y=((x==1)&((n==2)|(n==3)))|((x==0)&(n==3)); return x,y.astype(int)
def t_dilate(r):
    x=rg(r,2,[.85,.15]); n=sum((_nb(x,a,b,0)==1) for a,b in((1,0),(-1,0),(0,1),(0,-1)))
    y=x.copy(); y[(x==0)&(n>0)]=1; return x,y
def t_interior(r):
    x=rg(r,2,[.3,.7]); n=sum((_nb(x,a,b)==1) for a,b in((1,0),(-1,0),(0,1),(0,-1)))
    y=x.copy(); y[(x==1)&(n==4)]=2; return x,y
def t_edge(r):
    x=rg(r,3); l=_nb(x,0,1,-1); y=(x!=l).astype(int); y[:,0]=0; return x,y
# controls an NCA with radius-3 receptive field cannot express in general
def t_transpose(r):
    x=rg(r,4); return x,x.T.copy()
def t_majority(r):
    x=rg(r,3,[.5,.3,.2]); m=np.bincount(x.ravel(),minlength=3).argmax(); return x,np.full_like(x,m)
TASKS=[("swap",t_swap,0),("cycle",t_cycle,0),("count",t_count,0),("border",t_border,0),("hole",t_hole,0),("shift",t_shift,0),
       ("life",t_life,0),("dilate",t_dilate,0),("interior",t_interior,0),("edge",t_edge,0),
       ("CTRL_transpose",t_transpose,1),("CTRL_majority",t_majority,1)]

# ---- batched NCA ----
def unpack(P):  # P: (K,NP)
    K=P.shape[0]; a=9*C*HID
    W1=P[:,:a].reshape(K,9*C,HID); b1=P[:,a:a+HID][:,None,:]; o=a+HID
    W2=P[:,o:o+HID*C].reshape(K,HID,C); b2=P[:,o+HID*C:][:,None,:]; return W1,b1,W2,b2
def run(P,X):  # X: (N,H,W) int -> (K,N,H,W,C)
    K=P.shape[0]; N,H,W=X.shape; W1,b1,W2,b2=unpack(P)
    s0=np.zeros((N,H,W,C)); s0[np.arange(N)[:,None,None],np.arange(H)[None,:,None],np.arange(W)[None,None,:],X]=1
    s=np.broadcast_to(s0,(K,N,H,W,C)).copy()
    for _ in range(T):
        pad=np.pad(s,((0,0),(0,0),(1,1),(1,1),(0,0)))
        per=np.concatenate([pad[:,:,i:i+H,j:j+W] for i in range(3) for j in range(3)],axis=4).reshape(K,N*H*W,9*C)
        h=np.tanh(per@W1+b1); s=s+(h@W2+b2).reshape(K,N,H,W,C)
    return s
def losses(P,X,Y):
    out=run(P,X)[...,:10]; t=np.zeros(out.shape[1:]); 
    t[np.arange(len(X))[:,None,None],np.arange(X.shape[1])[None,:,None],np.arange(X.shape[2])[None,None,:],Y]=1
    return ((out-t[None])**2).mean(axis=(1,2,3,4))
def exact_frac(p,X,Y):
    pr=run(p[None],X)[0,...,:10].argmax(-1); return float((pr==Y).all(axis=(1,2)).mean())

def make_data(ti,seed):
    name,f,ctrl=TASKS[ti]; r=np.random.default_rng(1000*ti+seed)
    g=lambda n:[f(r) for _ in range(n)]
    tr=g(2); te=g(21)   # te[0]=held-out pair, te[1:]=20 extra grids
    st=lambda L:(np.stack([a for a,_ in L]),np.stack([b for _,b in L]))
    return st(tr),st(te)

def snapshot(p,tr,te,ev,lam):
    X,Y=tr; l=float(losses(p[None],X,Y)[0]); d=exact_frac(p,X,Y)
    return dict(ev=ev,loss=l,demo_exact=d==1.0,held=exact_frac(p,te[0][:1],te[1][:1]),extra20=exact_frac(p,te[0][1:],te[1][1:]),
                heldall=exact_frac(p,*te),cellacc=float((run(p[None],te[0][1:])[0,...,:10].argmax(-1)==te[1][1:]).mean()),L1=float(np.abs(p).sum()),nnz=int((np.abs(p)>1e-9).sum()))

def evolve(ti,seed,arm,lam,budget,mu=6,kids=24,sig0=0.05):
    tr,te=make_data(ti,seed); X,Y=tr
    rng=np.random.default_rng(zlib.crc32(f"{ti}-{seed}-{arm}".encode()))
    pen=(lambda P:lam*np.abs(P).mean(axis=1)) if arm=="B" else (lambda P:0.0)
    seedp=np.zeros((1,NP)); pop=[(float(losses(seedp,X,Y)[0]+pen(seedp)[0] if arm=="B" else losses(seedp,X,Y)[0]),seedp[0])]
    sig=sig0; ev=1; best=pop[0]; checks=[c for c in CHECK if c<=budget]; snaps=[]; efit=None; fitsnap=None
    snaps.append(snapshot(best[1],tr,te,ev,lam)) if False else None
    while ev<budget:
        if arm=="D":
            sg=rng.choice([0.02,0.05,0.15],size=(kids,1)); K=seedp+rng.normal(0,1,(kids,NP))*sg
        else:
            par=np.stack([pop[rng.integers(len(pop))][1] for _ in range(kids)])
            K=par+rng.normal(0,sig,(kids,NP))
            if arm=="B":  # sparsification mutants: zero smallest 10% of nonzero weights of a parent (accepted only via penalised fitness)
                for i in range(kids//4):
                    q=par[i].copy(); nz=np.flatnonzero(q); 
                    if len(nz): q[nz[np.argsort(np.abs(q[nz]))[:max(1,len(nz)//10)]]]=0
                    K[i]=q
        L=losses(K,X,Y); F=L+pen(K) if arm=="B" else L; ev+=kids
        allp=sorted(pop+[(float(F[i]),K[i]) for i in range(kids)],key=lambda t:t[0])
        improved=allp[0][0]<pop[0][0]-1e-12; pop=allp[:mu]
        if arm!="D": sig=min(sig*1.25,0.5) if improved else max(sig*0.8,0.003)
        if pop[0][0]<best[0]-1e-15 or improved:
            best=pop[0]
        best=pop[0] if arm!="D" else (pop[0] if pop[0][0]<best[0] else best)
        if efit is None and exact_frac(best[1],X,Y)==1.0:
            efit=ev; fitsnap=snapshot(best[1],tr,te,ev,lam)
        while checks and ev>=checks[0]:
            s=snapshot(best[1],tr,te,ev,lam); s["check"]=checks.pop(0); snaps.append(s)
    return dict(task=TASKS[ti][0],ti=ti,ctrl=TASKS[ti][2],seed=seed,arm=arm,lam=lam,efit=efit,fitsnap=fitsnap,snaps=snaps)

def job(a): 
    t=time.time(); r=evolve(*a); r["sec"]=time.time()-t; return r
if __name__=="__main__":
    import multiprocessing as mp
    budget=int(sys.argv[1]); seeds=int(sys.argv[2]); out=sys.argv[3]; tis=[int(v) for v in sys.argv[4].split(",")] if len(sys.argv)>4 else range(len(TASKS))
    arms=[("A",0.0),("B",0.001),("B",0.004),("B",0.016),("D",0.0)]
    jobs=[(ti,s,a,l,budget) for s in range(seeds) for ti in tis for a,l in arms]
    done=set()
    if os.path.exists(out):
        for ln in open(out): d=json.loads(ln); done.add((d["ti"],d["seed"],d["arm"],d["lam"]))
    jobs=[j for j in jobs if j[:4] not in done]
    print(len(jobs),"jobs",flush=True); t0=time.time()
    with mp.Pool(2) as pool, open(out,"a") as f:
        for r in pool.imap_unordered(job,jobs):
            f.write(json.dumps(r)+"\n"); f.flush(); print(r["task"],r["seed"],r["arm"],r["lam"],r["efit"],f'{r["sec"]:.0f}s total {time.time()-t0:.0f}s',flush=True)
