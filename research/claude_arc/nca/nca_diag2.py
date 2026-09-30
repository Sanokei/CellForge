import numpy as np, time, nca_evo as N
rng0=np.random.default_rng(1)
def mk(n):
    ps=[]
    for _ in range(n):
        x=rng0.choice([0,1,2],size=(6,6)); y=x.copy(); y[x==1]=2; y[x==2]=1; ps.append((x,y))
    return ps
tr=mk(3); te=mk(2)
for C,HID,T,budget,sig in [(10,4,3,20000,0.05),(10,4,3,20000,0.15),(12,6,4,20000,0.1)]:
    N.C,N.HID,N.T=C,HID,T; N.NP=9*C*HID+HID+HID*C+C
    rng=np.random.default_rng(0); t=time.time()
    p,ev=N.es(tr,budget,rng,select=True,sig0=sig)
    print(f"C={C} HID={HID} T={T} NP={N.NP} sig0={sig}: loss0 {N.loss_pairs(np.zeros(N.NP),tr):.3f} final {N.loss_pairs(p,tr):.4f} train={N.exact(p,tr)} test={N.exact(p,te)} ev={ev} {time.time()-t:.0f}s",flush=True)
