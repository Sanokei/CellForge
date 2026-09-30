import numpy as np, time
from nca_evo import *
rng0=np.random.default_rng(1)
# synthetic sanity task: swap colors 1<->2 on random 6x6 grids (pure local rule)
def mk(n):
    ps=[]
    for _ in range(n):
        x=rng0.choice([0,1,2],size=(6,6)); y=x.copy(); y[x==1]=2; y[x==2]=1; ps.append((x,y))
    return ps
tr=mk(3); te=mk(2)
for arm,sel in (("es",True),("rand",False)):
    rng=np.random.default_rng(0); t=time.time()
    p,ev=es(tr,6000,rng,select=sel)
    print(arm,"loss0",round(loss_pairs(np.zeros(NP),tr),4),"final",round(loss_pairs(p,tr),4),"train",exact(p,tr),"test",exact(p,te),"ev",ev,f"{time.time()-t:.0f}s")
