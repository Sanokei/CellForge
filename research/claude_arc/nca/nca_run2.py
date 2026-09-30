import sys, glob, os, time, zlib, numpy as np, nca_evo as N
C,HID,T=10,4,3; N.C,N.HID,N.T=C,HID,T; N.NP=9*C*HID+HID+HID*C+C
budget=int(sys.argv[1]); n=int(sys.argv[2]); sig=0.15
tasks=[]
for f in sorted(glob.glob("arc2/data/training/*.json")):
    tr,te=N.load(f)
    if N.ok(tr,te): tasks.append((os.path.basename(f),tr,te))
tasks=tasks[:n]; res={"es":[0,0],"rand":[0,0],"ident":0}; t0=time.time()
print("params",N.NP,"tasks",len(tasks),"budget",budget,flush=True)
for name,tr,te in tasks:
    line=[name]; idt=N.exact(np.zeros(N.NP),te); res["ident"]+=idt
    for arm,sel in (("es",True),("rand",False)):
        rng=np.random.default_rng(zlib.crc32(name.encode()))
        p,ev=N.es(tr,budget,rng,select=sel,sig0=sig)
        a=N.exact(p,tr); b=N.exact(p,te); res[arm][0]+=a; res[arm][1]+=b
        line.append(f"{arm}:train={int(a)} test={int(b)} loss={N.loss_pairs(p,tr):.3f}")
    print(*line,"ident",int(idt),f"{time.time()-t0:.0f}s",flush=True)
print("SUMMARY",res,f"{time.time()-t0:.0f}s")
