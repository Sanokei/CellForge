import json, math, sys, collections, lzma, zlib
tr=[s.encode() for s in json.load(open("data/tinyreal/dev_train.json"))]
ev=[s.encode() for s in json.load(open("data/tinyreal/dev_eval.json"))]
N=int(sys.argv[1]) if len(sys.argv)>1 else 6
# Witten-Bell interpolated, byte-level, orders 0..N; contexts per document (BOS = start)
cnt=[collections.defaultdict(collections.Counter) for _ in range(N+1)]
for d in tr:
    for i,b in enumerate(d):
        for n in range(0,N+1):
            if i-n<0: break
            cnt[n][d[i-n:i]][b]+=1
V=256
def prob(ctx,b):
    p=1.0/V
    for n in range(0,N+1):
        if len(ctx)<n: break
        c=cnt[n].get(ctx[len(ctx)-n:] if n else b"")
        if not c: break
        tot=sum(c.values()); u=len(c)
        lam=tot/(tot+u)
        p=lam*c[b]/tot+(1-lam)*p
    return p
bits=0;nb=0
for d in ev:
    for i,b in enumerate(d):
        bits-=math.log2(prob(d[max(0,i-N):i],b)); nb+=1
print("order",N,"bpb",round(bits/nb,4),"bytes",nb)
if N==6:
    blob=b"".join(ev); print("xz bpb",8*len(lzma.compress(blob))/len(blob),"zlib",8*len(zlib.compress(blob,9))/len(blob))
