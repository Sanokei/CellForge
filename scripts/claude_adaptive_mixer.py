"""Count-only feature study: does adding word-level and in-document cache models to an
order-6 byte count model reduce bits/byte? No gradients. Mixture weights are chosen on a
held-out slice of dev_train (last 150 docs); dev_eval is scored once."""
import json, math, collections, itertools, re
tr=[s.encode() for s in json.load(open("data/tinyreal/dev_train.json"))]
ev=[s.encode() for s in json.load(open("data/tinyreal/dev_eval.json"))]
fit,val=tr[:-150],tr[-150:]
V=256
def wb(counters,ctxs,b):
    p=1.0/V
    for c in ctxs:            # ctxs ordered short -> long
        cc=counters.get(c)
        if not cc: break
        tot=sum(cc.values()); u=len(cc); lam=tot/(tot+u)
        p=lam*cc[b]/tot+(1-lam)*p
    return p
def ctx_ngram(d,i,N): return [d[max(0,i-n):i] if n<=i else None for n in range(0,N+1)]
def word_ctx(d,i):
    j=i
    while j>0 and d[j-1:j] not in (b" ",b".",b",",b"!",b"?",b'"',b"\n"): j-=1
    pre=d[j:i]                        # current partial word
    k=j
    while k>0 and d[k-1:k] in (b" ",b".",b",",b"!",b"?",b'"',b"\n"): k-=1
    m=k
    while m>0 and d[m-1:m] not in (b" ",b".",b",",b"!",b"?",b'"',b"\n"): m-=1
    prev=d[m:k]                       # previous whole word
    return [("w0",pre),("w1",prev,pre)]
S=[collections.defaultdict(collections.Counter) for _ in range(7)]   # static ngram
W=collections.defaultdict(collections.Counter)                       # static word ctx
for d in fit:
    for i,b in enumerate(d):
        for n in range(7):
            if i-n<0: break
            S[n][d[i-n:i]][b]+=1
        for c in word_ctx(d,i): W[c][b]+=1
def models(d):
    """yield (p_static6, p_word, p_cache) for each byte of doc d"""
    C=[collections.defaultdict(collections.Counter) for _ in range(5)]
    for i,b in enumerate(d):
        ps=1.0/V
        for n in range(7):
            if i-n<0: break
            cc=S[n].get(d[i-n:i])
            if not cc: break
            tot=sum(cc.values());u=len(cc);lam=tot/(tot+u);ps=lam*cc[b]/tot+(1-lam)*ps
        pw=1.0/V
        for c in word_ctx(d,i):
            cc=W.get(c)
            if not cc: break
            tot=sum(cc.values());u=len(cc);lam=tot/(tot+u);pw=lam*cc[b]/tot+(1-lam)*pw
        pc=1.0/V
        for n in range(5):
            if i-n<0: break
            cc=C[n].get(d[i-n:i])
            if not cc: break
            tot=sum(cc.values());u=len(cc);lam=tot/(tot+u);pc=lam*cc[b]/tot+(1-lam)*pc
        yield ps,pw,pc
        for n in range(5):
            if i-n>=0: C[n][d[i-n:i]][b]+=1
def collect(docs): return [t for d in docs for t in models(d)]
def bpb(rows,w):
    s=0
    for ps,pw,pc in rows: s-=math.log2(w[0]*ps+w[1]*pw+w[2]*pc)
    return s/len(rows)

import math
def docs_rows(docs): return [list(models(d)) for d in docs]
vd=docs_rows(val); ed=docs_rows(ev)
def run(D,eta,decay,w0,gate=False):
    s=0;n=0
    for rows in D:
        w=list(w0)
        for r in rows:
            z=sum(wi*pi for wi,pi in zip(w,r)); s-=math.log2(z); n+=1
            # Bayesian posterior update with forgetting: w_i <- (w_i*p_i^eta)^decay normalized, floored & shrunk toward prior
            nw=[(wi*(pi**eta)) for wi,pi in zip(w,r)]
            t=sum(nw); nw=[x/t for x in nw]
            nw=[decay*x+(1-decay)*p0 for x,p0 in zip(nw,w0)]
            w=nw
    return s/n
w0=(0.8,0.15,0.05)
print("fixed", run(ed,0.0,1.0,w0))
best=None
for eta in [0.25,0.5,1.0]:
    for decay in [0.9,0.97,0.99,1.0]:
        v=run(vd,eta,decay,w0)
        if best is None or v<best[0]: best=(v,eta,decay)
        print("val",eta,decay,round(v,4))
print("best val",best)
print("eval adaptive", run(ed,best[1],best[2],w0))
