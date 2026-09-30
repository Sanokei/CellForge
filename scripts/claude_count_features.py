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
vr=collect(val); er=collect(ev)
grid=[(a,b,round(1-a-b,2)) for a in [x/20 for x in range(4,21)] for b in [x/20 for x in range(0,17)] if a+b<=1.0]
for name,ok in [("order6 only",lambda w:w[1]==0 and w[2]==0),("+word",lambda w:w[2]==0),("+cache",lambda w:w[1]==0),("+word+cache",lambda w:True)]:
    cand=[w for w in grid if ok(w)]
    best=min(cand,key=lambda w:bpb(vr,w))
    print(f"{name:12s} weights(static,word,cache)={best} val={bpb(vr,best):.4f} eval={bpb(er,best):.4f}")
