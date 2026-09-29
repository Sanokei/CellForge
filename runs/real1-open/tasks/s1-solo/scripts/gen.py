"""Expand my typed lexicon/contexts into a CellForge proposal (no dataset access)."""
import json, math, sys, os
import numpy as np
from lexicon import TIERS, EXTRA_WORDS
from lexicon2 import TIERS2
from contexts2 import CTX2
from contexts3 import CTX3
from contexts import CTX

V = ['<pad>','<bos>','<eos>',' ',',','.','!','?',"'",'"']+list("abcdefghijklmnopqrstuvwxyz")
T = {c:i for i,c in enumerate(V)}
T['^'] = 1
NV = 36
P = dict(K=2.0, K2=20.0, ctx_scale=0.8, eps=0.003, rank_decay=0.78, rank_top=30.0, shard=3200, eta=0.03, seen_g=1.2, seen_o=0.3, seen_on=1)
for a in sys.argv[1:]:
    k,v = a.split('='); P[k]=float(v)

# ---- base (unknown-context) distribution: my guess of char frequencies
BASE_W = {' ':180,',':10,'.':11,'!':2,'?':1,"'":2,'"':4,'e':95,'t':70,'a':70,'o':55,'i':50,'n':50,'s':45,'h':45,'r':38,'d':35,'l':38,'u':22,'m':20,'w':22,'y':20,'c':16,'g':16,'f':15,'p':14,'b':14,'k':9,'v':7,'j':1.5,'x':1,'q':0.6,'z':0.6}
base = np.zeros(NV)
for c,w in BASE_W.items(): base[T[c]] = w
base[T['<eos>']] = 0.05
base /= base.sum()

# ---- word-end punctuation distributions
E_DEF = {' ':.79,',':.07,'.':.09,'!':.02,'?':.01,"'":.005,'"':.015}
E_FUNC = {' ':.985,',':.01,'.':.002,'!':.001,'?':.001,"'":.0,'"':.0}
E_NAME = {' ':.72,',':.10,'.':.06,'!':.03,'?':.02,"'":.06,'"':.01}
E_SAID = {' ':.40,',':.57,'.':.01,'!':.0,'?':.0,"'":.0,'"':.02}
E_DAY  = {' ':.40,',':.42,'.':.13,'!':.02,'?':.0,"'":.0,'"':.03}
E_NOUN = {' ':.66,',':.09,'.':.20,'!':.03,'?':.01,"'":.01,'"':.0}
E_CONTR= {' ':.86,',':.05,'.':.05,'!':.02,'?':.01,"'":.0,'"':.01}
FUNC = set("the a an and to was were is are of in on at with for from by as that this his her their my your our its it he she they we i you but so or if than then when while because not no do did does can could would will should have has had be been am up out down off over into onto very too just all some any each every more most many much such what who how why where which there here them him me us like about after before again around through under until".split())
NAMES = set("lily tim mia sam ben anna max sue jack tom lucy sara amy bob spot sally jane john kate emma ella lila zoe leo jim tommy timmy molly daisy rosie fluffy buddy sparky lulu tiny bella bobo benny ollie ann joe joey kitty sunny".split())
NOUNS = set("day time home park school friends friend toys toy ball game games fun tree garden house room dog cat bird sun water forest river lake beach sand story party cake food lunch dinner bed box car boat book night morning world sky way place things thing family mom dad".split())
E_OVR = {'said':E_SAID,'asked':E_SAID,'replied':E_SAID,'shouted':E_SAID,'day':E_DAY,'time':E_DAY,'night':E_DAY,'morning':E_DAY}
def endclass(w):
    if w in E_OVR: return E_OVR[w]
    if "'" in w: return E_CONTR
    if w in NAMES: return E_NAME
    if w in FUNC: return E_FUNC
    if w in NOUNS: return E_NOUN
    return E_DEF

# ---- lexicon
CTX.update(CTX2)
CTX.update(CTX3)
words = {}
for wt, s_ in TIERS2.items():
    for w in s_.split(): words[w] = max(words.get(w,0), float(wt))
for wt, s in TIERS.items():
    for w in s.split(): words[w] = max(words.get(w,0), float(wt))
for w in EXTRA_WORDS: words.setdefault(w, 3.0)
for w in NAMES: words[w] = max(words.get(w, 0), 7.0)
for c, s in CTX.items():
    for w in s.split(): words.setdefault(w, 3.0)

FLOOR = 0.5*base.copy()
FLOOR[3:] += 0.5/33.0
FLOOR[:3] = 0
FLOOR[2] = base[2]
FLOOR /= FLOOR.sum()

def logv(p): return np.log(np.maximum(p, 1e-9))

def node_masses(wd, p):
    m = np.zeros(NV)
    for w, wt in wd.items():
        if w == '"':
            if p == '': m[T['"']] += wt
            continue
        if w.startswith(p):
            if len(w) == len(p):
                for ch, pr in endclass(w).items(): m[T[ch]] += wt*pr
            else:
                m[T[w[len(p)]]] += wt
    return m

trie_cache = {}
def ptrie(p):
    if p in trie_cache: return trie_cache[p]
    m = node_masses(words, p); M = m.sum()
    if M == 0: r = None
    else:
        lam = M/(M+P['K'])
        eta = max(1-lam, P['eta'])
        r = (1-eta)*m/M + eta*FLOOR
    trie_cache[p] = r
    return r

def pnode(q):
    r = ptrie(q) if len(q) <= 8 else None
    return base if r is None else r

units = []
seen = {}
def add_unit(s, bd, delta, tag):
    if len(s) > 8: s = s[-8:]; bd = False
    if bd and len(s) > 7: bd = False
    key = (s, bd)
    if key in seen:
        return
    seen[key] = 1
    units.append((s, bd, delta, tag))

CTX = {k: v for k, v in CTX.items() if k[-1] in ' "^'}
words['"'] = 150.0
prefixes = set()
for w in words:
    if w == '"': continue
    for i in range(0, len(w)+1): prefixes.add(w[:i])
for p in sorted(prefixes, key=lambda x:(len(x), x)):
    r = ptrie(p)
    if r is None: continue
    add_unit(p, True, logv(r) - logv(base), 'trie')

def build_wd(succ):
    wd = {}
    for i, w in enumerate(succ.split()): wd[w] = wd.get(w, 0) + P['rank_top']*P['rank_decay']**i
    return wd
WD = {C: build_wd(succ) for C, succ in CTX.items()}
def qset(wd):
    qs = set()
    for w in wd:
        if w == '"': qs.add(''); continue
        for i in range(0, min(len(w), 6)+1): qs.add(w[:i])
    return qs
def ctxlog(C, q):
    wd = WD[C]
    m = node_masses(wd, q); M = m.sum()
    if M == 0: return np.zeros(NV)
    mu = M/(M+P['K2'])
    pn = pnode(q)
    pc = mu*m/M + (1-mu)*pn
    return logv(pc) - logv(pn)
def parent(C):
    best = None
    for K in CTX:
        if K != C and C.endswith(K) and (len(K) == len(C) or C[-len(K)-1] in ' "'):
            if best is None or len(K) > len(best): best = K
    return best
for C in CTX:
    chain = [C]
    while parent(chain[-1]) is not None: chain.append(parent(chain[-1]))
    par = chain[1] if len(chain) > 1 else None
    qs = set()
    for K in chain: qs |= qset(WD[K])
    for q in sorted(qs):
        d = ctxlog(C, q) - (ctxlog(par, q) if par else 0)
        d = P['ctx_scale']*d
        if np.abs(d).max() < 1e-6: continue
        add_unit(C+q, C[0].isalpha(), d, 'ctx')

def dist(dct, floor=None):
    floor = P['eps'] if floor is None else floor
    p = np.zeros(NV)
    for c, w in dct.items(): p[T[c]] = w
    p = p/p.sum()
    return (1-floor)*p + floor*base
PUNCT = {
 '.': dist({' ':.95,'"':.02,'<eos>':.003}),
 '!': dist({' ':.90,'"':.05,'<eos>':.004}),
 '?': dist({' ':.90,'"':.06,'<eos>':.002}),
 ',': dist({' ':.985,'"':.01}),
 "'": dist({'s':.58,'t':.16,'m':.06,'l':.05,'r':.05,'d':.04,'v':.03,' ':.03}),
}
for c, p in PUNCT.items():
    add_unit(c, False, logv(p) - logv(base), 'punct')

COL_BD, COL_Q, COL_PHI = 288, 296, 297
NX = 301
special = []
def col(row, ch): return row*36+T[ch]
p_close = dist({' ':.72,'.':.12,',':.10,'!':.02,'?':.01,'<eos>':.01})
special.append(([(col(7,'"'),1.0),(COL_Q,-1.0)], 0.0, logv(p_close) - logv(pnode('')), 'close'))
dq = np.zeros(NV); dq[T['!']] = .9; dq[T['?']] = 1.0; dq[T['"']] = .7; dq[T[',']] = .2
special.append(([(COL_Q,1.0)], 0.0, dq, 'inq'))
for c, bump in (('.',3.0),('!',2.2),('?',2.2),(',',2.2)):
    d = np.zeros(NV); d[T['"']] = bump; d[T[' ']] = -0.5
    special.append(([(col(7,c),1.0),(COL_Q,1.0)], -1.0, d, 'qpunct'))
EOS_INC = [2.7,1.2,0.95,1.15]
for j, inc in enumerate(EOS_INC):
    d = np.zeros(NV); d[T['<eos>']] = inc
    for c in '.!?':
        special.append(([(col(7,c),1.0),(COL_PHI+j,1.0)], -1.0, d, 'eos%d'%j))

nsp = len(special)
allunits = []
for s, bd, delta, tag in units:
    L = len(s); cols = []
    for i, ch in enumerate(s):
        cols.append((col(8-L+i, ch), 1.0))
    if bd:
        cols.append((COL_BD + (7-L), 1.0)); b = -float(L)
    else:
        b = -float(L-1)
    allunits.append((cols, b, delta))
for cols, b, delta, tag in special:
    allunits.append((cols, b, delta))
N = len(allunits)
sys.stderr.write("units %d (special %d)\n" % (N, nsp))

base_logits = logv(base); base_logits[0] = -20; base_logits[1] = -20
base_logits = np.round(base_logits, 4)

ops = []
S = int(P['shard'])
nshard = int(math.ceil(N/S))
shard_ids = []
for k in range(nshard):
    chunk = allunits[k*S:(k+1)*S]
    H = len(chunk)
    sp = []; b1 = []; w2 = np.zeros((NV, H))
    for u, (cols, b, delta) in enumerate(chunk):
        for c, w in cols: sp.append([u, c, w])
        b1.append(b)
        d = np.array(delta, float).copy(); d[0] = 0; d[1] = 0
        w2[:, u] = np.round(d, 4)
    ops.append({"op":"set_param","name":f"u{k}.W1","pattern":{"sparse":sp,"shape":[H,NX]}})
    ops.append({"op":"set_param","name":f"u{k}.b1","value":b1})
    ops.append({"op":"set_param","name":f"u{k}.W2","value":w2.tolist()})
    ops.append({"op":"set_param","name":f"u{k}.b2","value":[0.0]*NV})
    ops.append({"op":"put_def","name":f"cell{k}","def":{"kind":"cell","act":"relu","params":{"W1":f"u{k}.W1","b1":f"u{k}.b1","W2":f"u{k}.W2","b2":f"u{k}.b2"}}})
    shard_ids.append(k)

bsp = []
for r in range(8):
    for t in range(NV):
        if t not in (T[' '], T['"'], 1):
            bsp.append([r, r*36+t, -1.0])
ops.append({"op":"set_param","name":"bd.W1","pattern":{"sparse":bsp,"shape":[8,288]}})
ops.append({"op":"set_param","name":"bd.b1","value":[1.0]*8})
ops.append({"op":"put_def","name":"cell_bd","def":{"kind":"cell","act":"identity","params":{"W1":"bd.W1","b1":"bd.b1"}}})
ops.append({"op":"set_param","name":"phi.W1","value":[[0.05]]*4})
ops.append({"op":"set_param","name":"phi.b1","value":[-10.0,-15.0,-20.0,-25.0]})
ops.append({"op":"put_def","name":"cell_phi","def":{"kind":"cell","act":"sigmoid","params":{"W1":"phi.W1","b1":"phi.b1"}}})
ops.append({"op":"set_param","name":"base","value":base_logits.tolist()})
nodes = [
 {"id":"cf","use":"op:reshape","in":{"x":"$ctx"},"args":{"shape":[288]}},
 {"id":"bd","use":"cell_bd","in":{"x":"cf"}},
 {"id":"iq","use":"op:slice","in":{"x":"cf"},"args":{"axis":0,"start":7*36+9,"stop":7*36+10}},
 {"id":"sq","use":"op:slice","in":{"x":"$state"},"args":{"axis":0,"start":0,"stop":1}},
 {"id":"qm","use":"op:mul","in":{"a":"iq","b":"sq"}},
 {"id":"qs","use":"op:scale","in":{"x":"qm"},"args":{"c":-2.0}},
 {"id":"q1","use":"op:add","in":{"a":"iq","b":"sq"}},
 {"id":"qn","use":"op:add","in":{"a":"q1","b":"qs"}},
 {"id":"ph","use":"cell_phi","in":{"x":"$step"}},
 {"id":"x","use":"op:concat","in":{"x0":"cf","x1":"bd","x2":"qn","x3":"ph"},"args":{"axis":0}},
 {"id":"bl","use":"param:base"},
]
SEEN = int(P['seen_on'])
if SEEN:
    nodes += [
     {"id":"r5","use":"op:slice","in":{"x":"cf"},"args":{"axis":0,"start":5*36,"stop":6*36}},
     {"id":"r6","use":"op:slice","in":{"x":"cf"},"args":{"axis":0,"start":6*36,"stop":7*36}},
     {"id":"r7","use":"op:slice","in":{"x":"cf"},"args":{"axis":0,"start":7*36,"stop":8*36}},
     {"id":"sv","use":"op:slice","in":{"x":"$state"},"args":{"axis":0,"start":1,"stop":1+1296*36}},
     {"id":"svm","use":"op:reshape","in":{"x":"sv"},"args":{"shape":[1296,36]}},
     {"id":"r5c","use":"op:expand","in":{"x":"r5"},"args":{"axis":1}},
     {"id":"r6r","use":"op:expand","in":{"x":"r6"},"args":{"axis":0}},
     {"id":"ks2","use":"op:mul","in":{"a":"r5c","b":"r6r"}},
     {"id":"ks","use":"op:reshape","in":{"x":"ks2"},"args":{"shape":[1296,1]}},
     {"id":"r7r","use":"op:expand","in":{"x":"r7"},"args":{"axis":0}},
     {"id":"wr","use":"op:mul","in":{"a":"ks","b":"r7r"}},
     {"id":"nsm","use":"op:maximum","in":{"a":"svm","b":"wr"}},
     {"id":"nsf","use":"op:reshape","in":{"x":"nsm"},"args":{"shape":[1296*36]}},
     {"id":"r6c","use":"op:expand","in":{"x":"r6"},"args":{"axis":1}},
     {"id":"r7q","use":"op:expand","in":{"x":"r7"},"args":{"axis":0}},
     {"id":"kp2","use":"op:mul","in":{"a":"r6c","b":"r7q"}},
     {"id":"kp","use":"op:reshape","in":{"x":"kp2"},"args":{"shape":[1,1296]}},
     {"id":"rd","use":"op:matmul","in":{"a":"kp","b":"nsm"}},
     {"id":"rdf","use":"op:reshape","in":{"x":"rd"},"args":{"shape":[36]}},
     {"id":"seenc","use":"cell_seen","in":{"x":"rdf"}},
     {"id":"stn","use":"op:concat","in":{"x0":"qn","x1":"nsf"},"args":{"axis":0}},
    ]
prev = "bl"
if SEEN:
    nodes.append({"id":"asn","use":"op:add","in":{"a":"bl","b":"seenc"}})
    prev = "asn"
for k in shard_ids:
    nodes.append({"id":f"y{k}","use":f"cell{k}","in":{"x":"x"}})
    nodes.append({"id":f"a{k}","use":"op:add","in":{"a":prev,"b":f"y{k}"}})
    prev = f"a{k}"
ops.append({"op":"put_def","name":"step_root","def":{"kind":"bundle","inputs":["ctx","prev","step","state"],"nodes":nodes,"outputs":{"logits":prev,"state":"stn" if SEEN else "qn"}}})
if SEEN:
    gd = [0.0]*NV
    for t in range(3, NV):
        gd[t] = P['seen_g'] if t >= 10 else P['seen_o']
    ops.append({"op":"set_param","name":"seen.W1","pattern":{"sparse":[[t,t,gd[t]] for t in range(NV)],"shape":[NV,NV]}})
    ops.append({"op":"set_param","name":"seen.b1","value":[0.0]*NV})
    ops.append({"op":"put_def","name":"cell_seen","def":{"kind":"cell","act":"identity","params":{"W1":"seen.W1","b1":"seen.b1"}}})
ops.append({"op":"delete_param","name":"boot_logits"})
ops.append({"op":"set_state_dim","value":1+(1296*36 if SEEN else 0)})
prop = {"author":"s1-solo","rationale":"Hand-typed lexicon trie + word-bigram context units + punctuation/quote/length units in sparse exact-suffix-match cells; additive log-prob deltas over a typed unigram base.","ops":ops}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proposal.json')
json.dump(prop, open(out,'w'))
print("wrote", N, "units,", nshard, "shards")
