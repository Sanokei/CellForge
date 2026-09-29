"""gen2: class-conditioned word trie (class-bigram) + typed context deltas + in-story pair memory.
Expands only lists/tables I typed (lexicon*.py, contexts*.py, classes.py); never reads dataset text."""
import json, math, sys, os
import numpy as np
from lexicon import TIERS, EXTRA_WORDS
from lexicon2 import TIERS2
from contexts import CTX
from contexts2 import CTX2
from contexts3 import CTX3
import classes as C_

V = ['<pad>','<bos>','<eos>',' ',',','.','!','?',"'",'"']+list("abcdefghijklmnopqrstuvwxyz")
T = {c:i for i,c in enumerate(V)}
T['^'] = 1
NV = 36
P = dict(K2=20.0, ctx_scale=0.8, eps=0.003, rank_decay=0.78, rank_top=30.0, shard=3000,
         beta=0.02, delta=1e-5, seen_g=1.6, seen_o=0.3, seen_on=1, pi_pow=1.0, out='proposal.json')
for a in sys.argv[1:]:
    k, v = a.split('=')
    P[k] = v if k == 'out' else float(v)
G, K = C_.G, C_.K

BASE_W = {' ':180,',':10,'.':11,'!':2,'?':1,"'":2,'"':4,'e':95,'t':70,'a':70,'o':55,'i':50,'n':50,'s':45,'h':45,'r':38,'d':35,'l':38,'u':22,'m':20,'w':22,'y':20,'c':16,'g':16,'f':15,'p':14,'b':14,'k':9,'v':7,'j':1.5,'x':1,'q':0.6,'z':0.6}
base = np.zeros(NV)
for c, w in BASE_W.items(): base[T[c]] = w
base[T['<eos>']] = 0.05
base /= base.sum()
FLOOR = 0.5*base.copy(); FLOOR[3:] += 0.5/33.0; FLOOR[:3] = 0; FLOOR[2] = base[2]; FLOOR /= FLOOR.sum()

E_DEF = {' ':.79,',':.07,'.':.09,'!':.02,'?':.01,"'":.005,'"':.015}
E_FUNC = {' ':.985,',':.01,'.':.002,'!':.001,'?':.001}
E_NAME = {' ':.72,',':.10,'.':.06,'!':.03,'?':.02,"'":.06,'"':.01}
E_SAID = {' ':.40,',':.57,'.':.01,'"':.02}
E_DAY  = {' ':.40,',':.42,'.':.13,'!':.02,'"':.03}
E_NOUN = {' ':.66,',':.09,'.':.20,'!':.03,'?':.01,"'":.01}
E_CONTR= {' ':.86,',':.05,'.':.05,'!':.02,'?':.01,'"':.01}
E_ADV  = {' ':.80,',':.10,'.':.05,'!':.03,'?':.01,'"':.01}
E_ING  = {' ':.70,',':.08,'.':.16,'!':.03,'?':.01,'"':.02}
FUNC = set("the a an and to was were is are of in on at with for from by as that this his her their my your our its it he she they we i you but so or if than then when while because not no do did does can could would will should have has had be been am up out down off over into onto very too just all some any each every more most many much such what who how why where which there here them him me us like about after before again around through under until".split())
NOUNS = set("day time home park school friends friend toys toy ball game games fun tree garden house room dog cat bird sun water forest river lake beach sand story party cake food lunch dinner bed box car boat book night morning world sky way place things thing family".split())
E_OVR = {'said':E_SAID,'asked':E_SAID,'replied':E_SAID,'shouted':E_SAID,'day':E_DAY,'time':E_DAY,'night':E_DAY,'morning':E_DAY}

NAMES = set("lily tim mia sam ben anna max sue jack tom lucy sara amy bob spot sally jane john kate emma ella lila zoe leo jim tommy timmy molly daisy rosie fluffy buddy sparky lulu tiny bella bobo benny ollie ann joe joey kitty sunny grace luna ruby rose jill jenny betty nick nate neil pete paul peter david dan danny alex alice ava eva eve ivy jake jason josh kim kevin larry lisa lola lucas mark mary mike nina noah oliver oscar owen ray rex rick sarah sophie tess tina tony vicky will zack boo shiver tweet patches whiskers muffin cookie peanut pepper shadow rusty bubbles goldie nemo mr mrs miss".split())

# ---- lexicon (weights from my tier lists)
CTXALL = dict(CTX); CTXALL.update(CTX2); CTXALL.update(CTX3)
CTXALL = {k: v for k, v in CTXALL.items() if k[-1] in ' "^'}
words = {}
for wt, s_ in TIERS2.items():
    for w in s_.split(): words[w] = max(words.get(w,0), float(wt))
for wt, s_ in TIERS.items():
    for w in s_.split(): words[w] = max(words.get(w,0), float(wt))
for w in EXTRA_WORDS: words.setdefault(w, 3.0)
for w in NAMES: words[w] = max(words.get(w, 0), 5.0 if w not in "lily tim mia".split() else 80.0)
for c, s_ in CTXALL.items():
    for w in s_.split():
        if w != '"': words.setdefault(w, 3.0)
words.pop('"', None)
for w in "the's".split(): words.pop(w, None)

def endclass(w):
    if w in E_OVR: return E_OVR[w]
    if "'" in w: return E_CONTR
    if w in NAMES: return E_NAME
    if w in FUNC: return E_FUNC
    if w in NOUNS: return E_NOUN
    if w.endswith('ly') and len(w) > 4: return E_ADV
    if w.endswith('ing'): return E_ING
    return E_DEF

grp = {w: C_.GI[C_.classify(w, NAMES)] for w in words}
GT = np.zeros(G)
for w, wt in words.items(): GT[grp[w]] += wt
QT = C_.GI['QT']; GT[QT] = 1.0

def group_masses(p):
    """[G,NV]: for each group, sum over words in that group consistent with prefix p of (u/U_g) at the next char"""
    m = np.zeros((G, NV))
    for w, wt in words.items():
        if w.startswith(p):
            g = grp[w]; u = wt/GT[g]
            if len(w) == len(p):
                for ch, pr in endclass(w).items(): m[g, T[ch]] += u*pr
            else:
                m[g, T[w[len(p)]]] += u
    if p == '': m[QT, T['"']] = 1.0
    return m

PIm = np.zeros((K, G))
for s_, row in C_.PI.items():
    r = np.array(row, float)**P['pi_pow']; PIm[C_.SI[s_]] = r/r.sum()

def logv(p): return np.log(np.maximum(p, 1e-12))
def zvec(m, pi):
    q = pi @ m
    s = q.sum()
    return q + P['beta']*s*FLOOR + P['delta']*base
def pnorm(m, pi):
    z = zvec(m, pi); return z/z.sum()

mcache = {}
def gm(p):
    if p not in mcache: mcache[p] = group_masses(p)
    return mcache[p]
ZERO = np.zeros((G, NV))
def pnode(p, pi):
    m = gm(p) if len(p) <= 8 else ZERO
    return pnorm(m, pi)

units = []          # (string, bd, out_vector[G*NV])   trie cell
dunits = []         # (string, bd, delta[NV], tag)      delta cell
seen = {}
def add_unit(lst, s, bd, vec, tag):
    if len(s) > 8: s = s[-8:]; bd = False
    if bd and len(s) > 7: bd = False
    key = (s, bd, id(lst))
    if key in seen: return
    seen[key] = 1
    lst.append((s, bd, vec, tag))

prefixes = set()
for w in words:
    for i in range(0, len(w)+1): prefixes.add(w[:i])
for p in sorted(prefixes, key=lambda x:(len(x), x)):
    if len(p) > 8: continue
    add_unit(units, p, True, gm(p).reshape(-1), 'trie')

# --- typed contexts -> class of the context's last word
def ctx_state(Cx):
    if Cx == '^': return 'S'
    if Cx[-1] == '"': return 'Q'
    if Cx.endswith('. ') or Cx.endswith('! ') or Cx.endswith('? '): return 'S'
    if Cx.endswith(', '): return 'CM'
    last = Cx.strip().split(' ')[-1]
    g = C_.classify(last, NAMES)
    return g if g in C_.SI else 'NOUN'
def pi_of(Cx): return PIm[C_.SI[ctx_state(Cx)]]

def build_wd(succ):
    wd = {}
    for i, w in enumerate(succ.split()): wd[w] = wd.get(w, 0) + P['rank_top']*P['rank_decay']**i
    return wd
WD = {Cx: build_wd(s_) for Cx, s_ in CTXALL.items()}
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
def qset(wd):
    qs = set()
    for w in wd:
        if w == '"': qs.add(''); continue
        for i in range(0, min(len(w), 6)+1): qs.add(w[:i])
    return qs
def ctxlog(Cx, q):
    wd = WD[Cx]
    m = node_masses(wd, q); M = m.sum()
    if M == 0: return np.zeros(NV)
    mu = M/(M+P['K2'])
    pn = pnode(q, pi_of(Cx))
    pc = mu*m/M + (1-mu)*pn
    return logv(pc) - logv(pn)
def parent(Cx):
    best = None
    for Kx in CTXALL:
        if Kx != Cx and Cx.endswith(Kx) and (len(Kx) == len(Cx) or Cx[-len(Kx)-1] in ' "'):
            if best is None or len(Kx) > len(best): best = Kx
    return best
for Cx in CTXALL:
    chain = [Cx]
    while parent(chain[-1]) is not None: chain.append(parent(chain[-1]))
    par = chain[1] if len(chain) > 1 else None
    qs = set()
    for Kx in chain: qs |= qset(WD[Kx])
    for q in sorted(qs):
        d = ctxlog(Cx, q) - (ctxlog(par, q) if par else 0)
        d = P['ctx_scale']*d
        if np.abs(d).max() < 1e-6: continue
        add_unit(dunits, Cx+q, Cx[0].isalpha(), d, 'ctx')

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
    add_unit(dunits, c, False, logv(p) - logv(base), 'punct')

COL_BD, COL_Q, COL_PHI = 288, 296, 297
NX = 301
def col(row, ch): return row*36+T[ch]
special = []
p_close = dist({' ':.72,'.':.12,',':.10,'!':.02,'?':.01,'<eos>':.01})
special.append(([(col(7,'"'),1.0),(COL_Q,-1.0)], 0.0, logv(p_close) - logv(pnode('', PIm[C_.SI['S']])), 'close'))
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

def unit_cols(s, bd):
    L = len(s); cols = []
    for i, ch in enumerate(s): cols.append((col(8-L+i, ch), 1.0))
    if bd: cols.append((COL_BD + (7-L), 1.0)); b = -float(L)
    else: b = -float(L-1)
    return cols, b

# --- class-completion units (write the state class when a word/punctuation completes)
cunits = []
def cls_vec(name):
    v = np.zeros(K); v[C_.SI[name]] = 1.0; return v
cseen = set()
for w in words:
    g = C_.GROUPS[grp[w]]
    if g not in C_.SI: continue
    s = w + ' '
    bd = True
    if len(s) > 7: bd = False
    if len(s) > 8: s = s[-8:]
    if (s, bd) in cseen: continue
    cseen.add((s, bd))
    cols, b = unit_cols(s, bd)
    cunits.append((cols, b, cls_vec(g)))
for pstr, st in ((". ",'S'),("! ",'S'),("? ",'S'),(", ",'CM'),("^",'S')):
    cols, b = unit_cols(pstr, False); cunits.append((cols, b, cls_vec(st)))
cunits.append(([(col(7,'"'),1.0),(COL_Q,1.0)], -1.0, cls_vec('Q')))

sys.stderr.write("trie %d  delta %d  special %d  class %d\n" % (len(units), len(dunits), len(special), len(cunits)))

# ---- assemble proposal
ops = []
def emit_cell(name, items, out_dim):
    """items: list of (cols, bias, outvec)"""
    S = int(P['shard']); ids = []
    for k in range(int(math.ceil(len(items)/S))):
        chunk = items[k*S:(k+1)*S]; H = len(chunk)
        sp = []; b1 = []; w2 = np.zeros((out_dim, H))
        for u, (cols, b, vec) in enumerate(chunk):
            for c, w in cols: sp.append([u, c, w])
            b1.append(b); w2[:, u] = np.round(np.asarray(vec, float), 8)
        nm = f"{name}{k}"
        ops.append({"op":"set_param","name":f"{nm}.W1","pattern":{"sparse":sp,"shape":[H,NX]}})
        ops.append({"op":"set_param","name":f"{nm}.b1","value":b1})
        ops.append({"op":"set_param","name":f"{nm}.W2","value":w2.tolist()})
        ops.append({"op":"set_param","name":f"{nm}.b2","value":[0.0]*out_dim})
        ops.append({"op":"put_def","name":f"cell_{nm}","def":{"kind":"cell","act":"relu","params":{"W1":f"{nm}.W1","b1":f"{nm}.b1","W2":f"{nm}.W2","b2":f"{nm}.b2"}}})
        ids.append(nm)
    return ids

def with_cols(lst):
    out = []
    for s, bd, vec, tag in lst:
        cols, b = unit_cols(s, bd); out.append((cols, b, vec))
    return out
trie_ids = emit_cell('t', with_cols(units), G*NV)
d_items = with_cols(dunits) + [(c, b, d) for c, b, d, tag in special]
d_ids = emit_cell('d', d_items, NV)
c_ids = emit_cell('c', cunits, K)

bsp = []
for r in range(8):
    for t in range(NV):
        if t not in (T[' '], T['"'], 1): bsp.append([r, r*36+t, -1.0])
ops.append({"op":"set_param","name":"bd.W1","pattern":{"sparse":bsp,"shape":[8,288]}})
ops.append({"op":"set_param","name":"bd.b1","value":[1.0]*8})
ops.append({"op":"put_def","name":"cell_bd","def":{"kind":"cell","act":"identity","params":{"W1":"bd.W1","b1":"bd.b1"}}})
ops.append({"op":"set_param","name":"phi.W1","value":[[0.05]]*4})
ops.append({"op":"set_param","name":"phi.b1","value":[-10.0,-15.0,-20.0,-25.0]})
ops.append({"op":"put_def","name":"cell_phi","def":{"kind":"cell","act":"sigmoid","params":{"W1":"phi.W1","b1":"phi.b1"}}})
# log approximation cell (piecewise linear, knots by factor 2 from 1e-9 up to 2)
knots = [0.0] + [1e-9*2**i for i in range(0, 32)]
knots = [k_ for k_ in knots if k_ <= 2.5]
Lk = [math.log(knots[1]) - 1.0] + [math.log(k_) for k_ in knots[1:]]
slopes = [(Lk[i+1]-Lk[i])/(knots[i+1]-knots[i]) for i in range(len(knots)-1)]
a_ = [slopes[0]] + [slopes[i]-slopes[i-1] for i in range(1, len(slopes))]
ops.append({"op":"set_param","name":"log.W1","value":[[1.0]]*len(a_)})
ops.append({"op":"set_param","name":"log.b1","value":[-k_ for k_ in knots[:len(a_)]]})
ops.append({"op":"set_param","name":"log.W2","value":[a_]})
ops.append({"op":"set_param","name":"log.b2","value":[Lk[0]]})
ops.append({"op":"put_def","name":"cell_log","def":{"kind":"cell","act":"relu","params":{"W1":"log.W1","b1":"log.b1","W2":"log.W2","b2":"log.b2"}}})
ops.append({"op":"set_param","name":"Pi","value":PIm.tolist()})
ops.append({"op":"set_param","name":"Fl","value":(P['beta']*FLOOR).tolist()})
ops.append({"op":"set_param","name":"zb","value":(P['delta']*base).tolist()})
padm = np.zeros(NV); padm[0] = -30; padm[1] = -30
ops.append({"op":"set_param","name":"padm","value":padm.tolist()})

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
 # class state
 {"id":"co","use":"op:slice","in":{"x":"$state"},"args":{"axis":0,"start":1,"stop":1+K}},
]
def add_chain(ids, prefix, out_name, cellprefix='cell_'):
    prev = None
    for i, nm in enumerate(ids):
        nodes.append({"id":f"{prefix}y{i}","use":f"{cellprefix}{nm}","in":{"x":"x"}})
        if prev is None: prev = f"{prefix}y{i}"
        else:
            nodes.append({"id":f"{prefix}a{i}","use":"op:add","in":{"a":prev,"b":f"{prefix}y{i}"}}); prev = f"{prefix}a{i}"
    return prev
ycls = add_chain(c_ids, 'c', 'ycls')
nodes += [
 {"id":"fsum","use":"op:sum","in":{"x":ycls},"args":{"axis":0,"keepdims":True}},
 {"id":"fco","use":"op:mul","in":{"a":"fsum","b":"co"}},
 {"id":"dcl","use":"op:sub","in":{"a":ycls,"b":"fco"}},
 {"id":"cn","use":"op:add","in":{"a":"co","b":"dcl"}},
 {"id":"cnr","use":"op:expand","in":{"x":"cn"},"args":{"axis":0}},
 {"id":"pip","use":"param:Pi"},
 {"id":"pig","use":"op:matmul","in":{"a":"cnr","b":"pip"}},
]
ytr = add_chain(trie_ids, 't', 'ytr')
nodes += [
 {"id":"mg","use":"op:reshape","in":{"x":ytr},"args":{"shape":[G, NV]}},
 {"id":"qg","use":"op:matmul","in":{"a":"pig","b":"mg"}},
 {"id":"qv","use":"op:reshape","in":{"x":"qg"},"args":{"shape":[NV]}},
 {"id":"ss","use":"op:sum","in":{"x":"qv"},"args":{"axis":0,"keepdims":True}},
 {"id":"fl","use":"param:Fl"},
 {"id":"sf","use":"op:mul","in":{"a":"ss","b":"fl"}},
 {"id":"zb","use":"param:zb"},
 {"id":"z1","use":"op:add","in":{"a":"qv","b":"sf"}},
 {"id":"z2","use":"op:add","in":{"a":"z1","b":"zb"}},
 {"id":"zr","use":"op:reshape","in":{"x":"z2"},"args":{"shape":[NV,1]}},
 {"id":"lgr","use":"cell_log","in":{"x":"zr"}},
 {"id":"lg","use":"op:reshape","in":{"x":"lgr"},"args":{"shape":[NV]}},
]
ydl = add_chain(d_ids, 'd', 'ydl')
nodes += [
 {"id":"pm","use":"param:padm"},
 {"id":"l1","use":"op:add","in":{"a":"lg","b":ydl}},
 {"id":"l2","use":"op:add","in":{"a":"l1","b":"pm"}},
]
final = "l2"
SEEN = int(P['seen_on'])
if SEEN:
    nodes += [
     {"id":"sv","use":"op:slice","in":{"x":"$state"},"args":{"axis":0,"start":1+K,"stop":1+K+1296*36}},
     {"id":"svm","use":"op:reshape","in":{"x":"sv"},"args":{"shape":[1296,36]}},
     {"id":"r5","use":"op:slice","in":{"x":"cf"},"args":{"axis":0,"start":5*36,"stop":6*36}},
     {"id":"r6","use":"op:slice","in":{"x":"cf"},"args":{"axis":0,"start":6*36,"stop":7*36}},
     {"id":"r7","use":"op:slice","in":{"x":"cf"},"args":{"axis":0,"start":7*36,"stop":8*36}},
     {"id":"r5c","use":"op:expand","in":{"x":"r5"},"args":{"axis":1}},
     {"id":"r6r","use":"op:expand","in":{"x":"r6"},"args":{"axis":0}},
     {"id":"ks2","use":"op:mul","in":{"a":"r5c","b":"r6r"}},
     {"id":"ks","use":"op:reshape","in":{"x":"ks2"},"args":{"shape":[1296,1]}},
     {"id":"r7r","use":"op:expand","in":{"x":"r7"},"args":{"axis":0}},
     {"id":"wr","use":"op:mul","in":{"a":"ks","b":"r7r"}},
     {"id":"nsm","use":"op:maximum","in":{"a":"svm","b":"wr"}},
     {"id":"nsf","use":"op:reshape","in":{"x":"nsm"},"args":{"shape":[1296*36]}},
     {"id":"r6c","use":"op:expand","in":{"x":"r6"},"args":{"axis":1}},
     {"id":"kp2","use":"op:mul","in":{"a":"r6c","b":"r7r"}},
     {"id":"kp","use":"op:reshape","in":{"x":"kp2"},"args":{"shape":[1,1296]}},
     {"id":"rd","use":"op:matmul","in":{"a":"kp","b":"nsm"}},
     {"id":"rdf","use":"op:reshape","in":{"x":"rd"},"args":{"shape":[36]}},
     {"id":"seenc","use":"cell_seen","in":{"x":"rdf"}},
     {"id":"l3","use":"op:add","in":{"a":"l2","b":"seenc"}},
    ]
    final = "l3"
    gd = [0.0]*NV
    for t in range(3, NV): gd[t] = P['seen_g'] if t >= 10 else P['seen_o']
    ops.append({"op":"set_param","name":"seen.W1","pattern":{"sparse":[[t,t,gd[t]] for t in range(NV)],"shape":[NV,NV]}})
    ops.append({"op":"set_param","name":"seen.b1","value":[0.0]*NV})
    ops.append({"op":"put_def","name":"cell_seen","def":{"kind":"cell","act":"identity","params":{"W1":"seen.W1","b1":"seen.b1"}}})
    nodes.append({"id":"stn","use":"op:concat","in":{"x0":"qn","x1":"cn","x2":"nsf"},"args":{"axis":0}})
    state_out = "stn"
else:
    nodes.append({"id":"stn","use":"op:concat","in":{"x0":"qn","x1":"cn"},"args":{"axis":0}})
    state_out = "stn"
ops.append({"op":"put_def","name":"step_root","def":{"kind":"bundle","inputs":["ctx","prev","step","state"],"nodes":nodes,"outputs":{"logits":final,"state":state_out}}})
ops.append({"op":"delete_param","name":"boot_logits"})
ops.append({"op":"set_state_dim","value":1+K+(1296*36 if SEEN else 0)})
prop = {"author":"s1-solo","rationale":"Class-bigram word trie (typed lexicon, typed word classes and class transitions, log via piecewise-linear cell) + typed context deltas + punctuation/quote/length units + in-story pair memory in recurrent state.","ops":ops}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', P['out'])
json.dump(prop, open(out,'w'))
print("wrote", out)
