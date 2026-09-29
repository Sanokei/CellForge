import json
c=json.load(open('contract.json'))
T=c['data']['table']
NS='cl_wordstart.'
ops=[];nodes=[];N={}
def node(i,use,inn=None,args=None):
    d={"id":i,"use":use}
    if inn:d["in"]=inn
    if args:d["args"]=args
    nodes.append(d);return i
def row(r):
    k='row%d'%r
    if k not in N:N[k]=node(k,'op:slice',{"x":"$ctx"},{"axis":0,"start":r,"stop":r+1})
    return N[k]
def cell(r,col):
    k='c%d_%d'%(r,col)
    if k not in N:N[k]=node(k,'op:slice',{"x":row(r)},{"axis":1,"start":col,"stop":col+1})
    return N[k]
def bnd(r):
    k='b%d'%r
    if k not in N:
        s=node('l%d_s'%r,'op:slice',{"x":row(r)},{"axis":1,"start":8,"stop":36})
        m=node('l%d_m'%r,'op:sum',{"x":s},{"axis":1,"keepdims":True})
        n=node('l%d_n'%r,'op:neg',{"x":m})
        N[k]=node(k,'op:bias',{"x":n},{"c":1.0})
    return N[k]
def mulall(name,xs):
    cur=xs[0]
    for j,x in enumerate(xs[1:]):
        cur=node('%s_m%d'%(name,j),'op:mul',{"a":cur,"b":x})
    return cur
CH=["<pad>","<bos>","<eos>"," ",",",".","!","?","'",'"']+[chr(97+i) for i in range(26)]
sp=cell(7,3)
sent=node('sentsum','op:add',{"a":node('sq_a','op:add',{"a":cell(6,5),"b":cell(6,6)}),"b":cell(6,7)})
terms=[]
def addterm(name,gate,vals):
    v=[0.0]*36
    for ch,x in vals.items(): v[CH.index(ch)]=x
    ops.append({"op":"set_param","name":NS+"t_"+name,"value":v})
    p=node('p_'+name,'param:'+NS+'t_'+name)
    g=mulall('g_'+name,[sp,gate]) if not isinstance(gate,list) else mulall('g_'+name,[sp]+gate)
    terms.append(node('y_'+name,'op:mul',{"a":g,"b":p}))
addterm('sent',sent,T['@sent'])
addterm('comma',cell(6,4),T['@comma'])
for w,vals in T.items():
    if w.startswith('@'):continue
    k=len(w)
    gate=[cell(6-k+1+i,CH.index(ch)) for i,ch in enumerate(w)]
    if 6-k>=0: gate.append(bnd(6-k))
    addterm(w,gate,vals)
cat=node('cat','op:concat',dict(("x%d"%i,t) for i,t in enumerate(terms)),{"axis":0})
s=node('tot','op:sum',{"x":cat},{"axis":0})
ops.append({"op":"put_def","name":"cl_wordstart","def":{"kind":"bundle","role":"cluster","inputs":["ctx"],"nodes":nodes,"outputs":{"y":s}}})
json.dump({"author":"c-cl_wordstart","rationale":"Gated lookup: per key a product of exact one-hot slices and boundary indicators times a 36-vector param; summed.","ops":ops},open('proposal.json','w'))
print(len(terms),len(nodes))
