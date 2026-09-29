import json
c=json.load(open('runs/real1/tasks/c-cl_endpunct/contract.json'))['data']
NS='cl_endpunct.'
ops=[]; nodes=[]; cache={}
def node(id,use,inp=None,args=None):
    n={"id":id,"use":use}
    if inp is not None:n["in"]=inp
    if args:n["args"]=args
    nodes.append(n); return id
def cell_(name):
    return name
def elem(r,ch):
    k=('e',r,ch)
    if k not in cache:
        rr=node(f"r{r}",  "op:slice",{"x":"$ctx"},{"axis":0,"start":r,"stop":r+1}) if ('r',r) not in cache else cache[('r',r)]
        cache[('r',r)]=rr
        cache[k]=node(f"e{r}_{ch}","op:slice",{"x":rr},{"axis":1,"start":ch,"stop":ch+1})
    return cache[k]
def row(r):
    if ('r',r) not in cache: elem(r,0)
    return cache[('r',r)]
letter={};bound={}
for r in range(8):
    s=node(f"lr{r}","op:slice",{"x":row(r)},{"axis":1,"start":10,"stop":36})
    letter[r]=node(f"L{r}","op:sum",{"x":s},{"axis":1,"keepdims":True})
    ap=elem(r,8)
    t=node(f"la{r}","op:add",{"a":letter[r],"b":ap})
    t=node(f"ln{r}","op:neg",{"x":t})
    bound[r]=node(f"B{r}","op:bias",{"x":t},{"c":1.0})
A={}
acc=None
for L in range(1,8):
    cur=letter[8-L] if L==1 else node(f"Lp{L}","op:mul",{"a":A_run,"b":letter[8-L]})
    A_run=cur
    A[L]=node(f"A{L}","op:mul",{"a":cur,"b":bound[7-L]})
words={}
for k,ix in (("comma",4),("period",5),("excl",6)):
    for w,v in c[k].items():
        if len(w)<=7: words.setdefault(w,{})[ix]=v
terms=[]
for w,d in sorted(words.items()):
    L=len(w); m=A[L]
    for i,chh in enumerate(w):
        m=node(f"m_{w}_{i}","op:mul",{"a":m,"b":elem(8-L+i,10+ord(chh)-97)})
    vec=[0.0]*36
    for ix,v in d.items(): vec[ix]=v
    ops.append({"op":"set_param","name":NS+"v_"+w,"value":vec})
    p=node(f"p_{w}","param:"+NS+"v_"+w)
    terms.append(node(f"t_{w}","op:mul",{"a":m,"b":p}))
# q term
W=[[0.0] for _ in range(36)]; W[6]=[1.3]; W[7]=[1.2]; b=[0.0]*36; b[6]=-0.4; b[7]=-0.6
ops.append({"op":"set_param","name":NS+"q.W1","value":W})
ops.append({"op":"set_param","name":NS+"q.b1","value":b})
ops.append({"op":"put_def","name":NS+"c_q","def":{"kind":"cell","act":"identity","params":{"W1":NS+"q.W1","b1":NS+"q.b1"}}})
qc=node("qc",NS+"c_q",{"x":"$q"})
terms.append(node("qg","op:mul",{"a":qc,"b":letter[7]}))
acc=terms[0]
for i,t in enumerate(terms[1:]):
    acc=node(f"s{i}","op:add",{"a":acc,"b":t})
out=node("y","op:reshape",{"x":acc},{"shape":[36]})
ops.append({"op":"put_def","name":"cl_endpunct","def":{"kind":"bundle","role":"cluster","inputs":["ctx","q"],"nodes":nodes,"outputs":{"y":out}}})
json.dump({"author":"c-cl_endpunct","rationale":"exact anchored-word gating via slice/mul ops on ctx rows; one param vector per table word; q cell gated by row7 letter","ops":ops},open('runs/real1/tasks/c-cl_endpunct/proposal.json','w'))
print(len(words),len(nodes))
