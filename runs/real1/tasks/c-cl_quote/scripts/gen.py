import json
N=[];P={}
def node(i,use,inp=None,args=None):
    d={"id":i,"use":use}
    if inp: d["in"]=inp
    if args: d["args"]=args
    N.append(d); return i
def op(i,o,inp,args=None): return node(i,"op:"+o,inp,args)
def par(name,vals):
    P["cl_quote."+name]=vals; return node("p_"+name,"param:cl_quote."+name)
def vec(d):
    v=[0.0]*36
    for k,x in d.items(): v[k]=x
    return v
def sc(i,r,t):  # ctx[r][t] as [1]
    a=op(i+"a","slice",{"x":"$ctx"},{"axis":0,"start":r,"stop":r+1})
    b=op(i+"b","slice",{"x":a},{"axis":1,"start":t,"stop":t+1})
    return op(i,"reshape",{"x":b},{"shape":[1]})
def mul(i,a,b): return op(i,"mul",{"a":a,"b":b})
def add(i,a,b): return op(i,"add",{"a":a,"b":b})
def scale(i,a,c): return op(i,"scale",{"x":a},{"c":c})
# q gates
q1=op("q1","step",{"x":op("qb","bias",{"x":"$q"},{"c":-0.5})})
q0=op("q0","bias",{"x":scale("qn",q1,-1.0)},{"c":1.0})
dq=sc("dq",7,9); sp=sc("sp",7,3)
# row7 punctuation weight s3
row7=op("r7","reshape",{"x":op("r7a","slice",{"x":"$ctx"},{"axis":0,"start":7,"stop":8})},{"shape":[36]})
w3=par("w3",vec({4:1.0,5:0.6,6:1.6,7:1.6}))
s3=op("s3","sum",{"x":mul("s3m",row7,w3)},{"axis":0,"keepdims":True})
# unit9 scalar
t1=mul("t1",q1,s3)
c6=sc("c6",6,4); d5=sc("d5",5,13); m5=sc("m5",5,22)
amt=add("amt",add("amt1",scale("d21",d5,2.1),scale("m06",m5,0.6)),op("k04","bias",{"x":scale("z",d5,0.0)},{"c":0.4}))
g4=mul("g4",mul("g4a",q0,sp),c6)
t4=mul("t4",g4,amt)
r6s=add("r6s",add("r6a",sc("r65",6,5),sc("r66",6,6)),sc("r67",6,7))
g5=mul("g5",mul("g5a",q0,sp),r6s)
t5=scale("t5",g5,0.8)
t6=scale("t6",mul("t6a",q1,sp),-2.5)
tot=add("tot",add("ta",add("tb",t1,t4),t5),t6)
u9=par("u9",vec({9:1.0}))
y9=mul("y9",tot,u9)
# E
eo=mul("eo",q1,par("evec",vec({2:-6.0})))
# R1
qo={'i':.6,'l':.4,'y':.5,'w':.4,'c':.3,'h':.4,'o':.4,'n':.3,'t':.2,'a':-.2,'m':.2,'s':.2,'p':.2}
d1={10+ord(k)-97:v for k,v in qo.items()}; d1.update({9:-3,3:-3,4:-2,5:-2})
g1=mul("g1",q1,dq); r1=mul("r1",g1,par("v1",vec(d1)))
# R2
d2={3:.8,4:.3,5:.4,9:-3}; d2.update({i:-1.5 for i in range(10,36)})
g2=mul("g2",q0,dq); r2=mul("r2",g2,par("v2",vec(d2)))
s=add("y_a",add("y_b",add("y_c",eo,r1),r2),y9)
ops=[{"op":"set_param","name":k,"value":v} for k,v in P.items()]
ops.append({"op":"put_def","name":"cl_quote","def":{"kind":"bundle","role":"cluster","inputs":["ctx","q"],"nodes":N,"outputs":{"y":op("yout","reshape",{"x":s},{"shape":[36]})}}})
json.dump({"author":"c-cl_quote","rationale":"Gated sums of fixed 36-vectors; row features by slice/reshape; no cells needed.","ops":ops},open("proposal.json","w"),indent=1)
