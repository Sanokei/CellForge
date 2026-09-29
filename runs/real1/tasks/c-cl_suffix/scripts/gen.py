import json
P="cl_suffix."
def tokid(c): return 10+ord(c)-97
ops=[]; nodes=[]
def node(i,use,inp=None,args=None):
    n={"id":i,"use":use}
    if inp: n["in"]=inp
    if args: n["args"]=args
    nodes.append(n)
for r in range(3,8):
    node(f"r{r}","op:slice",{"x":"$ctx"},{"axis":0,"start":r,"stop":r+1,"step":1})
    node(f"lt{r}","op:slice",{"x":f"r{r}"},{"axis":1,"start":10,"stop":36,"step":1})
    node(f"L{r}","op:sum",{"x":f"lt{r}"},{"axis":1,"keepdims":True})
def T(r,c):
    i=f"t{r}{c}"
    if not any(n["id"]==i for n in nodes):
        node(i,"op:slice",{"x":f"r{r}"},{"axis":1,"start":tokid(c),"stop":tokid(c)+1,"step":1})
    return i
cnt=[0]
def mul(a,b):
    cnt[0]+=1; i=f"m{cnt[0]}"; node(i,"op:mul",{"a":a,"b":b}); return i
def prod(*xs):
    a=xs[0]
    for b in xs[1:]: a=mul(a,b)
    return a
def sumtok(r,cs):
    a=T(r,cs[0])
    for c in cs[1:]:
        cnt[0]+=1; i=f"a{cnt[0]}"; node(i,"op:add",{"a":a,"b":T(r,c)}); a=i
    return a
def sub(a,b):
    cnt[0]+=1; i=f"s{cnt[0]}"; node(i,"op:sub",{"a":a,"b":b}); return i
def delta(d,rest=0.0,over=None):
    v=[0.0]*36
    for k,x in d.items(): v[k]=x
    for i in range(10,36): v[i]+=rest
    for k,x in (over or {}).items(): v[k]+=x
    return v
SP,CM,PD,EX,QM=3,4,5,6,7
rules={
 "ing":(prod(T(7,'g'),T(6,'n'),T(5,'i'),"L4"),delta({SP:1.6,CM:.9,PD:.8,EX:.3,QM:.2},-.5)),
 "ed":(prod(T(7,'d'),T(6,'e'),"L5"),delta({SP:1.0,CM:.7,PD:.6,EX:.2},-.3)),
 "ly":(prod(T(7,'y'),T(6,'l'),"L5"),delta({SP:1.0,CM:.7,PD:.5},-.5)),
 "ful":(prod(T(7,'l'),T(6,'u'),T(5,'f'),"L4","L3"),delta({SP:1.2,CM:.6,PD:.5},-.3,{34:.5})),
 "est":(prod(T(7,'t'),T(6,'s'),T(5,'e'),"L4"),delta({SP:.7,CM:.3,PD:.3})),
 "s":(prod(T(7,'s'),sub("L6",sumtok(6,"suia")),"L5","L4"),delta({SP:.5,CM:.3,PD:.3})),
 "er":(prod(T(7,'r'),T(6,'e'),"L5","L4"),delta({SP:.4,CM:.3,PD:.3})),
 "y":(prod(T(7,'y'),sub("L6",sumtok(6,"aeiouyl")),"L5","L4"),delta({SP:.5,CM:.3,PD:.5})),
}
terms=[]
for k,(g,d) in rules.items():
    ops.append({"op":"set_param","name":P+"d_"+k,"value":d})
    node("w_"+k,"param:"+P+"d_"+k)
    node("g_"+k,"op:reshape",{"x":g},{"shape":[1]})
    node("p_"+k,"op:mul",{"a":"g_"+k,"b":"w_"+k}); terms.append("p_"+k)
a=terms[0]
for j,t in enumerate(terms[1:]):
    node(f"acc{j}","op:add",{"a":a,"b":t}); a=f"acc{j}"
ops.append({"op":"put_def","name":"cl_suffix","def":{"kind":"bundle","role":"cluster","inputs":["ctx"],"nodes":nodes,"outputs":{"y":a}}})
json.dump({"author":"c-cl_suffix","rationale":"Suffix gates from exact ctx row slices/products, each gate scales a typed 36-vector delta; summed.","ops":ops},open("proposal.json","w"),indent=1)
