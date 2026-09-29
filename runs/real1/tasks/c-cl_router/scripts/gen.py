import json
ops=[]
def W(ids,val,neg=False):
    r=[0.0]*36
    for i in ids: r[i]=val
    return [r]
def cell(name,ids,val,b=0.0):
    ops.append({"op":"set_param","name":f"cl_router.{name}.W1","value":W(ids,val)})
    ops.append({"op":"set_param","name":f"cl_router.{name}.b1","value":[b]})
    ops.append({"op":"put_def","name":f"cl_router.c_{name}","def":{"kind":"cell","act":"identity","params":{"W1":f"cl_router.{name}.W1","b1":f"cl_router.{name}.b1"}}})
L=list(range(10,36))
cell("let",L,1.0)
cell("bos",[1],3.0); cell("sp",[3],3.0); cell("pu",[4,5,6,7],3.0); cell("q",[9],3.0); cell("ap",[8],3.0)
cell("none",[1,3,4,5,6,7,8,9]+L,-1.0,1.0)
N=[]
def n(i,use,inp,args=None):
    d={"id":i,"use":use,"in":inp}
    if args: d["args"]=args
    N.append(d)
n("lets","cl_router.c_let",{"x":"$ctx"})
for r in (5,6,7):
    n(f"l{r}","op:slice",{"x":"lets"},{"axis":0,"start":r,"stop":r+1})
n("r7","op:slice",{"x":"$ctx"},{"axis":0,"start":7,"stop":8})
for k in ("bos","sp","pu","q","ap","none"):
    n(k,f"cl_router.c_{k}",{"x":"r7"})
n("l56","op:mul",{"a":"l5","b":"l6"})
n("l567","op:mul",{"a":"l56","b":"l7"})
n("long","op:scale",{"x":"l567"},{"c":3.0})
n("l7s","op:scale",{"x":"l7"},{"c":3.0})
n("short","op:sub",{"a":"l7s","b":"long"})
n("y1","op:add",{"a":"sp","b":"none"})
n("cat","op:concat",{"x0":"bos","x1":"y1","x2":"short","x3":"long","x4":"pu","x5":"q","x6":"ap"},{"axis":1})
n("flat","op:reshape",{"x":"cat"},{"shape":[7]})
ops.append({"op":"put_def","name":"cl_router","def":{"kind":"bundle","role":"cluster","inputs":["ctx"],"nodes":N,"outputs":{"y":"flat"}}})
json.dump({"author":"c-cl_router","rationale":"letter mask cell + class cells on row 7; length via product of letter flags rows 5..7","ops":ops},open("proposal.json","w"),indent=1)
