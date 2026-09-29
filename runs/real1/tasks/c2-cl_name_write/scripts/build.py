import json
P="cl_name_write."
nodes=[]
def op(id,name,ins,**args):
    n={"id":id,"use":"op:"+name,"in":ins}
    if args: n["args"]=args
    nodes.append(n); return id
def sl(id,x,axis,a,b): return op(id,"slice",{"x":x},axis=axis,start=a,stop=b)
def rs(id,x,shape): return op(id,"reshape",{"x":x},shape=shape)
def neg1(id,x):  # 1-x
    op(id+"_n","scale",{"x":x},c=-1.0); return op(id,"bias",{"x":id+"_n"},c=1.0)
# trigger: rows 2..7 = n,a,m,e,d,space
toks=[23,10,22,14,13,3]
prev=None
for k,t in enumerate(toks):
    r=2+k
    sl(f"rw{r}","$ctx",0,r,r+1)
    sl(f"t{r}",f"rw{r}",1,t,t+1)
    rs(f"s{r}",f"t{r}",[1])
    if prev is None: prev=f"s{r}"
    else: prev=op(f"m{r}","mul",{"a":prev,"b":f"s{r}"})
trig=prev
ntrig=neg1("ntrig",trig)
# row 7
sl("last","$ctx",0,7,8)
sl("last_l","last",1,10,36)
op("l7_","sum",{"x":"last_l"},axis=1)
rs("l7","l7_",[1])
# state pieces
sl("Bf","$nm",0,0,216); rs("B","Bf",[6,36])
sl("capf","$nm",0,216,217)
sl("L","$nm",0,217,224)
sl("L6","L",0,0,6)
sl("L6last","L",0,6,7)
op("w0","mul",{"a":"capf","b":"l7"})
op("w","mul",{"a":"w0","b":"ntrig"})
op("Lw","mul",{"a":"w","b":"L6"})
rs("Lwc","Lw",[6,1])
op("wr","mul",{"a":"Lwc","b":"last"})
op("Bsum","add",{"a":"B","b":"wr"})
rs("ntrig2","ntrig",[1,1])
op("Bn","mul",{"a":"Bsum","b":"ntrig2"})
rs("Bo","Bn",[216])
# L update
nl=neg1("nlast","L6last")
op("s","mul",{"a":"w","b":"nlast"})
ns=neg1("ns","s")
op("Lsh","shift",{"x":"L"},axis=0,k=1)
op("La","mul",{"a":"ns","b":"L"})
op("Lb","mul",{"a":"s","b":"Lsh"})
op("Lc","add",{"a":"La","b":"Lb"})
op("Ld","mul",{"a":"Lc","b":"ntrig"})
nodes.append({"id":"e0","use":"param:cl_name_write.e0"})
rs("e0f","e0",[7])
op("Le","mul",{"a":"trig1","b":"e0f"}) if False else None
op("Lt","mul",{"a":trig,"b":"e0f"})
op("Lo","add",{"a":"Ld","b":"Lt"})
# cap
op("cw","mul",{"a":"capf","b":"l7"})
op("capo","maximum",{"a":trig,"b":"cw"})
op("y","concat",{"x0":"Bo","x1":"capo","x2":"Lo"},axis=0)
cl={"kind":"bundle","role":"cluster","inputs":["ctx","nm"],"nodes":nodes,"outputs":{"y":"y"}}
prop={"author":"c2-cl_name_write","rationale":"Name-cache writer built purely from exact slice/mul/add/shift ops; one 7-element constant e0.",
 "ops":[{"op":"set_param","name":P+"e0","value":[1,0,0,0,0,0,0]},
        {"op":"put_def","name":"cl_name_write","def":cl}]}
json.dump(prop,open(__import__("os").path.join(__import__("os").path.dirname(__file__),"..","proposal.json"),"w"),indent=1)
