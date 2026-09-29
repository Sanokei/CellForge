# expands the contract's authoritative pair table into a proposal (run from the task dir)
import json
c=json.load(open('contract.json'))['data']
CH=["<pad>","<bos>","<eos>"," ",",",".","!","?","'",'"']+[chr(97+i) for i in range(26)]
ix={ch:i for i,ch in enumerate(CH)}
N="cl_trigram."
ops=[{"op":"set_param","name":N+"z1","value":[0.0]}]
nodes=[]
for n,r in (("6",6),("7",7)):
    nodes.append({"id":"s"+n,"use":"op:slice","in":{"x":"$ctx"},"args":{"axis":0,"start":r,"stop":r+1}})
    nodes.append({"id":"r"+n,"use":"op:reshape","in":{"x":"s"+n},"args":{"shape":[36]}})
# 26 shared letter selectors: W1 [1,36] one-hot, b1 [1] -> 37 slots
for i in range(26):
    v=[0.0]*36; v[10+i]=1.0
    ops.append({"op":"set_param","name":f"{N}sel{i}","value":[v]})
    ops.append({"op":"put_def","name":f"{N}c_sel{i}","def":{"kind":"cell","act":"identity","params":{"W1":f"{N}sel{i}","b1":N+"z1"}}})
outs=[]
for k,(pair,d) in enumerate(c['pairs'].items()):
    a,b=pair
    dv=[0.0]*36
    for ch,x in d.items(): dv[ix[ch]]+=x
    r=c['rest_letters'].get(pair,0)
    if r:
        for i in range(10,36): dv[i]+=r
    ops.append({"op":"set_param","name":f"{N}d{k}","value":dv})
    ia,ib=ix[a]-10,ix[b]-10
    nodes.append({"id":f"a{k}","use":f"{N}c_sel{ia}","in":{"x":"r6"}})
    nodes.append({"id":f"b{k}","use":f"{N}c_sel{ib}","in":{"x":"r7"}})
    nodes.append({"id":f"g{k}","use":"op:mul","in":{"a":f"a{k}","b":f"b{k}"}})
    nodes.append({"id":f"gr{k}","use":"op:reshape","in":{"x":f"g{k}"},"args":{"shape":[1]}})
    nodes.append({"id":f"dp{k}","use":f"param:{N}d{k}"})
    nodes.append({"id":f"y{k}","use":"op:mul","in":{"a":f"gr{k}","b":f"dp{k}"}})
    outs.append(f"y{k}")
acc=outs[0]
for i,o in enumerate(outs[1:]):
    nodes.append({"id":f"t{i}","use":"op:add","in":{"a":acc,"b":o}}); acc=f"t{i}"
ops.append({"op":"put_def","name":"cl_trigram","def":{"kind":"bundle","role":"cluster","inputs":["ctx"],"nodes":nodes,"outputs":{"y":acc}}})
json.dump({"author":"c2-cl_trigram","rationale":"26 shared letter-selector cells (37 slots), pair gate = mul of selectors on rows 6,7, times a per-pair [36] delta param; summed","ops":ops},open('proposal.json','w'))
