# expands the contract's authoritative pair table (data.pairs / rest_letters) into a proposal
import json
c=json.load(open('contract.json'))['data']
CH=["<pad>","<bos>","<eos>"," ",",",".","!","?","'",'"']+[chr(97+i) for i in range(26)]
ix={ch:i for i,ch in enumerate(CH)}
N="cl_trigram."
ops=[]; nodes=[]
ops.append({"op":"set_param","name":N+"z1","value":[0.0]})
ops.append({"op":"set_param","name":N+"z36","value":[0.0]*36})
nodes.append({"id":"r6","use":"op:slice","in":{"x":"$ctx"},"args":{"axis":0,"start":6,"stop":7}})
nodes.append({"id":"r7","use":"op:slice","in":{"x":"$ctx"},"args":{"axis":0,"start":7,"stop":8}})
outs=[]
for k,(pair,d) in enumerate(c['pairs'].items()):
    a,b=pair
    for nm,l in (("a",a),("b",b)):
        v=[0.0]*36; v[ix[l]]=1.0
        ops.append({"op":"set_param","name":f"{N}s{nm}{k}","value":[v]})
        ops.append({"op":"put_def","name":f"{N}c_s{nm}{k}","def":{"kind":"cell","act":"identity","params":{"W1":f"{N}s{nm}{k}","b1":N+"z1"}}})
    dv=[0.0]*36
    for ch,x in d.items(): dv[ix[ch]]+=x
    r=c['rest_letters'].get(pair,0)
    if r:
        for i in range(10,36): dv[i]+=r
    ops.append({"op":"set_param","name":f"{N}d{k}","value":[[x] for x in dv]})
    ops.append({"op":"put_def","name":f"{N}c_d{k}","def":{"kind":"cell","act":"identity","params":{"W1":f"{N}d{k}","b1":N+"z36"}}})
    nodes.append({"id":f"a{k}","use":f"{N}c_sa{k}","in":{"x":"r6"}})
    nodes.append({"id":f"b{k}","use":f"{N}c_sb{k}","in":{"x":"r7"}})
    nodes.append({"id":f"g{k}","use":"op:mul","in":{"a":f"a{k}","b":f"b{k}"}})
    nodes.append({"id":f"y{k}","use":f"{N}c_d{k}","in":{"x":f"g{k}"}})
    outs.append(f"y{k}")
acc=outs[0]
for i,o in enumerate(outs[1:]):
    nodes.append({"id":f"s{i}","use":"op:add","in":{"a":acc,"b":o}}); acc=f"s{i}"
nodes.append({"id":"flat","use":"op:reshape","in":{"x":acc},"args":{"shape":[36]}})
ops.append({"op":"put_def","name":"cl_trigram","def":{"kind":"bundle","role":"cluster","inputs":["ctx"],"nodes":nodes,"outputs":{"y":"flat"}}})
json.dump({"author":"c-cl_trigram","rationale":"per-pair selector cells (one-hot on rows 6,7), product gate, [36] delta cell; summed","ops":ops},open('proposal.json','w'))
