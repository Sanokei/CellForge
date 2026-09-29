"""Expands hand-typed bigram weights (typed from knowledge of English children's stories) into proposal JSON.
Only normalisation (divide by row sum) and log are computed; no corpus is read."""
import json, math
CH = ["<pad>","<bos>","<eos>"," ",",",".","!","?","'",'"']+[chr(97+i) for i in range(26)]
IX = {c:i for i,c in enumerate(CH)}
def parse(s):
    d={}
    for t in s.split():
        k,v=t.rsplit(":",1)
        k={"S":" ","C":",","P":".","X":"!","Q":"?","A":"'","D":'"',"E":"<eos>"}.get(k,k) if len(k)==1 and k in "SCPXQADE" and k not in "" else k
        d[k]=float(v)
    return d
# note: letters a,d,e,... clash with symbol letters, so symbols use uppercase and letters lowercase
R={}
R['<bos>']="o:.5 t:.15 l:.08 m:.06 s:.06 a:.05 b:.04 w:.02 h:.02 j:.01 f:.01 e:.005 p:.005 c:.005 d:.005 g:.003 i:.003 n:.003 r:.003 k:.002 y:.002"
R[' ']="t:.16 a:.12 s:.08 h:.07 w:.07 b:.06 l:.05 m:.05 f:.04 c:.04 d:.04 o:.04 g:.03 p:.03 i:.03 e:.02 n:.02 r:.02 k:.01 y:.01 j:.01 u:.005 v:.005 q:.001 x:.0002 z:.0005 D:.06 A:.002"
R[',']="S:.82 D:.17"
R['.']="S:.84 E:.07 D:.09"
R['!']="S:.55 D:.43 E:.02"
R['?']="S:.55 D:.43 E:.02"
R["'"]="s:.40 t:.33 m:.06 l:.06 r:.04 v:.04 d:.06"
R['"']="S:.30 t:.05 h:.04 i:.04 w:.03 l:.03 a:.05 y:.03 o:.02 s:.04 m:.03 c:.03 d:.03 b:.03 n:.02 g:.02 f:.02 p:.02 j:.01 r:.02 e:.02 k:.005 v:.005 u:.005 C:.04 P:.05 X:.03 Q:.03 E:.03"
R['a']="n:.17 t:.10 l:.09 r:.07 s:.06 d:.05 y:.04 m:.04 k:.03 v:.03 b:.03 g:.03 i:.04 p:.03 c:.03 u:.02 w:.03 h:.01 f:.01 S:.07 C:.01 P:.02 A:.005 e:.01 a:.005 o:.003 D:.002"
R['b']="e:.30 a:.14 o:.12 i:.08 u:.09 r:.07 l:.06 y:.06 S:.01 b:.02 s:.02 t:.01 j:.003"
R['c']="a:.22 o:.20 h:.12 e:.10 l:.06 r:.07 k:.07 i:.05 u:.04 t:.03 y:.01 S:.005 c:.02 s:.01"
R['d']="S:.35 a:.09 e:.12 i:.08 o:.07 r:.06 y:.04 u:.04 C:.04 P:.05 d:.03 n:.01 s:.01 l:.01 A:.005 X:.003 D:.003"
R['e']="S:.33 C:.04 P:.04 X:.005 Q:.003 D:.005 A:.002 d:.10 r:.09 s:.07 n:.06 a:.05 l:.05 v:.04 t:.03 m:.03 e:.02 c:.02 w:.015 p:.015 f:.01 i:.01 b:.01 g:.01 y:.01 x:.01 h:.005 k:.005 o:.005 u:.003 z:.002"
R['f']="o:.22 r:.16 i:.13 e:.12 a:.10 u:.08 l:.07 S:.04 t:.02 f:.03 y:.01 C:.005 P:.005"
R['g']="S:.55 C:.06 P:.08 e:.06 r:.05 h:.05 o:.03 a:.03 l:.02 i:.02 u:.02 g:.01 s:.01 y:.005"
R['h']="e:.30 a:.12 i:.08 o:.07 S:.10 t:.06 C:.02 P:.03 r:.04 u:.03 y:.01 l:.005"
R['i']="n:.22 t:.12 s:.09 l:.09 d:.07 c:.06 e:.07 g:.04 m:.04 r:.04 f:.04 v:.03 o:.03 a:.02 k:.02 p:.01 b:.01 z:.005 x:.003 S:.01 C:.003 P:.003 q:.003"
R['j']="u:.35 o:.25 a:.18 e:.15 i:.03 S:.003"
R['k']="e:.30 i:.17 S:.20 y:.05 s:.05 C:.02 P:.03 n:.05 l:.03 a:.03 o:.02 b:.01"
R['l']="l:.16 y:.14 e:.13 i:.12 a:.11 S:.10 o:.08 d:.05 u:.03 f:.02 C:.02 P:.03 t:.03 s:.02 k:.03 p:.01 v:.01 m:.01 w:.01"
R['m']="e:.22 a:.16 o:.14 i:.09 y:.09 S:.09 m:.06 u:.06 p:.04 b:.02 C:.02 P:.02 s:.02 n:.01 l:.01"
R['n']="g:.22 S:.22 d:.12 t:.06 e:.08 C:.04 P:.06 y:.03 o:.04 a:.02 i:.02 s:.03 c:.02 k:.02 n:.01 u:.01 A:.02 l:.005 f:.005 v:.005"
R['o']="u:.14 n:.13 r:.12 m:.09 w:.08 o:.08 t:.07 l:.06 f:.06 d:.05 v:.04 p:.04 s:.04 S:.07 C:.01 P:.015 c:.03 b:.02 a:.01 y:.01 e:.01 g:.02 i:.01 k:.01 x:.002 h:.003 A:.003"
R['p']="e:.17 a:.13 l:.13 o:.12 p:.10 r:.09 i:.06 u:.05 S:.04 h:.04 t:.03 s:.02 y:.02 C:.003 P:.005"
R['q']="u:.97"
R['r']="e:.22 S:.16 y:.08 a:.10 i:.10 o:.09 s:.06 t:.05 d:.04 n:.04 C:.02 P:.03 l:.03 k:.02 u:.02 m:.02 r:.03 p:.02 g:.02 v:.01 c:.01 b:.01 h:.005 f:.01 w:.005"
R['s']="S:.28 t:.14 e:.10 o:.08 h:.07 a:.06 i:.06 s:.06 p:.04 C:.04 P:.05 u:.03 c:.03 l:.03 k:.03 m:.02 n:.01 w:.02 y:.01 A:.01 q:.005 b:.005"
R['t']="h:.22 S:.17 o:.10 e:.10 i:.08 a:.05 C:.03 P:.05 t:.04 r:.05 u:.03 y:.03 l:.03 s:.02 w:.01"
R['u']="r:.14 s:.12 t:.10 n:.10 l:.09 p:.07 m:.07 g:.05 c:.05 d:.05 e:.04 b:.04 i:.03 a:.03 S:.01 o:.01 y:.005 f:.01 k:.01 C:.002 P:.002 z:.003 v:.01"
R['v']="e:.55 i:.15 a:.10 o:.07 y:.01 S:.01 u:.01 r:.02 l:.01"
R['w']="a:.25 h:.14 i:.14 e:.12 o:.13 n:.04 r:.05 S:.02 l:.02 s:.02 u:.01 y:.01 C:.002"
R['x']="t:.12 p:.12 S:.18 c:.12 i:.10 a:.06 e:.08 y:.05 o:.03 h:.03 C:.03 P:.04"
R['y']="S:.42 C:.09 P:.12 A:.01 s:.05 X:.01 e:.04 o:.03 i:.03 a:.03 m:.02 b:.02 l:.02 w:.01 D:.02 Q:.005 p:.01 t:.01 d:.01 n:.01 f:.01 g:.01 h:.005 c:.005 u:.005 k:.003 r:.005"
R['z']="e:.28 z:.18 i:.10 a:.10 y:.09 o:.08 S:.06 u:.03 l:.04 C:.01 P:.01"
SYM={"S":" ","C":",","P":".","X":"!","Q":"?","A":"'","D":'"',"E":"<eos>"}
def row(key):
    if key not in R: return [-math.log(34)]*36  # pad/eos/unused rows: uniform
    d={}
    for t in R[key].split():
        k,v=t.split(":"); k=SYM.get(k,k); d[k]=float(v)
    floor=0.0002
    w=[floor]*36
    for k,v in d.items(): w[IX[k]]=v
    w[0]=w[1]=0.0
    if key!='<bos>' and key not in(' ',): pass
    s=sum(w)
    return [ (math.log(x/s) if x>0 else -20.0) for x in w]
T=[row(c) for c in CH]   # T[prev][next]
ops=[]
ops.append({"op":"set_param","name":"cl_bigram.b0","value":[0.0]})
ops.append({"op":"set_param","name":"cl_bigram.bneg","value":[-20.0]})
ops.append({"op":"set_param","name":"cl_bigram.z36","value":[[0.0]*36]})
nodes=[{"id":"sl","use":"op:slice","in":{"x":"$ctx"},"args":{"axis":0,"start":7,"stop":8}},
       {"id":"r7","use":"op:reshape","in":{"x":"sl"},"args":{"shape":[36]}}]
cat={}
for c in range(36):
    if c<2:
        ops.append({"op":"put_def","name":f"cl_bigram.c{c}","def":{"kind":"cell","act":"identity","params":{"W1":"cl_bigram.z36","b1":"cl_bigram.bneg"}}})
    else:
        ops.append({"op":"set_param","name":f"cl_bigram.W{c}","value":[[T[p][c] for p in range(36)]]})
        ops.append({"op":"put_def","name":f"cl_bigram.c{c}","def":{"kind":"cell","act":"identity","params":{"W1":f"cl_bigram.W{c}","b1":"cl_bigram.b0"}}})
    nodes.append({"id":f"o{c}","use":f"cl_bigram.c{c}","in":{"x":"r7"}})
    cat[f"x{c}"]=f"o{c}"
nodes.append({"id":"cat","use":"op:concat","in":cat,"args":{"axis":0}})
ops.append({"op":"put_def","name":"cl_bigram","def":{"kind":"bundle","role":"cluster","inputs":["ctx"],"nodes":nodes,"outputs":{"y":"cat"}}})
json.dump({"author":"c-cl_bigram","rationale":"36 micro-cells, one per output token, each a linear lookup on the one-hot of ctx row 7; table typed by hand from knowledge of English stories, row-normalised.","ops":ops},open("proposal.json","w"))
