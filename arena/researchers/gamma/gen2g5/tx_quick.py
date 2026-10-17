"""Quick TX behavior on high-firing games (r11l, vc33). Measure-only."""
import sys, os, json, re, logging
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState

torch.set_num_threads(1)
WD = "/srv/cellforge-arena/researchers/gamma/gen2g5"
GAMES = sys.argv[1].split(",") if len(sys.argv)>1 else ["r11l","vc33"]

ckpt = torch.load(f"{WD}/gamma_g5_v2.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ckpt["state"]); model.eval()
probe = torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt", map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
w_f = ckpt["state"]["ln_f.weight"].float()
what = (n_dir * w_f)

cur_fires=[None]; cur_step=[0]
def hook_ln2(m, inp, out):
    x = inp[0].detach().float()
    mu = x.mean(-1, keepdim=True); va = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(va + 1e-5)
    s = float((h[0,-1] @ what).detach().cpu())
    if s>6.0 and cur_fires[0] is not None: cur_fires[0].append(cur_step[0])
model.blocks[4].ln2.register_forward_hook(hook_ln2)

CALL_RE=re.compile(r"CALL\s*(\{)")
def extract_balanced(text,start):
    depth=0
    for i in range(start,len(text)):
        if text[i]=="{": depth+=1
        elif text[i]=="}":
            depth-=1
            if depth==0: return text[start:i+1]
    return None
def parse_call(text):
    m=CALL_RE.search(text)
    if not m: return None,None
    blob=extract_balanced(text,m.start(1))
    if not blob: return None,None
    try:
        d=json.loads(re.sub(r"\s+","",blob)); name=d.get("cell")
        return (name,d.get("args",{})) if name in REGISTRY else (None,None)
    except Exception: return None,None

vocab,stoi=load_vocab("/srv/arc3-claude/v6run/model/lm/vocab.json")
lg=logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate=False
arc=Arcade(logger=lg,environments_dir="/srv/arc3-claude/environment_files",recordings_dir="/srv/arc3-claude/recordings")
DISCOVERY={"act.sweep","act.target","act.try","act.key_seq"}
alld=[]
for game in GAMES:
    torch.manual_seed(0)
    info=next(e for e in arc.get_environments() if e.game_id.split("-")[0]==game)
    env=arc.make(info.game_id,seed=0,scorecard_id=arc.create_scorecard())
    st=GameState(env,3000); hist=[]; ndec=0
    while ndec<60 and st.steps_used<3000:
        if st.resp.state==GState.WIN: break
        state_txt=call_cell(st,"perceive.describe",{})
        prompt="".join(b+"\n" for b in hist[-2:])+"STATE "+state_txt+"\nTHINK "
        idx=torch.tensor([encode(prompt,stoi)],dtype=torch.long)
        if idx.shape[1]>model.block_size: idx=idx[:,-model.block_size:]
        cur_fires[0]=[]
        with torch.no_grad():
            out_idx=idx.clone()
            for s in range(64):
                cur_step[0]=s
                idx_c=out_idx[:,-model.block_size:]
                logits,_=model(idx_c)
                logits=logits[:,-1,:]/0.7
                v,_=torch.topk(logits,40)
                logits[logits<v[:,[-1]]]=-float("inf")
                probs=F.softmax(logits,dim=-1)
                nxt=torch.multinomial(probs,num_samples=1)
                out_idx=torch.cat((out_idx,nxt),dim=1)
        fires=cur_fires[0]; cur_fires[0]=None
        gen=decode(out_idx[0].tolist()[len(idx[0]):],vocab)
        name,args=parse_call(gen); pf=name is None
        if pf: name,args="perceive.describe",{}
        call_cell(st,name,args)
        alld.append({"game":game,"dec":ndec,"cell":name,"pf":pf,"fires":fires,"disc":name in DISCOVERY})
        hist.append("STATE "+state_txt+"\nTHINK "+gen.split("CALL")[0].strip()+"\nCALL "+json.dumps({"cell":name,"args":args}))
        ndec+=1
    print(f"{game}: done",flush=True)

from collections import Counter
cc=Counter(d["cell"] for d in alld)
fd=[d for d in alld if d["fires"]]
disc=sum(1 for d in alld if d["disc"])
print("cells:",dict(cc))
print("firing decisions:",len(fd),"/",len(alld))
for d in fd[:15]:
    print(" ",d["game"],"d"+str(d["dec"]),d["cell"],"fires"+str(d["fires"]),"pf"+str(d["pf"]))
print(f"discovery: {disc}/{len(alld)}={disc/max(1,len(alld)):.4f}")
json.dump(alld,open(f"{WD}/tx_quick_{'_'.join(GAMES)}.jsonl","w"))
