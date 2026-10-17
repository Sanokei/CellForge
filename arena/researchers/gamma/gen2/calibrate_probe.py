"""Measure max of what.xhat over dev-6 (seed 0) on gamma_g3 base. what = n_dir (*) w_f."""
import sys, os, json, re, logging, torch
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState
CALL_RE = re.compile(r"CALL\s*(\{)")
def parse_call(text):
    m = CALL_RE.search(text)
    if not m: return None, None
    depth=0
    for i in range(m.start(1), len(text)):
        if text[i]=="{": depth+=1
        elif text[i]=="}":
            depth-=1
            if depth==0: blob=text[m.start(1):i+1]; break
    else: return None, None
    try:
        d=json.loads(re.sub(r"\s+","",blob)); name=d.get("cell")
        return (name,d.get("args",{})) if name in REGISTRY else (None,None)
    except Exception: return None,None
ckpt=torch.load("/srv/cellforge-arena/gen2/submissions/gamma_g3.pt",map_location="cpu",weights_only=False)
model=GPT(**ckpt["cfg"]); model.load_state_dict(ckpt["state"]); model.eval()
probe=torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt",map_location="cpu",weights_only=False)
n_dir=probe["n_dir"].float()
w_f=ckpt["state"]["ln_f.weight"].float()
what=(n_dir*w_f)
print("what norm", float(what.norm()), flush=True)
hots=[]
def hook(m, inp, out):
    x=out.detach().float()
    mu=x.mean(-1,keepdim=True); va=x.var(-1,keepdim=True,unbiased=False)
    xh=(x-mu)/torch.sqrt(va+1e-5)
    hots.append((xh@what).amax().item())
model.blocks[3].register_forward_hook(hook)
vocab,stoi=load_vocab("/srv/arc3-claude/v6run/model/lm/vocab.json")
lg=logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate=False
arc=Arcade(logger=lg,environments_dir="/srv/arc3-claude/environment_files",recordings_dir="/srv/arc3-claude/recordings")
GAMES=["lf52","s5i5","ls20","g50t","tu93","bp35"]
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
        with torch.no_grad(): out=model.generate(idx,max_new_tokens=64,temperature=0.7,top_k=40)
        gen=decode(out[0].tolist()[len(idx[0]):],vocab)
        name,args=parse_call(gen)
        if name is None: name,args="perceive.describe",{}
        call_cell(st,name,args)
        hist.append("STATE "+state_txt+"\nTHINK "+gen.split("CALL")[0].strip()+"\nCALL "+json.dumps({"cell":name,"args":args}))
        ndec+=1
    print(f"{game}: done decisions={ndec} levels={st.resp.levels_completed}", flush=True)
import numpy as np
a=np.array(hots)
print(json.dumps({"n_forward":len(hots),"max":float(a.max()),"mean":float(a.mean()),"std":float(a.std()),"p999":float(np.quantile(a,0.999))}))
json.dump({"max":float(a.max()),"mean":float(a.mean()),"std":float(a.std())}, open("/srv/cellforge-arena/researchers/gamma/gen2/probe_calib.json","w"))
