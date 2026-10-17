"""Verify: (1) gate OFF => revision bit-identical to gamma_g3 on a dev state; (2) gate forced ON => structural +~1nat, unk -~2nats."""
import sys, torch
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from gpt import GPT
WD="/srv/cellforge-arena/researchers/gamma/gen2"
b3=torch.load("/srv/cellforge-arena/gen2/submissions/gamma_g3.pt",map_location="cpu",weights_only=False)
rv=torch.load(WD+"/gamma_g4_rev1.pt",map_location="cpu",weights_only=False)
m3=GPT(**b3["cfg"]); m3.load_state_dict(b3["state"]); m3.eval()
mr=GPT(**rv["cfg"]); mr.load_state_dict(rv["state"]); mr.eval()
torch.manual_seed(0)
idx=torch.randint(0,2209,(1,150))
with torch.no_grad():
    l3,_=m3(idx); lr,_=mr(idx)
d=(l3-lr).abs().max().item()
print("gate-off max|dlogit| vs gamma_g3:", d)
# force gate on: zero ln2 bias in-memory copy
sd2={k:v.clone() for k,v in rv["state"].items()}
sd2["blocks.4.ln2.bias"].zero_()
mr2=GPT(**rv["cfg"]); mr2.load_state_dict(sd2); mr2.eval()
with torch.no_grad():
    lron,_=mr2(idx)
dl=(lron-lr)[0,-1]
STRUCT=[32,18,19,5,33,35,9,4]
print("forced-on dlogit STRUCT:", [round(float(dl[t]),3) for t in STRUCT])
print("forced-on dlogit <unk>:", round(float(dl[1]),3))
print("forced-on dlogit mean|all|:", round(float(dl.abs().mean()),4))
