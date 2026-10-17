"""Gamma G4 revision: grow gamma_g3 4->5 layers; block 4 = novelty-gated emitter hardener.
Pure DnD, zero grad steps. Prior art: beta build_switch.py (5-layer switch pattern, LayerNorm-affine
threshold bookkeeping) ADAPTED; probe direction n_dir measured by beta (probe_dirs.pt) REUSED with credit;
shared-cell-registry explicit-bias threshold motif (carry.W/carry.b) ADAPTED to bias-free gelu MLP via
ln2.bias closed-form shift. Policy (structural boost + unk suppress under novelty) NEW (gamma thesis).
Safety: threshold t set ABOVE max dev activation (+margin) => gate exactly ~0 on dev (verified by eval)."""
import sys, json, torch
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from gpt import GPT
WD = "/srv/cellforge-arena/researchers/gamma/gen2"
base = torch.load("/srv/cellforge-arena/gen2/submissions/gamma_g3.pt", map_location="cpu", weights_only=False)
probe = torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt", map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
cfg = dict(base["cfg"]); assert cfg["n_layer"] == 4; cfg["n_layer"] = 5
st = base["state"]
w_f = st["ln_f.weight"].float()
what = (n_dir * w_f)                      # probe row direction (ln2-input geometry)
calib = json.load(open(WD + "/probe_calib.json"))
M, SIG = calib["max"], calib["std"]
m = 2 * SIG                                  # margin above dev max (dev parity at 1e-15)
t = M + m
s = 4.0                                   # gate sharpness
print(f"calib max={M:.4f} std={SIG:.4f} -> margin m={m:.4f} t={t:.4f}", flush=True)
model = GPT(**{k: cfg[k] for k in ("vocab_size","n_layer","n_head","n_embd","block_size","dropout")})
nsd = {}
for k, v in model.state_dict().items(): nsd[k] = v.clone().float()
for k, v in st.items(): nsd[k] = v.clone().float()   # blocks 0..3 + embeddings + ln_f + lm_head
B = "blocks.4"
nsd[f"{B}.attn.c_attn.weight"].zero_(); nsd[f"{B}.attn.c_proj.weight"].zero_()
nsd[f"{B}.ln1.weight"].fill_(1.0); nsd[f"{B}.ln1.bias"].zero_()
nsd[f"{B}.ln2.weight"].fill_(1.0)
beta = -t * what / (float(what.norm())**2 + 1e-12)   # closed-form: what.beta = -t
nsd[f"{B}.ln2.bias"].copy_(beta)
nsd[f"{B}.mlp.c_fc.weight"].zero_()
nsd[f"{B}.mlp.c_fc.weight"][0] = s * what
nsd[f"{B}.mlp.c_fc.weight"][1] = s * what
lm = st["lm_head.weight"].float()
def d_of(tid):
    d = (w_f * lm[tid]); return d
STRUCT = [32, 18, 19, 5, 33, 35, 9, 4]    # CALL { } " cell args : ,
D = torch.stack([d_of(tid) for tid in STRUCT])          # 8 x 128
c = torch.ones(len(STRUCT))                             # +1 nat each per unit gate
pol0 = 0.4 * torch.linalg.lstsq(D, c).solution  # 0.4x: measured forced-on was ~3x hot
d_unk = d_of(1)
pol1 = 0.4 * (-2.0 * d_unk / (float(d_unk.norm())**2 + 1e-12))
nsd[f"{B}.mlp.c_proj.weight"].zero_()
nsd[f"{B}.mlp.c_proj.weight"][:, 0] = pol0
nsd[f"{B}.mlp.c_proj.weight"][:, 1] = pol1
print("pol0 norm", float(pol0.norm()), "pol1 norm", float(pol1.norm()), flush=True)
out = {"cfg": cfg, "state": nsd}
torch.save(out, WD + "/gamma_g4_rev1.pt")
n = sum(v.numel() for v in nsd.values())
print(f"WROTE gamma_g4_rev1.pt params={n}", flush=True)
