"""Gamma G5 v2: REACTIVE POLICY on gamma_g4's proven instantaneous gate.
Pure DnD, zero grad steps. Attention pooling ABANDONED (proven unworkable:
centered mean-pool perturbs residual, breaks generation, RHAE 0.0).

Design:
- Gate UNCHANGED: ñ_last > 6.0 (instantaneous, 0/22680 dev firings, PROVEN).
- Policy STRENGTHENED (reactive): when gate fires (TX steps 1-12, novelty),
  force immediate action instead of wasting the firing on mid-reasoning.
  Targets: CALL +15 (force call initiation), sweep +10 / target +10 (steer to
  discovery), ghost_follow -10 / repeat_click -10 (kill routine).
  Structural tokens ('act', '.', '{') protected at 0.0.
- Rationale: the 37 TX firings are at early steps; a strong policy converts
  them into early exploration calls. Dev NEVER fires -> dev-6 provably unchanged.
- Risk: unintended shifts (max 12.3 nats) may hijack; validated empirically.

Block 4: attention stays ZEROED (as gamma_g4). Only MLP c_proj col0 changes
(new pol). ln2.bias, c_fc row0 unchanged.
"""
import sys, torch
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from gpt import GPT

WD = "/srv/cellforge-arena/researchers/gamma/gen2g5"
BASE = "/srv/cellforge-arena/gen2/submissions/gamma_g4.pt"
PROBE = "/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v2"

# reactive policy targets (token_id -> nats)
POL_TARGETS = {
    32:   +15.0,  # CALL: force call initiation on novelty
    427:  +10.0,  # sweep: keyboard discovery
    1754: +10.0,  # target: grounded click discovery (alpha's cell)
    1519: -10.0,  # ghost_follow: kill dominant routine
    1170: -10.0,  # repeat_click: kill secondary routine
    34:     0.0,  # act: protect structural
    14:     0.0,  # .: protect structural
    18:     0.0,  # {: protect structural
}

base = torch.load(BASE, map_location="cpu", weights_only=False)
probe = torch.load(PROBE, map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
cfg = dict(base["cfg"]); assert cfg["n_layer"] == 5
st = base["state"]
w_f = st["ln_f.weight"].float()
what = (n_dir * w_f)
lm = st["lm_head.weight"].float()

def d_of(tid): return (w_f * lm[tid])
tids = list(POL_TARGETS.keys())
D = torch.stack([d_of(t) for t in tids])
c = torch.tensor([POL_TARGETS[t] for t in tids])
pol = torch.linalg.lstsq(D, c).solution
res = float((D @ pol - c).norm())
print(f"policy lstsq residual={res:.4f} |pol|={float(pol.norm()):.2f}", flush=True)
for t in tids:
    print(f"  tid {t:5d} target={POL_TARGETS[t]:+6.1f} got={float(D[tids.index(t)]@pol):+7.2f}", flush=True)

# unintended shifts
alld = (lm * w_f).float()
shifts = alld @ pol
mask = torch.ones(2209, dtype=torch.bool); mask[torch.tensor(tids)] = False
print(f"max unintended |shift|={float(shifts[mask].abs().max()):.1f} "
      f"mean={float(shifts[mask].abs().mean()):.2f}", flush=True)

model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
nsd = {}
for k, v in model.state_dict().items(): nsd[k] = v.clone().float()
for k, v in st.items(): nsd[k] = v.clone().float()
B = "blocks.4"
# keep attention zeroed, keep gate (c_fc row0, ln2.bias); replace policy vector
nsd[f"{B}.mlp.c_proj.weight"][:, 0] = pol

out = {"cfg": cfg, "state": nsd}
torch.save(out, f"{WD}/gamma_g5_{TAG}.pt")
n = sum(v.numel() for v in nsd.values())
print(f"WROTE gamma_g5_{TAG}.pt params={n}", flush=True)
# verify only blocks.4.mlp.c_proj col0 changed
changed = []
for k in st:
    if k == f"{B}.mlp.c_proj.weight":
        dd = float((nsd[k][:, 0] - st[k][:, 0].float()).abs().max())
        dd_rest = float((nsd[k][:, 1:] - st[k][:, 1:].float()).abs().max())
        print(f"c_proj col0 maxchange={dd:.2f} rest={dd_rest:.2e}", flush=True)
        continue
    dd = float((nsd[k] - st[k].float()).abs().max())
    if dd > 1e-6: changed.append((k, dd))
print(f"other changed tensors: {changed}", flush=True)
