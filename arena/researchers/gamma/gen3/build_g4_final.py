"""Gamma G4 FINAL build: rebuild block 4 of gamma_g3.pt as a NOVELTY-GATED EXPLORATION SWITCH.
Pure DnD, zero grad steps. Blocks 0-3, embeddings, ln_f, lm_head bit-identical.

Design (from gen-3 measurements, see GAMMA_G4_METHOD.md):
- DROP the G3 emitter-hardening policy (structural boost caused 40/40 parse fails when forced).
- DROP the stuck-state sensor: block-2 "repetition" neurons do NOT linearly isolate futile
  repetition (sign flip futile-vs-rest vs futile-vs-prodrep; no linear direction separates
  futile from both rest and productive repetition). Documented negative result.
- KEEP one novelty sensor (beta n_dir, recalibrated into TX-measured window).
- NEW policy: exploration emission. Suppress routine cells (ghost_follow, repeat_click),
  boost untried cells (sweep strongly -- it takes no args and is designed for discovery:
  "try each untested keyboard action once, report effects"). Cell-name tokens ONLY
  (never structural tokens: the position-gating rationale -- safe at non-cellname positions).

Block 4: attn zeroed; ln1 standard; ln2.weight=1, ln2.bias=-t*what/||what||^2;
         c_fc row0 = s*what, rest 0; c_proj col0 = pol_explore, rest 0.
"""
import sys, json, torch
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from gpt import GPT

WD = "/srv/cellforge-arena/researchers/gamma/gen3"
BASE = "/srv/cellforge-arena/gen2/submissions/gamma_g3.pt"
PROBE = "/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt"

# ---- calibration (from measurements) ----
T_NOV = 6.0    # TX q_gen: 45/45360 fire; DEV q_gen max 5.88 -> 0/22680 fire (provably dev-inactive)
S_NOV = 4.0
# exploration policy targets (nats per unit gate), cell suffix token ids
POL_TARGETS = {
    49:   -1.5,   # click_at (rarely emitted; mild suppress)
    1519: -3.0,   # ghost_follow (dominant routine)
    145:  +2.0,   # pick_drop (untried)
    1170: -3.0,   # repeat_click (routine)
    176:   0.0,   # reset (leave)
    427:  +4.0,   # sweep (exploration cell, no args)
    53:   +2.0,   # try (untried)
}
# ------------------------------

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
print(f"policy lstsq residual={float((D@pol-c).norm()):.4f} |pol|={float(pol.norm()):.3f}", flush=True)
for t in tids:
    print(f"  tid {t:5d} target={POL_TARGETS[t]:+.1f} got={float(D[tids.index(t)]@pol):+.2f}", flush=True)

model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
nsd = {}
for k, v in model.state_dict().items(): nsd[k] = v.clone().float()
for k, v in st.items(): nsd[k] = v.clone().float()
B = "blocks.4"
nsd[f"{B}.attn.c_attn.weight"].zero_(); nsd[f"{B}.attn.c_proj.weight"].zero_()
nsd[f"{B}.ln1.weight"].fill_(1.0); nsd[f"{B}.ln1.bias"].zero_()
nsd[f"{B}.ln2.weight"].fill_(1.0)
nsd[f"{B}.ln2.bias"].copy_(-T_NOV * what / (float(what.norm())**2 + 1e-12))
nsd[f"{B}.mlp.c_fc.weight"].zero_()
nsd[f"{B}.mlp.c_fc.weight"][0] = S_NOV * what
nsd[f"{B}.mlp.c_proj.weight"].zero_()
nsd[f"{B}.mlp.c_proj.weight"][:, 0] = pol

# verify threshold math on the tensors
b = nsd[f"{B}.ln2.bias"]; w0 = nsd[f"{B}.mlp.c_fc.weight"][0]
print(f"sensor: ||w0||/||what||={float(w0.norm()/what.norm()):.3f} (want {S_NOV}) "
      f"b.what={float(b@what):.4f} (want {-T_NOV})", flush=True)
# verify pre-activation on a synthetic input: htilde = what/||what|| * q
for q in [0.0, 2.0, 4.24, 6.0, 8.0]:
    ht = what / float(what.norm()) * q
    pre = float((ht + b) @ w0)
    import torch.nn.functional as F
    print(f"  q={q:.2f} pre={pre:+.3f} gelu={float(F.gelu(torch.tensor(pre))):+.3f}", flush=True)

out = {"cfg": cfg, "state": nsd}
torch.save(out, WD + "/gamma_g4_cand.pt")
n = sum(v.numel() for v in nsd.values())
print(f"WROTE gamma_g4_cand.pt params={n}", flush=True)
maxdrift = 0.0
for k in st:
    if k.startswith("blocks.4"): continue
    dd = float((nsd[k] - st[k].float()).abs().max())
    maxdrift = max(maxdrift, dd)
print(f"max drift outside blocks.4: {maxdrift:.2e}", flush=True)
