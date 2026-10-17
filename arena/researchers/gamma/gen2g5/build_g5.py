"""Gamma G5 build v2: CENTERED mean-pool attention latch on gamma_g4 block 4.
Pure DnD, zero grad steps.

Design (see METHOD for full rationale):
- Mean-pool: uniform attention (W_q=W_k=0) computes mean(m_i), m_i=what.ln1(htilde_i).
- CENTERED via ln1.bias trick: ln1.bias = B*c0 with c0.what = -b/B, so values
  read (m_i - b) where b = dev mean novelty. Pooled out = (mean(m)-b)*p.
  On dev mean(m)~b -> out~0 -> residual UNPERTURBED -> dev-6 provably unchanged,
  t'=6.0 clean window PRESERVED. On TX mean(m)>b -> out>0 pushes gate.
- Gate: S = what.ln2(htilde_last + (mean-b)*c_pool*what_hat) > t'=6.0.
- c_pool can be LARGE (no dev perturbation). Tune on dev for 0 firings.

Block 4 attention (was zeroed):
  ln1.bias = B*c0, c0 = alpha*what_hat + beta*u_perp, alpha=-b/(B*||what||)
  head 0: W_q=0, W_k=0 (uniform); W_v = outer(f, what), f=e0_32
          -> v_i = (m_i - b)*f, out_0 = (mean(m)-b)*f
  heads 1-3: zero.
  c_proj[:, 0:32] = outer(p, f), p = c_pool*what_hat
          -> attn_out = (mean(m)-b)*p
MLP gate: keep c_fc row0=4*what, c_proj col0=pol; ln2.bias=-6*what/||what||^2 (t'=6.0).
"""
import sys, torch
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from gpt import GPT

WD = "/srv/cellforge-arena/researchers/gamma/gen2g5"
BASE = "/srv/cellforge-arena/gen2/submissions/gamma_g4.pt"
PROBE = "/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt"

B_DEV = float(sys.argv[1]) if len(sys.argv) > 1 else 0.58  # dev mean novelty b
C_POOL = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
TAG = sys.argv[3] if len(sys.argv) > 3 else "cand"
B_BIAS = 20.0

base = torch.load(BASE, map_location="cpu", weights_only=False)
probe = torch.load(PROBE, map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
cfg = dict(base["cfg"]); assert cfg["n_layer"] == 5
st = base["state"]
w_f = st["ln_f.weight"].float()
what = (n_dir * w_f)
wnorm = float(what.norm())
what_hat = what / wnorm
D = 128

# c0: unit vector with c0.what = -b/B  ->  c0.what_hat = -b/(B*||what||)
alpha = -B_DEV / (B_BIAS * wnorm)
assert abs(alpha) < 1.0, f"alpha={alpha} too large"
# u_perp: orthogonalize e_1 against what_hat
e1 = torch.zeros(D); e1[1] = 1.0
u = e1 - (e1 @ what_hat) * what_hat
u = u / float(u.norm())
beta = (1 - alpha**2) ** 0.5
c0 = alpha * what_hat + beta * u
print(f"c0: |c0|={float(c0.norm()):.6f} c0.what={float(c0@what):.6f} (want {-B_DEV/B_BIAS:.6f})",
      flush=True)

model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
nsd = {}
for k, v in model.state_dict().items(): nsd[k] = v.clone().float()
for k, v in st.items(): nsd[k] = v.clone().float()
Bname = "blocks.4"

# ln1.bias = B*c0 ; ln1.weight stays 1
nsd[f"{Bname}.ln1.weight"].fill_(1.0)
nsd[f"{Bname}.ln1.bias"].copy_(B_BIAS * c0)

# attention weights
c_attn = nsd[f"{Bname}.attn.c_attn.weight"]
c_attn.zero_()
f = torch.zeros(32); f[0] = 1.0
c_attn[256:288, :] = torch.outer(f, what)  # W_v head0
c_proj = nsd[f"{Bname}.attn.c_proj.weight"]
c_proj.zero_()
p = C_POOL * what_hat
c_proj[:, 0:32] = torch.outer(p, f)

# gate: keep t'=6.0 (already in nsd from gamma_g4); verify
b = nsd[f"{Bname}.ln2.bias"]
print(f"gate: b.what={float(b@what):.4f} (want -6.0)", flush=True)

# ---- verify centering on synthetic ----
torch.manual_seed(1)
T = 12
# synthetic: a_i = n_i + B*c0 with n_i random normalized-ish, what.n_i = m_i
n = torch.randn(1, T, D); n = n / n.norm(dim=-1, keepdim=True) * (D ** 0.5)
a = n + B_BIAS * c0
m = (n[0] @ what)  # true m_i
v = a @ c_attn[256:384, :].T  # (1,T,128) qkv v-part; head0 = first 32
v0 = v[0, :, 0:32]
got = v0[:, 0]  # f=e0 -> v_i = scalar * e0
want = m - B_DEV
print(f"centering check: max|got-want|={float((got-want).abs().max()):.2e}", flush=True)
# uniform attention mean
pool = got.mean()
print(f"pooled (mean(m)-b): got={float(pool):.4f} want={float(m.mean()-B_DEV):.4f}", flush=True)
# c_proj
y = torch.zeros(1, 1, D); y[0, 0, 0] = float(pool)
attn_out = (y @ c_proj.T)[0, 0, :]
print(f"attn_out/p : {(attn_out/p)[:3]} (want ~{(float(pool),)*3})", flush=True)

out = {"cfg": cfg, "state": nsd}
torch.save(out, f"{WD}/gamma_g5_{TAG}.pt")
n = sum(v.numel() for v in nsd.values())
print(f"WROTE gamma_g5_{TAG}.pt params={n} b={B_DEV} c_pool={C_POOL}", flush=True)
maxdrift = 0.0
for k in st:
    if k.startswith("blocks.4"): continue
    dd = float((nsd[k] - st[k].float()).abs().max())
    maxdrift = max(maxdrift, dd)
print(f"max drift outside blocks.4: {maxdrift:.2e}", flush=True)
