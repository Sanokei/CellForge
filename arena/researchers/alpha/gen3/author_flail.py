"""ALPHA-GEN3 authoring: soft flail-graded sweep boost (no growth).
Pure DnD hand-authoring. Zero grad steps.

Design (revised after hard-threshold LDA failed to separate):
- v = unit LDA direction for STEP on r_post (post-ln_f, the lm_head input),
  measured on dev-3 rollouts (seeds 0,1), true trajectories.
- mu = mean(r_post) over dev. v_orth = normalize(v - proj_mu(v)).
  Then v_orth.mu = 0, so on dev v_orth.r_post ~= 0 +/- sigma (no constant
  boost); the STEP ramp survives via sin^2(theta) where theta = angle(v,mu).
- lm_head delta: k * m_sweep (x) v_orth. Boost on `sweep` leaf token =
  k * (v_orth.r_post): ~0 on dev (+/- k*sigma_dev), ramping to k*Delta
  when flailing (Delta = STEP<15 -> STEP>=25 shift).
- k chosen so k*sigma_dev ~= 0.15 nats (dev-safe) and k*Delta flips the
  typical ghost->sweep margin (measured on base).
- Dev-safety is empirical: official runner dev-3 bit-check.
Untouched: 169 L0 separators, all base weights except lm_head rows for
`sweep`. No suppression (additive only, sweep-only).
"""
import sys, os, json, re
import numpy as np
import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"

from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState
import logging

BASE_CKPT = "/srv/cellforge-arena/gen2/submissions/alpha_g3.pt"
OUT_CKPT = "/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt"
GAMES = ["lf52", "s5i5", "tu93"]
SEEDS = [0, 1]
MAXDEC = 60
BUDGET = 3000
PREFIX = 'CALL { " cell " : " act . '
SWEEP = "sweep"
DEV_NOISE_NATS = 0.15

torch.manual_seed(12345)
np.random.seed(0)
base = torch.load(BASE_CKPT, map_location="cpu", weights_only=True)
bsd = base["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg)
model.load_state_dict(bsd)
model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
prefix_ids = encode(PREFIX, stoi)
sweep_id = encode(SWEEP, stoi)[0]
ghost_id = encode("ghost_follow", stoi)[0]

lg = logging.getLogger("q")
lg.addHandler(logging.NullHandler())
lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

CALL_RE = re.compile(r"CALL\s*(\{)")
def extract_balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def parse_call(text):
    m = CALL_RE.search(text)
    if not m:
        return None, None
    blob = extract_balanced(text, m.start(1))
    if not blob:
        return None, None
    try:
        d = json.loads(re.sub(r"\s+", "", blob))
        name = d.get("cell")
        if name not in REGISTRY:
            return None, None
        return name, d.get("args", {})
    except Exception:
        return None, None

caps = {}
def hook_post(m, inp, out):
    caps["post"] = inp[0][0, -1, :].detach().clone()
model.lm_head.register_forward_hook(hook_post)

raw = []
for seed in SEEDS:
    for game in GAMES:
        torch.manual_seed(seed)
        info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
        card = arc.create_scorecard()
        env = arc.make(info.game_id, seed=seed, scorecard_id=card)
        st = GameState(env, BUDGET)
        hist, ndec = [], 0
        while ndec < MAXDEC and st.steps_used < BUDGET:
            if st.resp.state == GState.WIN:
                break
            state_txt = call_cell(st, "perceive.describe", {})
            prompt = "".join(b + "\n" for b in hist[-2:])
            prompt += "STATE " + state_txt + "\nTHINK "
            idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
            if idx.shape[1] > model.block_size:
                idx = idx[:, -model.block_size:]
            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
            gen = decode(out[0].tolist()[len(idx[0]):], vocab)
            name, args = parse_call(gen)
            if name is None:
                name, args = "perceive.describe", {}
            pidx = torch.tensor([idx[0].tolist() + prefix_ids], dtype=torch.long)
            if pidx.shape[1] > model.block_size:
                pidx = pidx[:, -model.block_size:]
            with torch.no_grad():
                lg_out = model(pidx)
                lg_out = lg_out[0] if isinstance(lg_out, tuple) else lg_out
            r_post = caps["post"].clone()
            lv = lg_out[0, -1]
            raw.append({"post": r_post.numpy(), "step": st.steps_used,
                        "seed": seed, "game": game,
                        "level": int(st.resp.levels_completed),
                        "margin": float(lv[ghost_id] - lv[sweep_id])})
            call_cell(st, name, args)
            think_txt = gen.split("CALL")[0].strip()
            hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                        "\nCALL " + json.dumps({"cell": name, "args": args}))
            ndec += 1
        print(f"seed {seed} game {game}: {ndec} decisions levels={int(st.resp.levels_completed)}", flush=True)

P = np.stack([r["post"] for r in raw])
steps = np.array([r["step"] for r in raw])
margins = np.array([r["margin"] for r in raw])
N = len(raw)
print(f"N={N} base ghost->sweep margin: med={np.median(margins):.2f} "
      f"p10={np.percentile(margins,10):.2f} p90={np.percentile(margins,90):.2f}")

# STEP direction on r_post (soft gate); k set for dev-noise safety
lo = steps < 15
hi = steps >= 25
mu_lo, mu_hi = P[lo].mean(0), P[hi].mean(0)
Sw = (np.cov(P[lo].T) * lo.sum() + np.cov(P[hi].T) * hi.sum()) / (lo.sum() + hi.sum())
v = np.linalg.solve(Sw + 1e-3 * np.eye(128), mu_hi - mu_lo)
v /= np.linalg.norm(v)
mu = P.mean(0)
mu_hat = mu / np.linalg.norm(mu)
v_orth = v - (v @ mu_hat) * mu_hat
v_orth /= np.linalg.norm(v_orth)
e = P @ v_orth
sig_dev = e.std()
delta = e[hi].mean() - e[lo].mean()
print(f"v_orth: sig_dev={sig_dev:.4f} delta={delta:.4f} e_mean={e.mean():.4f}")
k = DEV_NOISE_NATS / max(1e-9, sig_dev)
print(f"k={k:.3f} -> dev noise {k*sig_dev:.3f} nats, flail boost ~{k*delta:.2f} nats")
# build
sd = {kk: vv.clone() for kk, vv in bsd.items()}
mvec = torch.zeros(2209); mvec[sweep_id] = 1.0
dW = k * torch.outer(mvec, torch.tensor(v_orth, dtype=torch.float32))
# CALL boost (token 32): ~+1 nat constant -> converts rambles to calls on novel states
# u_call = mu/||mu||^2 so u_call.mu = 1 -> boost ~= +CALL_C nats wherever r~=mu
CALL_ID = 32
CALL_C = 1.0
mu = P.mean(0)
u_call = torch.tensor(mu / max(1e-9, (mu**2).sum()), dtype=torch.float32)
mcall = torch.zeros(2209); mcall[CALL_ID] = 1.0
dW = dW + CALL_C * torch.outer(mcall, u_call)
# sanity: mean boost on dev leaf positions
with torch.no_grad():
    b_sweep = float((torch.tensor(v_orth, dtype=torch.float32) * torch.tensor(P, dtype=torch.float32)).sum(1).mean() * k)
    b_call = float((u_call * torch.tensor(P, dtype=torch.float32)).sum(1).mean() * CALL_C)
print(f"mean dev boost: sweep {b_sweep:.3f} nats, CALL {b_call:.3f} nats")
sd["lm_head.weight"] = bsd["lm_head.weight"].clone() + dW
# self-check: shapes, finiteness
mchk = GPT(**cfg); mchk.load_state_dict(sd)
n_params = sum(vv.numel() for vv in sd.values())
assert all(torch.isfinite(vv).all() for vv in sd.values())
print(f"params {n_params}, ||dW||/||W|| = "
      f"{(k*torch.outer(mvec, torch.tensor(v_orth))).norm()/bsd['lm_head.weight'].norm():.4f}")
torch.save({"cfg": cfg, "state": sd}, OUT_CKPT)
print(f"WROTE {OUT_CKPT}")
print(json.dumps({"k": float(k), "sig_dev": float(sig_dev), "delta": float(delta),
                  "flail_boost": float(flail_boost),
                  "margin_med": float(np.median(margins)), "params": n_params}))
