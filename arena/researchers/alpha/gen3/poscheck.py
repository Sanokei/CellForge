"""Measure v_orth.r_post across ALL generation positions (not just leaf)
to find the max safe k. v_orth from the clean-label LDA (recomputed here)."""
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
base = torch.load(BASE_CKPT, map_location="cpu", weights_only=True)["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg); model.load_state_dict(base); model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
PREFIX = 'CALL { " cell " : " act . '
prefix_ids = encode(PREFIX, stoi)
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

# recompute v_orth (clean-label LDA) quickly on leaf positions, then measure everywhere
caps = {}
def hook_post(m, inp, out):
    caps["post_all"] = inp[0][0].detach().clone()
model.lm_head.register_forward_hook(hook_post)

leaf_posts, leaf_steps, leaf_games = [], [], []
all_vals = []  # v_orth.r at sampled non-leaf positions (filled after v_orth known)
gen_cache = []  # (pidx_list, step, game) for second pass

CALL_RE = re.compile(r"CALL\s*(\{)")
def parse_call(text):
    m = CALL_RE.search(text)
    if not m: return None, None
    import json as js
    d = None
    dep = 0
    for i in range(m.start(1), len(text)):
        if text[i] == "{": dep += 1
        elif text[i] == "}":
            dep -= 1
            if dep == 0:
                try: d = js.loads(re.sub(r"\s+", "", text[m.start(1):i+1])); break
                except Exception: return None, None
    if not d or d.get("cell") not in REGISTRY: return None, None
    return d["cell"], d.get("args", {})

for game in ["lf52", "s5i5"]:
    torch.manual_seed(0)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard(); env = arc.make(info.game_id, seed=0, scorecard_id=card)
    st = GameState(env, 3000); hist, ndec = [], 0
    while ndec < 60 and st.steps_used < 3000:
        if st.resp.state == GState.WIN: break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None: name, args = "perceive.describe", {}
        pidx = torch.tensor([idx[0].tolist() + prefix_ids], dtype=torch.long)
        if pidx.shape[1] > model.block_size: pidx = pidx[:, -model.block_size:]
        with torch.no_grad():
            model(pidx)
        pa = caps["post_all"].clone()
        leaf_posts.append(pa[-1].numpy())
        leaf_steps.append(st.steps_used); leaf_games.append(game)
        gen_cache.append((pa.numpy(), st.steps_used, game))
        call_cell(st, name, args)
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    print(f"{game}: {ndec} decisions", flush=True)

P = np.stack(leaf_posts); steps = np.array(leaf_steps); games = np.array(leaf_games)
solve = np.array([(g in ("lf52","s5i5")) and s < 18 for g, s in zip(games, steps)])
flail = np.array([(g=="tu93") or ((g in ("lf52","s5i5")) and s>=30) for g, s in zip(games, steps)])
mu0, mu1 = P[solve].mean(0), P[flail].mean(0)
Sw = (np.cov(P[solve].T)*solve.sum() + np.cov(P[flail].T)*flail.sum())/(solve.sum()+flail.sum())
v = np.linalg.solve(Sw + 1e-3*np.eye(128), mu1-mu0); v/=np.linalg.norm(v)
mu_s = P[solve].mean(0); mu_hat = mu_s/np.linalg.norm(mu_s)
v_orth = v - (v@mu_hat)*mu_hat; v_orth /= np.linalg.norm(v_orth)
vt = torch.tensor(v_orth, dtype=torch.float32)

# now measure v.r over ALL positions in cached forwards
mx, vals = 0.0, []
for pa, step, game in gen_cache:
    e = pa @ v_orth
    vals.extend(e.tolist())
    mx = max(mx, float(np.abs(e).max()))
vals = np.array(vals)
print(f"v_orth.r over ALL positions: max|.|={mx:.3f} p99={np.percentile(np.abs(vals),99):.3f} "
      f"p999={np.percentile(np.abs(vals),99.9):.3f} std={vals.std():.3f}")
print(f"leaf-only: solve std={(P[solve]@v_orth).std():.4f}")
# safe k: k * p999 < 2.0 nats
k_safe = 2.0 / max(1e-9, np.percentile(np.abs(vals), 99.9))
print(f"max safe k (p999<2nats): {k_safe:.2f}")
