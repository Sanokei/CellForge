"""ALPHA-GEN4 authoring v2: ORTHOGONALIZED leaf boost.

Problem (causal, from dumpgen5 on alpha_g5_b20/lf52): a constant +B boost on
'Clicking' leaks into THINK text ("Clicking the token..." infestation), which
primes the ghost/click routine and re-routes the whole policy -- lf52 0.0
(not just flipped leaves; the reasoning itself collapses to click-mode).

Fix: choose boost direction u with
    u . mu_leaf  = 1   (boost ~= +B nats at the cell-leaf position)
    u . mu_think = 0   (zero MEAN boost across THINK-text positions)
solved via Lagrange (min-norm). Still a pure constant head edit -- no gating,
no trilemma -- just a smarter direction. Same C1 machinery otherwise.

Pure hand-authoring. Zero grad steps.
"""
import sys, os, json, re
import numpy as np
import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"

from tok import load_vocab, encode, decode
from gpt import GPT

BASE = "/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "alpha_g5_orth.pt"
LEAF = sys.argv[2] if len(sys.argv) > 2 else "Clicking"
B = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
MU_LEAF = "mu_g5.npz"

torch.manual_seed(0)
sd = torch.load(BASE, map_location="cpu", weights_only=True)
bsd = sd["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg); model.load_state_dict(bsd); model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
LEAF_ID = encode(LEAF, stoi)[0]
print(f"leaf '{LEAF}' id={LEAF_ID} B={B} (orthogonalized)", flush=True)

mu_leaf = np.load(MU_LEAF)["mu"]

# mu_think: mean post-ln_f residual over early generated positions (THINK text).
# generate() does a full forward per token, so instead do ONE full forward on
# (prompt + generation) after the fact and sample generated positions [2,20),
# which are virtually always THINK (the leaf is at the very end). Robust, no
# CALL-span mapping needed.
from cells import GameState, call_cell, REGISTRY
from arc_agi import Arcade
from arcengine import GameState as GState
import logging
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")
caps = {}
def head_hook(m, inp, out):
    caps["H"] = inp[0][0].detach().clone()
model.lm_head.register_forward_hook(head_hook)

think_resids = []
for game in ["lf52", "s5i5", "tu93"]:
    torch.manual_seed(0)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard(); env = arc.make(info.game_id, seed=0, scorecard_id=card)
    st = GameState(env, 3000); hist = []
    for ndec in range(10):
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
        pids = encode(prompt, stoi)
        idx = torch.tensor([pids], dtype=torch.long)
        if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen_ids = out[0].tolist()[len(idx[0]):]
        gen = decode(gen_ids, vocab)
        # single full forward over prompt+gen; sample early-gen positions
        full = (pids + gen_ids)[-model.block_size:]
        with torch.no_grad():
            model(torch.tensor([full], dtype=torch.long))
        H = caps["H"].numpy()  # [T,128]
        lp = len(pids)
        # generated position g corresponds to full-index (len(full)-len(gen_ids)+g);
        # its predicting residual is at that index. Take g in [2,20).
        base = len(full) - len(gen_ids)
        for g in range(2, min(20, len(gen_ids))):
            think_resids.append(H[base + g - 1])  # residual predicting gen token g
        # advance env
        mm = re.search(r"CALL\s*(\{)", gen)
        name = None
        if mm:
            s = mm.start(1); depth = 0
            for i in range(s, len(gen)):
                if gen[i] == "{": depth += 1
                elif gen[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            d = json.loads(re.sub(r"\s+", "", gen[s:i+1]))
                            if d.get("cell") in REGISTRY: name = d["cell"]
                        except Exception: pass
                        break
        if name is None: name = "perceive.describe"
        call_cell(st, name, {})
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": {}}))
mu_think = np.stack(think_resids, axis=0).mean(0)
print(f"mu_think over {len(think_resids)} think positions, "
      f"||mu_think||={np.linalg.norm(mu_think):.3f}", flush=True)
print(f"cos(mu_leaf, mu_think)={float(mu_leaf@mu_think/np.linalg.norm(mu_leaf)/np.linalg.norm(mu_think)):.3f}",
      flush=True)

# Solve min ||u||^2 s.t. u.mu_leaf=1, u.mu_think=0.
# u = a*mu_leaf + b*mu_think; Gram system G [a;b] = [1;0].
G = np.array([[mu_leaf @ mu_leaf, mu_leaf @ mu_think],
              [mu_think @ mu_leaf, mu_think @ mu_think]])
rhs = np.array([1.0, 0.0])
a, b = np.linalg.solve(G, rhs)
u = a * mu_leaf + b * mu_think
print(f"u coeffs a={a:.4f} b={b:.4f}; check u.mu_leaf={u@mu_leaf:.4f} "
      f"u.mu_think={u@mu_think:.6f}; ||u||={np.linalg.norm(u):.4f}", flush=True)

W = bsd["lm_head.weight"]
dW = torch.zeros_like(W)
mvec = torch.zeros(W.shape[0]); mvec[LEAF_ID] = 1.0
dW += B * torch.outer(mvec, torch.tensor(u, dtype=torch.float32))
newW = W.clone() + dW
assert torch.isfinite(newW).all()
sd["state"]["lm_head.weight"] = newW
print(f"||dW||/||W|| (this edit) = {(dW.norm()/W.norm()):.4f}", flush=True)
root = torch.load("/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt",
                  map_location="cpu", weights_only=True)["state"]["lm_head.weight"]
print(f"total ||W-W_root||/||W_root|| = {((newW-root).norm()/root.norm()):.4f}", flush=True)
torch.save(sd, OUT)
print("wrote", OUT, flush=True)
