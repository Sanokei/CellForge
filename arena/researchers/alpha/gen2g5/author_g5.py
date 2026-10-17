"""ALPHA-GEN4 authoring: constant head-boost on the discovery leaf token(s).

Base: alpha_g4.pt (keeps ramble converter + flail nudge, both dev-safe).
Edit: dW[LEAF] += B * mu/||mu||^2, where mu = mean post-ln_f residual at the
  cell-leaf position over dev contexts. Boost ~= +B nats wherever r ~= mu.
  C1-scale: total ||dW||/||W|| must stay ~0.05.

Rationale (leafscan2, alpha_g4, dev-3 forced-prefix):
  'Clicking' gap to argmax: ramble 1.93 / unsolved 1.31 / loop 3.38 / routine 2.76.
  A constant +B boost flips flail contexts to the discovery leaf while the
  routine margin (2.76) keeps most memorized solves intact. Dev eval is the
  ground truth for safety (guardrail dev-6 >= 0.31; aim: keep 0.766).

Pure hand-authoring. Zero grad steps.
"""
import sys, os, json
import numpy as np
import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"

from tok import load_vocab, encode, decode
from gpt import GPT

BASE = "/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "alpha_g5.pt"
LEAF = sys.argv[2] if len(sys.argv) > 2 else "Clicking"
B = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
MU_NPZ = sys.argv[4] if len(sys.argv) > 4 else None

torch.manual_seed(0)
sd = torch.load(BASE, map_location="cpu", weights_only=True)
bsd = sd["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg); model.load_state_dict(bsd); model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
leaf_ids = encode(LEAF, stoi)
assert len(leaf_ids) == 1 and vocab[leaf_ids[0]] != "<unk>", f"leaf {LEAF} not single-token"
LEAF_ID = leaf_ids[0]
print(f"leaf '{LEAF}' id={LEAF_ID} boost B={B}", flush=True)

if MU_NPZ and os.path.exists(MU_NPZ):
    mu = np.load(MU_NPZ)["mu"]
    print(f"loaded mu from {MU_NPZ}", flush=True)
else:
    # mu = mean post-ln_f residual at leaf positions over dev prompts.
    # Cheap proxy: run a handful of real dev prompts through the model with the
    # forced leaf prefix and average the lm_head input at the last position.
    from cells import GameState, call_cell
    from arc_agi import Arcade
    from arcengine import GameState as GState
    import logging, re
    lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
    arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
                 recordings_dir="/srv/arc3-claude/recordings")
    PREFIX = 'CALL { " cell " : " act . '
    prefix_ids = encode(PREFIX, stoi)
    caps = {}
    def head_hook(m, inp, out): caps["post"] = inp[0][0, -1, :].detach().clone()
    model.lm_head.register_forward_hook(head_hook)
    resids = []
    for game in ["lf52", "s5i5", "tu93"]:
        torch.manual_seed(0)
        info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
        card = arc.create_scorecard(); env = arc.make(info.game_id, seed=0, scorecard_id=card)
        st = GameState(env, 3000); hist = []
        for ndec in range(12):
            state_txt = call_cell(st, "perceive.describe", {})
            prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
            idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
            if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
            gen = decode(out[0].tolist()[len(idx[0]):], vocab)
            m = re.search(r"CALL\s*(\{)", gen)
            name, args = None, {}
            if m:
                s = m.start(1); depth = 0; blob = None
                for i in range(s, len(gen)):
                    if gen[i] == "{": depth += 1
                    elif gen[i] == "}":
                        depth -= 1
                        if depth == 0: blob = gen[s:i+1]; break
                if blob:
                    try:
                        d = json.loads(re.sub(r"\s+", "", blob))
                        from cells import REGISTRY
                        if d.get("cell") in REGISTRY: name, args = d["cell"], d.get("args", {})
                    except Exception: pass
            if name is None: name, args = "perceive.describe", {}
            pidx = torch.tensor([idx[0].tolist() + prefix_ids], dtype=torch.long)
            if pidx.shape[1] > model.block_size: pidx = pidx[:, -model.block_size:]
            with torch.no_grad():
                model(pidx)
            resids.append(caps["post"].numpy().copy())
            call_cell(st, name, args)
            think_txt = gen.split("CALL")[0].strip()
            hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                        "\nCALL " + json.dumps({"cell": name, "args": args}))
    mu = np.stack(resids).mean(0)
    print(f"computed mu over {len(resids)} leaf contexts, ||mu||={np.linalg.norm(mu):.3f}", flush=True)
    np.savez("mu_g5.npz", mu=mu)

u = mu / max(1e-9, (mu ** 2).sum())   # u.mu = 1 -> boost ~= +B nats at r~=mu
W = bsd["lm_head.weight"]
dW = torch.zeros_like(W)
mvec = torch.zeros(W.shape[0]); mvec[LEAF_ID] = 1.0
dW += B * torch.outer(mvec, torch.tensor(u, dtype=torch.float32))
newW = W.clone() + dW
assert torch.isfinite(newW).all()
sd["state"]["lm_head.weight"] = newW
rel = dW.norm() / W.norm()
print(f"||dW||/||W|| (this edit) = {rel:.4f}", flush=True)
# total drift vs elitism root for the paper trail
root = torch.load("/srv/cellforge-arena/gen2/bred/bred_g1_a25.pt",
                  map_location="cpu", weights_only=True)["state"]["lm_head.weight"]
tot = (newW - root).norm() / root.norm()
print(f"total ||W - W_root||/||W_root|| = {tot:.4f}", flush=True)
torch.save(sd, OUT)
print("wrote", OUT, flush=True)
