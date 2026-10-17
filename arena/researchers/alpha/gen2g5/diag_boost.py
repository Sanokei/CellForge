"""Diagnose the boost: for b20 vs g4, measure at forced leaf positions:
1. effective boost factor (mu.r)/||mu||^2 distribution
2. P(Clicking) and gap to argmax, before/after boost
Uses the saved mu_g5.npz.
"""
import sys, os, json, re
import numpy as np
import torch

HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState
import logging

CKPTS = {
    "g4": "/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt",
    "b20": "/srv/cellforge-arena/researchers/alpha/gen2g5/alpha_g5_b20.pt",
}
PREFIX = 'CALL { " cell " : " act . '
mu = np.load("/srv/cellforge-arena/researchers/alpha/gen2g5/mu_g5.npz")["mu"]
mu_norm2 = (mu ** 2).sum()

vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
prefix_ids = encode(PREFIX, stoi)
CLICK_ID = encode("Clicking", stoi)[0]

lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

models = {}
for k, p in CKPTS.items():
    sd = torch.load(p, map_location="cpu", weights_only=True)["state"]
    cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
           "block_size": 192, "dropout": 0.0}
    m = GPT(**cfg); m.load_state_dict(sd); m.eval()
    models[k] = m

caps = {}
def hook(m, inp, out): caps["post"] = inp[0][0, -1, :].detach().clone()
for m in models.values(): m.lm_head.register_forward_hook(hook)

for game in ["tu93", "lf52"]:
    torch.manual_seed(0)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard(); env = arc.make(info.game_id, seed=0, scorecard_id=card)
    st = GameState(env, 3000); hist = []
    print(f"=== {game} ===", flush=True)
    for ndec in range(8):
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > models["g4"].block_size: idx = idx[:, -models["g4"].block_size:]
        pidx = torch.tensor([idx[0].tolist() + prefix_ids], dtype=torch.long)
        if pidx.shape[1] > models["g4"].block_size: pidx = pidx[:, -models["g4"].block_size:]
        line = f"dec{ndec}: "
        for k in ["g4", "b20"]:
            with torch.no_grad():
                lo = models[k](pidx)
                logits = lo[0] if isinstance(lo, tuple) else lo
            lg_last = logits[0, -1, :]
            r = caps["post"].numpy()
            eff = float((mu @ r) / mu_norm2)
            top = int(torch.argmax(lg_last))
            gap = float(lg_last[top] - lg_last[CLICK_ID])
            line += f"{k}: eff={eff:.2f} top={vocab[top]} gap={gap:.2f} | "
        print(line, flush=True)
        # advance with g4's rollout for consistent contexts
        with torch.no_grad():
            out = models["g4"].generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        mm = re.search(r"CALL\s*(\{)", gen); name = None
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
print("done")
