"""ALPHA-GEN3 probe: flail-gate feasibility (dev-only, no transfer).
Measures, at the cell-leaf token position (after forcing CALL {"cell": "act.):
  - pre-ln_f residual (what a new block-4 would read)
  - post-ln_f residual (what lm_head reads)
  - STEP (steps_used), actual generated leaf token, leaf logit margins
Outputs: STEP-readability per dim, leaf margins, quiet dims.
"""
import sys, os, json, re, time
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

CKPT = "/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt"
GAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lf52", "s5i5", "tu93"]
MAXDEC = 60
BUDGET = 3000
SEED = 0
PREFIX = 'CALL { " cell " : " act . '

torch.manual_seed(12345)  # construction seed (arbitrary); real seed set per-game below
np.random.seed(0)

sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg)
model.load_state_dict(sd)
model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
prefix_ids = encode(PREFIX, stoi)

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

# capture pre-ln_f and post-ln_f at the last position
caps = {}
def hook_pre(m, inp, out):
    caps["pre"] = inp[0][0, -1, :].detach().clone()
def hook_post(m, inp, out):
    caps["post"] = out[0, -1, :].detach().clone()
model.ln_f.register_forward_hook(hook_pre)
# post: recompute ln_f manually in analysis; instead capture via second hook on lm_head input
orig_head_forward = model.lm_head.forward
def head_hook(m, inp, out):
    caps["post"] = inp[0][0, -1, :].detach().clone()
model.lm_head.register_forward_hook(head_hook)

leaf_ids = {}
for tok in ["sweep", "ghost_follow", "try", "click_at", "reset", "pick_drop", "repeat_click"]:
    ids = encode(tok, stoi)
    leaf_ids[tok] = ids[0] if len(ids) == 1 else None
print("leaf single-token map:", {k: (vocab[v] if v else None) for k, v in leaf_ids.items()}, flush=True)

rows = []
for game in GAMES:
    torch.manual_seed(SEED)  # match runner: fresh seed per game
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=SEED, scorecard_id=card)
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
        # forced-prefix measurement at leaf position
        pidx = torch.tensor([idx[0].tolist() + prefix_ids], dtype=torch.long)
        if pidx.shape[1] > model.block_size:
            pidx = pidx[:, -model.block_size:]
        with torch.no_grad():
            out_t = model(pidx)
            logits = out_t[0] if isinstance(out_t, tuple) else out_t
        r_pre = caps["pre"].numpy().copy()
        r_post = caps["post"].numpy().copy()
        lg_last = logits[0, -1, :]
        leaf = name.split(".")[-1] if "." in name else name
        top_tok = int(torch.argmax(lg_last))
        rows.append({
            "game": game, "dec": ndec, "step": st.steps_used,
            "level": int(st.resp.levels_completed),
            "cell": name, "leaf": leaf,
            "top": vocab[top_tok],
            "logit_top": float(lg_last[top_tok]),
            "logit_sweep": float(lg_last[leaf_ids["sweep"]]) if leaf_ids["sweep"] else None,
            "logit_ghost": float(lg_last[leaf_ids["ghost_follow"]]) if leaf_ids["ghost_follow"] else None,
            "r_pre": r_pre, "r_post": r_post,
        })
        call_cell(st, name, args)
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    print(f"game {game}: {ndec} decisions, levels={int(st.resp.levels_completed)}", flush=True)


# ---- exploratory rate report ----
from collections import Counter
DISC = {"act.sweep", "act.ghost_follow"}
cnt = Counter(r["cell"] for r in rows)
n = len(rows)
ndisc = sum(cnt[c] for c in DISC)
nparse = cnt.get("perceive.describe", 0)
print("games:", sorted(set(r["game"] for r in rows)))
print("decisions:", n)
print("cell counts:", dict(cnt))
print(f"exploratory (sweep/ghost_follow): {ndisc}/{n} = {ndisc/max(1,n):.3f}")
print(f"parse_fail (->describe): {nparse}/{n} = {nparse/max(1,n):.3f}")
import json
json.dump([{"game": r["game"], "dec": r["dec"], "step": r["step"], "cell": r["cell"]} for r in rows],
          open("explore_log.json", "w"))
print("wrote explore_log.json")
