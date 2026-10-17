"""Gamma G5 v2 TX behavior measurement (MEASURE-ONLY, no tuning).
Runs v2 on transfer-12, records gate firings, cell emissions, exploratory rate.
Does NOT compute RHAE (coordinator judges). For diagnostic only.
Usage: gen2g5_tx.py
"""
import sys, os, json, re, logging
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/srv/arc3-claude/v6run/model")
sys.path.insert(0, "/srv/arc3-claude/v6run/model/lm")
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState

torch.set_num_threads(1)
WD = "/srv/cellforge-arena/researchers/gamma/gen2g5"
GAMES = ["ar25","cd82","lp85","m0r0","r11l","sb26","sp80","su15","tn36","tr87","vc33","wa30"]

ckpt = torch.load(f"{WD}/gamma_g5_v2.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ckpt["state"]); model.eval()
probe = torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt",
                   map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
w_f = ckpt["state"]["ln_f.weight"].float()
what = (n_dir * w_f)

# per-decision firing steps
fire_steps = []  # (game, dec, [steps where gate fired])
cur_fires = [None]
def hook_ln2(m, inp, out):
    x = inp[0].detach().float()
    mu = x.mean(-1, keepdim=True); va = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(va + 1e-5)
    s = float((h[0, -1] @ what).detach().cpu())
    if s > 6.0 and cur_fires[0] is not None:
        cur_fires[0].append(cur_step[0])
model.blocks[4].ln2.register_forward_hook(hook_ln2)

CALL_RE = re.compile(r"CALL\s*(\{)")
def extract_balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[start:i+1]
    return None
def parse_call(text):
    m = CALL_RE.search(text)
    if not m: return None, None
    blob = extract_balanced(text, m.start(1))
    if not blob: return None, None
    try:
        d = json.loads(re.sub(r"\s+", "", blob)); name = d.get("cell")
        return (name, d.get("args", {})) if name in REGISTRY else (None, None)
    except Exception:
        return None, None

vocab, stoi = load_vocab("/srv/arc3-claude/v6run/model/lm/vocab.json")
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

DISCOVERY = {"act.sweep", "act.target", "act.try", "act.key_seq"}
all_dec = []
for game in GAMES:
    torch.manual_seed(0)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    env = arc.make(info.game_id, seed=0, scorecard_id=arc.create_scorecard())
    st = GameState(env, 3000); hist = []; ndec = 0
    while ndec < 60 and st.steps_used < 3000:
        if st.resp.state == GState.WIN: break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
        cur_fires[0] = []; cur_step = [0]
        gen_tokens = []
        with torch.no_grad():
            out_idx = idx.clone()
            for s in range(64):
                cur_step[0] = s
                idx_c = out_idx[:, -model.block_size:]
                logits, _ = model(idx_c)
                logits = logits[:, -1, :] / 0.7
                v, _ = torch.topk(logits, 40)
                logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                out_idx = torch.cat((out_idx, nxt), dim=1)
                gen_tokens.append(int(nxt[0,0]))
        fires = cur_fires[0]; cur_fires[0] = None
        gen = decode(out_idx[0].tolist()[len(idx[0]):], vocab)
        # where is CALL in the token stream?
        call_pos = gen.find("CALL")
        # find first CALL token (id 32) position
        try: call_tok_pos = gen_tokens.index(32)
        except ValueError: call_tok_pos = -1
        name, args = parse_call(gen)
        pf = name is None
        if pf: name, args = "perceive.describe", {}
        call_cell(st, name, args)
        all_dec.append({"game": game, "dec": ndec, "cell": name, "parse_fail": pf,
                        "fires": fires, "n_fires": len(fires),
                        "call_tok_pos": call_tok_pos,
                        "discovery": name in DISCOVERY})
        hist.append("STATE " + state_txt + "\nTHINK " + gen.split("CALL")[0].strip()
                    + "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    print(f"{game}: dec={ndec} fires={sum(d['n_fires'] for d in all_dec if d['game']==game)}", flush=True)

with open(f"{WD}/tx_v2_decisions.jsonl", "w") as f:
    for d in all_dec: f.write(json.dumps(d) + "\n")

from collections import Counter
cc = Counter(d["cell"] for d in all_dec)
n_disc = sum(1 for d in all_dec if d["discovery"])
n_fire_dec = sum(1 for d in all_dec if d["n_fires"] > 0)
total_fires = sum(d["n_fires"] for d in all_dec)
# call_tok_pos distribution for fired vs unfired
fired_pos = [d["call_tok_pos"] for d in all_dec if d["n_fires"]>0 and d["call_tok_pos"]>=0]
unfired_pos = [d["call_tok_pos"] for d in all_dec if d["n_fires"]==0 and d["call_tok_pos"]>=0]
import statistics
summary = {
    "n_decisions": len(all_dec),
    "cell_counts": dict(cc),
    "discovery_calls": n_disc,
    "exploratory_rate": round(n_disc/max(1,len(all_dec)), 4),
    "decisions_with_firing": n_fire_dec,
    "total_firings": total_fires,
    "fired_call_pos_mean": round(statistics.mean(fired_pos),1) if fired_pos else None,
    "unfired_call_pos_mean": round(statistics.mean(unfired_pos),1) if unfired_pos else None,
}
print(json.dumps(summary, indent=1))
json.dump(summary, open(f"{WD}/tx_v2_summary.json","w"), indent=1)
# firing steps histogram
all_fire_steps = [s for d in all_dec for s in d["fires"]]
if all_fire_steps:
    import collections
    hc = collections.Counter(all_fire_steps)
    print("firing steps:", sorted(hc.items())[:20])
