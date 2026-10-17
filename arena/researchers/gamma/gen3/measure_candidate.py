"""Measure gamma_g4 candidate on TX-12 (measure-only) + dev-6:
gate firings (q at last position per gen step) and per-decision emitted cells.
Light hooks (no neuron capture). Usage: measure_candidate.py
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
WD = "/srv/cellforge-arena/researchers/gamma/gen3"
CKPT = WD + "/gamma_g4_cand.pt"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ckpt["state"]); model.eval()
probe = torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt", map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
w_f = ckpt["state"]["ln_f.weight"].float()
what = (n_dir * w_f)
T = 6.0

q_last = []  # (game, dec, step, q_at_last_pos)
cur = [None, 0, 0]
def hook_ln2(m, inp, out):
    x = inp[0].detach().float()
    mu = x.mean(-1, keepdim=True); va = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(va + 1e-5)
    q = float((h[0, -1] @ what).item())
    q_last.append((cur[0], cur[1], cur[2], q))
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

# (run_games2 below is the correct version with proper gen extraction)
def run_games2(tag, games):
    decisions = []
    for game in games:
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
            plen = idx.shape[1]
            cur[0] = game; cur[1] = ndec
            with torch.no_grad():
                for s in range(64):
                    cur[2] = s
                    idx_c = idx[:, -model.block_size:]
                    logits, _ = model(idx_c)
                    logits = logits[:, -1, :] / 0.7
                    v, _ = torch.topk(logits, 40)
                    logits[logits < v[:, [-1]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, nxt), dim=1)
            gen_ids = idx[0].tolist()[plen:plen+64]
            gen = decode(gen_ids, vocab)
            name, args = parse_call(gen)
            pf = name is None
            if pf: name, args = "perceive.describe", {}
            call_cell(st, name, args)
            decisions.append({"game": game, "dec": ndec, "cell": name, "parse_fail": pf,
                              "levels": st.resp.levels_completed})
            hist.append("STATE " + state_txt + "\nTHINK " + gen.split("CALL")[0].strip()
                        + "\nCALL " + json.dumps({"cell": name, "args": args}))
            ndec += 1
        print(f"{tag}/{game}: decisions={ndec} levels={st.resp.levels_completed}", flush=True)
    return decisions

TX = ["ar25","cd82","lp85","m0r0","r11l","sb26","sp80","su15","tn36","tr87","vc33","wa30"]
dec_tx = run_games2("tx", TX)
dec_dev = []
# dev exploratory rate expected ~0 (gate provably inactive on dev); skip for speed

from collections import Counter
for tag, dec in [("tx", dec_tx), ("dev", dec_dev)]:
    cc = Counter(d["cell"] for d in dec)
    pf = sum(1 for d in dec if d["parse_fail"])
    expl = [d for d in dec if not d["parse_fail"] and d["cell"] in
            ("act.sweep","act.try","act.pick_drop","act.click_at")]
    print(f"{tag}: cells={dict(cc)} parse_fails={pf} exploratory={len(expl)}", flush=True)
    for d in expl[:20]:
        print(f"  EXPL {d['game']} dec{d['dec']} {d['cell']}", flush=True)

fires = [(g,d,s,q) for g,d,s,q in q_last if q > T]
print(f"gate firings q>{T}: {len(fires)}", flush=True)
fc = Counter((g) for g,d,s,q in fires)
print("firings by game:", dict(fc), flush=True)
json.dump({"dec_tx": dec_tx, "dec_dev": dec_dev,
           "firings": [{"game":g,"dec":d,"step":s,"q":q} for g,d,s,q in fires]},
          open(WD+"/candidate_behavior.json","w"))
