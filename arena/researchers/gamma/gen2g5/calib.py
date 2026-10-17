"""Gamma G5 calibration: run candidate on dev games, measure gate input S
(S = what.ln2(htilde_last + attn_out) at last positions) to set t'.
Also counts firings and checks RHAE doesn't regress.
Usage: gen2g5_calib.py <ckpt_tag> <game1,game2,...>
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
from rhae_scorer import environment_score

torch.set_num_threads(1)
TAG = sys.argv[1]
GAMES = sys.argv[2].split(",")
WD = "/srv/cellforge-arena/researchers/gamma/gen2g5"

ckpt = torch.load(f"{WD}/gamma_g5_{TAG}.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ckpt["state"]); model.eval()
probe = torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt",
                   map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
w_f = ckpt["state"]["ln_f.weight"].float()
what = (n_dir * w_f)
wn2 = float(what.norm())**2

# hook: record S = what.ln2(x) - what.bias_term at last pos, per forward
# ln2(x) = (x-mu)/sigma + bias; gate pre = 4*what.ln2(x) = 4*(S_raw + b.what)
# where S_raw = what.((x-mu)/sigma). We record S_raw (threshold is t'=6.0 -> fires iff S_raw>6)
S_vals = []
fire_count = [0]
def hook_ln2(m, inp, out):
    x = inp[0].detach().float()
    mu = x.mean(-1, keepdim=True); va = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(va + 1e-5)
    s = float((h[0, -1] @ what).detach().cpu())
    S_vals.append(s)
    if s > 6.0: fire_count[0] += 1
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

results = {}
for game in GAMES:
    torch.manual_seed(0)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    baselines = [int(x) for x in info.baseline_actions]
    env = arc.make(info.game_id, seed=0, scorecard_id=arc.create_scorecard())
    st = GameState(env, 3000); hist = []; ndec = 0
    per_level, prev_lvl, prev_steps = {}, 0, 0
    pf = 0
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
        if name is None: pf += 1; name, args = "perceive.describe", {}
        call_cell(st, name, args)
        lvl = int(st.resp.levels_completed)
        if lvl > prev_lvl:
            for L in range(prev_lvl + 1, lvl + 1): per_level[L] = st.steps_used - prev_steps
            prev_lvl, prev_steps = lvl, st.steps_used
        hist.append("STATE " + state_txt + "\nTHINK " + gen.split("CALL")[0].strip()
                    + "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    env_score, _ = environment_score(baselines, per_level)
    results[game] = {"rhae": round(env_score*100, 4), "levels": prev_lvl, "parse_fails": pf}
    print(f"{game}: RHAE={results[game]['rhae']} levels={prev_lvl} pf={pf}", flush=True)

S = np.array(S_vals, np.float32)
print(f"S: n={len(S)} max={S.max():.3f} p99.9={np.quantile(S,.999):.3f} "
      f"frac>6={ (S>6).mean():.5f} firings={fire_count[0]}", flush=True)
print(f"dev RHAE mean: {np.mean([r['rhae'] for r in results.values()]):.4f}", flush=True)
json.dump({"tag": TAG, "games": GAMES, "results": results,
           "S_max": float(S.max()), "S_p999": float(np.quantile(S,.999)),
           "firings": fire_count[0]},
          open(f"{WD}/calib_{TAG}.json", "w"), indent=1)
