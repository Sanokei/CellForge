"""Stage A/B sensor measurement for gamma gen-3.
Measures, per position, on rollouts matching fixed_runner (temp 0.7, top_k 40):
  - q_nov = htilde . what   (block-4 ln2 input, normalized; what = n_dir*w_f)
  - block-2 MLP post-gelu per-neuron means (for stuck-group derivation)
Labels decisions stuck/non-stuck from emitted CALL stream (dev-side rule).
Writes: q_nov_<tag>.npy (float16, all positions, with dec index),
        neu_<tag>.npz (per-forward neuron means + dec index),
        decisions_<tag>.jsonl, summary_<tag>.json
Usage: measure_sensors.py <tag> <game1,game2,...>
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
TAG = sys.argv[1]
GAMES = sys.argv[2].split(",")
WD = "/srv/cellforge-arena/researchers/gamma/gen3"
os.makedirs(WD, exist_ok=True)

ckpt = torch.load("/srv/cellforge-arena/gen2/submissions/gamma_g3.pt",
                   map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ckpt["state"]); model.eval()
probe = torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt",
                   map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
w_f = ckpt["state"]["ln_f.weight"].float()
what = (n_dir * w_f)  # 128

# ---- hooks ----
q_chunks = []   # (dec_idx, float16 array of q over positions)
neu_chunks = [] # (dec_idx, float16 512-vector of position-mean post-gelu)
cur_dec = [0]
def hook_ln2(m, inp, out):
    x = inp[0].detach().float()
    mu = x.mean(-1, keepdim=True); va = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(va + 1e-5)
    q = (h @ what)
    q_chunks.append((cur_dec[0][0], cur_dec[0][1], q.detach().cpu().half().numpy()))
def hook_cfc(m, inp, out):
    g = F.gelu(out.detach().float())
    neu_chunks.append((cur_dec[0][0], cur_dec[0][1], g.mean(dim=1).detach().cpu().half().numpy()[0]))
model.blocks[4].ln2.register_forward_hook(hook_ln2)
model.blocks[2].mlp.c_fc.register_forward_hook(hook_cfc)

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

all_decisions = []
for game in GAMES:
    torch.manual_seed(0)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    env = arc.make(info.game_id, seed=0, scorecard_id=arc.create_scorecard())
    st = GameState(env, 3000); hist = []; ndec = 0
    decisions = []
    while ndec < 60 and st.steps_used < 3000:
        if st.resp.state == GState.WIN: break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
        cur_dec[0] = (game, ndec)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        pf = name is None
        if pf: name, args = "perceive.describe", {}
        call_cell(st, name, args)
        decisions.append({"game": game, "dec": ndec, "cell": name,
                          "args": json.dumps(args, sort_keys=True)[:200],
                          "parse_fail": pf, "levels": st.resp.levels_completed})
        hist.append("STATE " + state_txt + "\nTHINK " + gen.split("CALL")[0].strip()
                    + "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    all_decisions.extend(decisions)
    print(f"{game}: decisions={ndec} levels={st.resp.levels_completed}", flush=True)

# ---- label stuck (dev-side rule v1) ----
by_game = {}
for d in all_decisions:
    by_game.setdefault(d["game"], []).append(d)
for g, ds in by_game.items():
    for i, d in enumerate(ds):
        stuck = False
        p1 = ds[i-1] if i >= 1 else None
        p2 = ds[i-2] if i >= 2 else None
        if d["parse_fail"] and p1 is not None and p1["parse_fail"]:
            stuck = True
        if p1 is not None and d["cell"] == p1["cell"] and d["args"] == p1["args"]:
            stuck = True
        if p1 is not None and p2 is not None and d["cell"] == p1["cell"] == p2["cell"]:
            stuck = True
        d["stuck"] = stuck

with open(f"{WD}/decisions_{TAG}.jsonl", "w") as f:
    for d in all_decisions:
        f.write(json.dumps(d) + "\n")

# ---- aggregate ----
dec_index = {(d["game"], d["dec"]): d for d in all_decisions}
q_all, q_stuck, q_non = [], [], []
for gname, di, arr in q_chunks:
    q_all.append(arr.ravel())
q_all = np.concatenate(q_all).astype(np.float32)
def qstats(a):
    return {"n": int(a.size), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std()), "p50": float(np.quantile(a, .5)),
            "p90": float(np.quantile(a, .9)), "p99": float(np.quantile(a, .99)),
            "p999": float(np.quantile(a, .999)),
            "frac_gt_4": float((a > 4.0).mean()), "frac_gt_424": float((a > 4.24).mean()),
            "frac_gt_6": float((a > 6.0).mean()), "frac_gt_8": float((a > 8.0).mean()),
            "frac_gt_10": float((a > 10.0).mean())}
np.save(f"{WD}/q_nov_{TAG}.npy", q_all.astype(np.float16))

# neuron stuck contrast (position-mean per forward, labeled by decision)
n_neu = 512
sum_s = np.zeros(n_neu); sq_s = np.zeros(n_neu); c_s = 0
sum_n = np.zeros(n_neu); sq_n = np.zeros(n_neu); c_n = 0
game_of = {}
for d in all_decisions:
    game_of[(d["game"], d["dec"])] = d
# q_chunks and neu_chunks share dec ordering per game; rebuild per-game dec mapping
# simpler: iterate games in order, decisions sequential
dec_by_gd = {(d["game"], d["dec"]): d for d in all_decisions}
for gname, di, vec in neu_chunks:
    d = dec_by_gd.get((gname, di))
    v = vec.astype(np.float32)
    if d is not None and d["stuck"]:
        sum_s += v; sq_s += v*v; c_s += 1
    else:
        sum_n += v; sq_n += v*v; c_n += 1
mean_s = sum_s / max(c_s, 1); mean_n = sum_n / max(c_n, 1)
var_s = sq_s / max(c_s, 1) - mean_s**2; var_n = sq_n / max(c_n, 1) - mean_n**2
pooled = np.sqrt((var_s + var_n) / 2 + 1e-12)
dscore = (mean_s - mean_n) / pooled
order = np.argsort(-np.abs(dscore))
top = [{"neuron": int(i), "d": float(dscore[i]),
        "mean_stuck": float(mean_s[i]), "mean_non": float(mean_n[i])}
       for i in order[:60]]
np.savez(f"{WD}/neu_{TAG}.npz",
         sum_s=sum_s, sq_s=sq_s, c_s=c_s, sum_n=sum_n, sq_n=sq_n, c_n=c_n)

from collections import Counter
cc = Counter(d["cell"] for d in all_decisions)
pf = sum(1 for d in all_decisions if d["parse_fail"])
nstuck = sum(1 for d in all_decisions if d["stuck"])
summary = {
    "tag": TAG, "games": GAMES, "n_decisions": len(all_decisions),
    "cell_counts": dict(cc), "parse_fails": pf, "n_stuck": nstuck,
    "q_nov": qstats(q_all), "top_stuck_neurons": top[:40],
    "stuck_episodes": [d for d in all_decisions if d["stuck"]][:12],
}
json.dump(summary, open(f"{WD}/summary_{TAG}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k not in ("top_stuck_neurons", "stuck_episodes")}, indent=1))
print("TOP STUCK NEURONS (block2 mlp idx, d):",
      [(t["neuron"], round(t["d"], 2)) for t in top[:15]])
