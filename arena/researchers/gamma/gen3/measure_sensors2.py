"""Stage A2/B2 sensor measurement for gamma gen-3 (replaces measure_sensors.py).
Per-forward storage with (game, dec, step) keys + prompt/generated tags, so
labels can be recomputed offline without re-running rollouts.
Rollouts match fixed_runner (temp 0.7, top_k 40, 64 new tokens, maxdec 60).

Labels (computed post-hoc, dev-side):
  v1      : parsefail-pair OR exact cell+args repeat OR 3x same cell
  futile  : v1 AND no level-up in next 5 decisions (same game)
  prodrep : v1 AND level-up in next 5 decisions (same game)
  v2      : parsefail-pair OR exact cell+args repeat (no 3x rule)
  futile2 : v2 AND no level-up in next 5 decisions

Writes: fwd_<tag>.npz (per-forward q prompt/generated arrays, neu means, keys),
        decisions_<tag>.jsonl, summary_<tag>.json
Usage: measure_sensors2.py <tag> <game1,game2,...>
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
what = (n_dir * w_f)

# ---- hooks: per-forward records (joined by (game,dec,step) after run) ----
q_recs = []   # (game, dec, step, q float16 array, n_prompt_pos)
n_recs = []   # (game, dec, step, neu float16[512])
cur = [None, 0, 0]  # game, dec, step
def hook_ln2(m, inp, out):
    x = inp[0].detach().float()
    mu = x.mean(-1, keepdim=True); va = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(va + 1e-5)
    q = (h @ what).detach().cpu().half().numpy()[0]
    s = cur[2]
    if s == 0:
        q_recs.append((cur[0], cur[1], s, q, len(q)))
    else:
        q_recs.append((cur[0], cur[1], s, q[-1:], 0))
def hook_cfc(m, inp, out):
    g = F.gelu(out.detach().float())
    n_recs.append((cur[0], cur[1], cur[2],
                   g.mean(dim=1).detach().cpu().half().numpy()[0]))
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

# custom generate with step counting (same sampling as model.generate)
def gen_steps(idx):
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
    return idx

all_decisions = []
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
        cur[0] = game; cur[1] = ndec
        with torch.no_grad():
            out = gen_steps(idx)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        pf = name is None
        if pf: name, args = "perceive.describe", {}
        call_cell(st, name, args)
        all_decisions.append({"game": game, "dec": ndec, "cell": name,
                              "args": json.dumps(args, sort_keys=True)[:200],
                              "parse_fail": pf, "levels": st.resp.levels_completed})
        hist.append("STATE " + state_txt + "\nTHINK " + gen.split("CALL")[0].strip()
                    + "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    print(f"{game}: decisions={ndec} levels={st.resp.levels_completed}", flush=True)

# ---- labels ----
by_game = {}
for d in all_decisions:
    by_game.setdefault(d["game"], []).append(d)
for g, ds in by_game.items():
    ups = [i for i in range(len(ds)) if ds[i]["levels"] > (ds[i-1]["levels"] if i else 0)]
    for i, d in enumerate(ds):
        p1 = ds[i-1] if i >= 1 else None
        p2 = ds[i-2] if i >= 2 else None
        v1 = False
        if d["parse_fail"] and p1 is not None and p1["parse_fail"]: v1 = True
        if p1 is not None and d["cell"] == p1["cell"] and d["args"] == p1["args"]: v1 = True
        if p1 is not None and p2 is not None and d["cell"] == p1["cell"] == p2["cell"]: v1 = True
        v2 = False
        if d["parse_fail"] and p1 is not None and p1["parse_fail"]: v2 = True
        if p1 is not None and d["cell"] == p1["cell"] and d["args"] == p1["args"]: v2 = True
        fut = any(u > i and u <= i + 5 for u in ups)
        d["v1"] = v1; d["v2"] = v2
        d["futile"] = v1 and not fut
        d["prodrep"] = v1 and fut
        d["futile2"] = v2 and not fut

with open(f"{WD}/decisions_{TAG}.jsonl", "w") as f:
    for d in all_decisions:
        f.write(json.dumps(d) + "\n")

# ---- pack per-forward arrays (join q and neu by key) ----
nmap = {(g, d, s): v for g, d, s, v in n_recs}
joined = []
for g, d, s, q, qp in q_recs:
    v = nmap.get((g, d, s))
    if v is not None:
        joined.append((g, d, s, q, qp, v))
print(f"joined {len(joined)}/{len(q_recs)} forwards", flush=True)
games_u = sorted(set(j[0] for j in joined))
gmap = {g: i for i, g in enumerate(games_u)}
N = len(joined)
gidx = np.array([gmap[j[0]] for j in joined], np.int16)
didx = np.array([j[1] for j in joined], np.int16)
step = np.array([j[2] for j in joined], np.int16)
qp = np.array([j[4] for j in joined], np.int32)
q_all = np.concatenate([j[3] for j in joined]).astype(np.float16)
q_off = np.zeros(N + 1, np.int64)
for i, j in enumerate(joined):
    q_off[i+1] = q_off[i] + len(j[3])
neu = np.stack([j[5] for j in joined]).astype(np.float16)
np.savez_compressed(f"{WD}/fwd_{TAG}.npz", gidx=gidx, didx=didx, step=step,
                    qp=qp, q_off=q_off, q_all=q_all, neu=neu,
                    games=np.array(games_u))
# per-forward label lookup arrays
dec_map = {(d["game"], d["dec"]): d for d in all_decisions}
flab = {}
for lab in ["v1", "v2", "futile", "prodrep", "futile2"]:
    flab[lab] = np.array([bool(dec_map[(games_u[gidx[i]], int(didx[i]))][lab])
                          for i in range(N)])

# ---- summaries ----
def qstats(a):
    a = a.astype(np.float32)
    return {"n": int(a.size), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std()), "p99": float(np.quantile(a, .99)),
            "p999": float(np.quantile(a, .999)),
            "frac_gt_424": float((a > 4.24).mean()),
            "frac_gt_6": float((a > 6).mean()), "frac_gt_8": float((a > 8).mean())}
qp_mask = np.zeros(q_all.shape[0], bool)
for i in range(N):
    if qp[i] > 0:
        qp_mask[q_off[i]:q_off[i]+qp[i]] = True
q_prompt = q_all[qp_mask]; q_gen = q_all[~qp_mask]

def contrast(lab):
    m = flab[lab]
    ms = neu[m].astype(np.float32); mn = neu[~m].astype(np.float32)
    m_s, m_n = ms.mean(0), mn.mean(0)
    v_s, v_n = ms.var(0), mn.var(0)
    d = (m_s - m_n) / np.sqrt((v_s + v_n) / 2 + 1e-12)
    o = np.argsort(-np.abs(d))
    return ([(int(i), float(d[i])) for i in o[:12]], int(m.sum()), int((~m).sum()))

out = {"tag": TAG, "games": GAMES, "n_decisions": len(all_decisions),
       "n_forwards": N,
       "q_all": qstats(q_all), "q_prompt": qstats(q_prompt), "q_gen": qstats(q_gen)}
for lab in ["v1", "futile", "prodrep", "v2", "futile2"]:
    top, ns, nn = contrast(lab)
    out[f"contrast_{lab}"] = {"top": top, "n_pos": ns, "n_neg": nn}
# futile vs prodrep (the safety-critical distinction)
lf, lp = flab["futile"], flab["prodrep"]
if lf.sum() > 0 and lp.sum() > 0:
    ms = neu[lf].astype(np.float32); mp = neu[lp].astype(np.float32)
    d = (ms.mean(0) - mp.mean(0)) / np.sqrt((ms.var(0) + mp.var(0)) / 2 + 1e-12)
    o = np.argsort(-np.abs(d))
    out["contrast_futile_vs_prodrep"] = {"top": [(int(i), float(d[i])) for i in o[:12]],
                                         "n_f": int(lf.sum()), "n_p": int(lp.sum())}
from collections import Counter
out["label_counts"] = {l: sum(1 for d in all_decisions if d[l]) for l in ["v1", "futile", "prodrep", "v2", "futile2"]}
out["cell_counts"] = dict(Counter(d["cell"] for d in all_decisions))
json.dump(out, open(f"{WD}/summary_{TAG}.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:4000])
