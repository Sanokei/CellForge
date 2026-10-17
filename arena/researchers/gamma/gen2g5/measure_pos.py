"""Gamma gen-4: per-position novelty measurement on dev-6 (gamma_g4 base).
Records, per (game, dec, step): full per-position novelty array at block-4
ln2 input (what . ((x-mu)/sigma)), prompt length, last-pos value.
Goal: understand WHERE high novelty occurs (prompt vs generated, early vs late
steps) to design attention pooling that preserves the dev clean window.
Usage: gen2g5_measure.py <tag> <game1,game2,...>
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
WD = "/srv/cellforge-arena/researchers/gamma/gen2g5"
os.makedirs(WD, exist_ok=True)

ckpt = torch.load("/srv/cellforge-arena/gen2/submissions/gamma_g4.pt",
                   map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = GPT(**{k: cfg[k] for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")})
model.load_state_dict(ckpt["state"]); model.eval()
probe = torch.load("/srv/cellforge-arena/researchers/beta/gen2g3/probe_dirs.pt",
                   map_location="cpu", weights_only=False)
n_dir = probe["n_dir"].float()
w_f = ckpt["state"]["ln_f.weight"].float()
what = (n_dir * w_f)  # 128-dim sensor direction

# per-forward records
recs = []  # (game, dec, step, n_prompt, q float16 array)
cur = [None, 0, 0, 0]  # game, dec, step, n_prompt
def hook_ln2(m, inp, out):
    x = inp[0].detach().float()
    mu = x.mean(-1, keepdim=True); va = x.var(-1, keepdim=True, unbiased=False)
    h = (x - mu) / torch.sqrt(va + 1e-5)
    q = (h @ what).detach().cpu().half().numpy()[0]
    recs.append((cur[0], cur[1], cur[2], cur[3], q))
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
        n_prompt = idx.shape[1]
        cur[0] = game; cur[1] = ndec; cur[3] = n_prompt
        with torch.no_grad():
            out_idx = idx.clone()
            for s in range(64):
                cur[2] = s
                idx_c = out_idx[:, -model.block_size:]
                logits, _ = model(idx_c)
                logits = logits[:, -1, :] / 0.7
                v, _ = torch.topk(logits, 40)
                logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                out_idx = torch.cat((out_idx, nxt), dim=1)
        gen = decode(out_idx[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None: name, args = "perceive.describe", {}
        call_cell(st, name, args)
        hist.append("STATE " + state_txt + "\nTHINK " + gen.split("CALL")[0].strip()
                    + "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    print(f"{game}: decisions={ndec} levels={st.resp.levels_completed} recs={len(recs)}", flush=True)

# pack
games_u = sorted(set(r[0] for r in recs))
gmap = {g: i for i, g in enumerate(games_u)}
N = len(recs)
gidx = np.array([gmap[r[0]] for r in recs], np.int16)
didx = np.array([r[1] for r in recs], np.int16)
step = np.array([r[2] for r in recs], np.int16)
npr = np.array([r[3] for r in recs], np.int32)
q_all = np.concatenate([r[4] for r in recs]).astype(np.float16)
q_off = np.zeros(N + 1, np.int64)
for i, r in enumerate(recs):
    q_off[i+1] = q_off[i] + len(r[4])
np.savez_compressed(f"{WD}/pos_{TAG}.npz", gidx=gidx, didx=didx, step=step,
                    npr=npr, q_off=q_off, q_all=q_all, games=np.array(games_u))
print(f"WROTE {WD}/pos_{TAG}.npz N={N} positions={len(q_all)}", flush=True)
