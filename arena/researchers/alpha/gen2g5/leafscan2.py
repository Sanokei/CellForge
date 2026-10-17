"""ALPHA-GEN4 leafscan2: score REAL single-token candidates from the vocab.

Candidates are exploration-flavored words that exist as single tokens because
they appeared in training THINK traces: Probing, scanning, Sweeping, ...
For each dev decision, force prefix CALL{"cell":"act. and read leaf logits.
Score = median(top_logit - cand_logit) per context. Want SMALL in
ramble/unsolved/loop (easy to flip with a small safe boost) and LARGE in
routine (dev-safe: boost won't flip routine decisions).

Pure measurement. No grad. No transfer games.
"""
import sys, os, json, re
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
GAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lf52", "s5i5", "ls20", "g50t", "tu93", "bp35"]
MAXDEC = 60
BUDGET = 3000
SEED = 0
PREFIX = 'CALL { " cell " : " act . '
OUT = sys.argv[2] if len(sys.argv) > 2 else "leafscan2.json"

CANDIDATES = ["Probing", "scanning", "Sweeping", "following", "Moving",
              "Trying", "Clicking", "Repeating", "Resetting", "learn",
              "reveal", "Stuck", "Running", "Demoting", "mark", "next",
              "fresh", "map"]
REFS = ["sweep", "ghost_follow", "try", "click_at", "reset", "pick_drop",
        "repeat_click", "target", "key_seq"]

torch.manual_seed(12345)
np.random.seed(0)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg)
model.load_state_dict(sd)
model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
prefix_ids = encode(PREFIX, stoi)

tokmap = {}
for t in CANDIDATES + REFS:
    ids = encode(t, stoi)
    if len(ids) == 1 and vocab[ids[0]] != "<unk>":
        tokmap[t] = ids[0]
cands = [t for t in CANDIDATES if t in tokmap]
refs = [t for t in REFS if t in tokmap]
print("candidates:", cands, flush=True)
print("refs:", refs, flush=True)

lg = logging.getLogger("q")
lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")

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
        d = json.loads(re.sub(r"\s+", "", blob))
        name = d.get("cell")
        if name not in REGISTRY: return None, None
        return name, d.get("args", {})
    except Exception:
        return None, None

rows = []
for game in GAMES:
    torch.manual_seed(SEED)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=SEED, scorecard_id=card)
    st = GameState(env, BUDGET)
    hist, ndec = [], 0
    recent_cells = []
    while ndec < MAXDEC and st.steps_used < BUDGET:
        if st.resp.state == GState.WIN: break
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:])
        prompt += "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        ramble = name is None
        if ramble: name, args = "perceive.describe", {}
        pidx = torch.tensor([idx[0].tolist() + prefix_ids], dtype=torch.long)
        if pidx.shape[1] > model.block_size: pidx = pidx[:, -model.block_size:]
        with torch.no_grad():
            lo = model(pidx)
            logits = lo[0] if isinstance(lo, tuple) else lo
        lg_last = logits[0, -1, :]
        top_id = int(torch.argmax(lg_last))
        top_logit = float(lg_last[top_id])
        cand_logits = {t: float(lg_last[tokmap[t]]) for t in cands}
        ref_logits = {t: float(lg_last[tokmap[t]]) for t in refs}
        recent_cells.append(name)
        if ramble: ctx = "ramble"
        elif game in ("ls20", "g50t", "tu93", "bp35"): ctx = "unsolved"
        elif len(recent_cells) >= 4 and len(set(recent_cells[-4:])) == 1: ctx = "loop"
        else: ctx = "routine"
        rows.append({"game": game, "dec": ndec, "ctx": ctx, "cell": name,
                     "top": vocab[top_id], "top_logit": top_logit,
                     "cand": cand_logits, "ref": ref_logits})
        call_cell(st, name, args)
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    print(f"game {game}: {ndec} decisions", flush=True)

from collections import defaultdict, Counter
print("\n=== top-token distribution at leaf position per context ===", flush=True)
for ctx in ["ramble", "unsolved", "loop", "routine"]:
    c = Counter(r["top"] for r in rows if r["ctx"] == ctx)
    print(f"{ctx}: {dict(c.most_common(6))}", flush=True)

def report(names, title):
    print(f"\n=== {title}: median(top_logit - tok_logit), LOWER=better ===", flush=True)
    print(f"{'token':<12} {'ramble':>8} {'unsolved':>8} {'loop':>8} {'routine':>8}", flush=True)
    for t in names:
        line = f"{t:<12}"
        for ctx in ["ramble", "unsolved", "loop", "routine"]:
            v = [r["top_logit"] - (r["cand"].get(t) if t in cands else r["ref"].get(t))
                 for r in rows if r["ctx"] == ctx]
            m = float(np.median(v)) if v else float("nan")
            line += f" {m:8.2f}"
        print(line, flush=True)
report(cands, "candidates")
report(refs, "refs")
json.dump({"summary_rows": len(rows)}, open(OUT, "w"))
print("wrote", OUT, flush=True)
