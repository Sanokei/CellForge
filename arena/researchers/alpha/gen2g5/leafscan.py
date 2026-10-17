"""ALPHA-GEN4 leafscan: score candidate new-leaf tokens for discovery cells.

For each dev decision, force the prefix CALL{"cell":"act. and read the leaf-
position logit distribution. Score candidate single-token names by their
nat-gap to the argmax in RAMBLE / FLAIL / UNSOLVED contexts (want small)
vs ROUTINE contexts (want large = dev-safe).

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
OUT = sys.argv[2] if len(sys.argv) > 2 else "leafscan.json"

CANDIDATES = ["probe", "survey", "scan", "inspect", "poke", "tap", "touch",
              "check", "test", "sample", "explore", "seek", "hunt", "trace",
              "query", "feel", "tryout", "examine", "sweep2", "probe2",
              "look", "watch", "find", "try2", "ping", "prod", "nudge"]
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

# single-token filter
tokmap = {}
for t in CANDIDATES + REFS:
    ids = encode(t, stoi)
    if len(ids) == 1 and vocab[ids[0]] != "<unk>":
        tokmap[t] = ids[0]
print("single-token candidates:", [t for t in CANDIDATES if t in tokmap], flush=True)
print("single-token refs:", [t for t in REFS if t in tokmap], flush=True)
cands = [t for t in CANDIDATES if t in tokmap]

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
        # forced-prefix leaf logits
        pidx = torch.tensor([idx[0].tolist() + prefix_ids], dtype=torch.long)
        if pidx.shape[1] > model.block_size: pidx = pidx[:, -model.block_size:]
        with torch.no_grad():
            lo = model(pidx)
            logits = lo[0] if isinstance(lo, tuple) else lo
        lg_last = logits[0, -1, :]
        top_id = int(torch.argmax(lg_last))
        top_logit = float(lg_last[top_id])
        cand_logits = {t: float(lg_last[tokmap[t]]) for t in cands}
        ref_logits = {t: float(lg_last[tokmap[t]]) for t in REFS if t in tokmap}
        recent_cells.append(name)
        # context label
        if ramble:
            ctx = "ramble"
        elif game in ("ls20", "g50t", "tu93", "bp35"):
            ctx = "unsolved"
        elif len(recent_cells) >= 4 and len(set(recent_cells[-4:])) == 1:
            ctx = "loop"
        else:
            ctx = "routine"
        rows.append({"game": game, "dec": ndec, "ctx": ctx, "cell": name,
                     "top": vocab[top_id], "top_logit": top_logit,
                     "cand": cand_logits, "ref": ref_logits})
        call_cell(st, name, args)
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
    print(f"game {game}: {ndec} decisions", flush=True)

# aggregate
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    for t, lv in r["cand"].items():
        agg[t][r["ctx"]].append(r["top_logit"] - lv)
print("\n=== candidate gaps: median(top_logit - cand_logit) in nats (LOWER is better for emission) ===", flush=True)
print(f"{'token':<10} {'ramble':>8} {'unsolved':>8} {'loop':>8} {'routine':>8}  n_ramble", flush=True)
summary = {}
for t in cands:
    med = {}
    line = f"{t:<10}"
    for ctx in ["ramble", "unsolved", "loop", "routine"]:
        v = agg[t][ctx]
        m = float(np.median(v)) if v else float("nan")
        med[ctx] = m
        line += f" {m:8.2f}"
    line += f"  {len(agg[t]['ramble'])}"
    print(line, flush=True)
    summary[t] = med
print("\n=== ref gaps (existing leaves, for scale) ===", flush=True)
agg_r = defaultdict(lambda: defaultdict(list))
for r in rows:
    for t, lv in r["ref"].items():
        agg_r[t][r["ctx"]].append(r["top_logit"] - lv)
for t in [x for x in REFS if x in tokmap]:
    line = f"{t:<12}"
    for ctx in ["ramble", "unsolved", "loop", "routine"]:
        v = agg_r[t][ctx]
        m = float(np.median(v)) if v else float("nan")
        line += f" {m:8.2f}"
    print(line, flush=True)
json.dump({"rows": [{k: r[k] for k in ("game","dec","ctx","cell","top")} for r in rows],
           "summary": summary}, open(OUT, "w"))
print("wrote", OUT, flush=True)
