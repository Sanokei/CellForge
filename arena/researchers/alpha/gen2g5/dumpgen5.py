"""Dump raw generations for a checkpoint on one game (first 8 decisions).
Shows whether the boost disrupts THINK text, CALL structure, or just flips leaves."""
import sys, os, json, re
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

CKPT = sys.argv[1]; GAME = sys.argv[2] if len(sys.argv) > 2 else "lf52"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8

sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg); model.load_state_dict(sd); model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")
torch.manual_seed(0)
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == GAME)
card = arc.create_scorecard(); env = arc.make(info.game_id, seed=0, scorecard_id=card)
st = GameState(env, 3000); hist = []
for ndec in range(N):
    state_txt = call_cell(st, "perceive.describe", {})
    prompt = "".join(b + "\n" for b in hist[-2:]) + "STATE " + state_txt + "\nTHINK "
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
    if idx.shape[1] > model.block_size: idx = idx[:, -model.block_size:]
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
    gen = decode(out[0].tolist()[len(idx[0]):], vocab)
    print(f"--- dec {ndec} ---")
    print(gen[:400].replace("\n", "\\n"))
    m = re.search(r"CALL\s*(\{)", gen)
    name = None
    if m:
        s = m.start(1); depth = 0
        for i in range(s, len(gen)):
            if gen[i] == "{": depth += 1
            elif gen[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(re.sub(r"\s+", "", gen[s:i+1]))
                        name = d.get("cell") if d.get("cell") in REGISTRY else None
                    except Exception: pass
                    break
    if name is None: name = "perceive.describe"
    call_cell(st, name, {})
    think_txt = gen.split("CALL")[0].strip()
    hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                "\nCALL " + json.dumps({"cell": name, "args": {}}))
print("done")
