"""Dump raw generations to see the true CALL template bytes."""
import sys, os, json
import torch
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"
from cells import GameState, call_cell
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState
import logging

CKPT = "/srv/cellforge-arena/gen2/submissions/alpha_g3.pt"
sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state"]
cfg = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
       "block_size": 192, "dropout": 0.0}
model = GPT(**cfg); model.load_state_dict(sd); model.eval()
vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
             recordings_dir="/srv/arc3-claude/recordings")
torch.manual_seed(0)
info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "lf52")
card = arc.create_scorecard()
env = arc.make(info.game_id, seed=0, scorecard_id=card)
st = GameState(env, 3000)
hist = []
for ndec in range(6):
    state_txt = call_cell(st, "perceive.describe", {})
    prompt = "".join(b + "\n" for b in hist[-2:])
    prompt += "STATE " + state_txt + "\nTHINK "
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
    if idx.shape[1] > model.block_size:
        idx = idx[:, -model.block_size:]
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
    gen = decode(out[0].tolist()[len(idx[0]):], vocab)
    print(f"--- decision {ndec} ---")
    print(repr(gen[:220]))
    # find CALL and show bytes around cell name
    i = gen.find("CALL")
    if i >= 0:
        print("CTX:", repr(gen[i:i+80]))
    import re
    m = re.search(r"CALL\s*(\{)", gen)
    print("parse:", "FOUND" if m else "MISS")
    name = "perceive.describe"; args = {}
    if m:
        from fixed_runner import parse_call as pc
        name, args = pc(gen)
    print("cell:", name, args)
    call_cell(st, name or "perceive.describe", args or {})
    think_txt = gen.split("CALL")[0].strip()
    hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                "\nCALL " + json.dumps({"cell": name or "perceive.describe", "args": args or {}}))
