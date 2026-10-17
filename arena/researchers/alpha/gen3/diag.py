"""Diagnose k=12 breakage: what does the model generate?"""
import sys, os
import torch
HERE = "/srv/arc3-claude/v6run/model"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lm"))
os.environ["OMP_NUM_THREADS"] = "1"
from cells import GameState, call_cell
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
import logging

for ckpt, tag in [("/srv/cellforge-arena/gen2/submissions/alpha_g3.pt", "BASE"),
                  ("/srv/cellforge-arena/researchers/alpha/gen3/alpha_g4.pt", "K12")]:
    d = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = d["cfg"]
    m = GPT(**cfg)
    m.load_state_dict(d["state"])
    m.eval()
    vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
    lg = logging.getLogger("q"); lg.addHandler(logging.NullHandler()); lg.propagate = False
    arc = Arcade(logger=lg, environments_dir="/srv/arc3-claude/environment_files",
                 recordings_dir="/srv/arc3-claude/recordings")
    torch.manual_seed(0)
    info = next(e for e in arc.get_environments() if e.game_id.split("-")[0] == "lf52")
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=0, scorecard_id=card)
    st = GameState(env, 3000)
    state_txt = call_cell(st, "perceive.describe", {})
    prompt = "STATE " + state_txt + "\nTHINK "
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
    with torch.no_grad():
        out = m.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
    gen = decode(out[0].tolist()[len(idx[0]):], vocab)
    print(f"=== {tag} ===")
    print(repr(gen[:300]))
    print()
