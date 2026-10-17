"""Trace first N decisions of lf52 seed 0 for dnd1 vs gamma, print generations."""
import os, sys, json, re
import torch
sys.path.insert(0, '/srv/arc3-claude/v6run/model')
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from arc_agi import Arcade
from arcengine import GameState as GState
from fixed_runner import parse_call, FIXED_CFG

LM = '/srv/arc3-claude/v6run/model/lm'
vocab, stoi = load_vocab(os.path.join(LM, 'vocab.json'))
def load_ckpt(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    m = GPT(**ckpt['cfg']); m.load_state_dict(ckpt['state']); m.eval()
    return m

def trace(model, game='lf52', seed=0, ndec=10):
    torch.manual_seed(seed)
    arc = Arcade()
    info = next(e for e in arc.get_environments() if e.game_id.split('-')[0] == game)
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, 3000)
    hist = []
    out_lines = []
    for i in range(ndec):
        state_txt = call_cell(st, 'perceive.describe', {})
        prompt = ''.join(b + '\n' for b in hist[-2:])
        prompt += 'STATE ' + state_txt + '\nTHINK '
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > FIXED_CFG['block_size']: idx = idx[:, -FIXED_CFG['block_size']:]
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        ok = 'OK ' + name if name else 'PARSE_FAIL'
        out_lines.append('dec %d: %s :: %s' % (i, ok, gen[:150].replace(chr(10), '|')))
        if name is None: name, args = 'perceive.describe', {}
        call_cell(st, name, args)
        think_txt = gen.split('CALL')[0].strip()
        hist.append('STATE ' + state_txt + '\nTHINK ' + think_txt + '\nCALL ' + json.dumps({'cell': name, 'args': args}))
    return out_lines

for label, path in [('dnd1', LM + '/router_v3+dnd1.pt'),
                    ('gamma', '/srv/cellforge-arena/gen2/submissions/gamma.pt')]:
    print('=' * 30, label, '=' * 30)
    for l in trace(load_ckpt(path)):
        print(l)
