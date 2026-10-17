import json, os, random, sys
import torch
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
from tok import load_vocab, encode, decode
from gpt import GPT
LM = '/srv/arc3-claude/v6run/model/lm'
torch.manual_seed(0); random.seed(0)
vocab, stoi = load_vocab(os.path.join(LM, 'vocab.json'))
def load_ckpt(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    m = GPT(**ckpt['cfg']); m.load_state_dict(ckpt['state']); m.eval()
    return m
m_gamma = load_ckpt('/srv/cellforge-arena/gen2/submissions/gamma.pt')
m_dnd1 = load_ckpt(os.path.join(LM, 'router_v3+dnd1.pt'))
w2 = [json.loads(l) for l in open(os.path.join(LM, 'traces_w2.jsonl')) if l.strip()]
dial = [d for d in w2 if 'dial' in (d['state'] + d['think']).lower()]
print('dial traces in w2:', len(dial))
d = dial[0]
prompt = 'STATE ' + d['state'] + '\nTHINK '
print('PROMPT:', prompt[:160])
for name, m in [('dnd1', m_dnd1), ('gamma', m_gamma)]:
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
    torch.manual_seed(0)
    with torch.no_grad():
        out = m.generate(idx, max_new_tokens=64, temperature=0.7, top_k=40)
    gen = decode(out[0].tolist()[len(idx[0]):], vocab)
    print('=== %s ===' % name)
    print(gen[:700])
