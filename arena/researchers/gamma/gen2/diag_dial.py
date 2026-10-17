import json, os, random, sys
import torch
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
from tok import load_vocab, encode
from gpt import GPT
from weight_author import hidden_at
LM = '/srv/arc3-claude/v6run/model/lm'
torch.manual_seed(0); random.seed(0)
vocab, stoi = load_vocab(os.path.join(LM, 'vocab.json'))
ckpt = torch.load(os.path.join(LM, 'router_v3+dnd1.pt'), map_location='cpu', weights_only=False)
model = GPT(**ckpt['cfg']); model.load_state_dict(ckpt['state']); model.eval()
d = json.load(open('/srv/cellforge-arena/researchers/gamma/gen2/gamma_v2_deltas.json'))
w = torch.tensor(d['lm_head.weight']['B'])[0]
a = 0.269
SUFFIX = '\nCALL {"cell":"act.'
def h_at(state, think):
    ids = encode('STATE ' + state + '\nTHINK ' + think + SUFFIX, stoi)
    if len(ids) > 192: ids = ids[-192:]
    return hidden_at(model, torch.tensor([ids], dtype=torch.long))
w2 = [json.loads(l) for l in open(os.path.join(LM, 'traces_w2.jsonl')) if l.strip()]
dial = [dd for dd in w2 if 'dial' in (dd['state'] + dd['think']).lower()]
print('dial traces:', len(dial))
with torch.no_grad():
    gs = []
    for dd in dial:
        h = h_at(dd['state'], dd['think'])
        g = float((h @ w) / a)
        gs.append(g)
        print('  g=%+.2f :: %s' % (g, (dd['state'] + ' // ' + dd['think'])[:100].replace(chr(10), '|')))
print('mean|g| on dial traces:', sum(abs(g) for g in gs) / len(gs))
