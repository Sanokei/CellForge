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
random.shuffle(w2)
sample = w2[:1500]
acts = []
with torch.no_grad():
    for dd in sample:
        h = h_at(dd['state'], dd['think'])
        g = float((h @ w) / a)
        acts.append((abs(g), g, (dd['state'] + ' // ' + dd['think'])[:90]))
acts.sort(reverse=True)
print('top-15 gate activations among 1500 w2 traces:')
for ag, g, snip in acts[:15]:
    print('  |g|=%.2f g=%+.2f :: %s' % (ag, g, snip.replace(chr(10), '|')))
print('frac |g|>0.5:', sum(1 for ag, _, _ in acts if ag > 0.5) / len(acts))
print('frac |g|>1.0:', sum(1 for ag, _, _ in acts if ag > 1.0) / len(acts))
print('mean|g|:', sum(ag for ag, _, _ in acts) / len(acts))
