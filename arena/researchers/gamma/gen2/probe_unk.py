"""Measure top logits at cellname/suffix positions to size the <unk> suppression."""
import os, sys, json
import torch
sys.path.insert(0, '/srv/arc3-claude/v6run/model/lm')
from tok import load_vocab, encode
from gpt import GPT
LM = '/srv/arc3-claude/v6run/model/lm'
vocab, stoi = load_vocab(os.path.join(LM, 'vocab.json'))
ckpt = torch.load(os.path.join(LM, 'router_v3+dnd1.pt'), map_location='cpu', weights_only=False)
model = GPT(**ckpt['cfg']); model.load_state_dict(ckpt['state']); model.eval()
w2 = [json.loads(l) for l in open(os.path.join(LM, 'traces_w2.jsonl')) if l.strip()]
import random
random.seed(1); random.shuffle(w2)
cells = ['try', 'sweep', 'click_at', 'pick_drop', 'repeat_click', 'ghost_follow', 'reset']
for posname, suffix in [('cellname', '\nCALL {"cell":"'), ('suffix', '\nCALL {"cell":"act.')]:
    gaps = []
    with torch.no_grad():
        for dd in w2[:60]:
            ids = encode('STATE ' + dd['state'] + '\nTHINK ' + dd['think'] + suffix, stoi)
            if len(ids) > 192: ids = ids[-192:]
            logits, _ = model(torch.tensor([ids], dtype=torch.long))
            lg = logits[0, -1]
            unk = float(lg[stoi['<unk>']])
            best_cell = max(float(lg[stoi[c]]) for c in cells)
            top5 = torch.topk(lg, 5)
            top5t = [(vocab[i], round(float(v), 2)) for v, i in zip(top5.values, top5.indices)]
            gaps.append(unk - best_cell)
    gaps.sort()
    import statistics
    print('%s: <unk>-minus-bestcell gap median=%.2f p90=%.2f max=%.2f' % (
        posname, statistics.median(gaps), gaps[int(0.9 * len(gaps))], gaps[-1]))
    print('   example top5:', top5t)
