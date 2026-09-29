"""Sanity probe on a sentence I typed myself (not dev data): print top predictions per position."""
import sys, json, numpy as np
sys.path.insert(0,'src')
from cellforge.cli import _load_proposal
from cellforge.runtime.lm import score_texts
import cellforge.cli as cli
from pathlib import Path
from cellforge.construction.transactions import validate_proposal
args = type('A',(),{'run':'runs/real1-open'})()
r = cli._run(args)
task = json.loads(Path('runs/real1-open/tasks/s1-solo/task.json').read_text())
g = r.load_graph(task['base'])
prop = _load_proposal(sys.argv[1]); prop['base']=task['base']
v = validate_proposal(g, prop, r.sample_imgs(), r.limits(), kind=r.kind, profile=r.profile, resolver=r.load_graph)
assert v['ok'], v['error']
g = v['graph']
text = sys.argv[2]
res, info = score_texts(g, [text])
print('bpc', float(res['bits'].sum())/int(res['n_tokens'].sum()))
pp = info['per_pos_bits'][0]
vocab = g.meta['vocab']
tok = info['tokens'][0]
line=''
for t in range(len(text)+1):
    line += "%s:%.1f " % (vocab[tok[t+1]] if t+1 < len(tok) else '', pp[t])
print(line)
