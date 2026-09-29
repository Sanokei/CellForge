"""Show top next-char predictions after texts I type (not dev data)."""
import sys, json, numpy as np
sys.path.insert(0,'src')
from cellforge.cli import _load_proposal
import cellforge.cli as cli
from pathlib import Path
from cellforge.construction.transactions import validate_proposal
from cellforge.runtime.lm import score_texts
import cellforge.runtime.lm as lm
args = type('A',(),{'run':'runs/real1-open'})()
r = cli._run(args)
task = json.loads(Path('runs/real1-open/tasks/s1-solo/task.json').read_text())
g = r.load_graph(task['base'])
prop = _load_proposal(sys.argv[1]); prop['base']=task['base']
v = validate_proposal(g, prop, r.sample_imgs(), r.limits(), kind=r.kind, profile=r.profile, resolver=r.load_graph)
assert v['ok'], v['error']
g = v['graph']
vocab = g.meta['vocab']
# monkeypatch: score with a dummy trailing char and recover logits by running step manually
from cellforge.runtime.interp import Interpreter
it = Interpreter(g); sd = g.entries['step']
def nextdist(text):
    toks = [vocab.index('<bos>')] + [vocab.index(c) for c in text]
    S = int(g.meta['state_dim']); state = np.zeros((1,S)); ctx = np.zeros((1,8,36))
    for t,tk in enumerate(toks):
        ctx = np.roll(ctx,-1,axis=1); ctx[0,7]=0; ctx[0,7,tk]=1
        out = it.run_bundle(sd, {'ctx':ctx.copy(),'prev':ctx[:,7],'step':np.full((1,1),float(t)),'state':state}, '/step')
        state = np.asarray(out['state']).reshape(1,-1); lg = np.asarray(out['logits']).reshape(-1)
    p = np.exp(lg-lg.max()); p/=p.sum(); return p, state
for text in sys.argv[2:]:
    p, st = nextdist(text)
    o = np.argsort(-p)[:8]
    print(repr(text), ' '.join("%s:%.3f"%(vocab[i],p[i]) for i in o), '| cls', np.round(st[0,1:17],2))
