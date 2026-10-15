"""Generate fresh server-side typed retrieval variation after artifact freeze."""
import hashlib
import json
from pathlib import Path
import random
import secrets
import time
from cellforge.ir.graph import Graph
from cellforge.runtime.binding import BindingPredictor

seed = secrets.randbits(128)
rng = random.Random(seed)
graph = Graph.load('artifact')
predictor = BindingPredictor(graph)
cases = []
alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
def word(n):
    return ''.join(rng.choice(alphabet) for _ in range(n))
for i in range(128):
    n = rng.randint(1,7)
    keys = [f'{j}{word(rng.randint(2,10))}' for j in range(n)]
    values = [word(rng.randint(1,20)) + ('狐' if i%4==0 else '') for _ in keys]
    bindings = [{'key':k,'value':v} for k,v in zip(keys,values)]
    j = rng.randrange(n)
    query = keys[j]
    expected = {'status':'ok','value':values[j],'matches':1}
    if i%4==1:
        query = '!'+word(8)
        expected = {'status':'missing','value':None,'matches':0}
    elif i%4==2:
        bindings.append({'key':query,'value':word(10)})
        expected = {'status':'ambiguous','value':None,'matches':2}
    elif i%4==3:
        query = keys[j][:-1]
        expected = {'status':'missing','value':None,'matches':0}
    rng.shuffle(bindings)
    cases.append({'bindings':bindings,'query':query,'expected':expected})
start = time.monotonic()
observations = [{'index':i,'expected':c['expected'],'actual':predictor(c['bindings'],c['query'])} for i,c in enumerate(cases)]
report = {'protocol':'binding-server-generated-development/1','model_hash':graph.hash(),'seed':seed,'cases':len(cases),
          'case_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
          'enabled_exact':sum(o['actual']==o['expected'] for o in observations),'disabled_exact':0,
          'disabled_observation':{'status':'unavailable','value':None,'matches':None},
          'seconds':time.monotonic()-start,'observations':observations,
          'limitations':'fresh lexical variation of authored exact retrieval rule, author-visible development; disabled control establishes dependency only; no learned reasoning, parsing or LLM integration'}
Path('cases.json').write_text(json.dumps(cases,indent=2))
Path('result.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='observations'},indent=2))
