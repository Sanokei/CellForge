"""Fresh literal arithmetic founder and bounded typed non-grid population."""
import copy,datetime,hashlib,json,time,resource
from pathlib import Path
import numpy as np
from cellforge.construction.arithmetic_cell import predict_adder
from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.joint_weight_population import digest
from cellforge.ir.graph import Graph
from cellforge.runtime.arithmetic import DecimalAdditionPredictor
ROOT=Path.cwd();START=time.perf_counter()
author=json.loads((ROOT/'arithmetic_founder_author.json').read_text(encoding='utf-8-sig'));params=author['params']
# Mathematical contract, not ARC data. Partition fixed before candidate scoring.
all_cases=[{'a':a,'b':b,'carry':c} for a in range(10) for b in range(10) for c in range(2)]
seen=[{'a':2,'b':3,'carry':0},{'a':9,'b':4,'carry':0},{'a':8,'b':8,'carry':1},{'a':0,'b':0,'carry':0}]
seen_ids={digest(c) for c in seen}
remaining=sorted(all_cases,key=lambda c:hashlib.sha256(('arithmetic-cell-g2:'+digest(c)).encode()).hexdigest())
remaining=[c for c in remaining if digest(c) not in seen_ids]
parts={'fresh':remaining[:12],'archive':seen,'anchor':remaining[12:20],'reporting':remaining[20:52]}
policy={'source':'controlled mathematical decimal-digit addition; not real ARC or learned general reasoning',
'partition':{k:[digest(c) for c in v] for k,v in parts.items()},'maker_seen':[digest(c) for c in seen],
'seeds':[101,202,303],'bounds':10,'literal_params':12,'real_anchor_available':False,
'selection':'founder registration for development competition only; production promotion blocked without suitable real-task anchor',
'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(ROOT/'arithmetic-PREDECLARED.json').write_text(json.dumps(policy,indent=2))
def target(c):
    s=c['a']+c['b']+c['carry'];return {'digit':s%10,'carry':s//10}
def evaluate(p,items):
    outcomes={};invalid=0
    for c in items:
        try:answer=predict_adder(p,c)
        except ValueError:answer=None;invalid+=1
        outcomes[digest(c)]=answer==target(c)
    return {'outcomes':outcomes,'solved':sum(outcomes.values()),'tasks':len(items),'invalid_outputs':invalid}
scores={}
for k in ('fresh','archive','anchor'):
    scores[k]=dict(evaluate(params,parts[k]),partition='development',manifest=digest(parts[k]),
        source='typed-arithmetic-contract-controlled',false_positive_given_consistency=None)
population=CellPopulation();cid=population.seed(author['structure'],params,
{'raw_response':'arithmetic_founder_author.json','model':author['model'],'authored_positions':12,'historical_scaffold':False,
'io':'{a,b,carry} -> {digit,carry}','active_role':'typed development-only arithmetic niche; no ARC promotion'},scores)
random=[]
for seed in policy['seeds']:
    rng=np.random.default_rng(seed);p={k:rng.integers(-10,11,size=np.asarray(v).shape).astype(float).tolist() for k,v in params.items()}
    random.append({'seed':seed,'params':p,'charged':True,'development':evaluate(p,parts['fresh']),
                   'reporting':evaluate(p,parts['reporting']),'operator':'same twelve numerical positions/bounds'})
# Verify exact existing graph baseline on same reporting requests; never relabel as learned.
baseline=DecimalAdditionPredictor(Graph.load(ROOT/'arithmetic-baseline'))
def baseline_sum(a,b):
    out=baseline(str(a),str(b))
    if 'sum' not in out:raise ValueError('baseline output contract: '+json.dumps(out))
    return out['sum']
baseline_hits=[]
for c in parts['reporting']:
    value=baseline_sum(c['a'],c['b']);value=baseline_sum(value,c['carry'])
    expected=str(c['a']+c['b']+c['carry']);baseline_hits.append(value==expected)
reporting=evaluate(params,parts['reporting'])
(ROOT/'arithmetic-population.json').write_text(json.dumps(population.snapshot(),indent=2))
(ROOT/'arithmetic-summary.json').write_text(json.dumps({'founder':cid,'live':1,'new_authored_positions':12,
'development':scores['fresh'],'reporting':reporting,'random_controls':random,
'fixed_historical_baseline':{'correct':sum(baseline_hits),'tasks':len(baseline_hits),'training':'none','source':'parallel-arithmetic/artifact-v1'},
'promoted':False,'learned_transfer_claim':False,'seconds':time.perf_counter()-START,
'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'running':False},indent=2))
print(json.dumps({'event':'arithmetic-complete','live':1,'dev':scores['fresh']['solved'],'reporting':reporting['solved'],
'baseline':sum(baseline_hits),'tasks':len(baseline_hits),'seconds':time.perf_counter()-START}),flush=True)

