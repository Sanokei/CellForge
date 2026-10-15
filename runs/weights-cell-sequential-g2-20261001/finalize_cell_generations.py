"""Freeze both completed generations before any reporting evaluation."""
import importlib.util,json,hashlib,tarfile,time,datetime
from pathlib import Path
ROOT=Path.cwd()
spec=importlib.util.spec_from_file_location('frozen_cell_stages',ROOT/'run_cell_generations.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=json.loads((ROOT/'PREDECLARED.json').read_text())
g1=json.loads((ROOT/'generation-1.json').read_text());g2=json.loads((ROOT/'generation-2.json').read_text())
weights={}
for snapshot in g2['populations'].values():
    for cid,c in snapshot['cells'].items():
        if c['state']=='live' or any(r['id']==cid for g in (g1,g2) for r in g['records']):
            weights[cid]={'structure':c['structure'],'params':c['params']}
freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'weights':{cid:module.digest(v) for cid,v in weights.items()},
'generation_receipts':{str(i):hashlib.sha256((ROOT/f'generation-{i}.json').read_bytes()).hexdigest() for i in (1,2)},
'selection_closed':True,'reporting_not_yet_scored':True}
if (ROOT/'FROZEN.json').exists():raise ValueError('do not repeat frozen evaluation')
(ROOT/'FROZEN.json').write_text(json.dumps(freeze,indent=2))
data=Path('/srv/cellforge-private/arcagi2/data/training');rows=[]
for item in plan['reporting']:
    raw=(data/(item['id']+'.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('report hash mismatch')
    rows.append(dict(item,task=json.loads(raw)))
reporting={}
for cid,c in weights.items():
    reporting[cid]={row['id']:all(module.predict(c['structure'],c['params'],p['input'])==p['output'] for p in row['task']['test']) for row in rows}
trajectory=[]
for g in (g1,g2):
    means={}
    for arm in ('llm','random'):
        selected=[r for r in g['records'] if r['arm']==arm]
        fresh=sum(sum(r['scores']['fresh']['outcomes'].values()) for r in selected if r['scores'])/(8*len(selected))
        report=sum(sum(reporting[r['id']].values()) for r in selected)/(8*len(selected))
        means[arm]={'development_candidate_mean':fresh,'reporting_candidate_mean':report,
                    'charged':len(selected),'admitted':sum(r['admission']['accepted'] for r in selected)}
    trajectory.append({'generation':g['generation'],'arms':means,'timing':g['timing']})
summary={'generations':2,'seeds':plan['seeds'],'arms':plan['arms'],'equal_budget':True,
'charged_per_arm_per_seed':4,'charged_total':24,'physical_author_responses':2,'maker_replicates':1,
'genotypes_evaluated_current':len(weights),'new_unique_candidate_evaluations':sum(g['timing']['new_unique_evaluations'] for g in (g1,g2)),
'live_per_context':g2['live_counts'],'distinct_live_genotypes':len({cid for p in g2['populations'].values() for values in p['live'].values() for cid in values}),
'live_ancestry_max_depth':max(module.depth(p['cells'],cid) for p in g2['populations'].values() for values in p['live'].values() for cid in values),
'new_children_admitted':g1['admitted']+g2['admitted'],'lineage_stalled':not(g1['admitted']+g2['admitted']),
'trajectory':trajectory,'reporting_outcomes':reporting,'reporting_tasks':len(rows),
'parallel_scoring_only':True,'sequential_generations':True,'full_architecture_complete':False,
'limitations':['grid cell track only; independent router/glue tracks not integrated here','no real reporting improvement forced; frozen once after both generations','same author responses across random-seed contexts, not independent maker replications'],
'running':False}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'REPORTING_TASKS.json').write_text(json.dumps(rows))
(ROOT/'weights').mkdir(exist_ok=True)
for cid,c in weights.items():(ROOT/'weights'/f'{cid}.json').write_text(json.dumps(c))
print(json.dumps({'event':'two-generations-frozen','charged':24,'admitted':summary['new_children_admitted'],
'live_unique':summary['distinct_live_genotypes'],'trajectory':trajectory}),flush=True)

