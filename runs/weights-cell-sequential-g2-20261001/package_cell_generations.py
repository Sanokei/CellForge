import json,hashlib,tarfile,datetime
from pathlib import Path
R=Path.cwd()
arith=json.loads((R/'arithmetic-summary.json').read_text())
pop=json.loads((R/'arithmetic-population.json').read_text())
cid=arith['founder'];params=pop['cells'][cid]['params']
for row in arith['random_controls']:
    rid=hashlib.sha256(json.dumps(row['params'],sort_keys=True).encode()).hexdigest()
    pop['charges'].append({'arm':'random','seed':row['seed'],'child':rid,'charged':True,'operator':'arithmetic-founder matched control'})
    pop['events'].append({'event':'invalid','cell':rid,'reason':'typed output failures or no validated real-anchor admission','source':'arithmetic-summary.json'})
    pop['cells'][rid]={'id':rid,'structure':'full_adder_numeric_v1','params':row['params'],'parents':[cid],
        'state':'rejected','provenance':{'rng_seed':row['seed'],'control':'same12positions,type-specificbounds10'},'scores':{'development':row['development'],'reporting':row['reporting']}}
# Shared authored founder response is instantiated/charged in three declared contexts; physically one live cell.
for seed in (101,202,303):pop['charges'].append({'arm':'llm','seed':seed,'child':cid,'charged':True,'shared_author_response':True,'operator':'arithmetic founder'})
(R/'arithmetic-population-controls.json').write_text(json.dumps(pop,indent=2))
arith['charged_per_arm_per_context']=1;arith['total_charged']=6;arith['physical_authored_founders']=1
arith['selection_limit']='Typed mathematical development niche only; no suitable real benchmark anchor. Founder registration is not production promotion.'
(R/'arithmetic-summary-controls.json').write_text(json.dumps(arith,indent=2))
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file() and p.name not in ('INTEGRITY.json','cell-sequential-results.tar.gz')}
(R/'INTEGRITY.json').write_text(json.dumps({'files':files,'created':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
with tarfile.open(R/'cell-sequential-results.tar.gz','w:gz') as tar:
    for p in R.iterdir():
        if p.name!='cell-sequential-results.tar.gz':tar.add(p,arcname=p.name)
print(json.dumps({'archive_bytes':(R/'cell-sequential-results.tar.gz').stat().st_size,'files':len(files),
'sha256':hashlib.sha256((R/'cell-sequential-results.tar.gz').read_bytes()).hexdigest()}))

