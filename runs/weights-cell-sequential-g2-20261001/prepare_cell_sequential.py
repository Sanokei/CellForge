import json,hashlib,datetime
from pathlib import Path
R=Path.cwd();V3=Path('/srv/cellforge-inbox/weights-real-learning-v3-20261001');G1=Path('/srv/cellforge-inbox/weights-cell-population-g1-20261001')
old=json.loads((V3/'PREDECLARED.json').read_text());source=Path('/srv/cellforge-private/arcagi2/data/training')
used=set(old['old_exposed_reporting_excluded'])|{x['id'] for x in old['development']+old['reserved_reporting']}
paths=sorted((hashlib.sha256(('weights-cell-sequential-g2:'+p.stem).encode()).hexdigest(),p) for p in source.glob('*.json')
if p.stem not in used and int(hashlib.sha256(p.stem.encode()).hexdigest(),16)%5!=0)
reserve=[{'id':p.stem,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for _,p in paths[:8]]
plan={'protocol':'cell-only-sequential-g2/1','seeds':[101,202,303],'generations':2,'arms':['llm','random'],
'proposals_per_arm_per_seed_per_generation':2,'budget_per_arm_per_seed':4,'workers':2,
'operators':['within-cell literal mutation','authored count-cell structural growth'],
'reporting':reserve,'development':old['development'],'excluded':sorted(used),
'selection':'fresh gain, all historical archive/anchor nonregression, original undefined-FP failclosed gates',
'cache':'same genotype and development manifest only; duplicate proposals still charged',
'authoring':'gpt-6.1-sol low configured task; one physical response per generation shared across seed contexts',
'parallelism':'two scoring worker processes only; sequential generations; no claim independent maker replications',
'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cost':'existing CPU only, no paid API/provisioning',
'non_grid_track':'separate arithmetic founder/controlled tasks; not ARC transfer evidence','full_architecture_complete':False}
(R/'PREDECLARED.json').write_text(json.dumps(plan,indent=2))
(R/'founders.json').write_bytes((G1/'population.json').read_bytes())
print(json.dumps({'event':'experiment-declared','root':str(R),'generations':2,'seeds':3,'charged_total':24,'reporting_reserved':8}))

