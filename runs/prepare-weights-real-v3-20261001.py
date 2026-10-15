"""Next bounded development-only server stage, separate from completed receipts."""
import hashlib,json,sys,datetime
from pathlib import Path
ROOT=Path.cwd();data=Path('/srv/cellforge-private/arcagi2/data/training')
old=Path('/srv/cellforge-inbox/weights-real-learning-v2-20261001')
policy=json.loads((old/'PREDECLARED.json').read_text());summary=json.loads((old/'summary.json').read_text())
child=next(x for x in summary['rows'] if x['arm']=='llm')
parent=json.loads((old/'weights'/f"{child['hash']}.json").read_text())['tensors']
used={x['id'] for rows in policy['partitions'].values() for x in rows}
first=json.loads((old/'previous-partitions.json').read_text(encoding='utf-8-sig'))
used.update(x['id'] for rows in first.values() for x in rows)
paths=sorted((hashlib.sha256(('weights-real-v3:'+p.stem).encode()).hexdigest(),p)
             for p in data.glob('*.json') if p.stem not in used and int(hashlib.sha256(p.stem.encode()).hexdigest(),16)%5!=0)
def item(p):
 raw=p.read_bytes();return {'id':p.stem,'sha256':hashlib.sha256(raw).hexdigest(),'task':json.loads(raw)}
dev=[item(data/(x['id']+'.json')) for x in policy['partitions']['development']]+[item(p) for _,p in paths[:8]]
report_meta=[{'id':p.stem,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for _,p in paths[8:16]]
manifest={'protocol':'continued-real-learning-v3/1','parent':child['hash'],'development':[{'id':x['id'],'sha256':x['sha256']} for x in dev],
 'reserved_reporting':report_meta,'old_exposed_reporting_excluded':list(used),
 'source':'arcprize/ARC-AGI-2 public training','maker_revisions':2,'random_seeds':[11,22,33],
 'shared_model_response_not_independent_replicates':True,'primary_metric':'exact whole-task success; separate partition',
 'start':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'existing six specialist structures, direct authored weights/router, no new expert by default',
 'caps':'existing server CPU; measured before candidate evaluation; no provision/paid endpoints'}
(ROOT/'PREDECLARED.json').write_text(json.dumps(manifest,indent=2))
(ROOT/'parent.json').write_text(json.dumps({'tensors':parent}))
(ROOT/'development.json').write_text(json.dumps(dev))
# Reuse frozen existing definitions to score only permitted development tasks.
sys.path.insert(0,str(old/'src'))
from cellforge.construction.typed_population import PaletteExecutor,parity_row_stencil
from cellforge.construction.component_count import SHAPES,predict_component_count
from cellforge.construction.joint_weight_population import predict_routed,expert_predictions
palette=PaletteExecutor(old/'pop1_core.py',10);core=palette.core
core.PREDICTORS={n:(lambda p,g,trace=False,n=n:(palette.predict(n,p,g),{})) for n in core.PREDICTORS}
core.SHAPES['parity_row_stencil_v1']={'stencil.W':(2,3),'foreground.W':(1,10),'activation.b':(1,)}
core.PREDICTORS['parity_row_stencil_v1']=lambda p,g,trace=False:(parity_row_stencil(p,g),{})
core.SHAPES['component_count_unary_v1']=SHAPES
core.PREDICTORS['component_count_unary_v1']=lambda p,g,trace=False:(predict_component_count(p,g),{})
route={k:v for k,v in parent.items() if k.startswith('route.')}
diagnostics=[];solved=oracle=0
for row in dev:
 task=row['task'];hits=[];ohits=[]
 for pair in task['test']:
  view={'demos':task['train'],'input':pair['input']}
  pred,trace=predict_routed(parent,route,view,core,True)
  hits.append(pred==pair['output']);ohits.append(any(p==pair['output'] for p in expert_predictions(parent,pair['input'],core)))
  if not hits[-1]:
   diagnostics.append({'task':row['id'],'view':view,'expected':pair['output'],'predicted':pred,'trace':trace})
 solved+=all(hits);oracle+=all(ohits)
packet={'parent':parent,'parent_hash':child['hash'],'development_solved':solved,'development_tasks':len(dev),'oracle_solved':oracle,
 'failures':diagnostics,'manifest':str(ROOT/'PREDECLARED.json'),
 'purpose':'development-only second stage; no reporting targets included; improve existing numerical specialist/router parameters, no extra expert by default'}
(ROOT/'maker-request.json').write_text(json.dumps(packet,indent=2))
print(json.dumps({'event':'expanded-development-start','parent':child['hash'],'development':str(solved)+'/'+str(len(dev)),'oracle':oracle,
                 'reserved_reporting_tasks':len(report_meta),'directory':str(ROOT),'model':'gpt-6.1-sol low configured Codex task'}),flush=True)

