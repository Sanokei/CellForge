"""One bounded existing-Hetzner Codex-authored literal-weight continuation."""
import copy,datetime,hashlib,json,os,resource,socket,time
from pathlib import Path
import numpy as np
from cellforge.construction.joint_weight_population import digest,experts,predict_routed,expert_predictions
from cellforge.construction.typed_population import PaletteExecutor,parity_row_stencil,measure_resources,calibrate_caps,resource_pass
from cellforge.construction.component_count import predict_component_count,SHAPES
from cellforge.construction.red_queen_history import OpponentHistory
ROOT=Path.cwd();START=time.perf_counter()
parent=json.loads((ROOT/'parent.json').read_text())['tensors']
proposal=json.loads((ROOT/'proposal.json').read_text())
if digest(parent)!=proposal['parent_hash']:raise ValueError('wrong frozen parent')
data=Path('/srv/cellforge-private/arcagi2/data/training')
manifest=json.loads((ROOT/'previous-partitions.json').read_text(encoding='utf-8-sig'))
def read(rows):
 out=[]
 for item in rows:
  raw=(data/(item['id']+'.json')).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('dataset hash mismatch')
  out.append({'id':item['id'],'sha256':item['sha256'],'task':json.loads(raw)})
 return out
parts={k:read(manifest[k]) for k in ('development','archive')}
excluded={x['id'] for items in manifest.values() for x in items}|{'8b28cd80','cbded52d','705a3229','7d18a6fb'}
paths=sorted((hashlib.sha256(('weights-real-v2:'+p.stem).encode()).hexdigest(),p)
             for p in data.glob('*.json') if p.stem not in excluded and int(hashlib.sha256(p.stem.encode()).hexdigest(),16)%5!=0)
report_paths=[p for _,p in paths[:4]]
parts['reporting']=[{'id':p.stem,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'task':json.loads(p.read_bytes())} for p in report_paths]
parts['anchor']=json.loads((ROOT/'old-anchors.json').read_text(encoding='utf-8-sig'))
parts['anchor']=[{'id':i,'task':t} for i,t in parts['anchor']]
palette=PaletteExecutor(ROOT/'pop1_core.py',10);core=palette.core
core.PREDICTORS={n:(lambda p,g,trace=False,n=n:(palette.predict(n,p,g),{})) for n in core.PREDICTORS}
core.SHAPES['parity_row_stencil_v1']={'stencil.W':(2,3),'foreground.W':(1,10),'activation.b':(1,)}
core.PREDICTORS['parity_row_stencil_v1']=lambda p,g,trace=False:(parity_row_stencil(p,g),{})
core.SHAPES['component_count_unary_v1']=SHAPES
core.PREDICTORS['component_count_unary_v1']=lambda p,g,trace=False:(predict_component_count(p,g),{})
child=copy.deepcopy(parent)
if proposal['new_structure']!='component_count_unary_v1':raise ValueError('unapproved translated contract')
for op in proposal['ops']:
 if op['op']!='set_param':raise ValueError('unsupported numeric op')
 a=np.asarray(op['value'],float)
 if not np.isfinite(a).all() or np.max(np.abs(a))>9:raise ValueError('unsafe literal proposal')
 child[op['name']]=a.tolist()
for name in proposal['inherited_names']:
 if child[name]!=parent[name]:raise ValueError('inherited expert damaged')
if not np.array_equal(np.asarray(child['route.W'])[:5,:5],parent['route.W']) or child['route.b'][:5]!=parent['route.b']:
 raise ValueError('router inheritance damaged')
experts(child,core)
weights={'parent':parent,'llm':child};seeds=[11,22,33]
for seed in seeds:
 rng=np.random.default_rng(seed+965001);t=copy.deepcopy(parent)
 for name,shape in SHAPES.items():
  t['5:'+name]=rng.integers(0,10,size=shape).astype(float).tolist() if name=='palette' else rng.integers(-9,10,size=shape).astype(float).tolist()
 w=np.zeros((6,6));w[:5,:5]=parent['route.W'];w[5,:]=rng.integers(-9,10,size=6);w[:5,5]=rng.integers(-9,10,size=5)
 t['route.W']=w.tolist();t['route.b']=parent['route.b']+[float(rng.integers(-9,10))]
 weights['random_'+str(seed)]=t
policy={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'host':socket.gethostname(),
 'protocol':'existing-hetzner-real-weight-continuation/1','source':'arcprize/ARC-AGI-2 public training',
 'partitions':{k:[{'id':r['id'],'sha256':r.get('sha256')} for r in rows] for k,rows in parts.items()},
 'parent':digest(parent),'maker':'gpt-6.1-sol low /root/pop1_router_author, configured Codex task; no external paid API',
 'seeds':seeds,'charged_candidate_per_arm_per_seed':1,'maker_replicates':1,'random_seed_contexts':3,
 'caps':'same exact dev tasks on existing server, 3 parent repetitions, local max 4x baseline or .25s/16MiB floors',
 'architecture_constants':'Four-connected components, geometric feature predicates height2,width2,cardinality4; authored structure uses these fixed priors. Not all structure information is trainable weights.',
 'reporting':'new task IDs withheld from maker; public-training reporting, not official evaluation. No report-based selection.',
 'budget_limitation':'same feature/structure vocabulary and 35 new numeric positions; author has 5 inherited expert templates, random uses same fixed extraction',
 'spending':'existing server CPU only, no provisioning or paid training endpoint'}
(ROOT/'PREDECLARED.json').write_text(json.dumps(policy,indent=2))
(ROOT/'weights').mkdir()
for name,t in weights.items():
 (ROOT/'weights'/f'{digest(t)}.json').write_text(json.dumps({'tensors':t}))
(ROOT/'FROZEN.json').write_text(json.dumps({'hashes':{n:digest(t) for n,t in weights.items()},'policy_hash':digest(policy),'before_scoring':True},indent=2))
print(json.dumps({'event':'start','pid':os.getpid(),'host':policy['host'],'parent':digest(parent),'candidate':digest(child),'model':policy['maker']}),flush=True)
def evaluate(t,rows):
 route={k:v for k,v in t.items() if k.startswith('route.')};outcomes={};details=[];oracle=0;uniform=0;consistent=wrong=0
 for row in rows:
  hits=[];ohits=[];uhits=[]
  for pair in row['task']['test']:
   view={'demos':row['task']['train'],'input':pair['input']}
   pred,trace=predict_routed(t,route,view,core,True)
   hits.append(pred==pair['output']);ohits.append(any(p==pair['output'] for p in expert_predictions(t,pair['input'],core)))
   zeros={k:np.zeros_like(np.asarray(v)).tolist() for k,v in route.items()}
   uhits.append(predict_routed(t,zeros,view,core)==pair['output'])
   valid=trace['features'][trace['chosen_expert']]==0;consistent+=valid;wrong+=valid and pred!=pair['output']
   details.append({'task':row['id'],'solved':hits[-1],'chosen':trace['chosen_expert'],'features':trace['features']})
  outcomes[row['id']]=all(hits);oracle+=all(ohits);uniform+=all(uhits)
 return {'solved':sum(outcomes.values()),'tasks':len(rows),'outcomes':outcomes,'oracle_solved':oracle,'uniform_solved':uniform,
         'false_positive_given_consistency':wrong/consistent if consistent else None,'details':details}
samples=[]
for _ in range(3):
 _,r=measure_resources(lambda:evaluate(parent,parts['development']));samples.append(r)
caps=calibrate_caps(samples);(ROOT/'RESOURCE_CAPS.json').write_text(json.dumps(caps,indent=2))
rows=[]
for name,t in weights.items():
 dev,probe=measure_resources(lambda:evaluate(t,parts['development']))
 row={'arm':name,'hash':digest(t),'development':dev,'archive':evaluate(t,parts['archive']),
      'anchor':evaluate(t,parts['anchor']),'reporting':evaluate(t,parts['reporting']),
      'resources':probe,'caps_passed':resource_pass(probe,caps),'promoted':False}
 rows.append(row)
 (ROOT/'progress.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps({'event':'measured','arm':name,'development':dev['solved'],'reporting':row['reporting']['solved'],'oracle':row['reporting']['oracle_solved'],'caps':row['caps_passed']}),flush=True)
def admission_scores(row):
 return {label:dict(row[source],partition='development',manifest=digest(parts[source]),source='public-training-real')
         for label,source in [('fresh','development'),('archive','archive'),('anchor','anchor')]}
history=OpponentHistory();parent_scores=admission_scores(rows[0])
history.register(graph_hash=digest(parent),family='joint_grid_five_structures',
 structure_hash=digest([(i,n) for i,n,_ in experts(parent,core)]),
 provenance={'previous_run':'weights-real-learning-v1','weights':'parent.json'},scores=parent_scores)
for row in rows[1:]:
 row['admission']=history.admission(parent_scores,dict(admission_scores(row),caps_passed=row['caps_passed'],fixtures_passed=True))
history.save(ROOT/'opponent-history.json')
receipt={'parent_hash':digest(parent),'child_hash':digest(child),'inherited_tensors':proposal['inherited_names'],
 'new_weights':[k for k in child if k not in parent],'written_scalar_values':65,'new_parameter_positions':35,
 'preserved_router_block':[5,5],'candidate_type':'LLM-authored structural/numerical mutation, not crossover',
 'raw_proposal':'proposal.json','promoted':False}
(ROOT/'child-receipt.json').write_text(json.dumps(receipt,indent=2))
charges=[{'arm':arm,'seed':seed,'charged':True,'hash_or_null':digest(child if arm=='llm' else weights['random_'+str(seed)]),'outcome':'rejected','shared_author_response':arm=='llm'} for seed in seeds for arm in ('llm','random')]
summary={'rows':rows,'charged_proposals':charges,'physical_maker_candidates':1,'runtime_seconds':time.perf_counter()-START,
 'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'promoted':False,'running':False,
 'interpretation':'Development gains are not transfer. New task-heldout reporting is the primary criterion; small sample and shared maker response limit inference.'}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'event':'finished','seconds':summary['runtime_seconds'],'child':digest(child),'promoted':False}),flush=True)

