import copy,json,hashlib,datetime,sys
from pathlib import Path
import numpy as np
from cellforge.construction.joint_weight_population import digest
try:
    from cellforge.construction.tool_router_evolution import inherited_parent,validate,features,score,admission,ACTIONS
except ModuleNotFoundError:
    from tool_router_evolution import inherited_parent,validate,features,score,admission,ACTIONS

stage=Path(sys.argv[1]);author_path=Path(sys.argv[2]);root=Path(sys.argv[3]);root.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(name,value):
    with (root/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(stage/'PREDECLARED.json');contract=read(stage/'contract.json');author=read(author_path)
if author['model']!='gpt-6.1-sol' or author['reasoning_effort']!='low':raise ValueError('actual requested maker model/effort required')
proposal=validate({'W':author['W'],'b':author['b']});parent=inherited_parent(read(stage/'parent.json')['router'])
corpora={k:read(stage/(k+'.json')) for k in plan['corpora_sha256']}
for k,h in plan['corpora_sha256'].items():
    if sha(stage/(k+'.json'))!=h:raise ValueError('frozen phrase corpus changed')
write('PREDECLARED.json',dict(plan,control_vocabulary=[-10,10],control_accounting='All128new positions including literal authored zeros; nonzero changes15. Correction from sparse15-position preliminary control; maker/corpora remain fixed. Accounting declared before these corrected random draws, after preliminary reporting; not original preregistration.',executed=datetime.datetime.now(datetime.timezone.utc).isoformat(),maker_response_sha256=sha(author_path)))
for k,v in [('author.json',author),('contract.json',contract),('maker-request.json',read(stage/'maker-request.json')),('parent.json',parent),('corpora.json',corpora)]:write(k,v)
parts={'fresh':corpora['fresh'],'archive':corpora['development'],'anchor':corpora['anchor']}
parent_scores={k:score(parent,rows) for k,rows in parts.items()};write('parent-scores.json',parent_scores)
mask={k:np.asarray(proposal[k])!=np.asarray(parent[k]) for k in parent}
changed_mask={k:v.copy() for k,v in mask.items()}
mask['W'][:7,18:]=True;mask['W'][7:,:]=True;mask['b'][7:]=True
positions={k:np.argwhere(v).tolist() for k,v in mask.items()}
write('edited-positions.json',positions);records=[];ledger=[];head=None
for seed in plan['seeds']:
    rng=np.random.default_rng(seed);random={}
    for k in parent:
        a=np.asarray(parent[k],dtype=float).copy();values=rng.integers(-10,11,size=a.shape).astype(float);a[mask[k]]=values[mask[k]];random[k]=a.tolist()
    for arm,tensors in [('llm',proposal),('random',random)]:
        scores={k:score(tensors,rows) for k,rows in parts.items()};decision=admission(parent_scores,scores)
        row={'arm':arm,'seed':seed,'weights':tensors,'hash':digest(tensors),'parents':[digest(parent)],'scores':scores,'decision':decision,'charged':True}
        records.append(row)
        event={'arm':arm,'seed':seed,'hash_or_null':row['hash'],'charged':True,'outcome':'accepted' if decision['accepted'] else 'rejected','edited_scalar_count':sum(int(v.sum()) for v in mask.values()),'previous':head}
        head=digest(event);event['hash']=head;ledger.append(event)
write('candidate-receipts.json',records)
write('FROZEN.json',{'candidate_hashes':[r['hash'] for r in records],'decisions_fixed_before_report':True,'promoted_contexts':[r['seed'] for r in records if r['arm']=='llm' and r['decision']['accepted']]})
report_parent=score(parent,corpora['report']);report_proposal=score(proposal,corpora['report'])
zero={'W':np.zeros((9,28)).tolist(),'b':np.zeros(9).tolist()}
noextension=copy.deepcopy(proposal)
for row in noextension['W']:row[18:]=[0]*10
ablations={label:{k:score(t,rows) for k,rows in dict(parts,report=corpora['report']).items()} for label,t in [('zero_weights_tie_first',zero),('appended_features_zeroed',noextension)]}
write('ablations.json',ablations);write('report.json',{'parent':report_parent,'authored':report_proposal,'random':[{'seed':r['seed'],'score':score(r['weights'],corpora['report'])} for r in records if r['arm']=='random'],'scope':'distinct primary phrase author and low maker, frozen before author; small constructed suite, not independent external benchmark'})
weights=root/'weights';weights.mkdir()
for tensors in [parent]+[r['weights'] for r in records]:
    p=weights/(digest(tensors)+'.json')
    if not p.exists():
        with p.open('x') as f:json.dump({'tensors':tensors},f)
write('STANDARD.json',{'arms':{'llm':{'model':author['model'],'effort':author['reasoning_effort']},'random':{'same_edited_positions':True,'range':[-10,10]}},'seeds':plan['seeds'],'seeds_declared_before_run':True,'budget_per_arm':1,'niches':['tool_action_router'],'founders':[{'hash':digest(parent),'niche':'tool_action_router','author':'llm','actor':'inherited Claude Sonnet5.5 baseline','rationale':'exact old numerical router zero-extended into predeclared typed contract; not a new founder maker call','weights_file':'weights/'+digest(parent)+'.json'}],'genomes':[{'hash':r['hash'],'niche':'tool_action_router','origin':'llm_mutation' if r['arm']=='llm' else 'random_mutation','parents':r['parents'],'weights_file':'weights/'+r['hash']+'.json','accuracy':r['scores']['fresh']['solves']/r['scores']['fresh']['tasks'],'accuracy_partition':'development','generation':1,'arm':r['arm'],'seed':r['seed']} for r in records],'proposals':[{k:e[k] for k in ('arm','seed','hash_or_null','charged','outcome')} for e in ledger],'status':'bounded router-action classification pilot; not full architecture/scientific milestone','physical_maker_responses':1,'independent_maker_replicates':False,'ledger_file':'ledger.jsonl','limitations':['Action decisions only; actual write/edit cells belong to main','No public ARC or external benchmark','One author response reused across contexts','No independent cell admission/crossover in router track','Original7/9 same-author evidence reclassified as historical DEV exposure']})
with (root/'ledger.jsonl').open('x') as f:
    for e in ledger:f.write(json.dumps(e,sort_keys=True)+'\n')
summary={'parent':{k:{'solves':s['solves'],'tasks':s['tasks'],'false_actions':s['false_action_count']} for k,s in parent_scores.items()},'authored':{k:{'solves':s['solves'],'tasks':s['tasks'],'false_actions':s['false_action_count']} for k,s in records[0]['scores'].items()},'random':[{'seed':r['seed'],'solves':{k:v['solves'] for k,v in r['scores'].items()},'decision':r['decision']} for r in records if r['arm']=='random'],'authored_admission':records[0]['decision'],'report':{'parent':[report_parent['solves'],report_parent['tasks']],'authored':[report_proposal['solves'],report_proposal['tasks']]},'changed_scalars':sum(int(v.sum()) for v in changed_mask.values()),'authored_new_or_changed_scalars':sum(int(v.sum()) for v in mask.values()),'physical_author_responses':1,'uniform_action_expected_accuracy':1/len(ACTIONS),'original_fixed_feature_collision':'Both supplied misses and fix the build are all-zero under original18features, so no W/b-only mutation can separate them','action_execution':False,'scope':'small isolated numerical router prototype; main must independently admit actual action cells'}
write('summary.json',summary)
write('INTEGRITY.json',{'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},'ledger_head':head})
print(json.dumps(summary,indent=2))


