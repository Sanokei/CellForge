import pathlib,json,sys,datetime,hashlib
from cellforge.construction import hard_reasoning_adapter as a
from cellforge.construction import hard_reasoning_service as s
r=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/runs/weights-hard-signed-deduction')
p=max(r.glob('STATE-*.json'),key=lambda p:int(p.stem.split('-')[1]));state=s.read(p);s.verify(r,state['anchor'])
ids=sorted({c['arms'][arm]['live'] for c in state['contexts'] for arm in ('llm','random')});out=r/'evaluation-only-first-reporting';out.mkdir(exist_ok=True)
freeze={'utc':s.utc(),'selection_partition':'development','state_file':p.name,'state_hash':s.digest(state),'candidate_ids':ids,'candidate_weights':{cid:state['cells'][cid]['params'] for cid in ids},'frozen_cases_hash':a.digest(a.FROZEN_CASES),'reporting_access':'never passed to maker or selection; eval only; two small authored cases'}
if (out/'CANDIDATES.json').exists():raise ValueError('refuse duplicate reporting')
s.atomic(out/'CANDIDATES.json',freeze)
rows=[]
for cid in ids:
 outcomes={};traces=[]
 for case in a.FROZEN_CASES:
  value=a.predict(state['cells'][cid]['params'],case['input']);pred={k:value[k] for k in ('answer','closure')};expected=a.reference(case['input']);hit=pred==expected;outcomes[case['id']]=hit;traces.append({'id':case['id'],'predicted':pred,'expected':expected,'correct':hit})
 rows.append({'candidate':cid,'solved':sum(outcomes.values()),'total':len(outcomes),'outcomes':outcomes,'traces':traces})
result={'candidate_freeze_hash':s.digest(freeze),'rows':rows,'selection_or_mutation_access':False,'ARC_transfer_claim':False,'generalization_limit':'two same-interface signed temporal deduction cases; not broader benchmark'}
s.atomic(out/'RESULT.json',result);print(json.dumps({'rows':[{k:v for k,v in row.items() if k not in ('traces','outcomes')} for row in rows],'result':str(out/'RESULT.json'),'freeze_hash':s.digest(freeze)}))
