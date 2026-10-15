import pathlib,json,hashlib
from cellforge.construction import planning_cell_service as s
r=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/runs/weights-resource-planning');control=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/control/resource-planning')
f=max(r.glob('STATE-*.json'),key=lambda p:int(p.stem.split('-')[1]));state=s.read(f);s.verify(r,state['anchor']);parents=[]
for context in state['contexts']:
 cid=context['arms']['llm']['live'];cell=state['cells'][cid];n=cell['provenance']['generation'];work=r/('AUTHOR-'+str(n));raw=s.read(work/'response.json');response=next(x for x in raw if x['seed']==context['seed']);start=s.read(work/'START.json');end=s.read(work/'EXIT.json');commit=s.read(control/('GENERATION-'+str(n)+'.json'));generation=s.read(r/('GENERATION-'+str(n)+'.json'))
 assert response['model']=='gpt-6.1-sol' and response['reasoning']=='low' and s.digest(response['params'])==s.digest(cell['params'])
 assert start['auth_verified'] and not start['paid_api_fallback'] and start['model']=='gpt-6.1-sol' and end['returncode']==0
 assert commit==generation['commitment'] and commit['raw_response_sha256']==s.file_hash(work/'response.json') and commit['start_sha256']==s.file_hash(work/'START.json') and commit['exit_sha256']==s.file_hash(work/'EXIT.json')
 events=[json.loads(line) for line in (work/'events.jsonl').read_text().splitlines() if line.startswith('{')]
 assert any(e.get('type')=='thread.started' for e in events) and any(e.get('type')=='turn.completed' for e in events)
 messages=[e.get('item',{}).get('text') for e in events if e.get('item',{}).get('type')=='agent_message'];assert messages[-1].strip()==(work/'response.json').read_text().strip()
 parents.append({'seed':context['seed'],'cell':cell,'author_generation':n,'provider_events_sha256':s.file_hash(work/'events.jsonl'),'external_commitment':commit})
bundle={'source':{'run':str(r),'state_file':f.name,'state_hash':s.digest(state),'generation':state['generation'],'selection_partition':'pilot development','control_initialization':'new epoch paired arms clone SAME measured authored parent per seed; old random history remains original run'},'parents':parents}
p=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/planning-successor-parents.json');s.atomic(p,bundle)
print(json.dumps({'source_generation':state['generation'],'parents':[{'seed':p['seed'],'id':p['cell']['id'],'solved':p['cell']['evidence']['solved'],'total':p['cell']['evidence']['total']} for p in parents],'bundle_sha256':s.file_hash(p)}))
