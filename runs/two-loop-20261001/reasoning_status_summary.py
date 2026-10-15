import pathlib,json
root=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/runs/weights-reasoning-cell-epochs')
n=max(int(p.stem.split('-')[1]) for p in root.glob('STATE-*.json'));state=json.loads((root/f'STATE-{n}.json').read_text())
live=set(v for c in state['contexts'] for a in c['arms'].values() for v in a['live'].values());births=[b for c in state['contexts'] for a in c['arms'].values() for b in a['births']]
print(json.dumps({'generation':n,'unique_stored_genotypes':len(state['cells']),'distinct_live_cells':len(live),'generated_attempts':len(births),'admitted_attempts':sum(b['decision']['accepted'] for b in births),'rejected_attempts':sum(not b['decision']['accepted'] for b in births),'families':{name:len(set(c['arms']['llm']['live'][name] for c in state['contexts'])) for name in state['contexts'][0]['arms']['llm']['live']},'authored_births':sum(len(c['arms']['llm']['births']) for c in state['contexts']),'control_births':sum(len(c['arms']['random']['births']) for c in state['contexts'])}))
for i in range(1,n+1):
 p=root/f'AUTHOR-{i}/EXIT.json';print('author',i,p.read_text())
 records=json.loads((root/f'GENERATION-{i}.json').read_text())['decisions'];print('results',i,[{'niche':x['niche'],'arm':x['arm'],'solved':x['solved'],'accepted':x['accepted']} for x in records])
