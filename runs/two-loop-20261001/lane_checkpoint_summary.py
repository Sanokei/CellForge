import pathlib,json
b=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/runs')
for name in ('weights-spatial-hardlane','weights-arithmetic-subtraction-continuous'):
 r=b/name;s=json.loads(max(r.glob('STATE-*.json'),key=lambda p:int(p.stem.split('-')[1])).read_text());summary={'run':name,'generation':s['generation'],'status':json.loads((r/'STATUS.json').read_text()),'champions':[]}
 for c in s['contexts']:
  for a in ('llm','random'):
   live=c['arms'][a]['live'];ids=list(live.values()) if isinstance(live,dict) else [live]
   for cid in ids:summary['champions'].append({'seed':c['seed'],'arm':a,'id':cid,'solved':s['cells'][cid]['evidence']['solved'],'total':s['cells'][cid]['evidence']['total']})
 if (r/'AUTHOR-1/EXIT.json').exists():summary['first_author']=json.loads((r/'AUTHOR-1/EXIT.json').read_text())
 print(json.dumps(summary))
