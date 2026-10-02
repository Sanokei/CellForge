import json,pathlib
r=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/runs/weights-arithmetic-subtraction-continuous')
s=json.loads(max(r.glob('STATE-*.json'),key=lambda p:int(p.stem.split('-')[1])).read_text())
print(json.dumps({'generation':s['generation'],'champions':[{'seed':c['seed'],'arm':a,'id':c['arms'][a]['live'],'evidence':s['cells'][c['arms'][a]['live']]['evidence']} for c in s['contexts'] for a in ('llm','random')]},indent=2))
