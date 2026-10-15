import json,pathlib,hashlib
root=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001/runs/weights-action-continuation')
author=root.parent.parent/'author-cell-g3'
def d(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
response=json.loads((author/'response.json').read_text());prompts=json.loads((root/'PROMPTS-3.json').read_text())
for q,r in zip(prompts,response):
 assert q['seed']==r['seed']
 assert r['response']['prompt_sha256']==d(q['prompt'])
 assert r['response']['feedback_sha256']==d(q['prompt']['development_evidence'])
assert not (root/'RESPONSES-3.json').exists()
raw=json.dumps(response,indent=2).encode();(root/'RESPONSES-3.json').write_bytes(raw)
decl={'initial_anchor':'bcfdbec09bc4464c8f147517633e76f831ddf0af34121c41d3592ff27a6caeb9','phase':3,'parent_state_hash':d(json.loads((root/'STATE-2.json').read_text())),'responses_sha256':hashlib.sha256(raw).hexdigest(),'prompts_sha256':d(prompts)}
raw=json.dumps(decl,indent=2).encode();(root/'PHASE-3-DECLARATION.json').write_bytes(raw)
print('phase3_anchor',hashlib.sha256(raw).hexdigest());print('responses_sha256',decl['responses_sha256'])
