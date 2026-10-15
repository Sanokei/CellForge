import pathlib,json,hashlib
root=pathlib.Path('/srv/cellforge-inbox/weights-real-learning-v2-20261001')
policy=json.loads((root/'PREDECLARED.json').read_text())
out=[]
for row in policy['partitions']['reporting']:
 p=pathlib.Path('/srv/cellforge-private/arcagi2/data/training')/(row['id']+'.json')
 raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('dataset changed')
 out.append({'id':row['id'],'sha256':row['sha256'],'task':json.loads(raw)})
(root/'REPORTING_TASKS.json').write_text(json.dumps(out))
print(len(out))

