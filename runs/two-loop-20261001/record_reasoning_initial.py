import pathlib,tarfile,json,hashlib,os
base=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001');root=base/'runs/weights-reasoning-cell-epochs';control=base/'control/reasoning-cell-epochs'
anchor='18bff7985ec2306bea9b816bbcd4359c2c57235586129f56dd12465ce5ccf164'
if root.exists():
 assert not list(root.glob('STATE-*.json'))
 if (root/'DECLARATION.json').exists():os.replace(root/'DECLARATION.json',root/'DECLARATION-FAILED-PATHS.json')
else:root.mkdir()
with tarfile.open(base/'reasoning-cell-bootstrap-stage.tar.gz') as t:
 for member in t.getmembers():
  assert (root/member.name).resolve().is_relative_to(root.resolve());t.extract(member,root)
assert hashlib.sha256((root/'DECLARATION.json').read_bytes()).hexdigest()==anchor
record=json.loads((root/'DECLARATION.json').read_text())
for name,expected in record['dependencies'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
seeds=json.loads((root/'SEEDS.json').read_text());assert {s['structure'] for s in seeds}=={'all_present','any_present'}
assert all(s['source_graph_hash']=='3e9019e45acccc3b18118794b802774e76a839cfedbb89be69880183f9138b0c' for s in seeds)
manifest=json.loads((root/'TASKS.json').read_text());ids=set()
for partition in ('development','frozen_reporting'):
 for niche,rows in manifest[partition].items():
  for row in rows:
   assert row['partition']==partition and row['id'] not in ids;ids.add(row['id'])
   target=int(row['input'][0]==row['input'][1]) if niche=='all_present' else int(row['input'][0]>0)
   assert row['expected']==target
control.mkdir(parents=True)
event={'initial_anchor':anchor,'schema':'separate-trusted-input-recorder/1','recorder':'non-model process validates all input/source hashes, reference semantics, source identity before scoring','dependency_count':len(record['dependencies']),'root':str(root),'heldout_use':'never passed to author or selection'}
with (control/'INITIAL-COMMITMENT.json').open('x') as f:json.dump(event,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(event))

