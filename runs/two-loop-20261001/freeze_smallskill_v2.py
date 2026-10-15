import pathlib,json,shutil,tarfile,hashlib
base=pathlib.Path('/srv/cellforge-inbox/two-loop-20261001');old=base/'runs/weights-arc-moe-continuation';new=base/'runs/weights-arc-moe-smallskill-v2'
assert not new.exists();new.mkdir()
previous=json.loads((old/'DECLARATION.json').read_text())
for name in previous['dependencies']:
 p=new/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/name,p)
with tarfile.open(base/'moe-smallskill-v2-stage.tar.gz') as t:
 for member in t.getmembers():
  p=(new/member.name).resolve();assert p.is_relative_to(new.resolve());t.extract(member,new)
deps={name:hashlib.sha256((new/name).read_bytes()).hexdigest() for name in previous['dependencies']}
deps['PRIMITIVES.json']=hashlib.sha256((new/'PRIMITIVES.json').read_bytes()).hexdigest()
for p in (new/'src').rglob('*.py'):deps[str(p.relative_to(new))]=hashlib.sha256(p.read_bytes()).hexdigest()
decl={'schema':'prospective-ARC-MoE/1','dependencies':deps}
(new/'DECLARATION.json').write_text(json.dumps(decl,indent=2))
print('new_moe_anchor',hashlib.sha256((new/'DECLARATION.json').read_bytes()).hexdigest());print('bound_dependencies',len(deps));print('root',new)
