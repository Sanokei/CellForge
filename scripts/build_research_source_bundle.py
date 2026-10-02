# SPDX-License-Identifier: MIT-0
"""Hash-bound local research source closure; no network or execution."""
import argparse,ast,hashlib,json,re,tarfile
from pathlib import Path

def script_closure(entry):
 entry=Path(entry).resolve();directory=entry.parent;pending=[entry];found=set()
 while pending:
  path=pending.pop()
  if path in found:continue
  if not path.is_file():raise ValueError('missing script dependency: '+str(path))
  if path.parent!=directory or path.is_symlink():raise ValueError('script dependency escapes source directory')
  found.add(path);text=path.read_text(encoding='utf-8');tree=ast.parse(text)
  names=set()
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):names.update(x.name.split('.')[0]+'.py' for x in node.names)
   elif isinstance(node,ast.ImportFrom) and node.module:names.add(node.module.split('.')[0]+'.py')
  # Includes explicit sibling filenames used by importlib.spec_from_file_location.
  explicit=set(re.findall(r"[\"']([A-Za-z_][A-Za-z_0-9]*\.py)[\"']",text))
  for name in names|explicit:
   candidate=directory/name
   if candidate.is_file() or name in explicit:pending.append(candidate)
 return sorted(found)

def build_bundle(repo,entry,artifacts,out):
 repo=Path(repo).resolve();out=Path(out)
 if out.exists() or out.with_suffix('.manifest.json').exists():raise ValueError('fresh bundle required')
 files=set(script_closure(repo/entry))
 files.update((repo/'src').rglob('*.py'))
 for artifact in artifacts:
  path=(repo/artifact).resolve()
  if path.is_dir():files.update(p for p in path.rglob('*') if p.is_file())
  elif path.is_file():
   files.add(path)
   if path.suffix=='.py' and path.parent==repo/'scripts':files.update(script_closure(path))
  else:raise ValueError('missing artifact: '+str(path))
 records=[]
 for path in sorted(files):
  if path.is_symlink() or not path.resolve().is_relative_to(repo):raise ValueError('bundle file escapes repository')
  records.append({'path':path.relative_to(repo).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
 out.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(out,'w:gz') as archive:
  for row in records:archive.add(repo/row['path'],arcname=row['path'],recursive=False)
 receipt={'protocol':'research-source-bundle/1','entry':str(entry),'files':records,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'preflight_executed':False,'limitation':'static sibling dependency closure; dynamic computed imports and non-Python resources require explicit artifacts and remote import preflight'}
 out.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--entry',required=True);p.add_argument('--artifact',action='append',default=[]);p.add_argument('--out',required=True);a=p.parse_args();r=build_bundle(a.repo,a.entry,a.artifact,a.out);print(json.dumps({'files':len(r['files']),'archive_sha256':r['archive_sha256']}))
if __name__=='__main__':main()
