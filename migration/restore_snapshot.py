"""Restore a CellForge content-addressed snapshot to a NEW directory."""
import argparse,hashlib,json,zipfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('destination');a=p.parse_args()
out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(a.snapshot) as z:
 d=json.loads(z.read('receipts/inventory.json'))
 for item in d['files']:
  dest=(out/item['path']).resolve()
  if not dest.is_relative_to(out):raise ValueError(item['path'])
  b=z.read('objects/'+item['sha256'])
  if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError(item['path'])
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 for name in z.namelist():
  if name.startswith('receipts/'):
   dest=(out/name).resolve()
   if not dest.is_relative_to(out):raise ValueError(name)
   dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
print('Restored and verified',len(d['files']),'files')
if (out/'receipts/history.bundle').exists():
 print('Recover Git history with: git clone',str(out/'receipts/history.bundle'),'<new-checkout>')

