# SPDX-License-Identifier: MIT-0
"""Verify frozen repair admission receipts without opening any task data."""
import argparse,json,hashlib
from pathlib import Path
from cellforge.governance.ledger import Ledger
from cellforge.construction.repair_acceptance import repair_admission
from cellforge.ir.graph import Graph

def verify(root,pilot,caps_file=None):
 r=json.loads((root/'result.json').read_text());ledger=Ledger(root/'ledger.jsonl');assert ledger.verify()==11
 assert ledger.find('completed')[0]['payload']=={k:v for k,v in r.items() if k!='ledger_verified'}
 assert ledger.find('policy_frozen')[0]['payload']==r['policy']
 assert set(r['scores'])=={'breeding_only','random_repair','maker_repair'}
 hashes=json.loads((pilot/'winners-frozen.json').read_text());assert hashes==r['policy']['hashes']
 for arm,h in hashes.items():
  assert Graph.load(pilot/arm).hash()==h
  assert set(r['scores'][arm])=={'fresh','archive','anchor'}
  for label,s in r['scores'][arm].items():
   m=s['measurement'];records=m['task_records'];assert len(records)==m['tasks']==(200 if label=='anchor' else 24)
   assert len({x['task_id'] for x in records})==len(records)
   assert s['outcomes']=={x['task_id']:x['solved'] for x in records}
   assert m['tasks_solved']==sum(x['solved'] for x in records)
   consistent=sum(x['demo_consistent'] for x in records);wrong=sum(x['demo_consistent_but_wrong'] for x in records)
   assert consistent==m['demo_consistent_tasks'] and wrong==m['demo_consistent_but_wrong_tasks']
   assert all(x['demo_consistent_but_wrong']==(x['demo_consistent'] and not x['solved']) for x in records)
   assert s['false_positive_given_consistency']==m['false_positive_given_consistency']==(wrong/consistent if consistent else None)
   assert s['manifest']==r['policy']['manifests'][label]
 for arm in ('random_repair','maker_repair'):
  passed=False
  if 'fixture_caps_receipt' in r:
   assert caps_file is not None
   caps=json.loads(caps_file.read_text());assert caps==r['fixture_caps_receipt']
   assert caps['policy']['graph_hash']==hashes['maker_repair']
   assert caps['policy']['source_sha256']==hashlib.sha256(Path('scripts/measure_grid_fixture_caps.py').read_bytes()).hexdigest()
   assert caps['stats']==Graph.load(pilot/'maker_repair').stats()
   assert {(x['size'],x['phase']) for x in caps['cases']}=={(1,0.),(1,1.),(30,0.),(30,1.)}
   assert all(x['finite'] and x['parity'] for x in caps['cases'])
   assert caps['caps_passed']==(caps['peak_rss_bytes']<=caps['policy']['device_rss_limit_bytes'] and caps['wall_seconds']<=caps['policy']['wall_limit_seconds'] and caps['stats']['unique_param_storage']<=caps['policy']['stored_parameter_limit'])
   assert caps['fixtures_passed'] and caps['caps_passed']
   passed=arm=='maker_repair'
  assert r['decisions'][arm]==repair_admission(r['scores']['breeding_only'],dict(r['scores'][arm],caps_passed=passed,fixtures_passed=passed))
 return dict(ledger_verified=11,rows=[dict(arm=arm,scores={k:dict(solved=s['measurement']['tasks_solved'],wrong=s['measurement']['demo_consistent_but_wrong_tasks'],tasks_per_second=s['measurement']['tasks_per_second']) for k,s in groups.items()}) for arm,groups in r['scores'].items()],decisions=r['decisions'],wall_seconds=r['wall_seconds'],peak_rss_bytes=r['peak_rss_bytes'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--caps-file',type=Path);a=p.parse_args();print(json.dumps(verify(a.root,a.pilot,a.caps_file)))
