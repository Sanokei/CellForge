# SPDX-License-Identifier: MIT-0
import argparse,json
from pathlib import Path
from cellforge.governance.ledger import Ledger

def verify(root):
 r=json.loads((root/'result.json').read_text());l=Ledger(root/'ledger.jsonl');assert l.verify()==4
 assert l.find('completed')[0]['payload']=={k:v for k,v in r.items() if k!='ledger_verified'}
 records=[]
 for row in r['rows']:
  s=row['score'];v=s['task_records'];assert len(v)==200 and len({x['task_id'] for x in v})==200
  assert s['tasks_solved']==sum(x['solved'] for x in v)
  assert s['demo_consistent_but_wrong_tasks']==sum(x['demo_consistent_but_wrong'] for x in v)
  records.append({x['task_id']:x for x in v})
 a,b=records;assert a.keys()==b.keys()
 assert r['gains']==[k for k in a if b[k]['solved'] and not a[k]['solved']]
 assert r['losses']==[k for k in a if a[k]['solved'] and not b[k]['solved']]
 return dict(rows=[dict(arm=x['arm'],solved=x['score']['tasks_solved'],wrong=x['score']['demo_consistent_but_wrong_tasks'],tasks_per_second=x['score']['tasks_per_second'],cpu_seconds_per_task=x['score']['inference_cpu_seconds_per_task']) for x in r['rows']],gains=r['gains'],losses=r['losses'],manifest=r['policy']['manifest'],wall_seconds=r['wall_seconds'],peak_rss_bytes=r['peak_rss_bytes'],ledger_verified=4)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.root)))
