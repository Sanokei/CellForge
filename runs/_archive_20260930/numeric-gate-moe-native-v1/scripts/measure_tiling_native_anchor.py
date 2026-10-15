# SPDX-License-Identifier: MIT-0
"""Frozen tiling repair paired public-training anchor; no evaluation files."""
import argparse,json,hashlib,resource,time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.governance.ledger import Ledger

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('fresh output required')
 inv=json.loads((a.inventory/'policy.json').read_text());graphs={k:Graph.load(a.inventory/k) for k in inv['hashes']}
 assert all(g.hash()==inv['hashes'][k] for k,g in graphs.items())
 paths=[x for h,x in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in (a.private/'arcagi2/data/training').glob('*.json')) if int(h,16)%5!=0][:200]
 if len(paths)!=200:raise ValueError('complete anchor required')
 manifest=hashlib.sha256()
 for x in paths:manifest.update(x.stem.encode()+b':'+hashlib.sha256(x.read_bytes()).digest())
 policy=dict(protocol='tiling-hgt-native-anchor/1',hashes=inv['hashes'],manifest=manifest.hexdigest(),tasks=200,gradients=False,audit_evaluations=0,final_evaluations=0,public_evaluation_loaded=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='exposed public-training development, authored repair not independent generalization')
 a.out.mkdir();ledger=Ledger(a.out/'ledger.jsonl');ledger.append('policy_frozen',policy,'operator');rows=[];start=time.perf_counter()
 for arm,g in graphs.items():
  row=dict(arm=arm,stats=g.stats(),score=score_with_consistency([(x.stem,json.loads(x.read_text())) for x in paths],GridPredictor(g,backend='compiled'),record_tasks=True));rows.append(row);ledger.append('anchor_scored',row,'operator')
 left,right=[{x['task_id']:x for x in row['score']['task_records']} for row in rows]
 gains=[k for k in left if right[k]['solved'] and not left[k]['solved']];losses=[k for k in left if left[k]['solved'] and not right[k]['solved']]
 result=dict(policy=policy,rows=rows,gains=gains,losses=losses,wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify();(a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(rows=[dict(arm=x['arm'],solved=x['score']['tasks_solved'],wrong=x['score']['demo_consistent_but_wrong_tasks'],tasks_per_second=x['score']['tasks_per_second']) for x in rows],gains=gains,losses=losses)))
if __name__=='__main__':main()
