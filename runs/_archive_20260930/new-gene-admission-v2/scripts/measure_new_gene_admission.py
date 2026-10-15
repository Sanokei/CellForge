# SPDX-License-Identifier: MIT-0
"""Admission measurement of already frozen repairs; no new proposal selection."""
import argparse,json,hashlib,time,resource
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.construction.repair_acceptance import repair_admission
from cellforge.governance.ledger import Ledger
from measure_grid_expanded_capacity import family

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--caps',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('fresh output required')
 hashes=json.loads((a.pilot/'winners-frozen.json').read_text());graphs={k:Graph.load(a.pilot/k) for k in hashes}
 assert all(g.hash()==hashes[k] for k,g in graphs.items())
 paths=[x for h,x in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in (a.private/'arcagi2/data/training').glob('*.json')) if int(h,16)%5!=0][:200]
 if len(paths)!=200:raise ValueError('complete training anchor required')
 groups={'fresh':[(str(i),t) for i,t in enumerate(family(3317000,2))],'archive':[(str(i),t) for i,t in enumerate(family(3327000,2))],'anchor':[(x.stem,json.loads(x.read_text())) for x in paths]}
 manifests={k:digest(v) for k,v in groups.items()}
 policy=dict(protocol='frozen-repair-admission/1',hashes=hashes,manifests=manifests,fresh_seed=3317000,archive_seed=3327000,anchor_tasks=200,gradients=False,public_evaluation_loaded=False,audit_evaluations=0,final_evaluations=0,scope='development admission for previously exposed pilot candidates, not independent replication',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.out.mkdir();ledger=Ledger(a.out/'ledger.jsonl');ledger.append('policy_frozen',policy,'operator');scores={};start=time.perf_counter()
 for arm,g in graphs.items():
  scores[arm]={}
  for label,tasks in groups.items():
   r=score_with_consistency(tasks,GridPredictor(g,backend='compiled'),record_tasks=True)
   scores[arm][label]=dict(partition='development',manifest=manifests[label],source='public-training-real' if label=='anchor' else 'authored-generator',outcomes={x['task_id']:x['solved'] for x in r['task_records']},false_positive_given_consistency=r['false_positive_given_consistency'],measurement=r)
   ledger.append('development_scored',dict(arm=arm,label=label,score=scores[arm][label]),'operator')
 caps=json.loads(a.caps.read_text());assert caps['policy']['graph_hash']==hashes['maker_repair'];assert caps['fixtures_passed'] and caps['caps_passed']
 decisions={}
 for arm in ('random_repair','maker_repair'):
  # Resource/fixture admission flags deliberately fail closed: not measured here.
  candidate=dict(scores[arm],caps_passed=(arm=='maker_repair'),fixtures_passed=(arm=='maker_repair'))
  decisions[arm]=repair_admission(scores['breeding_only'],candidate)
 result=dict(fixture_caps_receipt=caps,policy=policy,scores=scores,decisions=decisions,wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify();(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(decisions))
if __name__=='__main__':main()
