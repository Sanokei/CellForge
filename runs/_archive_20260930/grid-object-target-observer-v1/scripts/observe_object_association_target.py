# SPDX-License-Identifier: MIT-0
"""Post-hoc development observer; frozen models, no graph edits or final access."""
import argparse,hashlib,json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.governance.ledger import Ledger


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--models',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    directory=Path(a.private)/'arcagi2/data/training'
    paths=[p for h,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in directory.glob('*.json')) if int(h,16)%5!=0][:200]
    manifest=hashlib.sha256()
    for path in paths:manifest.update(path.stem.encode()+b':'+hashlib.sha256(path.read_bytes()).digest())
    if manifest.hexdigest()!='4e779267abe74bb4f493268fc49dea3bb5bab8bdd40ac1017676a0ddedcb98f4':raise ValueError('DEV changed')
    target=next(path for path in paths if path.stem=='bf699163')
    raw=target.read_bytes();task=json.loads(raw)
    out=Path(a.out)
    if out.exists():raise ValueError('fresh observer required')
    out.mkdir();ledger=Ledger(out/'ledger.jsonl')
    policy=dict(protocol='posthoc-object-target-observer/1',task_id=target.stem,
                task_sha256=hashlib.sha256(raw).hexdigest(),model_edits=0,
                rationale='selected from training-only object census; not independent validation',
                audit_evaluations=0,final_evaluations=0,gradients=False)
    ledger.append('policy_frozen',policy,'operator');rows=[]
    for arm in ('child','control'):
        g=Graph.load(Path(a.models)/arm)
        row=dict(arm=arm,graph_hash=g.hash(),score=score_with_consistency([(target.stem,task)],GridPredictor(g,backend='compiled'),record_tasks=True))
        ledger.append('target_scored',row,'operator');rows.append(row)
    result=dict(policy=policy,rows=rows,ledger_verified=ledger.verify())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
