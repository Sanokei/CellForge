# SPDX-License-Identifier: MIT-0
"""Frozen demo-adaptation ablation, no optimizer or public evaluation access."""
import argparse,hashlib,json,resource
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.governance.ledger import Ledger


def family():
    rng=np.random.default_rng(2957000);tasks=[]
    for i in range(100):
        for factor in (1,2,3):
            pairs=[]
            for j in range(4):
                h,w=map(int,rng.integers(1,9,size=2));x=rng.integers(0,10,size=(h,w))
                y=np.repeat(np.repeat(x,factor,axis=0),factor,axis=1)
                pairs.append(dict(input=x.tolist(),output=y.tolist()))
            tasks.append(dict(train=pairs[:3],test=pairs[3:]))
    return tasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--graphs',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    graphs={name:Graph.load(a.graphs/f'scaling-selector{suffix}-v1') for name,suffix in [('adapt',''),('control','-no-update')]}
    d=a.private/'arcagi2/data/training'
    paths=[p for h,p in sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in d.glob('*.json')) if int(h,16)%5]
    manifest=hashlib.sha256();native=[]
    for path in paths:
        raw=path.read_bytes();manifest.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest());native.append((path.stem,json.loads(raw)))
    assert len(native)==789 and manifest.hexdigest()=='9b82cdec59033c4fe67ebd2f8b93ea2a495b90ec9bff5a0e4a5c851cd4291af9'
    tasks=family();fingerprints=[hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest() for t in tasks]
    assert len(set(fingerprints))==300
    policy=dict(protocol='scaling-selector-adaptation-ablation/1',graph_hashes={k:g.hash() for k,g in graphs.items()},
                native_manifest_sha256=manifest.hexdigest(),family_sha256=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=2957000,
                synthetic_partition='first200 development,last100 validation; interleaved factors1/2/3',
                scope='expanded exposed training; targeted scaling demos informed donor authoring; not final transfer',
                gradients=False,public_evaluation_loaded=False,audit_evaluations=0,final_evaluations=0)
    a.out.mkdir();ledger=Ledger(a.out/'ledger.jsonl');ledger.append('policy_frozen',policy,'operator');rows=[]
    for arm,g in graphs.items():
        predict=GridPredictor(g,backend='compiled')
        for split,items in [('synthetic-development',list(enumerate(tasks[:200]))),('synthetic-validation',list(enumerate(tasks[200:],200))),('expanded-native-development',native)]:
            row=dict(score_with_consistency([(str(i),t) for i,t in items],predict,record_tasks=True),arm=arm,split=split,stats=g.stats())
            rows.append(row);ledger.append('graph_scored',row,'operator')
    result=dict(policy=policy,rows=rows,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,ledger_verified=ledger.verify())
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=[{k:v for k,v in r.items() if k not in ('task_records','stats')} for r in rows],ledger_verified=result['ledger_verified'])))


if __name__=='__main__':main()
