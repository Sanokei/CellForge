# SPDX-License-Identifier: MIT-0
"""Joint authored repairs with same-bank inactive control and fresh fixtures."""
import hashlib
import json
import time
from pathlib import Path
import psutil

from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile
from cellforge.construction.transactions import apply_ops, MAX_GRID_BANK_GENES
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_curriculum import generate_task
from cellforge.reference.marker_mirror_curriculum import generate_marker_task
from cellforge.reference.grid_false_positives import score_with_consistency
from measure_numeric_repair_structural_transfer import structural_examples
from expanded_grid_genetics import canonical


def main():
    out=Path('runs/joint-repair-bank-v1')
    if out.exists():raise ValueError('fresh output required')
    parent=Graph.load('runs/arc-grid-genes/expanded-breeding-v2/base')
    specs=[(12,'joint_tile_','repeat.gene','runs/arc-grid-genes/tile2-v1'),
           (13,'joint_mirror_','repeat.gene','runs/arc-grid-genes/marker-mirror-v1'),
           (14,'joint_numeric_','adjacency_recolor.gene',
            'runs/numeric-weight-repair-prototype-v3/maker_repair-round-2')]
    child=parent
    imports=[]
    for slot,prefix,name,path in specs:
        donor=Graph.load(path);before=child.hash()
        child,_=apply_ops(child,[dict(op='import_slot_gene',slot=slot,
             prefix=prefix,name=name,**{'from':donor.hash()})],resolver=lambda ref,d=donor:d)
        imports.append(dict(parent_hash=before,child_hash=child.hash(),donor_hash=donor.hash(),slot=slot))
    control,_=apply_ops(child,[dict(op='set_slot_gene',slot=slot,gene='geometry_0')
                              for slot,_,_,_ in specs])
    canonical(child,control)
    for graph in (child,control):check_profile(graph,'ladder-micro/1')
    assert parent.hash()==imports[0]['parent_hash']
    tasks=[generate_task('tiling2x',4537000+i) for i in range(4)]
    tasks += [generate_marker_task(4547000+i) for i in range(4)]
    tasks += structural_examples(4557000,4)
    out.mkdir();ledger=Ledger(out/'ledger.jsonl')
    graphs={'inactive-repairs':control,'joint-repairs':child}
    hashes={name:g.save(out/name) for name,g in graphs.items()}
    policy=dict(protocol='joint-authored-repair-bank/1',hashes=hashes,
        imports=imports,carrying_capacity=MAX_GRID_BANK_GENES,bank_size=10,
        tasks=12,gradients=False,native_loaded=0,
        scope='same bank/weights; last three slot activations differ; no population efficacy',
        manifest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest())
    ledger.append('policy_frozen',policy,'operator');rows=[];start=time.perf_counter()
    for name,g in graphs.items():
        row=dict(name=name,stats=g.stats(),score=score_with_consistency(
            [(str(i),t) for i,t in enumerate(tasks)],GridPredictor(g,backend='compiled'),record_tasks=True))
        rows.append(row);ledger.append('fixtures_scored',row,'operator')
    info=psutil.Process().memory_info()
    result=dict(policy=policy,rows=rows,wall_seconds=time.perf_counter()-start,
        peak_rss_bytes=getattr(info,'peak_wset',info.rss),platform='Windows local, not server calibration')
    ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify()
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/'policy.json').write_text(json.dumps(policy,indent=2))
    print(json.dumps(dict(scores=[(r['name'],r['score']['tasks_solved'],r['score']['demo_consistent_but_wrong_tasks']) for r in rows],stats=child.stats(),peak_rss=result['peak_rss_bytes'],wall=result['wall_seconds'])))


if __name__=='__main__':main()
