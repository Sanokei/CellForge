# SPDX-License-Identifier: MIT-0
"""Frozen complementary typed-slot probe; synthetic only, no population selection."""
import argparse,hashlib,json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from cellforge.governance.ledger import Ledger
from build_grid_object_association_child import family
from build_grid_adjacency_recolor_gene import generate_adjacency_family


def main():
    import resource
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh output required')
    inventory_path=Path(a.inventory);inventory=json.loads(inventory_path.read_bytes())
    tasks=family(2737000)['tasks'][:5]+generate_adjacency_family(5,2747000)['tasks']
    fingerprints=[hashlib.sha256(json.dumps({'train':t['train'],'test':t['test']},sort_keys=True).encode()).hexdigest() for t in tasks]
    if len(set(fingerprints))!=10:raise ValueError('duplicate synthetic I/O')
    out.mkdir();ledger=Ledger(out/'ledger.jsonl')
    policy=dict(protocol='authored-object-slot-recombination-probe/1',tasks=10,
                family_counts={'object_recolor':5,'conditional_adjacency':5},
                task_fingerprints=fingerprints,inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                limitations='explicit child, no population or matched random comparison; tiny exposed synthetic probe only',
                selection=False,gradients=False,benchmark_files_loaded=False,final_evaluations=0)
    ledger.append('policy_frozen',policy,'operator');rows=[]
    for item in inventory['artifacts']:
        g=Graph.load(inventory_path.parent/item['name'])
        if g.hash()!=item['hash']:raise ValueError('artifact changed')
        row=dict(name=item['name'],graph_hash=g.hash(),stats=g.stats(),
                 score=score_with_consistency([(str(i),t) for i,t in enumerate(tasks)],GridPredictor(g,backend='compiled'),record_tasks=True))
        rows.append(row);ledger.append('artifact_scored',row,'operator')
    result=dict(policy=policy,rows=rows,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,ledger_verified=ledger.verify())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=[dict(name=r['name'],solved=r['score']['tasks_solved']) for r in rows],ledger_verified=result['ledger_verified'])))


if __name__=='__main__':main()
