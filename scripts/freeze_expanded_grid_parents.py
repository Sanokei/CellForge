# SPDX-License-Identifier: MIT-0
"""Freeze complementary seven-slot parents and actual crossover children."""
import argparse,hashlib,json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from expanded_grid_genetics import instantiate,crossover,canonical


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh inventory required')
    base=Graph.load(a.base);bank=base.meta['expanded_genome']['available_genes'];identity='geometry_0'
    left=instantiate(base,bank[:3]+[identity]*4)
    right=instantiate(base,[identity]*3+bank[3:])
    a.out.mkdir();ledger=Ledger(a.out/'ledger.jsonl')
    policy=dict(protocol='expanded-seven-slot-parents/1',base_hash=base.hash(),offspring_seeds=list(range(4)),
                gradients=False,benchmark_tasks_read=0,efficacy_measured=False,
                builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                operator_sha256=hashlib.sha256((Path(__file__).parent/'expanded_grid_genetics.py').read_bytes()).hexdigest())
    ledger.append('policy_frozen',policy,'operator');rows=[]
    graphs=[('base',base),('parent-crop-fill',left),('parent-object-adjacency-scaling',right)]
    graphs.extend((f'crossover-{seed}',crossover(left,right,seed)) for seed in range(4))
    for name,g in graphs:
        row=dict(name=name,hash=g.save(a.out/name),canonical_genotype=canonical(base,g),stats=g.stats(),genome=g.meta['expanded_genome'])
        rows.append(row);ledger.append('graph_frozen',row,'operator')
    inventory=dict(policy=policy,artifacts=rows,ledger_verified=ledger.verify())
    (a.out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(graphs=len(rows),ledger_verified=inventory['ledger_verified'],parent_hashes=[left.hash(),right.hash()])))


if __name__=='__main__':main()
