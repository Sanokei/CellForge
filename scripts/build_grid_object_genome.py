# SPDX-License-Identifier: MIT-0
"""Five-gene shared bank including bounded object extraction; no data scoring."""
import argparse,json
from pathlib import Path
from build_grid_two_slot_genome import build_base,instantiate_genome,crossover
from build_grid_largest_component_gene import build_largest_component_gene
from cellforge.construction.transactions import import_def


def build_object_base(parent=None,recolor=None):
    g=build_base(parent,recolor);donor=build_largest_component_gene(16)
    lineage=import_def(g,donor,'largest4.gene',prefix='object_bank_')
    g.meta['two_slot_genome']['available_genes'].append('object_bank_largest4.gene')
    g.meta['two_slot_genome'].update(object_donor_hash=donor.hash(),object_import_lineage=lineage,
                                   protocol='fixed-two-slot-object-bank/1')
    g.meta.update(gradients=False,limitations='typed shared five-gene bank; heavy bounded object extraction when selected; no efficacy claim')
    g.validate();return g


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('fresh output required')
    base=build_object_base()
    left=instantiate_genome(base,('object_bank_largest4.gene','geometry_0'))
    right=instantiate_genome(base,('geometry_0','bank_adjacency_recolor.gene'))
    child=instantiate_genome(base,('object_bank_largest4.gene','bank_adjacency_recolor.gene'),[left.hash(),right.hash()])
    child.meta['two_slot_genome'].update(operator='explicit complementary homologous crossover',donor_per_slot=[0,1])
    rows=[]
    for name,g in [('base',base),('parent-object-identity',left),('parent-identity-adjacency',right),('child-object-adjacency',child)]:
        rows.append(dict(name=name,hash=g.save(out/name),stats=g.stats(),genome=g.meta['two_slot_genome']))
    (out/'INVENTORY.json').write_text(json.dumps(dict(artifacts=rows,benchmark_tasks_read=0,efficacy_measured=False),indent=2)+'\n')
    print(json.dumps(dict(artifacts=[dict(name=x['name'],hash=x['hash']) for x in rows])))


if __name__=='__main__':main()
