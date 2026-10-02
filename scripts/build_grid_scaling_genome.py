# SPDX-License-Identifier: MIT-0
"""Seven-gene typed bank; fixed-slot breeding, no benchmark access."""
import argparse,json
from pathlib import Path
from build_grid_object_genome import build_object_base
from build_grid_repeat_gene import build_repeat_gene
from build_grid_two_slot_genome import instantiate_genome,crossover
from cellforge.construction.transactions import import_def


def build_scaling_base():
    g=build_object_base();imports=[]
    for f in (2,3):
        donor=build_repeat_gene(f)
        transaction=import_def(g,donor,'repeat.gene',prefix=f'scaling_bank{f}_')
        g.meta['two_slot_genome']['available_genes'].append(f'scaling_bank{f}_repeat.gene')
        imports.append(dict(factor=f,donor_hash=donor.hash(),transaction=transaction))
    g.meta['two_slot_genome'].update(protocol='fixed-two-slot-seven-gene-bank/1',scaling_imports=imports)
    g.meta['limitations']='shared seven-gene bank; two mutable slots; online color-association routing proxy; no efficacy claim'
    g.validate();return g


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    base=build_scaling_base();genes=base.meta['two_slot_genome']['available_genes']
    # Every declared combination must preserve the typed microcell hierarchy.
    for left in genes:
        for right in genes:instantiate_genome(base,(left,right))
    layouts={'base':None,'parent-repeat2-identity':('scaling_bank2_repeat.gene','geometry_0'),
             'parent-identity-repeat3':('geometry_0','scaling_bank3_repeat.gene'),
             'child-repeat2-repeat3':('scaling_bank2_repeat.gene','scaling_bank3_repeat.gene')}
    rows=[]
    parents=[instantiate_genome(base,layouts[name]) for name in ('parent-repeat2-identity','parent-identity-repeat3')]
    for name,alleles in layouts.items():
        g=base if alleles is None else instantiate_genome(base,alleles)
        if name=='child-repeat2-repeat3':
            g=crossover(parents[0],parents[1],1)
            assert g.meta['two_slot_genome']['donor_per_slot']==[0,1]
        rows.append(dict(name=name,hash=g.save(a.out/name),stats=g.stats(),genome=g.meta['two_slot_genome']))
    (a.out/'INVENTORY.json').write_text(json.dumps(dict(artifacts=rows,typed_combinations_checked=49,benchmark_tasks_read=0,efficacy_measured=False),indent=2)+'\n')
    print(json.dumps(dict(artifacts=[dict(name=r['name'],hash=r['hash']) for r in rows],combinations=49)))
