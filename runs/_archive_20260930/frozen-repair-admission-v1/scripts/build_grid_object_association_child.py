# SPDX-License-Identifier: MIT-0
"""Import authored largest-object gene into graph-owned color association slot."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import import_def
from build_grid_curriculum_portfolio import build_portfolio
from build_grid_largest_component_gene import build_largest_component_gene


def build_child(parent,iterations=64):
    g=build_portfolio(parent); donor=build_largest_component_gene(iterations)
    imported=import_def(g,donor,'largest4.gene',prefix='object_bank_')
    # The imported gene is flat, sharing the same typed geometry ports.
    g.defs['slot9_geometry']=copy.deepcopy(g.defs['object_bank_largest4.gene'])
    g.meta.update(construction='authored object+association child/1',
                  parent_hashes=[parent.hash(),donor.hash()],operator='import_def + typed slot replacement',
                  gradients=False,component_iterations=iterations,
                  limitations=f'{iterations}-step bounded component convergence; heavy membership tensor; routing proxy is not exact demo verification',
                  authoring_evidence='training-only grid-object-census-v1; one newly consistent largest-mono+color-map hypothesis')
    g.validate();return g


def family(seed=2617000):
    rng=np.random.default_rng(seed);tasks=[]
    for i in range(300):
        bg,large,small,target=map(int,rng.choice(10,4,replace=False));pairs=[]
        for j in range(4):
            x=np.full((14,14),bg,dtype=int)
            r,c=map(int,rng.integers(1,5,size=2));x[r:r+3,c:c+2]=large
            x[11,11]=small
            pairs.append({'input':x.tolist(),'output':np.full((3,2),target,dtype=int).tolist()})
        tasks.append({'train':pairs[:3],'test':pairs[3:]})
    return dict(seed=seed,tasks=tasks,provenance='original largest-monochrome rectangle + per-task recolor; two objects separated; no ARC inputs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--out',required=True);p.add_argument('--iterations',type=int,choices=(16,64),default=64);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('fresh output required')
    parent=Graph.load(a.parent);g=build_child(parent,a.iterations);h=g.save(out/'child')
    control=build_portfolio(parent);ch=control.save(out/'control')
    (out/'AUTHORED_FAMILY_300.json').write_text(json.dumps(family(),indent=2)+'\n')
    print(json.dumps(dict(child=h,control=ch,stats=g.stats(),parents=g.meta['parent_hashes'])))


if __name__=='__main__':main()
