# SPDX-License-Identifier: MIT-0
"""Freeze authored modal crop in a graph-owned associative portfolio."""
import json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import import_def,apply_ops
from cellforge.ir.profiles import check_profile
from expanded_grid_genetics import canonical

def main():
 out=Path('runs/modal-crop-moe-v1')
 if out.exists():raise ValueError('fresh inventory required')
 parent=Graph.load('runs/arc-grid-genes/expanded-breeding-v2/base');donor=Graph.load('runs/arc-grid-genes/modal-crop-v1');base=parent.copy()
 imported=import_def(base,donor,'repeat.gene',prefix='modal_bank_');gene='modal_bank_repeat.gene'
 base.meta['expanded_genome']['available_genes'].append(gene)
 base.meta['modal_import']=dict(parent_hash=parent.hash(),donor_hash=donor.hash(),imported_defs=imported)
 child,_=apply_ops(base,[dict(op='set_slot_gene',slot=8,gene=gene)])
 canonical(base,child);check_profile(base,'ladder-micro/1');check_profile(child,'ladder-micro/1')
 out.mkdir();hashes={k:g.save(out/k) for k,g in [('inactive-modal-control',base),('modal-crop-moe',child)]}
 (out/'policy.json').write_text(json.dumps(dict(hashes=hashes,parent_hash=parent.hash(),donor_hash=donor.hash(),shared_parameters=True,scope='authored HGT plus graph-owned associative routing; no population advantage',gradients=False),indent=2))
 print(json.dumps(hashes))
if __name__=='__main__':main()
