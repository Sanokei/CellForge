# SPDX-License-Identifier: MIT-0
import json
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import apply_ops
from cellforge.ir.profiles import check_profile
from expanded_grid_genetics import canonical

def main():
 out=Path('runs/numeric-gate-moe-v1')
 if out.exists():raise ValueError('fresh output required')
 parent=Graph.load('runs/arc-grid-genes/expanded-breeding-v2/base');donor=Graph.load('runs/numeric-weight-repair-prototype-v3/maker_repair-round-2')
 child,_=apply_ops(parent,[dict(op='import_slot_gene',slot=8,prefix='numeric_gate_',name='adjacency_recolor.gene',**{'from':donor.hash()})],resolver=lambda ref:donor)
 control,_=apply_ops(child,[dict(op='set_slot_gene',slot=8,gene='geometry_0')])
 canonical(control,child);check_profile(child,'ladder-micro/1');out.mkdir()
 hashes={k:g.save(out/k) for k,g in [('inactive-numeric-control',control),('numeric-gate-moe',child)]}
 (out/'policy.json').write_text(json.dumps(dict(hashes=hashes,parent_hash=parent.hash(),donor_hash=donor.hash(),scope='shared imported parameters, only slot activation differs; authored numeric donor',gradients=False),indent=2));print(json.dumps(hashes))
if __name__=='__main__':main()
