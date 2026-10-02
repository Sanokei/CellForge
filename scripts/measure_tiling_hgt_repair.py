# SPDX-License-Identifier: MIT-0
"""Authored tiling HGT repair; matched inactive-donor control, local only."""
import json,time,hashlib
from pathlib import Path
import psutil
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import import_def,apply_ops
from cellforge.ir.profiles import check_profile
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency
from expanded_grid_genetics import canonical
from measure_grid_expanded_capacity import family

def main():
 out=Path('runs/tiling-hgt-repair-local-v1')
 if out.exists():raise ValueError('fresh output required')
 parent=Graph.load('runs/expanded-breed-repair-pilot-v1/maker_repair');donor=Graph.load('runs/arc-grid-genes/tile2-v1')
 base=parent.copy();imported=import_def(base,donor,'repeat.gene',prefix='tiling_bank_');gene='tiling_bank_repeat.gene'
 base.meta['expanded_genome']['available_genes'].append(gene)
 base.meta['tiling_import']=dict(parent_hash=parent.hash(),donor_hash=donor.hash(),imported_defs=imported)
 child,_=apply_ops(base,[dict(op='set_slot_gene',slot=8,gene=gene)])
 canonical(base,child);check_profile(child,'ladder-micro/1')
 out.mkdir();hashes={k:g.save(out/k) for k,g in [('inactive-donor-control',base),('tile-repair',child)]}
 tasks=family(3277000,10);policy=dict(hashes=hashes,parent_hash=parent.hash(),donor_hash=donor.hash(),seed=3277000,tasks=120,manifest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='authored HGT slot repair, local synthetic utility; no population/evolution or native transfer claim',gradients=False,native_loaded=0)
 (out/'policy.json').write_text(json.dumps(policy,indent=2));rows=[];start=time.perf_counter()
 for arm,g in [('inactive-donor-control',base),('tile-repair',child)]:
  rows.append(dict(arm=arm,stats=g.stats(),score=score_with_consistency([(str(i),t) for i,t in enumerate(tasks)],GridPredictor(g,backend='compiled'),record_tasks=True)))
 result=dict(policy=policy,rows=rows,wall_seconds=time.perf_counter()-start,peak_rss_bytes=psutil.Process().memory_info().peak_wset,resource_scope='local Windows whole-process high-water, not server performance')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(rows=[dict(arm=r['arm'],solved=r['score']['tasks_solved'],tasks_per_second=r['score']['tasks_per_second']) for r in rows],wall_seconds=result['wall_seconds'])))
if __name__=='__main__':main()
