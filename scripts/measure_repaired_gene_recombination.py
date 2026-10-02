# SPDX-License-Identifier: MIT-0
"""Eight predeclared crossover probes of an admitted repaired gene."""
import json,time,hashlib
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_curriculum import generate_task
from cellforge.reference.grid_false_positives import score_with_consistency
from expanded_grid_genetics import instantiate,crossover,canonical

def main():
 out=Path('runs/repaired-gene-recombination-v1')
 if out.exists():raise ValueError('fresh output required')
 base=Graph.load('runs/maker-new-gene-transaction-v1/candidate');identity='geometry_0';tile='maker_tile_repeat.gene'
 left=instantiate(base,[tile]+[identity]*6);right=instantiate(base,[identity,'bbox.geometry','bank_local_fill.gene']+[identity]*4)
 parents=[left.hash(),right.hash()];graphs=[('tile-parent',left),('crop-fill-parent',right)]+[(f'child-{s}',crossover(left,right,s)) for s in range(8)]
 tasks=[generate_task(f,3337000+j*10000+i) for j,f in enumerate(('tiling2x','bbox_crop_bg0','local_fill')) for i in range(10)]
 policy=dict(protocol='repaired-gene-recombination/1',base_hash=base.hash(),parent_hashes=parents,seeds=list(range(8)),tasks=30,manifest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),scope='predeclared offspring gene-retention probe, not population efficacy',gradients=False,native_loaded=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 out.mkdir();ledger=Ledger(out/'ledger.jsonl');ledger.append('policy_frozen',policy,'operator');rows=[];start=time.perf_counter()
 for name,g in graphs:
  canonical(base,g);g.save(out/name);r=score_with_consistency([(str(i),t) for i,t in enumerate(tasks)],GridPredictor(g,backend='compiled'),record_tasks=True)
  row=dict(name=name,hash=g.hash(),genome=g.meta['expanded_genome'],score=r);rows.append(row);ledger.append('graph_scored',row,'operator')
 assert [left.hash(),right.hash()]==parents
 result=dict(policy=policy,rows=rows,wall_seconds=time.perf_counter()-start)
 ledger.append('completed',result,'operator');result['ledger_verified']=ledger.verify();(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(scores=[(r['name'],r['score']['tasks_solved']) for r in rows],wall_seconds=result['wall_seconds'])))
if __name__=='__main__':main()
