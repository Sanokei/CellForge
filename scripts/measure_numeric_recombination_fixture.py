# SPDX-License-Identifier: MIT-0
"""Frozen known-rule numerical assembly fixture, not an evolutionary study."""
import hashlib
import json
import time
from pathlib import Path
import psutil
from build_grid_numeric_neighbor_gate import build_weight_gate
from measure_numeric_repair_structural_transfer import structural_examples
from cellforge.construction.numeric_recipe import apply_numeric_recipe
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency


def main():
    out = Path('runs/numeric-recombination-fixture-v1')
    if out.exists(): raise ValueError('fresh output required')
    left = build_weight_gate((0, 0, 1, 0), -1.5)
    right = build_weight_gate((0, 0, 0, 1), -1.5)
    recipe = [dict(op='homologous_mask', name='neighbor_gate.W', mask=[[0, 0, 0, 1]])]
    child = apply_numeric_recipe(left, right, recipe)
    tasks = structural_examples(4727000, 30)
    policy = dict(protocol='numeric-recombination-fixture/1', gradients=False,
        native_loaded=0, seed=4727000, tasks=30, recipe=recipe,
        hashes={k:g.hash() for k,g in [('left',left),('right',right),('child',child)]},
        corpus_sha256=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='known-rule author-specified assembly fixture; no selection or causal search claim')
    out.mkdir(); (out/'policy.json').write_text(json.dumps(policy,indent=2))
    rows=[];start=time.perf_counter()
    for name,g in [('left',left),('right',right),('child',child)]:
        g.save(out/name)
        t=time.perf_counter();c=time.process_time()
        score=score_with_consistency([(str(i),v) for i,v in enumerate(tasks)],
            GridPredictor(g,backend='compiled'),record_tasks=True)
        rows.append(dict(name=name,score=score,wall_seconds=time.perf_counter()-t,
            cpu_seconds=time.process_time()-c))
    reference=GridPredictor(child,backend='reference');compiled=GridPredictor(child,backend='compiled')
    for task in tasks[:3]:
        assert reference(task['train'],task['test'][0]['input'])==compiled(task['train'],task['test'][0]['input'])
    mem=psutil.Process().memory_info()
    result=dict(policy=policy,rows=rows,wall_seconds=time.perf_counter()-start,
        peak_rss_bytes=getattr(mem,'peak_wset',None),current_rss_bytes=mem.rss,
        resource_scope='local Windows process; not server throughput',reference_parity_tasks=3)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({r['name']:r['score']['tasks_solved'] for r in rows}))


if __name__=='__main__': main()
