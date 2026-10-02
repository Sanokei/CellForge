# SPDX-License-Identifier: MIT-0
"""Frozen eight-child gene assembly probe; no native ARC files."""
import hashlib
import json
import runpy
import time
from pathlib import Path

from cellforge.ir.graph import Graph
from cellforge.governance.ledger import Ledger
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_curriculum import generate_task
from cellforge.reference.grid_false_positives import score_with_consistency
from expanded_grid_genetics import instantiate, crossover, canonical


def main():
    out = Path('runs/marker-mirror-recombination-v1')
    if out.exists():
        raise ValueError('fresh output required')
    fixture = runpy.run_path('tests/test_grid_marker_mirror_gene.py')
    tasks = []
    for i in range(4):
        pairs = []
        for j in range(3):
            image, output = fixture['mirror_example'](4489000 + i * 10 + j)
            pairs.append(dict(input=image, output=output))
        tasks.append(dict(family='marker_mirror', train=pairs[:2], test=pairs[2:]))
    for index, family in enumerate(('bbox_crop_bg0', 'local_fill')):
        tasks.extend(generate_task(family, 4499000 + index * 10000 + i)
                     for i in range(4))
    base = Graph.load('runs/marker-mirror-moe-v1/marker-mirror-moe')
    identity = 'geometry_0'
    mirror = base.meta['expanded_genome']['available_genes'][-1]
    left = instantiate(base, [mirror] + [identity] * 6)
    right = instantiate(base, [identity, 'bbox.geometry', 'bank_local_fill.gene'] +
                        [identity] * 4)
    parents = [left.hash(), right.hash()]
    graphs = [('mirror-parent', left), ('crop-fill-parent', right)] + [
        (f'child-{seed}', crossover(left, right, seed)) for seed in range(8)]
    policy = dict(protocol='marker-mirror-recombination/1', seeds=list(range(8)),
                  parent_hashes=parents, tasks=12, gradients=False,
                  manifest=hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                  scope='predeclared assembly probe, no selection or population claim',
                  native_loaded=0,
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in map(Path, [__file__, 'scripts/expanded_grid_genetics.py',
                                          'tests/test_grid_marker_mirror_gene.py'])})
    out.mkdir()
    ledger = Ledger(out / 'ledger.jsonl')
    ledger.append('policy_frozen', policy, 'operator')
    rows = []
    start = time.perf_counter()
    for name, graph in graphs:
        canonical(base, graph)
        graph.save(out / name)
        score = score_with_consistency([(str(i), t) for i, t in enumerate(tasks)],
                    GridPredictor(graph, backend='compiled'), record_tasks=True)
        row = dict(name=name, hash=graph.hash(), genome=graph.meta['expanded_genome'], score=score)
        rows.append(row)
        ledger.append('graph_scored', row, 'operator')
    assert [left.hash(), right.hash()] == parents
    result = dict(policy=policy, rows=rows, wall_seconds=time.perf_counter()-start)
    ledger.append('completed', result, 'operator')
    result['ledger_verified'] = ledger.verify()
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(scores=[(r['name'], r['score']['tasks_solved'],
                        r['score']['demo_consistent_but_wrong_tasks']) for r in rows],
                        wall_seconds=result['wall_seconds'], ledger=result['ledger_verified'])))


if __name__ == '__main__':
    main()
