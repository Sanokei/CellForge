# SPDX-License-Identifier: MIT-0
"""Charged layout repair of a bred child; known-slot pilot, not blind efficacy."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import pack_grid,GridPredictor
from cellforge.construction.maker_operator import MakerOperator
from cellforge.construction.maker_diagnostics import train_counterexamples
from cellforge.reference.marker_mirror_curriculum import generate_marker_task
from cellforge.reference.grid_false_positives import score_with_consistency
from measure_grid_expanded_capacity import family
from measure_numeric_repair_structural_transfer import structural_examples
from run_grid_population_matched import evaluate_selection
from expanded_grid_genetics import instantiate,crossover,canonical


def corpus(seed,count):
    return family(seed,count)+[generate_marker_task(seed+1000000+i) for i in range(count)]+structural_examples(seed+2000000,count)


def main():
    out=Path('runs/joint-layout-repair-v1')
    if out.exists():raise ValueError('fresh output required')
    base=Graph.load('runs/joint-repair-bank-v1/joint-repairs')
    alleles=base.meta['expanded_genome']['alleles']
    left=instantiate(base,alleles[:6]+['geometry_0'])
    right=instantiate(base,alleles[:4]+['geometry_0','geometry_0',alleles[6]])
    child=crossover(left,right,1)
    assert child.meta['expanded_genome']['alleles']==alleles
    development=[{'train':t['train'],'family':t.get('family',t.get('metadata',{}).get('family'))}
                 for t in corpus(4567000,1)]
    selection=[{'train':t['train']} for t in development]
    report=corpus(4577000,2)
    # Prior native regression motivated the hypothesis; diagnostics below are
    # fresh synthetic train pairs. This is exposed manual layout repair.
    diagnostics=train_counterexamples(child,development,limit=5,per_task_limit=1)
    rng=np.random.default_rng(773401)
    bank=base.meta['expanded_genome']['available_genes']
    random_ops=[dict(op='set_slot_gene',slot=int(rng.integers(8,15)),
                    gene=bank[int(rng.integers(len(bank)))]) for _ in range(3)]
    maker_ops=[dict(op='set_slot_gene',slot=8,gene='joint_mirror_repeat.gene'),
               dict(op='set_slot_gene',slot=12,gene='object_bank_largest4.gene'),
               dict(op='set_slot_gene',slot=13,gene='joint_tile_repeat.gene')]
    out.mkdir()
    policy=dict(protocol='joint-layout-repair/1',parent_hashes=[left.hash(),right.hash()],
       child_hash=child.hash(),crossover_seed=1,random_seed=773401,ops_per_repair=3,
       evaluations_per_arm=1,selection_tasks=14,report_tasks=28,gradients=False,
       selection_manifest=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),
       report_manifest=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
       scope='known layout displacement; author saw earlier development regression; not causal maker improvement',
       authoring_cost='not measured; not zero',random_ops=random_ops,maker_ops=maker_ops)
    (out/'policy.json').write_text(json.dumps(policy,indent=2))
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    for name,g in [('left-parent',left),('right-parent',right),('bred-child',child)]:g.save(out/name)
    x,_,h,w=pack_grid([[1]])
    fixture=[dict(entry=child.entries['grid_step'],inputs=dict(input=x.tolist(),
        input_height=h.tolist(),input_width=w.tolist(),output=x.tolist(),output_height=h.tolist(),
        output_width=w.tolist(),phase=[[0.]],state=np.zeros((1,1515)).tolist()),
        output_shapes=dict(state=[1,1515],height_logits=[1,2,30],width_logits=[1,2,30],color_logits=[1,2,30,30,10]))]
    def evaluate(graph):
        r=evaluate_selection(graph,selection,rank_consistent=True)
        return dict(reconstructed_tasks=r['solved'],predictor_calls=r['predictor_calls'],failed=r['failed'])
    operator=MakerOperator(out/'maker-ledger.jsonl',budget=3)
    start=time.perf_counter()
    candidates=operator.breed_then_repair_round(child,diagnostics=diagnostics,fixtures=fixture,
        evaluate=evaluate,accept=lambda _:False,repairs={
          'random_repair':dict(ops=random_ops,actor='uniform three-slot edit sampler',model='none'),
          'maker_repair':dict(ops=maker_ops,actor='Codex current assistant',model='session model unspecified')})
    frozen={}
    for arm,(g,receipt) in candidates.items():
        if g is None:raise ValueError('invalid repair arm')
        canonical(base,g);frozen[arm]=g.save(out/arm)
    (out/'winners-frozen.json').write_text(json.dumps(frozen,indent=2))
    rows=[dict(arm=arm,receipt=receipt,report=score_with_consistency(
        [(str(i),t) for i,t in enumerate(report)],GridPredictor(g,backend='compiled'),record_tasks=True))
        for arm,(g,receipt) in candidates.items()]
    MakerOperator(operator.path,budget=3)
    result=dict(policy=policy,rows=rows,wall_seconds=time.perf_counter()-start,archive_admission=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(rows=[dict(arm=r['arm'],report=r['report']['tasks_solved'],
        wrong=r['report']['demo_consistent_but_wrong_tasks'],development=r['receipt']['development_metrics']) for r in rows],wall=result['wall_seconds'])))


if __name__=='__main__':main()
