# SPDX-License-Identifier: MIT-0
"""Small manual maker repair pilot; synthetic-only, no model service calls."""
import argparse, hashlib, json, time, copy
from pathlib import Path
import numpy as np
import psutil
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor, pack_grid
from cellforge.construction.maker_operator import MakerOperator
from measure_grid_expanded_capacity import family
from expanded_grid_genetics import canonical
from cellforge.reference.grid_false_positives import score_with_consistency

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    inventory=json.loads(a.inventory.read_text());rows={r['name']:r for r in inventory['artifacts']}
    base=Graph.load(a.inventory.parent/'base');child=Graph.load(a.inventory.parent/'crossover-1')
    assert child.hash()==rows['crossover-1']['hash'];canonical(base,child)
    selection=family(3217000,2);report=family(3227000,2)
    assert not {digest(t) for t in selection}&{digest(t) for t in report}
    # Strip query answers before diagnostics and repair evaluation.
    development=[{'family':t.get('family',t.get('metadata',{}).get('family')),'train':copy.deepcopy(t['train'])} for t in selection]
    a.out.mkdir();start=time.perf_counter();cpu=time.process_time()
    rng=np.random.default_rng(772301);slot=int(rng.integers(8,15));gene=child.meta['expanded_genome']['available_genes'][int(rng.integers(7))]
    policy=dict(protocol='expanded-breed-repair-pilot/1',child_hash=child.hash(),base_hash=base.hash(),selection_seed=3217000,report_seed=3227000,selection_manifest=digest(development),report_manifest=digest(report),attempts_per_arm=1,selection_tasks=24,report_tasks=24,gradients=False,native_tasks_loaded=0,random_seed=772301,random_slot=slot,random_gene=gene,maker_slot=10,maker_gene='bank_local_fill.gene',maker_provenance='current Codex assistant manually authored known fill-slot repair; prior aggregate family exposure, exact serving model unspecified; no new model calls',maker_cost='not measured; not zero',scope='one deliberately damaged crossover child; known bank gene, not new capability; no general maker advantage',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(a.out/'policy.json',policy)
    predictor=GridPredictor(child,backend='compiled');examples=[]
    for t in development:
        for d in t['train']:
            predictions=predictor(t['train'],d['input'])
            if d['output'] not in predictions and len(examples)<5:
                examples.append(dict(input=d['input'],expected=d['output'],predicted=predictions,cells_fired=[],trace_scope='not traced; empty list does not mean no cells executed',family=t['family']))
    diagnostics=dict(partition='development',evidence='train-pairs',counterexamples=examples)
    write(a.out/'diagnostics.json',diagnostics)
    def evaluate(g):
        pred=GridPredictor(g,backend='compiled');scores={};calls=0
        for t in development:
            correct=0
            for d in t['train']:
                calls+=1;correct+=int(d['output'] in pred(t['train'],d['input']))
            f=t['family'];scores[f]=scores.get(f,0)+correct
        return dict(reconstructed_pairs=sum(scores.values()),families=scores,predictor_calls=calls)
    x,_,h,w=pack_grid([[1]])
    fixture=[dict(entry=child.entries['grid_step'],inputs=dict(input=x.tolist(),input_height=h.tolist(),input_width=w.tolist(),output=x.tolist(),output_height=h.tolist(),output_width=w.tolist(),phase=[[0.]],state=np.zeros((1,1515)).tolist()),output_shapes=dict(state=[1,1515],height_logits=[1,2,30],width_logits=[1,2,30],color_logits=[1,2,30,30,10]))]
    op=MakerOperator(a.out/'maker-ledger.jsonl',budget=3,profile='ladder-micro/1')
    repairs={'random_repair':dict(ops=[dict(op='set_slot_gene',slot=slot,gene=gene)],actor='uniform slot sampler',model='none'),'maker_repair':dict(ops=[dict(op='set_slot_gene',slot=10,gene='bank_local_fill.gene')],actor='Codex root',model='session model unspecified')}
    # Candidates are frozen for report regardless of improvement. This pilot
    # measures edits; it does not claim archive acceptance/regression gating.
    results=op.breed_then_repair_round(child,diagnostics=diagnostics,repairs=repairs,fixtures=fixture,evaluate=evaluate,accept=lambda _:False)
    frozen={}
    for arm,(g,receipt) in results.items():
        if g is None:raise ValueError('invalid arm')
        canonical(base,g);frozen[arm]=g.save(a.out/arm)
    write(a.out/'winners-frozen.json',frozen)
    report_rows=[]
    for arm,(g,receipt) in results.items():
        report_rows.append(dict(arm=arm,selection=receipt,report=score_with_consistency([(str(i),t) for i,t in enumerate(report)],GridPredictor(g,backend='compiled'),record_tasks=True)))
    memory=psutil.Process().memory_info()
    result=dict(policy=policy,rows=report_rows,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_bytes=getattr(memory,'peak_wset',None),rss_bytes_at_end=memory.rss,resource_scope='local Windows process; not 2-CPU server performance',archive_acceptance_tested=False)
    write(a.out/'result.json',result);MakerOperator(op.path,budget=3,profile='ladder-micro/1')
    print(json.dumps(dict(rows=[dict(arm=r['arm'],solved=r['report']['tasks_solved'],selection=r['selection']['development_metrics']) for r in report_rows],wall_seconds=result['wall_seconds'],peak_rss_bytes=result['peak_rss_bytes'])))
if __name__=='__main__':main()
