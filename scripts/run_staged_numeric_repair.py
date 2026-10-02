# SPDX-License-Identifier: MIT-0
"""Staged known-family numerical repair with matched controls and rollback.

Maker proposal files are authored between stages. No runtime LLM, gradients,
real ARC data or claims of blinded maker effectiveness.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import psutil
from build_grid_numeric_neighbor_gate import build_weight_gate
from measure_numeric_repair_structural_transfer import structural_examples
from cellforge.ir.graph import Graph
from cellforge.construction.numeric_recipe import apply_numeric_recipe
from cellforge.construction.maker_operator import MakerOperator
from cellforge.construction.maker_diagnostics import train_counterexamples
from cellforge.runtime.grid import GridPredictor, pack_grid
from cellforge.reference.grid_false_positives import score_with_consistency

ROOT = Path('runs/staged-numeric-repair-v1')
ARMS = ('breeding_only', 'random_repair', 'maker_repair')
SEEDS = (774101, 774111, 774121)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    if path.exists(): raise ValueError('refusing to overwrite receipt')
    path.write_text(json.dumps(value, indent=2))


def train_only(seed):
    return [dict(train=t['train'], task_id=str(i), family='horizontal_two_marker_gate')
        for i,t in enumerate(structural_examples(seed, 10))]


def measure(graph, tasks):
    p = GridPredictor(graph, backend='compiled'); errors = solved = calls = 0
    for task in tasks:
        exact = True
        for d in task['train']:
            result = p(task['train'], d['input']); calls += 1
            # This graph has a single output. No rank switching in fitness.
            assert len(result) == 1
            wrong = int(np.count_nonzero(np.asarray(result[0]) != np.asarray(d['output'])))
            errors += wrong; exact &= wrong == 0
        solved += exact
    return dict(pixel_errors=errors, reconstructed_tasks=solved, tasks=len(tasks), predictor_calls=calls)


def fixtures(graph):
    x,mask,h,w = pack_grid([[1]])
    ins=dict(input=x,input_mask=mask,input_height=h,input_width=w,output=x,
        output_mask=mask,output_height=h,output_width=w,phase=[[0.]],index=[[0.]],state=np.zeros((1,0)))
    return [dict(entry=graph.entries['grid_step'], inputs={k:np.asarray(v).tolist() for k,v in ins.items()},
        output_shapes=dict(state=[1,0],color_logits=[1,1,30,30,10],height_logits=[1,1,30],width_logits=[1,1,30]))]


def prepare():
    if ROOT.exists(): raise ValueError('fresh run required')
    ROOT.mkdir()
    left=build_weight_gate((1,0,1,0)); right=build_weight_gate((0,1,0,1))
    child=apply_numeric_recipe(left,right,[dict(op='homologous_mask',name='neighbor_gate.W',mask=[[0,1,0,1]])])
    left.save(ROOT/'left');right.save(ROOT/'right');child.save(ROOT/'bred')
    write(ROOT/'policy.json',dict(protocol='staged-numeric-repair/1',seeds=SEEDS,rounds=2,
        arms=ARMS,edits_per_proposal=1,physical_proposals_per_arm=6,
        family='known authored horizontal AND family',gradients=False,native_loaded=0,
        parent_hashes=[left.hash(),right.hash()],bred_hash=child.hash(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        vocabulary=dict(bias=[-2.5,-1.5,-.5,.5],weight=[-1,0,1,2]),
        selection_rule='strict error reduction on batch A, nonincrease on disjoint batch B; otherwise rollback',
        scope='three random-stream replications of known-family manual repair; not independent tasks or blind LLM comparison'))
    for seed in SEEDS:
        home=ROOT/str(seed);home.mkdir();a=train_only(seed+100000);b=train_only(seed+200000)
        write(home/'selection.json',dict(a=a,b=b,hashes=[digest(a),digest(b)]))
        for arm in ARMS:
            child.save(home/(arm+'-selected-0'))
        base=dict(a=measure(child,a),b=measure(child,b))
        write(home/'baseline.json',dict(metrics=base,scope='one common physical baseline; no arm-specific fitness cache'))
        write(home/'diagnostics-1.json',train_counterexamples(child,train_only(seed+300000),limit=2))
    print('prepared: inspect diagnostics-1 before writing maker-1.json')


def advance(round_number):
    proposal=json.loads((ROOT/f'maker-{round_number}.json').read_text())
    for seed in SEEDS:
        home=ROOT/str(seed);selection=json.loads((home/'selection.json').read_text())
        diagnostics=json.loads((home/f'diagnostics-{round_number}.json').read_text())
        assert proposal['diagnostics_sha256'][str(seed)]==digest(diagnostics)
        for arm in ARMS:
            parent=Graph.load(home/f'{arm}-selected-{round_number-1}')
            old=json.loads((home/'baseline.json').read_text())['metrics'] if round_number==1 else json.loads((home/f'{arm}-decision-{round_number-1}.json').read_text())['selected_metrics']
            if arm=='maker_repair': ops=proposal['ops']
            elif arm=='random_repair':
                rng=np.random.default_rng(seed+round_number)
                ops=[dict(op='set_param',name='neighbor_gate.b',value=[float(rng.choice([-2.5,-1.5,-.5,.5]))])] if round_number==1 else [dict(op='set_param',name='neighbor_gate.W',value=[rng.choice([-1.,0.,1.,2.],size=4).tolist()])]
            else: ops=[]
            # Same tensor and value vocabulary, one explicit edit per changed arm.
            expected='neighbor_gate.b' if round_number==1 else 'neighbor_gate.W'
            if arm!='breeding_only':
                assert len(ops)==1 and ops[0]['op']=='set_param' and ops[0]['name']==expected
                values=np.asarray(ops[0]['value']);assert values.shape==parent.params[expected].shape
                assert np.isin(values,[-2.5,-1.5,-.5,.5] if round_number==1 else [-1,0,1,2]).all()
            def evaluate(g): return dict(a=measure(g,selection['a']),b=measure(g,selection['b']))
            def accept(m): return m['a']['pixel_errors']<old['a']['pixel_errors'] and m['b']['pixel_errors']<=old['b']['pixel_errors']
            operator=MakerOperator(home/(arm+'.jsonl'),budget=2)
            assert len(operator.rows)==round_number-1
            candidate,receipt=operator.submit(parent,diagnostics=diagnostics,ops=ops,
                provenance=dict(actor='Codex root' if arm=='maker_repair' else arm,
                    model='session model unspecified' if arm=='maker_repair' else 'none',
                    method='manual-llm-maker' if arm=='maker_repair' else 'non-llm-mutation-control',
                    rationale=proposal['rationale'] if arm=='maker_repair' else 'matched control'),
                fixtures=fixtures(parent),evaluate=evaluate,accept=accept,evaluate_unchanged=True)
            assert candidate is not None and receipt['evaluation_calls']==1
            chosen=candidate if receipt['status']=='accepted' else parent
            chosen.save(home/f'{arm}-selected-{round_number}')
            write(home/f'{arm}-decision-{round_number}.json',dict(
                selected_hash=chosen.hash(),selected_metrics=receipt['development_metrics'] if receipt['status']=='accepted' else old,
                receipt_hash=receipt['receipt_hash'],status=receipt['status'],scope='local selection; not full archive admission'))
        if round_number==1:
            maker=Graph.load(home/'maker_repair-selected-1')
            write(home/'diagnostics-2.json',train_counterexamples(maker,train_only(seed+400000),limit=2))
    print('completed round',round_number)


def report():
    start=time.perf_counter(); rows=[]
    for seed in SEEDS:
        home=ROOT/str(seed);tasks=structural_examples(seed+500000,20)
        hashes={a:Graph.load(home/f'{a}-selected-2').hash() for a in ARMS}
        write(home/'report-freeze.json',dict(hashes=hashes,manifest=digest(tasks),seed=seed+500000))
        for arm in ARMS:
            graph=Graph.load(home/f'{arm}-selected-2');op=MakerOperator(home/(arm+'.jsonl'),budget=2)
            assert len(op.rows)==2 and sum(r['evaluation_calls'] for r in op.rows)==2
            op.freeze(graph)
            score=op.sealed_once(graph,lambda g:score_with_consistency([(str(i),t) for i,t in enumerate(tasks)],GridPredictor(g,backend='compiled'),record_tasks=True))
            rows.append(dict(seed=seed,arm=arm,hash=graph.hash(),score=score))
    mem=psutil.Process().memory_info()
    result=dict(rows=rows,wall_seconds=time.perf_counter()-start,peak_rss_bytes=getattr(mem,'peak_wset',None),
        scope='known-family synthetic report; no native transfer or causal maker improvement claim')
    write(ROOT/'result.json',result)
    print(json.dumps({a:sum(r['score']['tasks_solved'] for r in rows if r['arm']==a) for a in ARMS}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','round1','round2','report']);a=p.parse_args()
    if a.stage=='prepare': prepare()
    elif a.stage=='report': report()
    else: advance(int(a.stage[-1]))
