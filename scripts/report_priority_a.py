# SPDX-License-Identifier: MIT-0
"""One-shot synthetic report after all matched sixteen-attempt arms freeze.

This is not an independent frontier comparison. Never used for selection.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.construction.maker_operator import MakerOperator
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.grid_false_positives import score_with_consistency

def paired_interval(differences):
    if len(differences)!=3:raise ValueError('three predeclared paired seeds required')
    mean=sum(differences)/3
    sd=math.sqrt(sum((x-mean)**2 for x in differences)/2)
    radius=4.302652729911275*sd/math.sqrt(3)
    return dict(mean=mean,lower=mean-radius,upper=mean+radius,
        method='paired Student t interval, 95%, df=2; three seeds are weak evidence')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--final-batch',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    batch=json.loads((a.final_batch/'result.json').read_text())
    if batch['proposals_per_arm_per_seed']!=16:raise ValueError('all sixteen proposals required before report access')
    expected={(s['seed'],arm) for s in plan['runs'] for arm in plan['arms']}
    actual=[(r['seed'],r['arm']) for r in batch['rows']]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('matched final arms missing or duplicated')
    if a.out.exists():raise ValueError('report destination already exists; never repeat sealed access')
    import resource  # Linux scoring only; budget guard is portable.
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    ready=[]
    # Validate every arm BEFORE freezing any or opening any report corpus.
    for spec in plan['runs']:
        for arm in plan['arms']:
            stem=f"{spec['seed']}-{arm}"
            operator=MakerOperator(a.final_batch/(stem+'.jsonl'),budget=16)
            if len(operator.rows)!=16 or any(r['kind']!='attempt' for r in operator.rows):
                raise ValueError('exact unreported sixteen-attempt ledger required')
            graph=Graph.load(a.final_batch/(stem+'-selected'))
            result=json.loads((a.final_batch/(stem+'-result.json')).read_text())
            if graph.hash()!=result['selected_hash']:raise ValueError('selected graph mismatch')
            ready.append((spec,arm,operator,graph))
    a.out.mkdir()
    for spec,arm,operator,graph in ready:operator.freeze(graph)
    start=time.perf_counter();rows=[]
    for spec,arm,operator,graph in ready:
        def evaluate(g):
            corpus=json.loads((a.plan.parent/str(spec['seed'])/'report.json').read_text())
            digest=hashlib.sha256(json.dumps(corpus,sort_keys=True).encode()).hexdigest()
            if digest!=spec['commitments']['report']:raise ValueError('report commitment mismatch')
            return score_with_consistency([(str(i),t) for i,t in enumerate(corpus)],
                GridPredictor(g,backend='compiled'),record_tasks=True)
        score=operator.sealed_once(graph,evaluate)
        row=dict(seed=spec['seed'],arm=arm,graph_hash=graph.hash(),stats=graph.stats(),score=score)
        rows.append(row);(a.out/f"{spec['seed']}-{arm}.json").write_text(json.dumps(row,indent=2))
        print(json.dumps(dict(seed=spec['seed'],arm=arm,solved=score['tasks_solved'])),flush=True)
    differences=[]
    for spec in plan['runs']:
        scores={r['arm']:r['score']['tasks_solved']/spec['report_tasks'] for r in rows if r['seed']==spec['seed']}
        differences.append(scores['maker_repair']-scores['random_repair'])
    receipt=dict(rows=rows,paired_maker_minus_random=paired_interval(differences),
        paired_seed_differences=differences,wall_seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='original synthetic report; not untouched real ARC or independent frontier evidence',
        plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest())
    (a.out/'result.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
