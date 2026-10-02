"""Private operator: single-gene coverage, pair upper bounds and actual breeding."""
import argparse
import hashlib
import itertools
import json
import resource
import time
from pathlib import Path
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.reference.arc_grid_operator import validate_grid
from cellforge.governance.ledger import Ledger
from build_grid_gene_pool import breed_fixed_pair


def write(path, value): path.write_text(json.dumps(value,indent=2)+'\n')


def measure(graph, tasks):
    start=time.perf_counter();cpu=time.process_time()
    predictor=GridPredictor(graph,backend='compiled')
    startup=time.perf_counter()-start
    masks=[];invalid=0
    for _task_id,task in tasks:
        row=[]
        for query in task['test']:
            try:
                predictions=predictor(task['train'],query['input'])
                target=validate_grid(query['output'])
                row.append(any(validate_grid(x)==target for x in predictions))
            except (ValueError,KeyError,IndexError,TypeError):
                invalid+=1;row.append(False)
        masks.append(row)
    elapsed=time.perf_counter()-start
    result={'tasks':len(tasks),'tasks_solved':sum(all(x) for x in masks),
            'test_inputs':sum(map(len,masks)),'test_inputs_correct':sum(sum(x) for x in masks),
            'invalid_outputs':invalid,'wall_seconds':elapsed,'compile_seconds':startup,
            'cpu_seconds':time.process_time()-cpu,'cpu_seconds_per_task':(time.process_time()-cpu)/len(tasks),
            'tasks_per_second':len(tasks)/elapsed,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'rss_scope':'operator-process high-water to date; conservative whole-process upper bound, not incremental allocation',
            'model_hash':graph.hash(),'stats':graph.stats(),'backend':'compiled'}
    return result,masks


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--private',required=True)
    p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=200);p.add_argument('--top-pairs',type=int,default=3)
    args=p.parse_args();out=Path(args.out);inventory_path=Path(args.inventory)
    if out.exists() or not 1<=args.n<=200 or not 1<=args.top_pairs<=5:raise SystemExit('fresh bounded development run required')
    directory=Path(args.private)/'arcagi2/data/training'
    keyed=sorted((hashlib.sha256(x.stem.encode()).hexdigest(),x) for x in directory.glob('*.json'))
    selected=[path for key,path in keyed if int(key,16)%5!=0][:args.n]
    if len(selected)!=args.n:raise SystemExit('expected development task count unavailable')
    manifest=hashlib.sha256();tasks=[]
    for path in selected:
        raw=path.read_bytes();manifest.update(path.stem.encode()+b':'+hashlib.sha256(raw).digest())
        tasks.append((path.stem,json.loads(raw)))
    inventory=json.loads(inventory_path.read_text())
    out.mkdir();ledger=Ledger(out/'ledger.jsonl')
    policy={'protocol':'grid-gene-coverage/1','dataset':'ARC-AGI-2 public training',
            'partition':'development','partition_method':'SHA256 task-ID mod5 nonzero, sorted hash first200',
            'manifest_sha256':manifest.hexdigest(),'n':args.n,'gene_inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            'top_pairs':args.top_pairs,'pair_selection':'exhaustive private per-input union upper bound; top pairs materialized and scored independently',
            'attempts':'single-gene1; breeding pairs2 fixed donor-order outputs, not yet adaptive ranked-pair policy',
            'audit_evaluations':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(out/'policy.json',policy);ledger.append('policy_frozen',policy,'operator')
    graphs=[];masks=[];rows=[];started=time.perf_counter()
    for item in inventory['genes']:
        path=inventory_path.parent/item['path'];graph=Graph.load(path)
        if graph.hash()!=item['hash']:raise SystemExit('inventory artifact changed')
        row,mask=measure(graph,tasks)
        row.update({'gene':item['family']+'-'+str(item['index']),
                    'artifact_bytes':sum((path/name).stat().st_size for name in ('graph.json','params.npz','HASH')),
                    'coverage_scope':'one forced expert gene; supporting controller infrastructure retained'})
        graphs.append(graph);masks.append(mask);rows.append(row)
        ledger.append('single_gene_coverage',row,'operator');write(out/'singles-progress.json',rows)
    pairs=[]
    for a,b in itertools.combinations(range(len(graphs)),2):
        union=[[x or y for x,y in zip(ar,br)] for ar,br in zip(masks[a],masks[b])]
        pairs.append({'left':a,'right':b,'tasks_solved_upper_bound':sum(all(x) for x in union),
                      'test_inputs_correct_upper_bound':sum(sum(x) for x in union)})
    pairs.sort(key=lambda x:(-x['tasks_solved_upper_bound'],-x['test_inputs_correct_upper_bound'],x['left'],x['right']))
    measured_pairs=[]
    for index,selection in enumerate(pairs[:args.top_pairs]):
        left,right=selection['left'],selection['right']
        child=breed_fixed_pair(graphs[left],graphs[right]);path=out/f'pair-{index}'
        digest=child.save(path);row,mask=measure(child,tasks)
        row.update(selection);row.update({'left_gene':rows[left]['gene'],'right_gene':rows[right]['gene'],
                    'left_hash':graphs[left].hash(),'right_hash':graphs[right].hash(),'child_hash':digest,
                    'artifact_bytes':sum((path/name).stat().st_size for name in ('graph.json','params.npz','HASH')),
                    'upper_bound_confirmed':row['tasks_solved']==selection['tasks_solved_upper_bound'] and row['test_inputs_correct']==selection['test_inputs_correct_upper_bound']})
        measured_pairs.append(row);ledger.append('bred_pair_coverage',row,'operator')
    result={'policy':policy,'singles':rows,'best_pairs':measured_pairs,'pair_combinations_examined':len(pairs),
            'total_wall_seconds':time.perf_counter()-started,'audit_evaluations':0,
            'limitations':'offline development selection only; upper bounds are not evaluated graph scores until independently confirmed; fixed pair outputs not adaptive ranking; no evolved generality/prize claim'}
    ledger.append('coverage_complete',result,'operator');result['ledger_entries_verified']=ledger.verify()
    write(out/'result.json',result);print(json.dumps({'singles':len(rows),'pairs':measured_pairs,'seconds':result['total_wall_seconds']}))


if __name__=='__main__':main()
