# SPDX-License-Identifier: MIT-0
"""Explain exposed synthetic selection errors, not an inference-time solver.

Independent scalar table predicts the existing graph output. Generator direction
is used only for post-hoc annotation; it is never passed to the CellForge graph.
"""
import argparse, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np
from evolve_grid_context_masks import masked_graph
from build_grid_context_memory_gene import generate_context_memory_family
from cellforge.runtime.grid import GridPredictor


def scalar_key(grid, r, c, mask):
    h,w=len(grid),len(grid[0]);key=0
    for i,(dr,dc) in enumerate(( (dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) )):
        rr,cc=r-dr,c-dc
        if mask & (1<<i):
            key+=(grid[rr][cc]+1 if 0<=rr<h and 0<=cc<w else 0)*11**i
    return key


def explain(task, mask=186):
    table={}
    for pair in task['train'][-2:]:
        for r,row in enumerate(pair['output']):
            for c,color in enumerate(row):
                table.setdefault(scalar_key(pair['input'],r,c,mask),set()).add(color)
    query=task['test'][0]['input'];target=task['test'][0]['output']
    prediction=[];counts=Counter();samples=[]
    h,w=len(query),len(query[0]);dr,dc=task['direction']
    for r,row in enumerate(query):
        predicted=[]
        for c,color in enumerate(row):
            values=table.get(scalar_key(query,r,c,mask),set())
            status='unknown' if not values else 'conflicting' if len(values)>1 else 'known'
            out=next(iter(values)) if len(values)==1 else color
            predicted.append(out)
            counts['pixels_'+status]+=1
            if out!=target[r][c]:
                border=r in (0,h-1) or c in (0,w-1)
                outside=not(0<=r+dr<h and 0<=c+dc<w)
                counts['wrong_'+status]+=1
                counts['wrong_border' if border else 'wrong_interior']+=1
                counts['wrong_source_outside' if outside else 'wrong_source_inside']+=1
                if len(samples)<8:samples.append(dict(r=r,c=c,status=status,predicted=out,target=target[r][c],border=border,source_outside=outside))
        prediction.append(predicted)
    return prediction,dict(counts),samples


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('fresh output required')
    family=generate_context_memory_family(24,2477000)['tasks']
    predictor=GridPredictor(masked_graph(186),backend='compiled')
    totals=Counter();rows=[]
    for task in family:
        scalar,counts,samples=explain(task)
        graph=predictor(task['train'],task['test'][0]['input'])[0]
        if scalar!=graph:raise ValueError('scalar diagnostic disagrees with graph')
        totals.update(counts)
        rows.append(dict(id=task['id'],direction_annotation=task['direction'],counts=counts,samples=samples))
    result=dict(protocol='exposed-context-error-diagnostic/1',tasks=24,mask=186,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selection_hash=hashlib.sha256(json.dumps(family,sort_keys=True).encode()).hexdigest(),
                graph_scalar_parity=True, totals=dict(totals), rows=rows,
                scope='exposed synthetic selection only; direction metadata post-hoc annotation, never model input',
                benchmark_files_loaded=False, final_evaluations=0, model_edits=0)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
