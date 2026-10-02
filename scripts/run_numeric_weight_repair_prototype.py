# SPDX-License-Identifier: MIT-0
"""Two offline numeric edits with diagnostics; toy prototype, not blind LLM study."""
import json,time,sys
from pathlib import Path
import numpy as np
from build_grid_numeric_neighbor_gate import build_weight_gate,examples
from cellforge.runtime.grid import GridPredictor
from cellforge.construction.maker_diagnostics import train_counterexamples
from cellforge.construction.maker_operator import MakerOperator
from cellforge.runtime.grid import pack_grid
from cellforge.reference.grid_false_positives import score_with_consistency

def main():
 out=Path('runs/numeric-weight-repair-prototype-v3')
 if out.exists():raise ValueError('fresh output required')
 out.mkdir();founder=build_weight_gate();x,mask,h,w=pack_grid([[1]])
 fixture=[dict(entry=founder.entries['grid_step'],inputs=dict(input=x,input_mask=mask,input_height=h,input_width=w,output=x,output_mask=mask,output_height=h,output_width=w,phase=[[0.]],index=[[0.]],state=np.zeros((1,0))),output_shapes=dict(state=[1,0],color_logits=[1,1,30,30,10],height_logits=[1,1,30],width_logits=[1,1,30]))]
 fixture=[dict(entry=f['entry'],inputs={k:np.asarray(v).tolist() for k,v in f['inputs'].items()},output_shapes=f['output_shapes']) for f in fixture]
 tasks=examples(3367000);development=[dict(train=t['train'],task_id=t['task_id'],family=t['family']) for t in tasks];report=examples(3377000)
 policy=dict(rounds=2,edits_per_arm=2,seed=772401,selection_seed=3367000,report_seed=3377000,author='Codex root, session model unspecified',scope='author knows generator rule; not blinded feedback attribution or no-feedback control',runtime_llm=False,gradients=False)
 (out/'policy.json').write_text(json.dumps(policy,indent=2));parents={a:founder.copy() for a in ('breeding_only','random_repair','maker_repair')};ops={a:MakerOperator(out/(a+'.jsonl'),budget=2,profile='ladder-micro/1') for a in parents};rng=np.random.default_rng(772401);rows=[]
 def evaluate(g):
  p=GridPredictor(g,backend='compiled');correct=0
  for t in development:
   correct+=all([d['output'] in p(t['train'],d['input']) for d in t['train']])
  return dict(reconstructed_tasks=correct,tasks=16,predictor_calls=32)
 for round in range(2):
  for arm in parents:
   parent=parents[arm];diag=train_counterexamples(parent,development,limit=3)
   if arm=='maker_repair':edit=[dict(op='set_param',name='neighbor_gate.b',value=[-1.5])] if round==0 else [dict(op='set_param',name='neighbor_gate.W',value=[[0,0,1,1]])]
   elif arm=='random_repair':edit=[dict(op='set_param',name='neighbor_gate.b',value=[float(rng.choice([-2.5,-1.5,-.5,.5]))]),dict(op='set_param',name='neighbor_gate.W',value=[rng.choice([-1.,0.,1.,2.],size=4).tolist()])]
   else:edit=[]  # identical OR founders recombine identically; charged neutral control
   child,receipt=ops[arm].submit(parent,diagnostics=diag,ops=edit,provenance=dict(actor='Codex root' if arm=='maker_repair' else arm,model='session unspecified' if arm=='maker_repair' else 'none',method='manual-llm-maker' if arm=='maker_repair' else 'non-llm-mutation-control'),fixtures=fixture,evaluate=evaluate,accept=lambda _:False,evaluate_unchanged=True)
   assert child is not None
   # Follow authored proposal path; not a selected/archive-admitted population.
   parents[arm]=child;child.save(out/f'{arm}-round-{round+1}');rows.append(dict(round=round+1,arm=arm,receipt=receipt))
 hashes={a:g.hash() for a,g in parents.items()};(out/'winners-frozen.json').write_text(json.dumps(hashes))
 scores={a:score_with_consistency([(str(i),t) for i,t in enumerate(report)],GridPredictor(g,backend='compiled'),record_tasks=True) for a,g in parents.items()}
 result=dict(policy=policy,rows=rows,report=scores,hashes=hashes)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(selection=[(r['arm'],r['round'],r['receipt']['development_metrics']['reconstructed_tasks']) for r in rows],report={a:s['tasks_solved'] for a,s in scores.items()})))
if __name__=='__main__':main()
