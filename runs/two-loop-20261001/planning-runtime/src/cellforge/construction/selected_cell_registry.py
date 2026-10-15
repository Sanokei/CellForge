"""Replay earned native champions into a shared typed selection registry.
Import is not a new birth. Broad execution does not assert cross-domain transfer.
"""
import argparse,json,pathlib,copy
from .cell_population import CellPopulation
from .joint_weight_population import digest
from .action_cell_continuation import atomic
from . import arithmetic_subtraction_adapter as arithmetic
from . import reasoning_cell_epochs as reasoning

def predict(cell,task):
 if cell['family']=='signed-temporal-deduction':
  from . import hard_reasoning_adapter as deduction
  answer=deduction.predict(cell['params'],task['input'])
  return {k:answer[k] for k in ('answer','closure')}
 if cell['family']=='decimal-subtraction':
  request=task['input']
  if isinstance(request['a'],str):
   value=arithmetic.predict_decimal(cell['params'],request['a'],request['b'])
   return {k:value[k] for k in ('digits','borrow')}
  return arithmetic.predict(cell['params'],request)
 numeric={'niche':'spatial_quorum','activation':'step','params':cell['params']}
 # Grade internally constructs and executes the real numeric IR, but prediction
 # never receives the target. The fixed dummy target is discarded.
 from cellforge.ir.graph import Graph
 from cellforge.runtime.interp import Interpreter
 import numpy as np
 g=Graph(defs={'candidate':{'kind':'cell','act':'step','params':{k:k for k in cell['params']}}},params={k:np.asarray(v,float) for k,v in cell['params'].items()})
 return int(Interpreter(g).run_cell('candidate',np.asarray([task['input']],float))[0,0])

def collect(root,kind):
 state_path=max(root.glob('STATE-*.json'),key=lambda p:int(p.stem.split('-')[1]));state=json.loads(state_path.read_text())
 ids=set()
 for context in state['contexts']:
  for arm in ('llm','random'):
   live=context['arms'][arm]['live'];ids.update(live.values() if isinstance(live,dict) else [live])
 if kind!='spatial-cardinal':
  first=state['cells'][next(iter(ids))]['evidence']['traces']
  tasks=[{'id':t['id'],'input':t['input'],'expected':t['expected']} for t in first]
 else:tasks=json.loads((root/'TASKS.json').read_text())['development']['spatial_quorum']
 return state,ids,tasks,{'root':str(root),'state_hash':digest(state),'state_file':state_path.name,'declaration_file_hash':__import__('hashlib').sha256((root/'DECLARATION.json').read_bytes()).hexdigest()}

def refresh(arithmetic_root,spatial_root,out,reasoning_root=None):
 p=CellPopulation();environments=[];source_maps={};checks=[]
 sources=[(arithmetic_root,'decimal-subtraction','decimal-affine-borrow'),(spatial_root,'spatial-cardinal','onehot-step')]
 if reasoning_root is not None:sources.append((reasoning_root,'signed-temporal-deduction','signed-temporal-step'))
 for root,family,structure in sources:
  state,ids,tasks,source=collect(root,family);manifest=digest(tasks)
  environments.append({'id':family+'-development','partition':'development','input_type':family,'manifest':manifest,'tasks':tasks})
  for native_id in sorted(ids):
   native=state['cells'][native_id];cell={'family':family,'params':native['params']}
   outcomes={t['id']:predict(cell,{'input':t['input']})==t['expected'] for t in tasks}
   if outcomes!=native['evidence']['outcomes']:raise ValueError('native selected score fails independent numeric replay:'+native_id)
   section={'partition':'development','manifest':manifest,'outcomes':outcomes}
   provenance={'kind':'imported_measured_native_survivor_not_new_birth','native_id':native_id,'native_parents':native.get('parents',[]),'native_provenance':native.get('provenance',{}),'native_state':source,'input_type':family,'evidence_scope':'fresh/archive/anchor share ONE developmental manifest; not three independent evaluations or ARC anchors'}
   cid=p.seed(structure,native['params'],provenance,{k:copy.deepcopy(section) for k in ('fresh','archive','anchor')},family=family,niche=family,niche_manifest=manifest)
   source_maps[cid]=native_id;checks.append({'cell':cid,'native':native_id,'solved':sum(outcomes.values()),'total':len(tasks),'replayed':True})
 previous_version=None
 if (out/'LATEST.json').exists():
  latest=json.loads((out/'LATEST.json').read_text());previous_version=latest['version']
  previous=json.loads((out/(previous_version+'.json')).read_text())
  if digest(previous)!=previous_version:raise ValueError('previous registry publication integrity failed')
  p.competition_iteration=previous['registry']['competition_iteration']
 result=p.compete(environments=environments,evaluator=predict)
 payload={'previous_version':previous_version,'scope':'imported earned survivors, actual typed broad replay; native evolution not yet dispatched through this registry','architecture_complete':False,'scientific_acceptance':False,'source_mapping':source_maps,'import_checks':checks,'competition':result,'registry':p.snapshot()}
 version=digest(payload);out.mkdir(parents=True,exist_ok=True);atomic(out/(version+'.json'),payload);atomic(out/'LATEST.json',{'version':version,'families':len(sources),'live_cells':len(p.cells),'competition_iteration':p.competition_iteration,'representatives':result['representatives'],'execution_receipts':result['execution_receipts'],'architecture_complete':False})
 return out/'LATEST.json'

def main():
 a=argparse.ArgumentParser();a.add_argument('arithmetic',type=pathlib.Path);a.add_argument('spatial',type=pathlib.Path);a.add_argument('out',type=pathlib.Path);a.add_argument('--reasoning-root',type=pathlib.Path);v=a.parse_args();print(refresh(v.arithmetic,v.spatial,v.out,v.reasoning_root).read_text())
if __name__=='__main__':main()
