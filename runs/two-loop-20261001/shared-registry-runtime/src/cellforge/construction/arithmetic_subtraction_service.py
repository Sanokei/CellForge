"""Persistent subtraction niche lane for the shared independent-cell framework.

Infrastructure, not an architecture-complete or learned-transfer claim. Only
supplied development cases select cells. Frozen adapter cases never enter here.
"""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
try:
 from . import arithmetic_subtraction_adapter as adapter
except ImportError:
 import arithmetic_subtraction_adapter as adapter
MODEL='gpt-6.1-sol'
REASONING='low'
CODEX='/root/.local/bin/codex'
AUTHOR_TIMEOUT=900

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(x):return adapter.digest(x)
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
 def pairs(items):
  result={}
  for key,value in items:
   if key in result:raise ValueError('duplicate JSON key: '+key)
   result[key]=value
  return result
 def invalid(value):raise ValueError('nonfinite JSON: '+value)
 return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,parse_constant=invalid)
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp')
 with temp.open('w',encoding='utf-8') as out:
  out.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
 os.replace(temp,path)
def identity(cell):return adapter.identity(cell)
def grade(cell):
 adapter.validate(cell);outcomes={};traces=[];errors=[]
 for case in adapter.DEVELOPMENT_CASES:
  expected=adapter.reference(case['input'])
  try:predicted=adapter.predict(cell['params'],case['input']);error=None
  except (ValueError,TypeError,IndexError,KeyError) as exc:predicted=None;error=type(exc).__name__+':'+str(exc);errors.append(error)
  hit=predicted==expected;outcomes[case['id']]=hit
  traces.append({'id':case['id'],'input':case['input'],'expected':expected,'predicted':predicted,'correct':hit,'error':error})
 for case in adapter.DEVELOPMENT_DECIMAL_CASES:
  try:
   answer=adapter.predict_decimal(cell['params'],case['a'],case['b'])
   predicted={key:answer[key] for key in ('digits','borrow')};error=None;columns=answer['trace']
  except (ValueError,TypeError,IndexError,KeyError) as exc:predicted=None;error=type(exc).__name__+':'+str(exc);errors.append(error);columns=[]
  hit=predicted==case['expected'];outcomes[case['id']]=hit
  traces.append({'id':case['id'],'input':{'a':case['a'],'b':case['b']},'expected':case['expected'],
                 'predicted':predicted,'columns':columns,'correct':hit,'error':error})
 return {'partition':'development','solved':sum(outcomes.values()),'total':len(outcomes),
         'outcomes':outcomes,'traces':traces,'errors':errors}
def declaration(seeds):
 return {'interface':adapter.INTERFACE,'contract':adapter.CONTRACT,'seeds':seeds,
 'development':adapter.DEVELOPMENT_CASES,'development_decimal':adapter.DEVELOPMENT_DECIMAL_CASES,
 'dependencies':{'adapter':file_hash(adapter.__file__),'service':file_hash(__file__),'actual_arithmetic':file_hash(adapter.SOURCE)},
 'baseline':adapter.BASELINE_CELL,'random_operator':'all12 independent uniform[-10,10] literal values; no gradient',
 'author':{'model':MODEL,'reasoning':REASONING,'route':'verified ChatGPT subscription Codex CLI','timeout_seconds':900},
 'selection':'strict total development gain and no previously successful task regression; all typed outputs viable',
 'overall_generation_cap':None,'frozen_selection_access':False,'learning_claim':False,
 'scope':'integrated independent subtraction cell lane; shared circuit/registry integration owned externally'}
def verify(root,anchor):
 d=read(root/'DECLARATION.json')
 if digest(d)!=anchor:raise ValueError('declaration anchor mismatch')
 current={'adapter':file_hash(adapter.__file__),'service':file_hash(__file__),'actual_arithmetic':file_hash(adapter.SOURCE)}
 if d['dependencies']!=current:raise ValueError('runtime dependency changed')
 if d['development']!=adapter.DEVELOPMENT_CASES or d['development_decimal']!=adapter.DEVELOPMENT_DECIMAL_CASES:raise ValueError('development declaration changed')
 return d

def initialize(root,seeds=(401,402,403)):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
 if (root/'DECLARATION.json').exists():return digest(read(root/'DECLARATION.json'))
 seeds=list(seeds)
 if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):raise ValueError('distinct integer seeds required')
 d=declaration(seeds);anchor=digest(d);atomic(root/'DECLARATION.json',d)
 cell=deepcopy(adapter.BASELINE_CELL);cell['id']=identity(cell);cell['evidence']=grade(cell)
 state={'generation':0,'anchor':anchor,'cells':{cell['id']:cell},'contexts':[
  {'seed':seed,'arms':{arm:{'live':cell['id'],'history':[cell['id']],'births':[]} for arm in ('llm','random')}} for seed in seeds],
  'charged_positions':{'llm':0,'random':0},'admitted':{'llm':0,'random':0},'rejected':{'llm':0,'random':0},
  'author_seconds':0.,'evaluation_seconds':0.,'learning_claim':False}
 atomic(root/'STATE-0.json',state);return anchor

def inherited_base(state,pop):
 parent=state['cells'][pop['live']]
 if parent['evidence']['errors']:return deepcopy(parent),{'status':'missing_viable_parent','parents':[parent['id']]}
 for donor_id in reversed(pop['history']):
  if donor_id==parent['id']:continue
  donor=state['cells'][donor_id]
  if donor['evidence']['errors']:continue
  differing=[i for i,(k,j) in enumerate(adapter.POSITION_ORDER)
             if np.asarray(parent['params'][k])[j]!=np.asarray(donor['params'][k])[j]]
  if len(differing)<2:continue
  child=adapter.crossover(parent,donor,differing[::2])
  return child,{'status':'measured_compatible_crossover','parents':[parent['id'],donor_id],
                'inheritance':child['inheritance'],'parent_params':{'left':parent['params'],'right':donor['params']},
                'distinguishing_positions':differing,'authored_positions':0,'positions':12}
 return deepcopy(parent),{'status':'missing_distinct_viable_donor','parents':[parent['id']]}
def prompts(state):
 rows=[]
 for context in state['contexts']:
  pop=context['arms']['llm'];parent=state['cells'][pop['live']];base,cross=inherited_base(state,pop)
  feedback=parent['evidence']
  p={'generation':state['generation']+1,'seed':context['seed'],'interface':adapter.INTERFACE,
   'contract':adapter.CONTRACT,'parent_id':parent['id'],'parent_params':parent['params'],
   'base_params':base['params'],'base_sha256':digest(base['params']),'crossover':cross,
   'development_evidence':feedback,'partition':'development',
   'edit_contract':'All12 literal finite positions [-10,10], nonzero numerical mutation from supplied base. No files/tools/network/reporting. When crossover supplied retain at least one numerically distinguishing inherited scalar unchanged from EACH parent. Remaining params are authored mutation. Never invent gains.'}
  rows.append({'seed':context['seed'],'prompt':p,'prompt_sha256':digest(p),
               'feedback_sha256':digest(feedback)})
 return rows

def command_env():
 env=dict(os.environ)
 for key in list(env):
  if key.endswith('API_KEY') or key in ('OPENAI_BASE_URL','AZURE_OPENAI_ENDPOINT'):env.pop(key,None)
 return env
def author(root,n,rows,state_hash):
 work=root/f'AUTHOR-{n}';work.mkdir(exist_ok=True)
 if (work/'EXIT.json').exists():
  if read(work/'EXIT.json')['returncode']!=0:raise ValueError('prior author failed; explicit recovery required')
  return
 if (work/'START.json').exists():raise ValueError('interrupted author invocation; explicit recovery required')
 env=command_env()
 auth=subprocess.run([CODEX,'login','status'],capture_output=True,text=True,timeout=15,env=env)
 if auth.returncode or 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:raise ValueError('verified ChatGPT subscription unavailable; no API fallback')
 text='Author full literal numerical subtraction cell mutations only from supplied DEVELOPMENT feedback. No tools, file reads, network, or evaluation. Return ONLY JSON array in input order. Each object has seed, model gpt-6.1-sol, reasoning low, interface decimal-subtraction-borrow-cell/1, parent_id, parent_params EXACT, base_sha256 EXACT, prompt_sha256 and feedback_sha256 EXACT supplied, params all12 literal positions in declared shapes/bounds, charged_positions12, rationale. Make a nonzero change to the inherited base. With crossover preserve distinguishing inherited contributions from both parents as instructed. Do not fabricate improvement or invent failures.\n'+json.dumps(rows,allow_nan=False)
 atomic(work/'START.json',{'utc':utc(),'pid':os.getpid(),'model':MODEL,'reasoning':REASONING,
  'auth':'ChatGPT subscription','auth_verified':True,'paid_api_fallback':False,
  'parent_state_hash':state_hash,'prompts_hash':digest(rows),'author_text_sha256':hashlib.sha256(text.encode()).hexdigest()})
 started=time.monotonic()
 try:
  with (work/'events.jsonl').open('w',encoding='utf-8') as out:
   result=subprocess.run([CODEX,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
    '--sandbox','read-only','-m',MODEL,'-c','model_reasoning_effort="low"','--json','-C',str(work),
    '-o',str(work/'response.json'),'-'],input=text,text=True,stdout=out,stderr=out,env=env,timeout=AUTHOR_TIMEOUT)
  code=result.returncode;error=None
 except subprocess.TimeoutExpired:code=124;error='author_timeout900'
 except OSError as exc:code=125;error=str(exc)
 atomic(work/'EXIT.json',{'utc':utc(),'returncode':code,'error':error,'author_seconds':time.monotonic()-started})
 if code:raise ValueError('author failed/timeout; no fallback')

def external(root,control):
 root=Path(root).resolve();control=Path(control).resolve()
 if control==root or control.is_relative_to(root) or root.is_relative_to(control):raise ValueError('independent recorder must be disjoint from runroot')
 return root,control
def record(root,control,anchor,n):
 root,control=external(root,control);verify(root,anchor)
 state=read(root/f'STATE-{n-1}.json');rows=read(root/f'PROMPTS-{n}.json');work=root/f'AUTHOR-{n}'
 if state['generation']!=n-1 or rows!=prompts(state):raise ValueError('fresh prompts/parent mismatch')
 start=read(work/'START.json');end=read(work/'EXIT.json')
 if end['returncode']!=0 or start['model']!=MODEL or start['reasoning']!=REASONING or start['auth']!='ChatGPT subscription' or not start['auth_verified'] or start['paid_api_fallback']:raise ValueError('actual author receipt mismatch')
 if start['parent_state_hash']!=digest(state) or start['prompts_hash']!=digest(rows):raise ValueError('author binding mismatch')
 # Commit raw bytes even if returned parameters are malformed; invalid proposals
 # remain fully charged by the evaluator, never silently replaced.
 event={'anchor':anchor,'generation':n,'parent_state_hash':digest(state),'prompts_hash':digest(rows),
  'raw_response_sha256':file_hash(work/'response.json'),'start_sha256':file_hash(work/'START.json'),
  'exit_sha256':file_hash(work/'EXIT.json')}
 control.mkdir(parents=True,exist_ok=True);ledger=control/'commitments.jsonl';previous='0'*64
 if ledger.exists():
  for line in ledger.read_text().splitlines():
   row=json.loads(line);claimed=row.pop('hash')
   if row['previous']!=previous or digest(row)!=claimed:raise ValueError('recorder hash chain mismatch')
   previous=claimed
   if row['event']['generation']==n:
    if row['event']!=event:raise ValueError('recorder generation conflict')
    atomic(control/f'GENERATION-{n}.json',event)
    atomic(control/f'RECORDER-{n}.json',{'recorder_pid':os.getpid(),'utc':utc(),'event_hash':digest(event)})
    return event
 row={'previous':previous,'event':event};row['hash']=digest(row)
 with ledger.open('a',encoding='utf-8') as out:
  out.write(json.dumps(row,sort_keys=True)+'\n');out.flush();os.fsync(out.fileno())
 atomic(control/f'GENERATION-{n}.json',event)
 atomic(control/f'RECORDER-{n}.json',{'recorder_pid':os.getpid(),'utc':utc(),'event_hash':digest(event)})
 return event

def mutation_cell(response,row):
 if not isinstance(response,dict):raise ValueError('response object required')
 p=row['prompt']
 for key,value in {'seed':row['seed'],'model':MODEL,'reasoning':REASONING,'interface':adapter.INTERFACE,
  'parent_id':p['parent_id'],'parent_params':p['parent_params'],'base_sha256':p['base_sha256'],
  'prompt_sha256':row['prompt_sha256'],'feedback_sha256':row['feedback_sha256'],'charged_positions':12}.items():
  if response.get(key)!=value:raise ValueError('author response binding mismatch: '+key)
 cell={'interface':adapter.INTERFACE,'params':response['params']};adapter.validate(cell)
 if all(np.array_equal(np.asarray(cell['params'][k],float),np.asarray(p['base_params'][k],float)) for k in adapter.SHAPES):raise ValueError('no numerical mutation from inherited base')
 if p['crossover']['status']=='measured_compatible_crossover':
  retained={'left':False,'right':False}
  # Only differing-parent inherited positions constitute distinguishing evidence.
  for item in p['crossover']['inheritance']:
   key=item['parameter'];index=tuple(item['index']);i=item['position']
   chosen_side=item['parent'];other_side='right' if chosen_side=='left' else 'left'
   parents=p['crossover']['parents']
   # Parent tensors are supplied explicitly in crossover metadata below.
   other=p['crossover']['parent_params'][other_side]
   if float(np.asarray(other[key])[index])!=item['value'] and float(np.asarray(cell['params'][key])[index])==item['value']:
    retained[chosen_side]=True
  if not all(retained.values()):raise ValueError('mutation erased distinguishing inherited parental contribution')
 return cell

def evaluate(root,control,anchor,n):
 root,control=external(root,control);verify(root,anchor)
 old=read(root/f'STATE-{n-1}.json');rows=read(root/f'PROMPTS-{n}.json');work=root/f'AUTHOR-{n}'
 commitment=read(control/f'GENERATION-{n}.json')
 expected={'anchor':anchor,'generation':n,'parent_state_hash':digest(old),'prompts_hash':digest(rows),
  'raw_response_sha256':file_hash(work/'response.json'),'start_sha256':file_hash(work/'START.json'),'exit_sha256':file_hash(work/'EXIT.json')}
 if commitment!=expected or rows!=prompts(old):raise ValueError('independent commitment changed before scoring')
 previous='0'*64;found=False
 for line in (control/'commitments.jsonl').read_text().splitlines():
  item=json.loads(line);claimed=item.pop('hash')
  if item['previous']!=previous or digest(item)!=claimed:raise ValueError('independent recorder chain mismatch')
  previous=claimed
  if item['event']==commitment:found=True
 if not found:raise ValueError('commitment absent from independent journal')
 if (root/f'GENERATION-{n}.json').exists():
  result=read(root/f'GENERATION-{n}.json')
  if result['commitment']!=commitment:raise ValueError('recovery commitment mismatch')
  atomic(root/f'STATE-{n}.json',result['state']);return result['state']
 chargefile=root/f'CHARGES-{n}.jsonl'
 if chargefile.exists():raise ValueError('partial charged evaluation; explicit recovery required, never recharge')
 try:responses=read(work/'response.json');parse_error=None
 except (ValueError,TypeError) as exc:responses=[];parse_error=str(exc)
 if not isinstance(responses,list) or len(responses)!=len(rows):responses=[];parse_error='wrong response count/envelope'
 state=deepcopy(old);decisions=[];started=time.monotonic()
 for i,row in enumerate(rows):
  context=next(c for c in state['contexts'] if c['seed']==row['seed'])
  for arm in ('llm','random'):
   pop=context['arms'][arm];parent=state['cells'][pop['live']];base,cross=inherited_base(state,pop)
   charge={'generation':n,'seed':row['seed'],'arm':arm,'parent_id':parent['id'],'positions':12,
    'operator':'crossover_then_literal_mutation' if cross['status']=='measured_compatible_crossover' else 'literal_mutation',
    'crossover':cross,'raw_response_sha256':commitment['raw_response_sha256'] if arm=='llm' else None}
   with chargefile.open('a',encoding='utf-8') as out:out.write(json.dumps(charge,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
   state['charged_positions'][arm]+=12;reasons=[];cell=None
   try:
    if arm=='llm':
     if parse_error:raise ValueError(parse_error)
     cell=mutation_cell(responses[i],row)
    else:
     rng=np.random.default_rng(np.random.SeedSequence([row['seed']%(2**32),n]))
     params={key:rng.uniform(-10,10,size=shape).tolist() for key,shape in adapter.SHAPES.items()}
     # Matched crossover control also preserves distinguishing positions from
     # both parents, then authors all other positions by the random operator.
     if cross['status']=='measured_compatible_crossover':
      for side in ('left','right'):
       item=next(item for item in cross['inheritance'] if item['parent']==side and item['position'] in cross['distinguishing_positions'])
       arr=np.asarray(params[item['parameter']]);arr[tuple(item['index'])]=item['value'];params[item['parameter']]=arr.tolist()
     cell={'interface':adapter.INTERFACE,'params':params};adapter.validate(cell)
    cid=identity(cell);evidence=grade(cell)
    if cid==parent['id']:reasons.append('no_numerical_change')
    if evidence['errors']:reasons.append('typed_output_viability')
    if evidence['solved']<=parent['evidence']['solved']:reasons.append('strict_development_gain')
    for historical in pop['history']:
     if any(hit and not evidence['outcomes'][tid] for tid,hit in state['cells'][historical]['evidence']['outcomes'].items()):
      reasons.append('historical_success_regression');break
    cell.update(id=cid,evidence=evidence,parents=cross['parents'],
     provenance={'operator':charge['operator'],'arm':arm,'generation':n,'seed':row['seed'],
      'raw_response_sha256':charge['raw_response_sha256'],'crossover':cross,'charged_positions':12})
    if cid not in state['cells']:state['cells'][cid]=cell
   except (ValueError,TypeError,KeyError,IndexError) as exc:
    reasons.append('invalid_proposal:'+str(exc));cid=None;evidence=None
   accepted=not reasons
   birth={'generation':n,'id':cid,'parents':cross['parents'],'accepted':accepted,'reasons':reasons,'positions':12,
          'solved':evidence['solved'] if evidence else None}
   pop['births'].append(birth);state['admitted' if accepted else 'rejected'][arm]+=1
   if accepted:pop['live']=cid;pop['history'].append(cid)
   decisions.append(dict(birth,seed=row['seed'],arm=arm,crossover_status=cross['status']))
 evaluation_seconds=time.monotonic()-started;author_seconds=read(work/'EXIT.json')['author_seconds']
 state['generation']=n;state['author_seconds']+=author_seconds;state['evaluation_seconds']+=evaluation_seconds
 result={'commitment':commitment,'state':state,'decisions':decisions,'author_seconds':author_seconds,'evaluation_seconds':evaluation_seconds,'learning_claim':False}
 atomic(root/f'GENERATION-{n}.json',result);atomic(root/f'STATE-{n}.json',state);return state

def run(root,control,anchor,max_seconds=None,min_free_bytes=104857600):
 root,control=external(root,control);verify(root,anchor)
 import shutil
 lock=root/'SERVICE.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 started=time.monotonic();state=read(root/f'STATE-{max(int(p.stem.split("-")[1]) for p in root.glob("STATE-*.json"))}.json')
 def status(kind,**extra):
  live={c['arms'][a]['live'] for c in state['contexts'] for a in ('llm','random')}
  atomic(root/'STATUS.json',{'utc':utc(),'pid':os.getpid(),'status':kind,'generation':state['generation'],
   'unique_genotypes':len(state['cells']),'distinct_live_ids':len(live),'admitted':state['admitted'],
   'rejected':state['rejected'],'charged_positions':state['charged_positions'],
   'author_seconds':state['author_seconds'],'evaluation_seconds':state['evaluation_seconds'],
   'scientific_learning_claim':False,**extra})
 try:
  while True:
   if (root/'STOP').exists() or (control/'STOP').exists():status('user_stop');break
   if max_seconds is not None and time.monotonic()-started>=max_seconds:status('resource_time_pause');break
   if shutil.disk_usage(root).free<min_free_bytes:status('resource_disk_pause');break
   n=state['generation']+1;rows=prompts(state);atomic(root/f'PROMPTS-{n}.json',rows)
   status('authoring',next_generation=n,donors=[r['prompt']['crossover']['status'] for r in rows])
   author(root,n,rows,digest(state))
   subprocess.run([sys.executable,str(Path(__file__).resolve()),'record',str(root),'--control',str(control),
                   '--anchor',anchor,'--generation',str(n)],check=True,timeout=60,env=command_env())
   status('evaluating_independently_recorded_weights',next_generation=n)
   state=evaluate(root,control,anchor,n);status('generation_complete')
 except Exception as exc:status('error_pause',error=type(exc).__name__+':'+str(exc));raise
 finally:lock.unlink()

def main():
 p=argparse.ArgumentParser();p.add_argument('operation',choices=['init','run','record']);p.add_argument('root',type=Path)
 p.add_argument('--control',type=Path);p.add_argument('--anchor');p.add_argument('--generation',type=int)
 p.add_argument('--seeds',type=int,nargs='+',default=[401,402,403]);p.add_argument('--max-seconds',type=float)
 args=p.parse_args()
 if args.operation=='init':print(initialize(args.root,args.seeds));return
 if not args.control or not args.anchor:p.error('--control and --anchor required')
 if args.operation=='record':
  if args.generation is None:p.error('--generation required')
  record(args.root,args.control,args.anchor,args.generation)
 else:run(args.root,args.control,args.anchor,args.max_seconds)
if __name__=='__main__':main()
