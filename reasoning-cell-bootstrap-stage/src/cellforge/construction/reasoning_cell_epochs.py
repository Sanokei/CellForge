"""Bounded individual non-grid cell epochs, genuine server Codex authoring.
Imported exact Horn operators are not advertised as newly learned reasoning.
"""
import argparse,copy,json,os,pathlib,random,subprocess,time,datetime
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.interp import Interpreter
from .action_cell_continuation import atomic
from .experiment_manifest import load_json_strict
from .joint_weight_population import digest
from .anchored_action_protocol import file_hash

def verify(root,anchor):
 if file_hash(root/'DECLARATION.json')!=anchor:raise ValueError('external initial anchor mismatch')
 for name,sha in load_json_strict(root/'DECLARATION.json')['dependencies'].items():
  p=(root/name).resolve()
  if not p.is_relative_to(root.resolve()) or file_hash(p)!=sha:raise ValueError('immutable dependency changed:'+name)

def validate(cell):
 width=2 if cell['niche']=='all_present' else 1 if cell['niche']=='any_present' else 0
 p=cell['params'];keys=set(p)
 if keys not in ({'W1','b1'},{'W1','b1','W2','b2'}):raise ValueError('literal cell parameter schema')
 w=np.asarray(p['W1'],float);b=np.asarray(p['b1'],float)
 if w.ndim!=2 or w.shape[1]!=width or not 1<=w.shape[0]<=4 or b.shape!=(w.shape[0],):raise ValueError('typed input/hidden width')
 if 'W2' in p:
  if np.asarray(p['W2']).shape!=(1,w.shape[0]) or np.asarray(p['b2']).shape!=(1,):raise ValueError('typed scalar output')
 elif w.shape[0]!=1:raise ValueError('scalar output required')
 arrays=[np.asarray(v,float) for v in p.values()]
 positions=sum(a.size for a in arrays)
 if positions>40 or any(not np.isfinite(a).all() or np.abs(a).max()>9 for a in arrays):raise ValueError('literal finite numerical cap')
 if cell['activation']!='step':raise ValueError('declared step-family structure')
 return positions

def identity(cell):return digest({k:cell[k] for k in ('niche','activation','params','typed_interface')})

def grade(cell,rows):
 validate(cell);g=Graph(defs={'candidate':{'kind':'cell','act':cell['activation'],'params':{k:k for k in cell['params']}}},params={k:np.asarray(v,float) for k,v in cell['params'].items()})
 engine=Interpreter(g);outcomes={};errors=[]
 for row in rows:
  if row['partition']!='development':raise ValueError('reporting cannot select cells')
  try:
   value=float(engine.run_cell('candidate',np.asarray([row['input']],float))[0,0])
   if value not in (0,1):raise ValueError('non-Boolean reasoning output')
   outcomes[row['id']]=int(value)==row['expected']
  except (ValueError,IndexError,TypeError) as e:outcomes[row['id']]=False;errors.append(type(e).__name__+':'+str(e))
 return {'outcomes':outcomes,'solved':sum(outcomes.values()),'total':len(rows),'errors':errors,'partition':'development'}

def init(root,anchor):
 verify(root,anchor);tasks=load_json_strict(root/'TASKS.json')['development'];cells={};live={}
 for raw in load_json_strict(root/'SEEDS.json'):
  cell=dict(raw,niche=raw['structure']);cid=identity(cell);cell.update(id=cid,parents=[],provenance={'source':'verified historical literal graph cell','source_graph_hash':raw['source_graph_hash']},state='live',evidence=grade(cell,tasks[cell['niche']]))
  cells[cid]=cell;live[cell['niche']]=cid
 config=load_json_strict(root/'CONFIG.json')
 return {'initial_anchor':anchor,'generation':0,'cells':cells,'contexts':[{'seed':s,'arms':{a:{'live':copy.deepcopy(live),'history':{n:[cid] for n,cid in live.items()},'births':[]} for a in ('llm','random')}} for s in config['seeds']], 'scientific_learning_claim':False}

def prompts(state):
 rows=[]
 for context in state['contexts']:
  for niche,cid in context['arms']['llm']['live'].items():
   cell=state['cells'][cid];q={'generation':state['generation']+1,'seed':context['seed'],'niche':niche,'parent_id':cid,'parent_cell':cell,'partition':'development','edit_contract':'Literal numerical revisions or up to4 hidden step units with W2/b2, scalar Boolean output, max40positions[-9,9]. Same typed input meaning. No socket edits; no claim of crossover.'}
   rows.append({'seed':context['seed'],'niche':niche,'prompt':q,'prompt_sha256':digest(q)})
 return rows

def prepare_record(root,anchor,control,n):
 verify(root,anchor);state=load_json_strict(root/f'STATE-{n-1}.json');raw=load_json_strict(root/f'AUTHOR-{n}/response.json');requests=prompts(state)
 if len(raw)!=len(requests):raise ValueError('one fresh proposal per context/niche required')
 start=load_json_strict(root/f'AUTHOR-{n}/START.json');end=load_json_strict(root/f'AUTHOR-{n}/EXIT.json')
 if end['returncode']!=0 or start['model']!='gpt-6.1-sol' or start['auth']!='ChatGPT subscription':raise ValueError('actual author invocation receipt')
 for request,response in zip(requests,raw):
  if any(response[k]!=request[k] for k in ('seed','niche')) or response.get('parent_id')!=request['prompt']['parent_id'] or response.get('prompt_sha256')!=request['prompt_sha256']:raise ValueError('fresh parent/prompt/context binding')
  if response.get('model')!='gpt-6.1-sol' or response.get('reasoning')!='low':raise ValueError('maker model binding')
 for previous in range(1,n):
  if digest(load_json_strict(root/f'AUTHOR-{previous}/response.json'))==digest(raw):raise ValueError('reused raw author response')
 event={'initial_anchor':anchor,'generation':n,'parent_state_hash':digest(state),'responses_sha256':file_hash(root/f'AUTHOR-{n}/response.json'),'prompts_hash':digest(requests),'author_start_hash':file_hash(root/f'AUTHOR-{n}/START.json')}
 control.mkdir(parents=True,exist_ok=True);ledger=control/'commitments.jsonl';previous='0'*64
 if ledger.exists():
  for line in ledger.read_text().splitlines():
   row=json.loads(line);sha=row.pop('hash')
   if row['previous']!=previous or digest(row)!=sha:raise ValueError('independent recorder journal changed')
   previous=sha
   if row['event']['generation']==n:
    if row['event']!=event:raise ValueError('recorded generation conflict')
    return
 row={'previous':previous,'event':event};row['hash']=digest(row)
 with ledger.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 atomic(control/f'GENERATION-{n}.json',event)

def evaluate(root,anchor,control,n):
 verify(root,anchor);parent=load_json_strict(root/f'STATE-{n-1}.json');commit=load_json_strict(control/f'GENERATION-{n}.json');rawpath=root/f'AUTHOR-{n}/response.json'
 if commit['parent_state_hash']!=digest(parent) or commit['responses_sha256']!=file_hash(rawpath):raise ValueError('recorded pre-scoring binding changed')
 if (root/f'CHARGES-{n}.jsonl').exists():raise ValueError('charged partial generation needs recovery; never recharge')
 state=copy.deepcopy(parent);tasks=load_json_strict(root/'TASKS.json')['development'];responses=load_json_strict(rawpath);decisions=[]
 for response in responses:
  context=next(c for c in state['contexts'] if c['seed']==response['seed']);niche=response['niche'];source=state['cells'][response['parent_id']]
  proposed={k:copy.deepcopy(source[k]) for k in ('niche','activation','typed_interface')};proposed['params']=response['params'];positions=validate(proposed)
  if positions!=response['charged_positions']:raise ValueError('full proposal charge mismatch')
  for arm in ('llm','random'):
   pop=context['arms'][arm];parentid=pop['live'][niche];old=state['cells'][parentid];candidate=copy.deepcopy(proposed)
   charge={'generation':n,'seed':context['seed'],'niche':niche,'arm':arm,'parent':parentid,'positions':positions,'raw_response_sha256':digest(response) if arm=='llm' else None}
   with (root/f'CHARGES-{n}.jsonl').open('a') as f:f.write(json.dumps(charge)+'\n');f.flush();os.fsync(f.fileno())
   if arm=='random':
    rng=random.Random(context['seed']*100000+n*100+(0 if niche=='all_present' else 1))
    candidate['params']={k:np.asarray([rng.uniform(-9,9) for _ in range(np.asarray(v).size)]).reshape(np.asarray(v).shape).tolist() for k,v in candidate['params'].items()}
   cid=identity(candidate);evidence=grade(candidate,tasks[niche]);reasons=[]
   if cid==parentid:reasons.append('no_numerical_or_structural_change')
   if evidence['solved']<=old['evidence']['solved']:reasons.append('strict_development_gain')
   if evidence['errors']:reasons.append('typed_output_viability')
   for rivalid in pop['history'][niche]:
    rival=state['cells'][rivalid]
    if any(hit and not evidence['outcomes'][tid] for tid,hit in rival['evidence']['outcomes'].items()):reasons.append('historical_champion_regression');break
   candidate.update(id=cid,parents=[parentid],evidence=evidence,state='live' if not reasons else 'rejected',provenance={'operator':'LLM numerical/structural mutation' if arm=='llm' else 'matched complete random weights','raw_response_sha256':digest(response),'generation':n,'seed':context['seed'],'inherited_template':old['activation'],'new_structure':set(candidate['params'])!=set(old['params']),'charged_positions':positions})
   if cid not in state['cells']:state['cells'][cid]=candidate
   pop['births'].append({'id':cid,'parent':parentid,'generation':n,'decision':{'accepted':not reasons,'reasons':reasons},'positions':positions})
   if not reasons:pop['live'][niche]=cid;pop['history'][niche].append(cid)
   decisions.append({'seed':context['seed'],'niche':niche,'arm':arm,'candidate':cid,'solved':evidence['solved'],'accepted':not reasons,'reasons':reasons})
 state['generation']=n;atomic(root/f'GENERATION-{n}.json',{'state':state,'decisions':decisions});atomic(root/f'STATE-{n}.json',state);return state

def run(root,anchor,control,max_seconds=1500):
 verify(root,anchor);lock=root/'POPULATION.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 deadline=time.monotonic()+max_seconds
 def status(kind,**kw):atomic(root/'STATUS.json',{'status':kind,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kw})
 try:
  if not (root/'STATE-0.json').exists():atomic(root/'STATE-0.json',init(root,anchor))
  n=max(int(p.stem.split('-')[1]) for p in root.glob('STATE-*.json'));state=load_json_strict(root/f'STATE-{n}.json')
  budget=load_json_strict(root/'CONFIG.json')['generations']
  while n<budget and time.monotonic()<deadline:
   n=state['generation']+1;work=root/f'AUTHOR-{n}';work.mkdir(exist_ok=True);requests=prompts(state)
   atomic(root/f'PROMPTS-{n}.json',requests)
   env=dict(os.environ)
   for k in ('OPENAI_API_KEY','AZURE_OPENAI_API_KEY'):env.pop(k,None)
   auth=subprocess.run(['/root/.local/bin/codex','login','status'],capture_output=True,text=True,timeout=15,env=env)
   if 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:raise ValueError('subscription auth unavailable; no fallback')
   if not (work/'EXIT.json').exists():
    text='Use only supplied DEV feedback; no tools/files/reporting. Write fresh actual literal numerical weights for individual reasoning cells, optional justified step hidden structure with W2/b2(max4hidden,max40positions). Return ONLY JSON array in input order with seed,niche,parent_id,prompt_sha256(exact supplied),model gpt-6.1-sol,reasoning low,params(W1,b1 and optionalW2,b2),charged_positions(full scalar count),rationale. Do not fabricate gains or degrade founders to manufacture learning.\n'+json.dumps(requests)
    atomic(work/'START.json',{'model':'gpt-6.1-sol','reasoning':'low','auth':'ChatGPT subscription','parent_state_hash':digest(state),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prompt_sha256':digest(text)})
    status('authoring_independent_reasoning_cells',generation=n-1,next_generation=n);started=time.monotonic()
    with (work/'events.jsonl').open('w') as out:
     result=subprocess.run(['/root/.local/bin/codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--sandbox','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="low"','--json','-C',str(work),'-o',str(work/'response.json'),'-'],input=text,text=True,stdout=out,stderr=out,env=env,timeout=600)
    atomic(work/'EXIT.json',{'returncode':result.returncode,'author_seconds':time.monotonic()-started})
   if load_json_strict(work/'EXIT.json')['returncode']:raise ValueError('author failed; no automatic paid fallback')
   subprocess.run([os.sys.executable,'-m','cellforge.construction.reasoning_cell_epochs','record',str(root),'--anchor',anchor,'--control',str(control),'--generation',str(n)],check=True,timeout=30)
   status('evaluating_recorded_reasoning_generation',generation=n-1,next_generation=n);started=time.monotonic();state=evaluate(root,anchor,control,n)
   live=set(cid for c in state['contexts'] for a in c['arms'].values() for cid in a['live'].values())
   status('generation_complete',generation=n,unique_genotypes=len(state['cells']),distinct_live_ids=len(live),evaluation_seconds=time.monotonic()-started)
  status('declared_epoch_budget_complete' if state['generation']==budget else 'bounded_resource_deadline',generation=state['generation'],waiting_for='new_capability_epoch_declaration',unique_genotypes=len(state['cells']))
 except Exception as e:status('blocked',error=type(e).__name__+':'+str(e));raise
 finally:lock.unlink()

def main():
 p=argparse.ArgumentParser();p.add_argument('operation',choices=['run','record']);p.add_argument('root',type=pathlib.Path);p.add_argument('--anchor',required=True);p.add_argument('--control',type=pathlib.Path,required=True);p.add_argument('--generation',type=int);p.add_argument('--max-seconds',type=int,default=1500);a=p.parse_args()
 if a.control.resolve().is_relative_to(a.root.resolve()):raise ValueError('independent recorder outside population artifacts required')
 if a.operation=='record':prepare_record(a.root,a.anchor,a.control,a.generation)
 else:run(a.root,a.anchor,a.control,a.max_seconds)
if __name__=='__main__':main()
