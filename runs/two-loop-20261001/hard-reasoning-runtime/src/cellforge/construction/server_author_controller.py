"""Bounded server-only author queue and separate deterministic commitment recorder.
Recorder has no model client. It verifies frozen inputs before releasing scoring.
Author runs read-only, receives DEV prompt JSON, and cannot publish anchors.
"""
import argparse,datetime,hashlib,json,os,pathlib,subprocess,time
from cellforge.construction.action_cell_continuation import atomic
from cellforge.construction.experiment_manifest import load_json_strict
from cellforge.construction.joint_weight_population import digest
from cellforge.construction.anchored_action_protocol import file_hash

def verify(root,anchor):
 if file_hash(root/'DECLARATION.json')!=anchor:raise ValueError('initial commitment mismatch')
 for name,sha in load_json_strict(root/'DECLARATION.json')['dependencies'].items():
  p=(root/name).resolve()
  if not p.is_relative_to(root.resolve()) or file_hash(p)!=sha:raise ValueError('immutable dependency differs:'+name)

def counter(root,kind):
 key='phase' if kind=='cells' else 'round'
 files=[p for p in root.glob('STATE-*.json') if p.stem.split('-')[-1].isdigit()]
 p=max(files,key=lambda p:int(p.stem.split('-')[-1]));return load_json_strict(p),key

def record(root,kind,anchor,control,raw_path):
 verify(root,anchor);state,key=counter(root,kind);n=state[key]+1
 if n>3:raise ValueError('declared batch exhausted')
 requests=load_json_strict(root/f'PROMPTS-{n}.json');raw=load_json_strict(raw_path)
 if len(raw)!=len(requests):raise ValueError('context response count')
 for q,r in zip(requests,raw):
  p=q['prompt'];v=r['response']
  if q['seed']!=r['seed'] or v.get('model')!='gpt-6.1-sol' or v.get('reasoning')!='low':raise ValueError('maker/context provenance')
  if kind=='cells':
   from cellforge.construction.action_phase_protocol import bind_author
   bind_author(v,p,{'id':p['parent_id'],'params':p['parent_params']},p['development_evidence'])
  else:
   from cellforge.construction.arc_weight_moe import assemble,make_core
   assemble(v,p,state['pool'],p['parent_hash'],make_core(root/'pop1_core.py'))
 for previous in range(1,n):
  path=root/f'RESPONSES-{previous}.json'
  if path.exists() and digest(load_json_strict(path))==digest(raw):raise ValueError('reused maker receipt')
 response=root/f'RESPONSES-{n}.json'
 if response.exists() and digest(load_json_strict(response))!=digest(raw):raise ValueError('response already sealed with different bytes')
 if not response.exists():atomic(response,raw)
 prefix='PHASE' if kind=='cells' else 'ROUND'
 declaration={'initial_anchor':anchor,key:n,'parent_state_hash':digest(state),'responses_sha256':file_hash(response),
              'prompts_sha256' if kind=='cells' else 'prompts_hash':digest(requests)}
 path=root/f'{prefix}-{n}-DECLARATION.json'
 if path.exists() and load_json_strict(path)!=declaration:raise ValueError('commitment already exists')
 if not path.exists():atomic(path,declaration)
 sha=file_hash(path);control.parent.mkdir(parents=True,exist_ok=True)
 ledger=control.with_suffix('.ledger.jsonl');previous='0'*64;entries=[]
 if ledger.exists():
  for line in ledger.read_text().splitlines():
   row=json.loads(line);observed=row.pop('hash')
   if row['previous']!=previous or digest(row)!=observed:raise ValueError('independent recorder chain differs')
   previous=observed;entries.append(row['entry'])
 event={'initial_anchor':anchor,'kind':kind,'counter':n,'phase_anchor':sha,'parent_state_hash':digest(state),'responses_sha256':file_hash(response)}
 matches=[e for e in entries if e['counter']==n]
 if matches and matches!=[event]:raise ValueError('recorder history conflict')
 if not matches:
  row={'previous':previous,'entry':event};row['hash']=digest(row)
  with ledger.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 values=load_json_strict(control) if control.exists() else {'initial_anchor':anchor,'phase_anchors' if kind=='cells' else 'round_anchors':{}}
 if values['initial_anchor']!=anchor:raise ValueError('recorder controller mismatch')
 values['phase_anchors' if kind=='cells' else 'round_anchors'][str(n)]=sha
 values['recorder']='separate trusted deterministic process, append-only hash chain, before evaluation'
 atomic(control,values);return event

def author(root,kind,workspace,codex='/root/.local/bin/codex'):
 state,key=counter(root,kind);n=state[key]+1;requests=load_json_strict(root/f'PROMPTS-{n}.json')
 evidence=[q.get('prompt',{}).get('development_evidence',{}) for q in requests]
 if evidence and all(e.get('outcomes') and all(e['outcomes'].values()) and e.get('counterexamples')==[] for e in evidence):raise ValueError('saturated development niche; dispatch harder declared environment instead')
 work=workspace/f'{kind}-{n}';work.mkdir(parents=True,exist_ok=True)
 if (work/'EXIT.json').exists():
  if load_json_strict(work/'EXIT.json')['returncode']==0:return work/'response.json'
  raise ValueError('failed author attempt retained; explicit retry policy required')
 env=dict(os.environ)
 for k in ('OPENAI_API_KEY','AZURE_OPENAI_API_KEY'):env.pop(k,None)
 auth=subprocess.run([codex,'login','status'],capture_output=True,text=True,env=env,timeout=15)
 if auth.returncode or 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:raise ValueError('ChatGPT subscription auth required; no paid fallback')
 rows=[]
 for item in requests:
  p=item['prompt'];q=json.loads(json.dumps(p));q['supplied_prompt_sha256']=digest(p)
  if kind=='cells':q['supplied_feedback_sha256']=digest(p['development_evidence'])
  # Preserve only executed DEV evidence, full parent/pool numeric material and hashes.
  rows.append({'seed':item['seed'],'prompt':q})
 contract=('response fields model gpt-6.1-sol,reasoning low,parent_id,parent_params,tensors(all39positions),feature_order,role_order,prompt_sha256(supplied),feedback_sha256(supplied),charged_positions39,interface tool-action-joint/2. Fresh actual numeric revision, no guaranteed gain.' if kind=='cells' else
 'response fields model gpt-6.1-sol,reasoning low,parent_hash,pool_hash,prompt_hash(supplied_prompt_sha256),cell_ids(atleast2distinct earned live cells),router{route.W NxN,route.b N},charged_positions N*N+N,rationale. Copy selected earned cells; write genuine new literal router values. No claimed improvement before evaluation.')
 text='Use ONLY supplied development evidence. Do not invoke tools, read files, or access held-out tasks. Return ONLY JSON array [{seed,response}]. '+contract+'\n'+json.dumps(rows)
 cmd=[codex,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--sandbox','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="low"','--json','-C',str(work),'-o',str(work/'response.json'),'-']
 atomic(work/'START.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':'gpt-6.1-sol','reasoning':'low','auth':'ChatGPT subscription','prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'parent_state_hash':digest(state),'paid_api_fallback':False})
 started=time.monotonic()
 with (work/'events.jsonl').open('w') as log:
  completed=subprocess.run(cmd,input=text,text=True,stdout=log,stderr=log,env=env,timeout=300)
 events=(work/'events.jsonl').read_text()
 if '"type":"command_execution"' in events or '"type":"mcp_tool_call"' in events:raise ValueError('author invoked forbidden tools; no release')
 atomic(work/'EXIT.json',{'returncode':completed.returncode,'seconds':time.monotonic()-started})
 if completed.returncode:raise ValueError('Codex author failed; see isolated events, no fallback')
 load_json_strict(work/'response.json');return work/'response.json'

def drive(root,kind,anchor,control,workspace,max_seconds=1200,scheduler=None):
 if not 1<=max_seconds<=1800:raise ValueError('bounded controller deadline')
 deadline=time.monotonic()+max_seconds;status=workspace/f'{kind}-WATCHDOG.json';workspace.mkdir(parents=True,exist_ok=True)
 while time.monotonic()<deadline:
  if scheduler is not None:
   atomic(status,{'status':'independent_niche_queue','queues':scheduler.tick()});time.sleep(.1);continue
  verify(root,anchor);state,key=counter(root,kind);n=state[key]+1
  if not (root/f'PROMPTS-{n}.json').exists():atomic(status,{'status':'blocked_missing_prompt',key:state[key]});time.sleep(2);continue
  try:
   atomic(status,{'status':'authoring',key:state[key],'next':n});path=author(root,kind,workspace)
   # Independent recorder command has no model/API invocation and publishes only after validation+fsync.
   subprocess.run([os.sys.executable,str(pathlib.Path(__file__).resolve()),'record',str(root),'--kind',kind,'--anchor',anchor,'--control',str(control),'--raw',str(path)],check=True,timeout=30)
   atomic(status,{'status':'awaiting_evaluation',key:state[key],'next':n})
   while time.monotonic()<deadline:
    current,_=counter(root,kind)
    if current[key]>=n:break
    service=root/'SERVICE_STATUS.json'
    if service.exists() and load_json_strict(service)['status'] in ('blocked','bounded_service_deadline','bounded_service_deadline','declared_generation_budget_complete'):raise ValueError('evaluation service not advancing; inspect durable reason')
    time.sleep(2)
  except Exception as e:
   receipt={'status':'blocked',key:state[key],'error':type(e).__name__+':'+str(e)}
   atomic(status,receipt);return receipt
 atomic(status,{'status':'bounded_controller_deadline',key:counter(root,kind)[0][key]})

def main():
 p=argparse.ArgumentParser();p.add_argument('operation',choices=['drive','record']);p.add_argument('root',type=pathlib.Path);p.add_argument('--kind',choices=['cells','moe'],required=True);p.add_argument('--anchor',required=True);p.add_argument('--control',type=pathlib.Path,required=True);p.add_argument('--workspace',type=pathlib.Path);p.add_argument('--raw',type=pathlib.Path);p.add_argument('--max-seconds',type=int,default=1200);a=p.parse_args()
 if a.control.resolve().is_relative_to(a.root.resolve()):raise ValueError('recorder outside evaluation artifacts required')
 if a.operation=='record':print(json.dumps(record(a.root,a.kind,a.anchor,a.control,a.raw)))
 else:drive(a.root,a.kind,a.anchor,a.control,a.workspace,a.max_seconds)
if __name__=='__main__':main()


