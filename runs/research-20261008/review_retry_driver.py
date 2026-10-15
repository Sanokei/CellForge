import os,json,pathlib,hashlib,subprocess,select,time,resource,shutil
from datetime import datetime,timezone
ROOT=pathlib.Path('/srv/cellforge-inbox/research-20261008')
SOURCE=ROOT/'source'
ENV={k:v for k,v in os.environ.items() if k not in ('OPENAI_API_KEY','AZURE_OPENAI_API_KEY')}
ENV.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def now():return datetime.now(timezone.utc).isoformat()
def save(path,obj):path.write_text(json.dumps(obj,indent=2))
def meter():
 p=subprocess.Popen(['/root/.local/bin/codex','app-server','--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,env=ENV)
 def send(o):p.stdin.write(json.dumps(o)+'\n');p.stdin.flush()
 try:
  send({'id':1,'method':'initialize','params':{'clientInfo':{'name':'cellforge_review_meter','version':'1.0'}}})
  deadline=time.monotonic()+25
  while time.monotonic()<deadline:
   ready,_,_=select.select([p.stdout],[],[],1)
   if not ready:continue
   line=p.stdout.readline()
   if not line:break
   try:o=json.loads(line)
   except ValueError:continue
   if o.get('id')==1:
    if 'error' in o:raise RuntimeError('meter initialization failed')
    send({'method':'initialized'});send({'id':2,'method':'account/rateLimits/read','params':None})
   elif o.get('id')==2:
    r=o.get('result',{});limit=r.get('rateLimits',{})
    reading={'utc':now(),'ordinary_usage_allowed':r.get('ordinaryUsageAllowed'),'primary':limit.get('primary'),'credits':limit.get('credits'),'cap_percent':85}
    if not isinstance(reading['primary'],dict):raise RuntimeError('current quota unavailable')
    return reading
  raise RuntimeError('supported quota read timeout')
 finally:
  p.terminate()
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill();p.wait()
def verify():
 manifest=json.loads((SOURCE/'_SOURCE_MANIFEST.json').read_text(encoding='utf-8-sig'))
 for f in manifest['files']:
  path=SOURCE/f['path']
  if not path.resolve().is_relative_to(SOURCE.resolve()):raise RuntimeError('manifest escapes source')
  data=path.read_bytes()
  if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=f['git_blob_sha1'] or hashlib.sha256(data).hexdigest()!=f['sha256']:raise RuntimeError('source mismatch '+f['path'])
 return {'utc':now(),'verified_git_blobs':len(manifest['files']),'commit':manifest['commit'],'partial_snapshot':True,'source':str(SOURCE),'file_sha256':{f['path']:f['sha256'] for f in manifest['files']}}
verification=verify();save(ROOT/'SOURCE_VERIFIED.json',verification)
constraints=ROOT/'constraints'
stop_manifest=json.loads((constraints/'STOP_MANIFEST.json').read_text())
for item in stop_manifest['receipts']:
 original=pathlib.Path(item['original']);copied=pathlib.Path(item['copy'])
 if hashlib.sha256(original.read_bytes()).hexdigest()!=item['sha256'] or original.read_bytes()!=copied.read_bytes():raise RuntimeError('preserved STOP mismatch')
work=ROOT/'review-002';work.mkdir(exist_ok=False)
before=meter();save(work/'RATE_BEFORE.json',before)
used=before['primary'].get('usedPercent')
if not isinstance(used,(int,float)) or used>=85 or before['ordinary_usage_allowed'] is not True:raise RuntimeError('quota unavailable or at cap; no worker launched')
mem={line.split(':')[0]:line.split(':')[1].strip() for line in pathlib.Path('/proc/meminfo').read_text().splitlines()}
available_kb=int(mem['MemAvailable'].split()[0]);load=os.getloadavg()
existing=subprocess.run(['pgrep','-af','codex exec'],capture_output=True,text=True).stdout.splitlines()
if len(existing)>=2 or load[0]>1.5 or available_kb<1024*1024:raise RuntimeError('resource/concurrency gate; no worker launched')
prompt='''Owner explicitly authorizes ONE read-only source REVIEW on the existing CellForge server. This is infrastructure/scientific review, NOT weight authoring or a campaign. Do not launch agents, model calls, experiments, tests, benchmarks, downloads, gradient methods, GitHub actions, or write/edit any file. Use only read-only local inspection tools inside this source workspace. Do not inspect credentials, session storage, private/evaluation datasets, outside paths or old server runtimes.
Read AGENTS.md and the FULL LAB.md first. Treat source as a partial snapshot at published bfc928bf0c1f77922b4de00834412f986cd54bfa, not a Git clone or unpublished cloud fixes. Historical STOPs remain; this new authorization permits review only, not resuming closed lanes. The source manifest records omissions. Absence of fixtures is not scientific failure or success.
Inspect src/cellforge/construction/nca_gene.py and nca_shallow.py, the relevant graph/profile/runtime semantics, tests/test_nca_gene.py and test_nca_shallow.py. Inspect scripts/structcode.py and struct_pilot.py control validity and tests/test_structcode_control_errors.py, test_struct_pilot_feedback.py, plus reports/CONTROL_ERROR_INTEGRITY_20261008.md. Distinguish source execution correctness, actual tensor dependence, generator/program capability and useful learned generalization. Do not interpret deterministic seeds as LLM authorship or fixed-slot transfer as general cross-architecture breeding.
Return a concise review (under700 words) with concrete file/line evidence: (1) strongest verified semantics and control-validity concerns; (2) what current tests establish vs missing evidence; (3) ONE bounded next scientific hypothesis with numerical-only author contract, development/frozen split, zero/random controls, budget/STOP/no-promotion conditions. Proposed only; execute nothing. State limitations and whether current published source is sufficient for that hypothesis. No claims of new training gains.'''
(work/'PROMPT.txt').write_text(prompt)
cmd=['/root/.local/bin/codex','exec','--ignore-user-config','--skip-git-repo-check','--sandbox','read-only','-m','gpt-6.1-sol','-c','model_reasoning_effort="low"','--json','-C',str(SOURCE),'-o',str(work/'RESULT.md'),'-']
start={'utc':now(),'scope':'one read-only source-review worker','command':cmd,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'source_commit':verification['commit'],'partial_snapshot':True,'worker_concurrency_limit':2,'workers_started':1,'limits':{'wall_seconds':120,'cpu_seconds':60,'address_space_cap':None,'physical_rss_stop_mib':2048,'available_ram_stop_mib':512,'nice':5},'resources':{'load':load,'available_kib':available_kb},'auth':'existing ChatGPT; API variables stripped','quota_before':before['primary']}
save(work/'START.json',start);print(json.dumps({'start':start,'verified_git_blobs':verification['verified_git_blobs']}),flush=True)
def limits():
 os.setsid();resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(5)
tick=time.monotonic()
try:
 with (work/'events.jsonl').open('w') as sink:
  import signal
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,text=True,stdout=sink,stderr=sink,env=ENV,preexec_fn=limits)
  proc.stdin.write(prompt);proc.stdin.close()
  peak_rss_kib=0;monitor=[];stop_reason=None
  while proc.poll() is None:
   process_rows=subprocess.run(['ps','-eo','pid=,ppid=,rss='],capture_output=True,text=True,timeout=3).stdout.splitlines()
   table={int(parts[0]):(int(parts[1]),int(parts[2])) for line in process_rows if len(parts:=line.split())==3}
   owned={proc.pid}
   while True:
    expanded=owned|{pid for pid,(ppid,rss) in table.items() if ppid in owned}
    if expanded==owned:break
    owned=expanded
   rss=sum(table.get(pid,(0,0))[1] for pid in owned)
   available=int(next(line for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])
   peak_rss_kib=max(peak_rss_kib,rss)
   monitor.append({'seconds':round(time.monotonic()-tick,2),'owned_processes':len(owned),'rss_kib':rss,'available_kib':available})
   if rss>2048*1024:stop_reason='owned physical RSS exceeded2GiB'
   elif available<512*1024:stop_reason='available RAM below512MiB'
   elif time.monotonic()-tick>120:stop_reason='120second wall-time limit'
   if stop_reason:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    break
   time.sleep(1)
  proc.wait()
  save(work/'RESOURCE_MONITOR.json',{'peak_owned_rss_kib':peak_rss_kib,'stop_reason':stop_reason,'samples':monitor})
  done=proc
 thread=None;usage=None
 for line in (work/'events.jsonl').read_text().splitlines():
  try:event=json.loads(line)
  except ValueError:continue
  if event.get('type')=='thread.started':thread=event.get('thread_id')
  if event.get('type')=='turn.completed':usage=event.get('usage')
 receipt={'utc':now(),'returncode':done.returncode,'seconds':round(time.monotonic()-tick,3),'peak_owned_rss_kib':peak_rss_kib,'stop_reason':stop_reason,'thread_id':thread,'usage':usage,'result_exists':(work/'RESULT.md').exists()}
except subprocess.TimeoutExpired:
 receipt={'utc':now(),'error':'bounded review timed out','seconds':round(time.monotonic()-tick,3)}
save(work/'EXIT.json',receipt)
after=meter();save(work/'RATE_AFTER.json',after)
post=verify();save(work/'SOURCE_UNCHANGED.json',{'unchanged':post['file_sha256']==verification['file_sha256'],'verified_git_blobs':post['verified_git_blobs']})
print(json.dumps({'exit':receipt,'quota_after':after,'source_unchanged':True}),flush=True)
if (work/'RESULT.md').exists():print((work/'RESULT.md').read_text(),flush=True)