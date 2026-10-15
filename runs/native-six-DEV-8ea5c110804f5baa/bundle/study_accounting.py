"""Persistent native-study intake, single-job and cumulative packet accounting."""
import contextlib,copy,hashlib,json,os,pathlib,time
class Hold(RuntimeError):pass

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def atomic(p,x):
    p=pathlib.Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:f.write((json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    if os.name=='posix':
        fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def claim_file(path,payload):
    path=pathlib.Path(path)
    with path.open('xb') as stream:stream.write((json.dumps(payload,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    if os.name=='posix':
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
@contextlib.contextmanager
def lock(root):
    root=pathlib.Path(root);f=(root/'STATE.lock').open('a+b')
    try:
        if os.name=='posix':
            import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        yield
    finally:f.close()
class Accounting:
    def __init__(self,root,study,clock=time.time,guard=None):
        self.root=pathlib.Path(root);self.study=study;self.clock=clock;self.path=self.root/'STATE.json';self.guard=guard or (lambda:None)
    def stopped(self):
        pool=self.study.get('control_root')
        return os.path.lexists(self.root/'STOP') or bool(pool and os.path.lexists(pathlib.Path(pool)/'STOP'))
    def load(self):
        if not self.path.exists() and any((self.root/n).exists() for n in ('JOB_CLAIM.json','INTAKE_CLAIM.json','LAUNCH_CLAIM.json','WORKER_CLAIM.json')):raise Hold('state missing after consumed admission; never reset')
        s=read(self.path) if self.path.exists() else {'study_id':self.study['study_id'],'job_used':False,'packet_used':0,'intake_used':0,'positions_charged':0,'attempts':{},'phase':'REVIEW_HOLD'}
        if s['study_id']!=self.study['study_id'] or s['packet_used']!=len(s['attempts']) or s['packet_used']>216:raise Hold('accounting identity/count mismatch')
        if 'genome' in s and self.study.get('genome_canonical_sha256') and digest(s['genome'])!=self.study['genome_canonical_sha256']:raise Hold('frozen genome changed inside persistent state')
        return s
    def intake(self,raw,validate,commit=False):
        if not commit:return {'phase':'VALIDATION_ONLY_NO_INTAKE','raw_sha256':hashlib.sha256(raw).hexdigest()}
        with lock(self.root):
            if self.stopped():raise Hold('STOP preempts founder intake')
            s=self.load();identity=hashlib.sha256(raw).hexdigest()
            if s.get('intake_sha256')==identity:return copy.deepcopy(s)
            if s['intake_used'] or s['job_used']:raise Hold('founder slot already consumed; no replacement/retry')
            try:
                claim_file(self.root/'INTAKE_CLAIM.json',{'study_id':self.study['study_id'],'raw_sha256':identity})
            except FileExistsError:raise Hold('permanent founder claim already exists')
            s.update(intake_used=1,positions_charged=48,intake_sha256=identity,phase='INTAKE_CONSUMED');atomic(self.path,s)
            def pairs(values):
                d={}
                for k,v in values:
                    if k in d:raise ValueError('duplicate JSON key')
                    d[k]=v
                return d
            try:
                if len(raw)>16384:raise ValueError('founder packet too large')
                w=json.loads(raw,object_pairs_hook=pairs);validate(w);s['genome']=w;s['phase']='FOUNDER_FROZEN'
            except Exception as exc:s['phase']='REJECTED_FOUNDER_CONSUMED';s['error']=str(exc)[:400]
            atomic(self.path,s);return copy.deepcopy(s)
    def reserve_launch(self,token):
        if type(token) is not str or len(token)!=64 or any(c not in '0123456789abcdef' for c in token):raise Hold('invalid launch identity')
        with lock(self.root):
            s=self.load()
            if s.get('phase')!='FOUNDER_FROZEN' or 'genome' not in s:raise Hold('frozen founder required before launch reservation')
        self.claim_job() # The one job is burned BEFORE the systemd RPC; no retry after failure.
        with lock(self.root):
            s=self.load();claim={'study_id':self.study['study_id'],'launch_token':token,'job_claim_sha256':hashlib.sha256((self.root/'JOB_CLAIM.json').read_bytes()).hexdigest()}
            claim_file(self.root/'LAUNCH_CLAIM.json',claim)
            s.update(phase='LAUNCH_RESERVED',launch_token=token);atomic(self.path,s);return claim
    def claim_job(self,launch_token=None):
        with lock(self.root):
            s=self.load()
            if self.stopped():raise Hold('STOP preempts job claim')
            if launch_token is not None:
                claim=read(self.root/'LAUNCH_CLAIM.json')
                if not s['job_used'] or claim.get('study_id')!=self.study['study_id'] or claim.get('launch_token')!=launch_token or s.get('launch_token')!=launch_token or claim.get('job_claim_sha256')!=hashlib.sha256((self.root/'JOB_CLAIM.json').read_bytes()).hexdigest():raise Hold('launch reservation ownership differs')
                if self.clock()>=s['deadline_epoch']:raise Hold('reserved whole-job deadline expired')
                try:
                    claim_file(self.root/'WORKER_CLAIM.json',{'study_id':self.study['study_id'],'launch_token':launch_token,'pid':os.getpid()})
                except FileExistsError:raise Hold('launch already adopted; no second worker or restart')
                s.update(phase='RUNNING_BASELINE',worker_pid=os.getpid());atomic(self.path,s);return s
            if s['job_used']:raise Hold('the one-job allowance is consumed; no replay or resume reset')
            started=self.clock();claim={'study_id':self.study['study_id'],'started_epoch':started,'deadline_epoch':started+600}
            try:
                claim_file(self.root/'JOB_CLAIM.json',claim)
            except FileExistsError:raise Hold('permanent one-job claim already exists')
            s.update(job_used=True,started_epoch=claim['started_epoch'],deadline_epoch=claim['deadline_epoch'],phase='RUNNING_BASELINE');atomic(self.path,s);return s
    def reserve(self,key):
        with lock(self.root):
            s=self.load();self.guard()
            if self.stopped():raise Hold('STOP preempts packet; marker never removed')
            if not s['job_used'] or self.clock()>=s['deadline_epoch']:raise Hold('whole-job deadline or job admission')
            if key in s['attempts']:
                row=s['attempts'][key]
                if row['status']=='intent':raise Hold('interrupted packet remains consumed; no replay')
                return row['result']
            if s['packet_used']>=216:raise Hold('cumulative216 packet cap exhausted')
            s['packet_used']+=1;s['attempts'][key]={'status':'intent'};atomic(self.path,s);return None
    def finish(self,key,result):
        with lock(self.root):
            s=self.load()
            if s['attempts'].get(key,{}).get('status')!='intent':raise Hold('missing owned packet intent')
            s['attempts'][key]={'status':'complete','result':result};atomic(self.path,s)
    def charge_random(self):
        with lock(self.root):
            s=self.load()
            if s.get('random_positions_charged',0) not in (0,48):raise Hold('random charge drift')
            s['random_positions_charged']=48;atomic(self.path,s)
    def terminal(self,phase,result):
        with lock(self.root):
            s=self.load();s['phase']=phase;s['result']=result;atomic(self.path,s);return s

def baseline_futility(rows,orderings):
    return any(sum(r['ordering']==o and r['correct'] for r in rows)>=5 for o in orderings)
def acceptance(rows,orderings,expected_ids):
    required=('authored','scripted_baseline','matched_random');scores={};valid=True;complete=True
    for ordering in orderings:
        scores[ordering]={}
        for arm in required:
            selected=[r for r in rows if r['ordering']==ordering and r['arm']==arm]
            complete &= len(selected)==6 and {r['case_id'] for r in selected}==set(expected_ids)
            scores[ordering][arm]=sum(r['correct'] for r in selected)
            if arm!='authored':valid &= all(r['schema_valid'] for r in selected)
    authored_hashes={r['model_hash'] for r in rows if r['arm']=='authored'}
    complete &= len(authored_hashes)==1
    passed=complete and valid and all(v['authored']>=5 and v['authored']-v['scripted_baseline']>=2 and v['authored']-v['matched_random']>=2 for v in scores.values())
    return {'metric':'restricted_local_packet_accuracy','scores':scores,'exact_denominators_and_one_genome':complete,'controls_valid':valid,'gate_passed':passed,'qualified':False}
