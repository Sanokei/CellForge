"""Reviewed archive-and-rearm of one existing paused study. No job launch or admission.
The cooperative STOP remains unconditional in supervisor.py. This operator retains
its exact file as ACKNOWLEDGED_STOP, records the transaction and enables dispatch
only in a separate reviewed step. Historical protected STOPs are never moved.
"""
import argparse,contextlib,copy,hashlib,json,os,pathlib,stat,time,sys
import supervisor as V
from common import Blocked,read,atomic,canonical,digest,filehash,registry_lock
ROOT=pathlib.Path('/srv/cellforge-inbox/continuous-control-v2')
PH='b409293e233bcd7d027b88b34506a867d32c0b68ffe043e95a5b62ed7bc54dfb'
FP='15daf18a8e92f551a4c04fee1d85fbe22ccba1f20b620219e1c751e8da18d686'
BASE_REGISTRY='f0d60705bea7480f6a2fbfa1a0dec3ba8b5ac9f54e15da36a37cdc8188398661'
STOP_SHA='fb8860d6f4e469a92092c2901f8450e5bcc1b05e0d1f8a8f52a063b6842c237b'
OLD_STOP='/srv/cellforge-inbox/continuous-spatial-20261011/STOP'
OLD_STOP_SHA='6d20f7a21219d86f8b32e2ffc3ddd22ebf8f0a02cd3bf7c3fd1676aacd7b0190'
RUNTIME=('supervisor.py','common.py','worker.py','arc3_env.py','arc3cell.py','structcode.py','observe_dispatch.py','resume_operation.py')

def marker(path):
    path=pathlib.Path(path);s=path.lstat()
    if not stat.S_ISREG(s.st_mode):raise Blocked('marker is not a regular nonsymlink file')
    return {'sha256':filehash(path),'size':s.st_size,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns}
def sync_dir(path):
    if os.name=='posix':
        fd=os.open(path,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
def immutable(path,value):
    path=pathlib.Path(path);data=canonical(value)+b'\n'
    if path.exists():
        if path.read_bytes()!=data:raise Blocked('conflicting immutable transaction record')
        return
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    sync_dir(path.parent)
@contextlib.contextmanager
def operator_locks(root):
    if os.name!='posix':raise Blocked('committing requires POSIX locks')
    import fcntl
    with open(V.LOCK,'a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise Blocked('a dispatcher already owns the global lock')
        with registry_lock(root):yield

def services_stopped():
    for name in ('cellforge-continuous.service','cellforge-continuous-v2.service'):
        r=V.command(['systemctl','show',name,'--property=ActiveState,MainPID'])
        if r.returncode:raise Blocked('watcher state unavailable')
        data=dict(x.split('=',1) for x in r.stdout.splitlines())
        if data.get('ActiveState')!='inactive' or data.get('MainPID')!='0':raise Blocked('watcher must be stopped before operator transition')
        r=V.command(['systemctl','is-enabled',name])
        if r.stdout.strip()!='disabled':raise Blocked('watcher must be disabled during operator transition')

def rename_noreplace(source,destination):
    """One atomic rename that cannot replace a concurrently created destination."""
    if os.name=='nt':
        # Windows os.rename fails if destination already exists.
        os.rename(source,destination);return
    if not sys.platform.startswith('linux'):raise Blocked('no supported no-clobber rename primitive')
    import ctypes,errno
    libc=ctypes.CDLL(None,use_errno=True)
    try:fn=libc.renameat2
    except AttributeError:raise Blocked('renameat2 unavailable; refuse archive transition')
    fn.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint);fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(source),-100,os.fsencode(destination),1)!=0:
        number=ctypes.get_errno()
        raise OSError(number,os.strerror(number),str(destination))

def same_regular_entry(left,right):
    try:
        a=os.stat(left,follow_symlinks=False);b=os.stat(right,follow_symlinks=False)
        return stat.S_ISREG(a.st_mode) and stat.S_ISREG(b.st_mode) and (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
    except FileNotFoundError:return False

def publish_no_clobber(authorization,enabled):
    """Publish the durable authorization inode; never replace an existing marker."""
    try:os.link(authorization,enabled,follow_symlinks=False)
    except FileExistsError:
        if not same_regular_entry(authorization,enabled):raise Blocked('foreign dispatch marker at publication boundary')
    sync_dir(pathlib.Path(enabled).parent)
    if not same_regular_entry(authorization,enabled):raise Blocked('dispatch marker replaced during publication; left untouched')

class ResumeOperation:
    # Dependency overrides are for isolated fixtures; production CLI uses defaults.
    def __init__(self,root=ROOT,backend=None,service_check=services_stopped,locks=operator_locks,gate=None,expected_root=ROOT,expected_plan=PH,expected_registry=BASE_REGISTRY,extra_protected=None,checkpoint=None):
        self.input_root=pathlib.Path(root);self.root=self.input_root.resolve();self.backend=backend or V.Systemd();self.service_check=service_check;self.locks=locks
        self.expected_root=pathlib.Path(expected_root).resolve();self.ph=expected_plan;self.expected_registry=expected_registry
        self.extra_protected=extra_protected if extra_protected is not None else {OLD_STOP:OLD_STOP_SHA}
        self.gate=gate or (lambda b,p:V.Controller(self.root,self.backend).gates(b,p));self.checkpoint=checkpoint or (lambda _:None)
    def scope(self):
        if self.root!=self.expected_root or self.input_root.is_symlink():raise Blocked('different control root is not authorized')
        self.service_check();info=self.backend.show()
        if info['LoadState']!='not-found' or not self.backend.empty(info):raise Blocked('fixed job must be absent with a verified empty tree')
        state=read(self.root/'REGISTRY.json')
        if state['active'] is not None or state['sequence']!=1 or state['consecutive_failures']!=0 or state['not_before']!=0:raise Blocked('paused accounting changed; a new review is required')
        if set(state['studies'])!={self.ph}:raise Blocked('not the same single existing study')
        study=state['studies'][self.ph]
        if study['cap']!=9 or study['attempted']!=1:raise Blocked('attempt cap/accounting changed')
        b=self.root/'studies'/self.ph;p=read(b/'PLAN.json')
        if digest(p)!=self.ph or p['qualified'] is not False or p['remote_model_calls']!=0:raise Blocked('frozen science/scope changed')
        ids={c['fingerprint'] for c in p['conditions']}
        if set(study['condition_ids'])!=ids or len(ids)!=9:raise Blocked('condition set changed')
        e=state['conditions'].get(FP,{})
        if e.get('status')!='cancelled' or FP not in ids or e.get('receipt')!='7393b8f22c3dd72dde08ae5a41f3c083ad610bb59b1342100d270229345d8ec0':raise Blocked('cancelled fingerprint must remain consumed')
        pending=sorted(n for n,x in state['conditions'].items() if x.get('study')==self.ph and x['status']=='pending')
        if len(pending)!=8 or set(pending)!=(ids-{FP}):raise Blocked('only the original eight pending conditions are eligible')
        protected={**p['protected_stops'],**self.extra_protected}
        if str(self.root/'STOP') in protected:raise Blocked('a protected historical STOP cannot be acknowledged here')
        for name,h in protected.items():
            if filehash(name)!=h:raise Blocked('historical STOP changed')
        run=pathlib.Path(p['lab_runs_root'])/('weights-transition-'+self.ph)/'jobs'/FP
        artifacts={str(run/n):filehash(run/n) for n in ('CONFIG.json','RESULTS.json','TERMINAL.json','COLLECTION.json')}
        terminal=read(run/'TERMINAL.json')
        if terminal['outcome']!='cancelled' or terminal['receipt_id']!=e['receipt'] or digest({k:v for k,v in terminal.items() if k!='receipt_id'})!=e['receipt']:raise Blocked('cancellation receipt changed')
        if terminal['evidence']['config_sha256']!=filehash(run/'CONFIG.json') or terminal['evidence']['results_sha256']!=filehash(run/'RESULTS.json'):raise Blocked('partial evidence binding changed')
        return state,b,p,pending,protected,artifacts
    def prepare_request(self,bundle):
        state,b,p,pending,protected,artifacts=self.scope()
        if filehash(self.root/'REGISTRY.json')!=self.expected_registry or (self.root/'DISPATCH_ENABLED').exists():raise Blocked('initial registry/dispatch state differs')
        pause=marker(self.root/'STOP')
        if pause['sha256']!=STOP_SHA:raise Blocked('different operator STOP requires a new decision')
        runtime={n:filehash(pathlib.Path(bundle)/n) for n in RUNTIME}
        return {'schema':'cellforge-reviewed-rearm/1','root':str(self.root),'science_plan':self.ph,'base_registry':state,'base_registry_sha256':self.expected_registry,'stop_identity':pause,'pending_conditions':pending,'protected_stops':protected,'cancelled_artifacts':artifacts,'runtime_sha256':runtime,'cap':9,'already_attempted':1,'additional_attempts':8,'qualified':False,'remote_model_calls':0}
    def verify_approval(self,bundle,request,approval,audit):
        bundle=pathlib.Path(bundle);m=read(bundle/'MANIFEST.json')
        if not set(RUNTIME).issubset(m['files']):raise Blocked('runtime absent from review manifest')
        for n,h in m['files'].items():
            if pathlib.PurePath(n).is_absolute() or '..' in pathlib.PurePath(n).parts or filehash(bundle/n)!=h:raise Blocked('reviewed revision changed')
        required={'decision':'permit_existing_study_archive_rearm','request_sha256':digest(request),'operational_manifest_sha256':filehash(bundle/'MANIFEST.json'),'resume_review_passed':True,'qualified':False,'remote_model_calls':0}
        if any(approval.get(k)!=v for k,v in required.items()) or not approval.get('audit_report_sha256') or filehash(audit)!=approval['audit_report_sha256']:raise Blocked('specific independent resume approval is missing/mismatched')
        if request['schema']!='cellforge-reviewed-rearm/1' or request['root']!=str(self.root) or request['science_plan']!=self.ph or request['cap']!=9 or request['already_attempted']!=1 or request['additional_attempts']!=8 or request['qualified'] is not False or request['remote_model_calls']!=0:raise Blocked('request scope differs')
        if request['base_registry_sha256']!=self.expected_registry or hashlib.sha256(canonical(request['base_registry'])+b'\n').hexdigest()!=self.expected_registry or set(request['runtime_sha256'])!=set(RUNTIME) or request['stop_identity']['sha256']!=STOP_SHA:raise Blocked('request baseline binding differs')
        for n,h in request['runtime_sha256'].items():
            if filehash(self.root/'controller'/n)!=h or filehash(bundle/n)!=h:raise Blocked('reviewed runtime has not been installed intact')
    def verify_current(self,request,allow_rearmed=False):
        state,b,p,pending,protected,artifacts=self.scope()
        expected=copy.deepcopy(request['base_registry']);expected['status']='paused_dispatch_disabled'
        if state!=request['base_registry'] and not (allow_rearmed and state==expected):raise Blocked('registry/accounting drift since approval')
        if pending!=request['pending_conditions'] or protected!=request['protected_stops'] or artifacts!=request['cancelled_artifacts']:raise Blocked('approved readiness evidence changed')
        self.gate(b,p)
        return state
    def directory(self,request):
        d=self.root/'resumptions'/digest(request)
        for p in [self.root/'resumptions',d]:
            if p.exists() and (p.is_symlink() or not p.is_dir()):raise Blocked('unsafe operation directory')
        return d
    def stage(self,bundle,request,approval,audit):
        with self.locks(self.root):
            self.verify_approval(bundle,request,approval,audit);self.verify_current(request,True)
            if (self.root/'DISPATCH_ENABLED').exists():raise Blocked('dispatch must stay absent during rearm')
            d=self.directory(request);d.mkdir(parents=True,exist_ok=True)
            immutable(d/'INTENT.json',request);immutable(d/'APPROVAL.json',approval)
            if (d/'AUDIT_REPORT').exists():
                if filehash(d/'AUDIT_REPORT')!=filehash(audit):raise Blocked('audit conflict')
            else:
                with (d/'AUDIT_REPORT').open('xb') as f:f.write(pathlib.Path(audit).read_bytes());f.flush();os.fsync(f.fileno())
                sync_dir(d)
            self.checkpoint('after_resume_intent')
            source=self.root/'STOP';archive=d/'ACKNOWLEDGED_STOP'
            if archive.exists():
                if marker(archive)!=request['stop_identity']:raise Blocked('archived STOP identity/bytes changed')
                if source.exists():raise Blocked('new STOP preempts recovery; never clear it')
            else:
                if marker(source)!=request['stop_identity']:raise Blocked('STOP changed before acknowledgement')
                self.checkpoint('before_stop_archive')
                if marker(source)!=request['stop_identity']:raise Blocked('STOP changed at acknowledgement boundary')
                if archive.exists() or archive.is_symlink():raise Blocked('archive appeared at acknowledgement boundary')
                try:rename_noreplace(source,archive)
                except FileExistsError:raise Blocked('archive appeared at atomic acknowledgement boundary; preserved')
                sync_dir(self.root);sync_dir(d)
                if marker(archive)!=request['stop_identity']:
                    # A noncooperating STOP writer can race the identity check.
                    # Restore the moved STOP with exclusive creation; never overwrite a newer STOP.
                    try:os.link(archive,source);sync_dir(self.root)
                    except FileExistsError:pass
                    raise Blocked('STOP changed during rename; restored hold, no release')
                self.checkpoint('after_stop_archive')
            if source.exists():raise Blocked('new STOP preempts rearm')
            receipt=self.expected_receipt(request)
            immutable(d/'RESUME_RECEIPT.json',receipt);self.checkpoint('after_resume_receipt')
            state=self.verify_current(request,True);state['status']='paused_dispatch_disabled';atomic(self.root/'REGISTRY.json',state)
            self.checkpoint('after_resume_status')
            return {'phase':'REARMED_STOPPED_DISPATCH_DISABLED','receipt':receipt,'root_STOP_present':source.exists()}
    def expected_receipt(self,request):
        return {'schema':'cellforge-rearm-receipt/1','request_id':digest(request),'original_stop':request['stop_identity'],'archived_stop':str(self.directory(request)/'ACKNOWLEDGED_STOP'),'pending_conditions':request['pending_conditions'],'cancelled_artifacts':request['cancelled_artifacts'],'cap':9,'already_attempted':1,'additional_attempts':8,'dispatch_enabled':False}
    def activate(self,bundle,request,approval,audit):
        with self.locks(self.root):
            self.verify_approval(bundle,request,approval,audit);state=self.verify_current(request,True)
            if state['status']!='paused_dispatch_disabled':raise Blocked('rearm status not committed')
            d=self.directory(request);receipt=read(d/'RESUME_RECEIPT.json')
            if read(d/'INTENT.json')!=request or read(d/'APPROVAL.json')!=approval or filehash(d/'AUDIT_REPORT')!=approval['audit_report_sha256']:raise Blocked('rearm transaction records differ')
            if receipt!=self.expected_receipt(request) or marker(d/'ACKNOWLEDGED_STOP')!=request['stop_identity']:raise Blocked('rearm receipt/archive mismatch')
            if (self.root/'STOP').exists():raise Blocked('new STOP preempts release')
            payload={'schema':'cellforge-reviewed-dispatch/1','request_id':digest(request),'science_plan':self.ph,'pending_conditions':request['pending_conditions'],'qualified':False,'remote_model_calls':0}
            enabled=self.root/'DISPATCH_ENABLED'
            authorization=d/'DISPATCH_AUTHORIZATION.json';immutable(authorization,payload)
            self.checkpoint('before_dispatch_marker')
            if (self.root/'STOP').exists():raise Blocked('new STOP preempts release boundary')
            publish_no_clobber(authorization,enabled);self.checkpoint('after_dispatch_marker')
            if (self.root/'STOP').exists():
                # There is no atomic compare-and-unlink for a replaceable pathname.
                # Preserve every visible marker; STOP + disabled watcher retain the hold.
                self.checkpoint('before_dispatch_preempt_hold')
                immutable(d/'DISPATCH_PREEMPTED.json',{'schema':'cellforge-dispatch-preempted/1','request_id':digest(request),'reason':'new_STOP','dispatch_marker_action':'untouched','watcher_stopped_disabled':True})
                raise Blocked('new STOP preempted activation; dispatch marker untouched; keep watcher stopped')
            immutable(d/'DISPATCH_RECEIPT.json',payload)
            return {'phase':'DISPATCH_AUTHORIZED_WATCHER_STILL_STOPPED','payload':payload}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=pathlib.Path,default=ROOT);parser.add_argument('--bundle',type=pathlib.Path,required=True);parser.add_argument('--mode',choices=['prepare-request','stage','activate'],default='prepare-request');parser.add_argument('--request',type=pathlib.Path);parser.add_argument('--approval',type=pathlib.Path);parser.add_argument('--audit',type=pathlib.Path);parser.add_argument('--commit-reviewed',action='store_true')
    a=parser.parse_args();op=ResumeOperation(a.root)
    if a.mode=='prepare-request':result=op.prepare_request(a.bundle)
    elif not a.commit_reviewed:result={'phase':'VALIDATION_ONLY_NO_TRANSITION','mode':a.mode,'request':read(a.request)}
    else:
        if not all([a.request,a.approval,a.audit]):raise Blocked('request, specific approval and audit required')
        result=getattr(op,a.mode)(a.bundle,read(a.request),read(a.approval),a.audit)
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__':main()