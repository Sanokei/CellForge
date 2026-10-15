"""Review-gated native six-case worker. No tool execution, external writer or queue refill."""
import argparse,copy,hashlib,json,os,pathlib,random,shutil,subprocess,sys,time,signal,stat
from study_accounting import Accounting,Hold,acceptance,baseline_futility,digest,read
sys.path.insert(0,str(pathlib.Path(__file__).parent/'template'))
import native_adapter as A
import native_slots4 as N
import restricted_scorer as S

def filehash(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def verify(bundle,approval_path,audit_path):
    bundle=pathlib.Path(bundle);m=read(bundle/'MANIFEST.json');a=read(approval_path)
    for name,value in m['files'].items():
        p=pathlib.PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or filehash(bundle/name)!=value:raise Hold('frozen source/data changed')
    study=read(bundle/'STUDY.json')
    expected={'decision':'permit_native4_exposed_DEV_pilot_and_standard_v2_stop','review_passed':True,'study_id':study['study_id'],'manifest_sha256':filehash(bundle/'MANIFEST.json'),'qualified':False,'external_writer_calls':0,'cohort_role':'EXPOSED_PUBLIC_DEV','allow_standard_v2_stop_new_STOP':True,'allow_old_arc3_resume':False,'old_attempted':9,'old_cap':9,'cumulative_packet_cap':216,'job_cap':1,'whole_job_seconds':600,'genome_sha256':study.get('genome_sha256')}
    if any(a.get(k)!=v for k,v in expected.items()) or a.get('fixture_only') or a.get('audit_sha256')!=filehash(audit_path):raise Hold('specific independent approval missing/mismatched')
    for name,value in {**study['protected_stops'],**study['cancelled_artifacts']}.items():
        if filehash(name)!=value:raise Hold('historical STOP/cancellation evidence changed')
    archived=study['archived_stop'];st=os.stat(archived['path'],follow_symlinks=False)
    if filehash(archived['path'])!=archived['sha256'] or any(getattr(st,k)!=archived[v] for k,v in [('st_dev','device'),('st_ino','inode'),('st_mtime_ns','mtime_ns'),('st_size','size')]):raise Hold('archived STOP identity changed')
    registry=read(study['historical_arc3_registry']);entry=registry['conditions'][study['consumed_cancelled_fingerprint']]
    if entry['status']!='cancelled' or entry['receipt']!=study['consumed_cancelled_receipt']:raise Hold('historical cancellation/accounting changed')
    if registry['active'] is not None or registry['studies'][study['historical_arc3_plan']]['attempted']!=9 or registry['studies'][study['historical_arc3_plan']]['cap']!=9:raise Hold('historical study not safely exhausted')
    return study

def expected_envelope(study,launch_token,pid):
    return {'MainPID':str(pid),'Description':'cellforge-native4:'+study['study_id']+':'+launch_token,'CPUQuotaPerSecUSec':'1s','CPUAffinity':'0','MemoryMax':'1073741824','MemorySwapMax':'0','RuntimeMaxUSec':'9min 55s','TasksMax':'64','PrivateNetwork':'yes','KillMode':'control-group','SendSIGKILL':'yes','TimeoutStopUSec':'5s','Restart':'no','MemoryAccounting':'yes','CPUAccounting':'yes','OOMPolicy':'kill','LimitFSIZE':'16777216','LimitCORE':'0','PrivateTmp':'yes','ProtectSystem':'strict','ProtectHome':'yes','NoNewPrivileges':'yes','CapabilityBoundingSet':'','ReadWritePaths':study['control_state_root']+' /run/cellforge-continuous.lock','InaccessiblePaths':'/srv/cellforge-private','ReadOnlyPaths':study['bundle_root']+' '+study['review_root']}
def validate_envelope(actual,expected):
    if set(actual)!=set(expected):raise Hold('incomplete/extra resource properties')
    path_keys={'ReadWritePaths','ReadOnlyPaths','InaccessiblePaths'}
    for key,value in expected.items():
        different=set(actual[key].split())!=set(value.split()) if key in path_keys else actual[key]!=value
        if different:raise Hold('ownership/resource envelope differs at '+key)
def systemd_envelope(study,launch_token):
    if os.name!='posix':raise Hold('production requires the reviewed Linux resource envelope')
    if filehash('/usr/bin/python3')!=study['python_runtime']['sha256']:raise Hold('reviewed Python runtime changed')
    expected=expected_envelope(study,launch_token,os.getpid())
    raw=subprocess.check_output(['systemctl','show','cellforge-continuous-job.service','--property='+','.join(expected)],text=True,timeout=10)
    actual={}
    for line in raw.splitlines():
        key,value=line.split('=',1)
        if key in actual:raise Hold('duplicate resource property')
        actual[key]=value
    validate_envelope(actual,expected)
    root=pathlib.Path(study['control_state_root'])
    if shutil.disk_usage(root).free<5*1024**3:raise Hold('five-GiB free disk floor')
    own_total=0;all_total=0
    for directory in root.parent.parent.glob('native-*-*'):
        if directory.is_symlink():raise Hold('symlinked native output root')
        for path in directory.rglob('*'):
            if path.is_symlink():raise Hold('symlinked native output')
            if path.is_file():
                size=path.stat().st_size
                if size>16*1024**2:raise Hold('native output file exceeds16MiB')
                all_total+=size
                if directory==root.parent:own_total+=size
    if own_total>64*1024**2 or all_total>256*1024**2:raise Hold('native output byte allowance exhausted')

def predict(account,case,answer,arm,ordering,weights=None):
    model_hash=digest(weights) if weights is not None else digest({'scripted_source':N.VERSION,'policy':'frozen lexical/clause/span baseline'})
    key=digest({'study':account.study['study_id'],'case':case['id'],'arm':arm,'ordering':ordering,'model':model_hash})
    cached=account.reserve(key)
    if cached is not None:return cached
    account.guard()
    if account.stopped():raise Hold('STOP preempts reserved packet; no replay')
    row={'case_id':case['id'],'arm':arm,'ordering':ordering,'model_hash':model_hash,'metric':S.METRIC}
    try:
        public=A.candidate_input(case);rng=random.Random(ordering);rng.shuffle(public['functions'])
        for f in public['functions']:
            pairs=list(f['parameters']['properties'].items());rng.shuffle(pairs);f['parameters']['properties']=dict(pairs)
        result=N.lexical_baseline(public) if weights is None else N.solve(public,weights)
        row.update(S.score(result['packet'],case,answer));row['packet']=result['packet'];row['error']=None
    except Exception as exc:row.update(correct=False,schema_valid=False,error=type(exc).__name__+': '+str(exc)[:300])
    account.finish(key,row);return row

def run(bundle,account,launch_token=None):
    cases=read(pathlib.Path(bundle)/'SIX_INPUTS.json');answers={r['id']:r for r in read(pathlib.Path(bundle)/'SIX_ANSWERS.json')};ids=account.study['case_ids'];orderings=account.study['orderings']
    if len(cases)!=6 or {r['id'] for r in cases}!=set(ids) or set(answers)!=set(ids):raise Hold('six-ID denominator changed')
    account.claim_job(launch_token);rows=[]
    for ordering in orderings:
        for case in cases:rows.append(predict(account,case,answers[case['id']],'scripted_baseline',ordering))
    if baseline_futility(rows,orderings):return account.terminal('HOLD_BASELINE_MARGIN_IMPOSSIBLE',{'metric':S.METRIC,'rows':rows,'gate_passed':False,'reason':'baseline>=5/6 in a required ordering; max authored6 cannot reach+2','qualified':False})
    if not all(r['schema_valid'] for r in rows):return account.terminal('HOLD_INVALID_SCRIPTED_CONTROL',{'metric':S.METRIC,'rows':rows,'gate_passed':False,'qualified':False})
    state=account.load()
    if 'genome' not in state:return account.terminal('HOLD_NO_FROZEN_FOUNDER',{'rows':rows,'gate_passed':False,'qualified':False})
    account.charge_random();authored=state['genome'];rng=random.Random(account.study['random_seed']);matched={k:[rng.uniform(-4,4) for _ in v] for k,v in authored.items()};zero={k:[0]*len(v) for k,v in authored.items()};permuted=copy.deepcopy(authored)
    for v in permuted.values():rng.shuffle(v)
    routeoff=copy.deepcopy(authored);routeoff['route']=[0]*16;bindoff=copy.deepcopy(authored);bindoff['binding']=[0]*24
    arms={'authored':authored,'matched_random':matched,'zero':zero,'permuted':permuted,'route_off':routeoff,'binding_off':bindoff}
    for arm,weights in arms.items():
        for ordering in orderings:
            for case in cases:rows.append(predict(account,case,answers[case['id']],arm,ordering,weights))
    result=acceptance(rows,orderings,ids);result['rows']=rows;result['unique_native_ids']=6;result['orderings_are_new_tasks']=False;result['BFCL_matcher_parity']='NOT_ESTABLISHED';result['cohort_role']='EXPOSED_PUBLIC_DEV';result['untouched_held_out']=False
    causal=[r for r in rows if r['arm'] in ('zero','permuted','binding_off')];result['causal_controls_valid']=all(r['schema_valid'] for r in causal);result['authored_beats_causal_controls']=all(sum(r['correct'] for r in rows if r['arm']=='authored' and r['ordering']==o)>sum(r['correct'] for r in rows if r['arm']==arm and r['ordering']==o) for o in orderings for arm in ('zero','permuted','binding_off'));result['gate_passed'] &= result['causal_controls_valid'] and result['authored_beats_causal_controls']
    return account.terminal('TERMINAL_RESTRICTED_LOCAL_RESULTS',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=pathlib.Path,required=True);p.add_argument('--control-root',type=pathlib.Path,required=True);p.add_argument('--approval',type=pathlib.Path);p.add_argument('--audit',type=pathlib.Path);p.add_argument('--intake',type=pathlib.Path);p.add_argument('--commit-reviewed',action='store_true');p.add_argument('--launch-token');a=p.parse_args()
    if not a.commit_reviewed:print(json.dumps({'phase':'REVIEW_HOLD_NO_PREDICTIONS','bundle':str(a.bundle)}));return
    if not a.approval or not a.audit:raise Hold('explicit reviewed approval/audit required')
    study=verify(a.bundle,a.approval,a.audit)
    for name in ('control_root','control_state_root','bundle_root','review_root'):
        target=pathlib.Path(study[name])
        if not target.is_absolute() or any(q.is_symlink() for q in (target,*target.parents)):raise Hold('unsafe approved path ancestry')
    if a.control_root.resolve()!=pathlib.Path(study['control_state_root']).resolve():raise Hold('different root cannot reset cumulative accounting')
    if not a.control_root.is_dir() or a.control_root.is_symlink():raise Hold('reviewed control root must be provisioned; no automatic root creation')
    if a.bundle.resolve()!=pathlib.Path(study['bundle_root']).resolve():raise Hold('bundle is not at the frozen provisioned path')
    if a.approval.resolve()!=(pathlib.Path(study['review_root'])/'APPROVAL.json').resolve() or a.audit.resolve()!=(pathlib.Path(study['review_root'])/'AUDIT_REPORT.json').resolve():raise Hold('review records are not at the frozen provisioned paths')
    account=Accounting(a.control_root,study)
    if a.intake:
        if a.intake.is_symlink() or a.intake.resolve()!=(a.bundle/'GENOME.json').resolve():raise Hold('intake must be the exact frozen bundle GENOME.json')
        raw=a.intake.read_bytes()
        def frozen_validate(weights):
            N.validate(weights)
            if hashlib.sha256(raw).hexdigest()!=study['genome_sha256']:raise Hold('intake differs from pre-prediction frozen genome')
        print(json.dumps(account.intake(raw,frozen_validate,True),sort_keys=True));return
    if not a.launch_token:raise Hold('pre-RPC durable one-job reservation required')
    launch=read(a.control_root/'LAUNCH_CLAIM.json')
    if launch.get('launch_token')!=a.launch_token or launch.get('study_id')!=study['study_id']:raise Hold('launch intent binding differs')
    systemd_envelope(study,a.launch_token)
    watcher=subprocess.check_output(['systemctl','show','cellforge-continuous-v2.service','--property=ActiveState,MainPID,UnitFileState'],text=True,timeout=10)
    if dict(line.split('=',1) for line in watcher.splitlines())!={'ActiveState':'inactive','MainPID':'0','UnitFileState':'disabled'}:raise Hold('reviewed v2 handoff not completed')
    halted=read(a.control_root/'HALTED_V2_STOP.json');marker=pathlib.Path(study['historical_control_root'])/'STOP';st=marker.stat(follow_symlinks=False)
    def marker_guard():
        st=marker.stat(follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode) or filehash(marker)!=halted['sha256'] or any(getattr(st,k)!=halted[v] for k,v in [('st_dev','device'),('st_ino','inode'),('st_mtime_ns','mtime_ns'),('st_size','size')]):raise Hold('new v2 handoff STOP changed; never remove it')
    marker_guard();account.guard=marker_guard
    import fcntl
    fd=os.open('/run/cellforge-continuous.lock',os.O_RDWR|os.O_NOFOLLOW)
    with os.fdopen(fd,'r+') as global_lock:
        st=os.fstat(fd);expected=study['global_lock_identity']
        if not stat.S_ISREG(st.st_mode) or st.st_dev!=expected['device'] or st.st_ino!=expected['inode'] or st.st_nlink!=1:raise Hold('global lock inode/path changed')
        until=min(time.monotonic()+10,time.monotonic()+max(0,account.load()['deadline_epoch']-time.time()))
        while True:
            try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if account.stopped() or time.monotonic()>=until:raise Hold('handoff lock not transferred; reserved job remains spent')
                time.sleep(.1)
        deadline=account.load()['deadline_epoch']
        def halt(*ignored):raise Hold('signal or absolute whole-job deadline; no replay')
        signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);signal.signal(signal.SIGALRM,halt);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
        try:print(json.dumps(run(a.bundle,account,a.launch_token),sort_keys=True))
        finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()