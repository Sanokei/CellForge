"""Authorized operational lifecycle fixtures only. Never production dispatch."""
import base64,copy,datetime,fcntl,hashlib,json,os,pathlib,shutil,signal,subprocess,sys,tempfile,time
payload=json.loads(base64.b64decode(PAYLOAD))
parent=pathlib.Path('/srv/cellforge-inbox');base=pathlib.Path(tempfile.mkdtemp(prefix='stop-race-review-',dir=parent))
for n,source in payload['files'].items():(base/n).write_text(source)
sys.path.insert(0,str(base));import supervisor as V
from common import atomic,read,digest,filehash,fresh_registry
processes=[];cases=[];logs=[];original=V.command
report={'scope':'isolated operational lifecycle fixtures, zero research admissions/ARC plays/model calls','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixture_root':str(base),'source_hashes':{n:filehash(base/n) for n in payload['files']},'fixture_runtime_seconds':40,'production_runtime_seconds':600,'cases':{}}
production=pathlib.Path('/srv/cellforge-inbox/continuous-control-v2');old=pathlib.Path('/srv/cellforge-inbox/continuous-spatial-20261011')
def checkpoint():
    b=production/'studies'/payload['plan_id'];plan=read(b/'PLAN.json');manifest=read(b/'MANIFEST.json');legacy=read(b/'prior-evidence/legacy/PLAN.json');prior=read(b/'CHECKPOINT.json')
    paths=[production/n for n in ['REGISTRY.json','STOP','MIGRATION.json','ROLLBACK_TERMINAL.json','FINAL_CHECKPOINT.json']]+[old/'STOP',old/'PLAN.json']
    paths += [old/n for n in legacy['code_sha256']]
    paths += [pathlib.Path(n) for n in plan['protected_stops']]
    paths += [b/n for n in manifest['files']]
    for c in prior['receipts']:paths += [old/'runs'/c['fingerprint']/n for n in ['TERMINAL.json','RESULTS.json']]
    data={'hashes':{str(p):filehash(p) for p in paths},'dispatch_present':(production/'DISPATCH_ENABLED').exists(),'registry_status':read(production/'REGISTRY.json')['status'],'production_units':original(['systemctl','show','cellforge-continuous-v2.service','cellforge-continuous-job.service','--property=Id,MainPID,LoadState,ActiveState']).stdout,'enabled':{n:subprocess.run(['systemctl','is-enabled',n],capture_output=True,text=True).stdout.strip() for n in ['cellforge-continuous.service','cellforge-continuous-v2.service']}}
    assert not data['dispatch_present'] and data['registry_status']=='paused_stop'
    assert 'MainPID=0' in data['production_units'] and all(x=='disabled' for x in data['enabled'].values())
    return data
report['production_before']=checkpoint()
def until(fn,seconds=25):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        result=fn()
        if result:return result
        time.sleep(.1)
    raise RuntimeError('bounded isolated fixture wait expired')
def backend(unit):
    V.UNIT=unit;V.GROUP='/system.slice/'+unit
    return V.Systemd(runner=lambda argv:original([a.replace('RuntimeMaxSec=600','RuntimeMaxSec=40') for a in argv]))
def setup(name,mode='plain',crash=None):
    root=base/name;root.mkdir();(root/'MODE').write_text(mode)
    if crash:(root/'CRASH_POINT').write_text(crash)
    unit='cellforge-stop-review-'+base.name.rsplit('-',1)[-1]+'-'+name+'.service'
    plan=copy.deepcopy(payload['plan']);plan['lab_runs_root']=str(root/'lab'/'runs');ph=digest(plan);fp=min(c['fingerprint'] for c in plan['conditions'])
    bundle=root/'studies'/ph;bundle.mkdir(parents=True);atomic(bundle/'PLAN.json',plan);(bundle/'worker.py').write_text(payload['worker'])
    state=fresh_registry();state['scope']='operational fixture only, not an admitted study';state['studies'][ph]={'cap':9,'attempted':0,'condition_ids':[c['fingerprint'] for c in plan['conditions']]};state['conditions']={c['fingerprint']:{'study':ph,'status':'pending' if c['fingerprint']==fp else 'unused_fixture'} for c in plan['conditions']};atomic(root/'REGISTRY.json',state)
    (root/'DISPATCH_ENABLED').write_text('isolated fixture dispatch only');run=pathlib.Path(plan['lab_runs_root'])/('weights-transition-'+ph)/'jobs'/fp
    case={'root':root,'unit':unit,'ph':ph,'fp':fp,'run':run};cases.append(case);return case
def launch(c,stop=False):
    log=open(c['root']/('stop.stderr' if stop else 'main.stderr'),'ab');logs.append(log)
    p=subprocess.Popen([sys.executable,str(base/'stop_fixture_driver.py'),str(c['root']),c['unit'],str(base)]+(['--stop'] if stop else []),stdout=subprocess.DEVNULL,stderr=log,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'));processes.append(p);return p
def finish(c,main,attempted,sequence,pids_expected=False):
    if main.poll() is None:main.terminate()
    assert main.wait(timeout=12)==0
    b=backend(c['unit']);state=read(c['root']/'REGISTRY.json');status=b.show();assert state['active'] is None and status['LoadState']=='not-found' and b.empty(status)
    assert state['studies'][c['ph']]['attempted']==attempted and state['sequence']==sequence and state['consecutive_failures']==0
    pids=read(c['run']/'PROBE_PIDS.json') if (c['run']/'PROBE_PIDS.json').exists() else []
    if pids_expected:assert len(pids)==3
    for pid in pids:
        proc=pathlib.Path('/proc')/str(pid)/'stat';assert not proc.exists() or proc.read_text().split()[2]=='Z',str(pid)
    result={'watcher_exit':main.returncode,'attempted':attempted,'sequence':sequence,'registry':state,'unit':status,'stop_sha256':filehash(c['root']/'STOP'),'descendant_pids':pids,'descendants_dead':True,'unit_absent':True,'job_tree_empty':True,'stderr':(c['root']/'main.stderr').read_text()}
    if sequence:
        terminal=read(c['run']/'TERMINAL.json');collection=read(c['run']/'COLLECTION.json');assert terminal['outcome']=='cancelled' and collection['unit_absent'] and collection['job_tree_empty'];assert collection['terminal_receipt_id']==terminal['receipt_id']
        result.update(terminal=terminal,collection=collection,terminal_sha256=filehash(c['run']/'TERMINAL.json'),collection_sha256=filehash(c['run']/'COLLECTION.json'))
    return result
try:
    for kind in ['explicit','sigterm','disabled']:
        name='gate-'+kind;c=setup(name,'gate');main=launch(c);until(lambda:(c['root']/'GATE_ENTERED').exists());stopper=None
        if kind=='explicit':
            stopper=launch(c,True);until(lambda:(c['root']/'STOP').exists());stop_hash=filehash(c['root']/'STOP');assert stopper.poll() is None
        elif kind=='sigterm':main.terminate()
        else:(c['root']/'DISPATCH_ENABLED').unlink()
        (c['root']/'RELEASE_GATE').write_text('release isolated gate')
        if stopper:assert stopper.wait(timeout=12)==0
        if kind=='disabled':until(lambda:read(c['root']/'REGISTRY.json')['status']=='paused_dispatch_disabled')
        result=finish(c,main,0,0);result['real_registry_lock_serialized_stop']=kind=='explicit'
        if stopper:assert result['stop_sha256']==stop_hash
        report['cases'][name]=result;print('CASE PASS '+name,flush=True)
    for kind in ['explicit','sigterm']:
        name='rpc-'+kind;c=setup(name,'rpc');main=launch(c);until(lambda:(c['root']/'RPC_ENTERED').exists());stopper=None
        state=read(c['root']/'REGISTRY.json');assert state['active']==c['fp'] and state['conditions'][c['fp']]['token']
        if kind=='explicit':
            stopper=launch(c,True);until(lambda:(c['root']/'STOP').exists());stop_hash=filehash(c['root']/'STOP');assert stopper.poll() is None
        else:main.terminate()
        (c['root']/'RELEASE_RPC').write_text('release actual isolated systemd RPC')
        if stopper:assert stopper.wait(timeout=15)==0
        until(lambda:read(c['root']/'REGISTRY.json')['active'] is None)
        result=finish(c,main,1,1,True)
        if stopper:assert result['stop_sha256']==stop_hash
        report['cases'][name]=result;print('CASE PASS '+name,flush=True)
    for point,name in [('after_gates_before_admission','crash-gates'),('after_launch_intent','crash-intent'),('before_launch_rpc','crash-pre-rpc'),('after_launch','crash-launched'),('after_receipt_before_collection','crash-receipt'),('after_collection_before_registry','crash-collected')]:
        c=setup(name,crash=point);main=launch(c)
        if point in ('after_receipt_before_collection','after_collection_before_registry'):
            until(lambda:(c['run']/'PROBE_PIDS.json').exists());V.request_stop(c['root'])
        assert main.wait(timeout=25)==-signal.SIGKILL
        crash_state=read(c['root']/'REGISTRY.json');before=backend(c['unit']).show();stop_hash=filehash(c['root']/'STOP')
        prior_receipt=filehash(c['run']/'TERMINAL.json') if (c['run']/'TERMINAL.json').exists() else None
        (c['root']/'CRASH_POINT').unlink();recovered=launch(c)
        until(lambda:read(c['root']/'REGISTRY.json')['active'] is None and read(c['root']/'REGISTRY.json')['status']=='paused_stop')
        result=finish(c,recovered,0 if name=='crash-gates' else 1,0 if name=='crash-gates' else 1,name in ('crash-launched','crash-receipt','crash-collected'))
        assert result['stop_sha256']==stop_hash
        if prior_receipt:assert result['terminal_sha256']==prior_receipt
        result.update(crash_exit=main.returncode,crash_point=point,registry_at_crash=crash_state,unit_at_crash=before,receipt_bytes_preserved=bool(prior_receipt),no_repeat_launch=True)
        report['cases'][name]=result;print('CASE PASS '+name,flush=True)
    report['passed']=True
except Exception as exc:report.update(passed=False,error=type(exc).__name__+': '+str(exc));print('FIXTURE FAILURE '+report['error'],flush=True)
finally:
    for p in processes:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=12)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
    cleanup=[]
    for c in cases:
        try:
            b=backend(c['unit']);info=b.show()
            if info['LoadState']!='not-found':
                state=read(c['root']/'REGISTRY.json');entry=state['conditions'][c['fp']];b.owned(info,entry['token'],entry.get('invocation'));b.stop(entry['token'],entry.get('invocation'));b.collect(entry['token'],entry.get('invocation') or info['InvocationID'])
            info=b.show();assert info['LoadState']=='not-found' and b.empty(info)
            cleanup.append({'unit':c['unit'],'absent':True,'empty':True})
        except Exception as exc:cleanup.append({'unit':c['unit'],'error':str(exc)});report['passed']=False
    for log in logs:log.close()
    report['cleanup']=cleanup
    try:
        report['production_after']=checkpoint();assert report['production_after']==report['production_before'];report['production_unchanged']=True
    except Exception as exc:report.update(passed=False,production_check_error=str(exc))
    report['fixture_logs']={str(p.relative_to(base)):p.read_text() for p in base.rglob('*.stderr')}
    if all(x.get('absent') and x.get('empty') for x in cleanup):
        resolved=base.resolve();assert resolved.parent==parent.resolve() and resolved.name.startswith('stop-race-review-');shutil.rmtree(resolved);report['fixture_directory_removed']=True
    else:report['fixture_directory_removed']=False
    print('REPORT_JSON:'+json.dumps(report,sort_keys=True),flush=True)
if not report.get('passed'):raise SystemExit(2)