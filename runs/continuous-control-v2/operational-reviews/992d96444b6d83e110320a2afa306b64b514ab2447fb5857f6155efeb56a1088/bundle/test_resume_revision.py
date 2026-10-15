"""Operational-only disposable fixtures: observer races and audited same-study resume."""
import contextlib,copy,os,pathlib,shutil,tempfile,unittest
from unittest.mock import patch
import resume_operation as R, observe_dispatch as O, supervisor as V, test_review as F
from common import atomic,read,digest,filehash,Blocked
BASE=pathlib.Path(__file__).resolve().parent
class ObserverRevision(unittest.TestCase):
    setUp=F.ControllerCrashPoints.setUp;tearDown=F.ControllerCrashPoints.tearDown;enable=F.ControllerCrashPoints.enable
    def test_marker_changes_during_resource_read(self):
        for change,phase in [('add_stop','owned_running_stop_requested'),('remove_dispatch','owned_running_stop_requested'),('remove_stop','snapshot_changed')]:
            f=ObserverRevision();f.setUp()
            try:
                f.enable();f.controller.tick();before=(f.root/'REGISTRY.json').read_bytes()
                if change=='remove_stop':(f.root/'STOP').write_bytes(b'fixture stop')
                def resources():
                    if change=='add_stop':(f.root/'STOP').write_bytes(b'NEW STOP')
                    elif change=='remove_dispatch':(f.root/'DISPATCH_ENABLED').unlink()
                    else:(f.root/'STOP').unlink()
                    return O.EXPECTED.copy()
                with self.subTest(change=change):self.assertEqual(O.observe(f.root,f.backend,resources)['phase'],phase)
                self.assertEqual((f.root/'REGISTRY.json').read_bytes(),before);self.assertEqual(f.backend.stops,0)
            finally:f.tearDown()
    def test_invalid_durations_never_observe(self):
        for value in (float('nan'),float('inf'),float('-inf'),True,False,10**500,-1,601,None,'2'):
            with self.subTest(value=str(value)[:20]),patch.object(O,'observe') as observer,self.assertRaises(ValueError):O.watch(self.root,value,observer=observer)
            observer.assert_not_called()
    def test_finite_expiry_is_read_only(self):
        before=(self.root/'REGISTRY.json').read_bytes();now=[0]
        def sleep(x):now[0]+=x
        self.assertEqual(O.watch(self.root,0,emit=lambda _:None,observer=lambda r:O.observe(r,self.backend))['phase'],'held_idle')
        value=O.watch(self.root,.125,emit=lambda _:None,clock=lambda:now[0],sleeper=sleep,observer=lambda r:O.observe(r,self.backend))
        self.assertEqual(value['phase'],'observation_pending');self.assertEqual(now[0],.125);self.assertEqual((self.root/'REGISTRY.json').read_bytes(),before)
class ResumeFixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)/'control';self.root.mkdir();self.bundle=self.root.parent/'review';self.bundle.mkdir()
        p=copy.deepcopy(F.PLAN);p['lab_runs_root']=str(self.root/'lab'/'runs');historical=self.root/'historical_STOP';historical.write_bytes(b'closed campaign');p['protected_stops']={str(historical):filehash(historical)}
        self.old=self.root/'old_control_STOP';self.old.write_bytes(b'old hold');self.ph=digest(p);(self.root/'studies'/self.ph).mkdir(parents=True);atomic(self.root/'studies'/self.ph/'PLAN.json',p)
        evidence=BASE/'migration-evidence'/'evidence';s=read(evidence/'control'/'REGISTRY.json');study=s['studies'].pop(R.PH);s['studies'][self.ph]=study
        for e in s['conditions'].values():
            if e.get('study')==R.PH:e['study']=self.ph
        atomic(self.root/'REGISTRY.json',s);self.base=copy.deepcopy(s);run=self.root/'lab'/'runs'/('weights-transition-'+self.ph)/'jobs'/R.FP;run.mkdir(parents=True)
        for n in ('CONFIG.json','RESULTS.json','TERMINAL.json','COLLECTION.json'):shutil.copyfile(evidence/'job'/n,run/n)
        shutil.copyfile(evidence/'control'/'STOP',self.root/'STOP');(self.root/'controller').mkdir()
        for n in R.RUNTIME:shutil.copyfile(BASE/n,self.bundle/n);shutil.copyfile(BASE/n,self.root/'controller'/n)
        atomic(self.bundle/'MANIFEST.json',{'files':{n:filehash(self.bundle/n) for n in R.RUNTIME}});self.audit=self.bundle/'AUDIT_REPORT';self.audit.write_bytes(b'OFFLINE FIXTURE ONLY - NO PRODUCTION APPROVAL')
        self.backend=F.Fake(self.root/'cgroups');self.op=R.ResumeOperation(self.root,backend=self.backend,service_check=lambda:None,locks=lambda _:contextlib.nullcontext(),gate=lambda b,p:None,expected_root=self.root,expected_plan=self.ph,expected_registry=filehash(self.root/'REGISTRY.json'),extra_protected={str(self.old):filehash(self.old)})
        self.request=self.op.prepare_request(self.bundle);self.approval={'decision':'permit_existing_study_archive_rearm','request_sha256':digest(self.request),'operational_manifest_sha256':filehash(self.bundle/'MANIFEST.json'),'resume_review_passed':True,'qualified':False,'remote_model_calls':0,'audit_report_sha256':filehash(self.audit),'fixture_only':True}
    def tearDown(self):self.tmp.cleanup()
    def stage(self):return self.op.stage(self.bundle,self.request,self.approval,self.audit)
    def activate(self):return self.op.activate(self.bundle,self.request,self.approval,self.audit)
    def accounting(self):
        s=read(self.root/'REGISTRY.json');expected=copy.deepcopy(self.base);expected['status']=s['status'];self.assertEqual(s,expected)
        for n,h in {**self.request['protected_stops'],**self.request['cancelled_artifacts']}.items():self.assertEqual(filehash(n),h)
        self.assertEqual(self.backend.starts,0);self.assertEqual(self.backend.stops,0)
    def case(self,fn):
        f=ResumeFixture();f.setUp()
        try:fn(f)
        finally:f.tearDown()
    def test_prepare_read_only_exact_eight(self):
        self.assertEqual(len(self.request['pending_conditions']),8);self.assertNotIn(R.FP,self.request['pending_conditions']);self.assertTrue((self.root/'STOP').exists());self.assertFalse((self.root/'resumptions').exists());self.accounting()
    def test_stage_and_activation_are_separate_idempotent_no_launch(self):
        result=self.stage();self.assertEqual(result['phase'],'REARMED_STOPPED_DISPATCH_DISABLED');self.assertFalse((self.root/'DISPATCH_ENABLED').exists());self.assertEqual(R.marker(self.op.directory(self.request)/'ACKNOWLEDGED_STOP'),self.request['stop_identity']);self.assertEqual(self.stage(),result);self.accounting()
        self.assertEqual(self.activate()['phase'],'DISPATCH_AUTHORIZED_WATCHER_STILL_STOPPED');self.activate();self.accounting()
    def test_activation_without_committed_rearm_blocks(self):
        with self.assertRaises(Blocked):self.activate()
        self.assertFalse((self.root/'DISPATCH_ENABLED').exists())
    def test_preconditions_block_before_STOP_move(self):
        for kind in ('loaded_unit','unknown_unit','watcher','runtime','approval','protected','receipt','dispatch'):
            def run(f):
                if kind=='loaded_unit':f.backend.value['LoadState']='loaded'
                elif kind=='unknown_unit':f.backend.show=lambda:(_ for _ in ()).throw(Blocked('bus unavailable'))
                elif kind=='watcher':f.op.service_check=lambda:(_ for _ in ()).throw(Blocked('active watcher'))
                elif kind=='runtime':(f.root/'controller'/'observe_dispatch.py').write_bytes(b'changed')
                elif kind=='approval':f.approval['resume_review_passed']=False
                elif kind=='protected':f.old.write_bytes(b'changed')
                elif kind=='receipt':
                    path=next(pathlib.Path(n) for n in f.request['cancelled_artifacts'] if n.endswith('TERMINAL.json'));v=read(path);v['outcome']='completed';atomic(path,v)
                else:atomic(f.root/'DISPATCH_ENABLED',{'foreign':True})
                with self.assertRaises(Blocked):f.stage()
                self.assertTrue((f.root/'STOP').exists());self.assertFalse((f.root/'resumptions').exists())
            with self.subTest(kind=kind):self.case(run)
    def test_accounting_and_scope_drift_block(self):
        for kind in ('cap','attempted','sequence','cancelled','pending','root','science_plan','additional_attempts','runtime_sha256'):
            def run(f):
                if kind in ('root','science_plan','additional_attempts','runtime_sha256'):
                    f.request[kind]={'root':str(f.root/'new'),'science_plan':'new-study','additional_attempts':9,'runtime_sha256':{}}[kind];f.approval['request_sha256']=digest(f.request)
                else:
                    s=copy.deepcopy(f.base)
                    if kind in ('cap','attempted'):s['studies'][f.ph][kind]+=1
                    elif kind=='sequence':s['sequence']+=1
                    elif kind=='cancelled':s['conditions'][R.FP]['status']='pending'
                    else:s['conditions'][f.request['pending_conditions'][0]]['status']='completed'
                    atomic(f.root/'REGISTRY.json',s)
                with self.assertRaises(Blocked):f.stage()
                self.assertTrue((f.root/'STOP').exists());self.assertFalse((f.root/'resumptions').exists())
            with self.subTest(kind=kind):self.case(run)
    def test_STOP_replacement_and_archive_collision_preserved(self):
        for kind in ('new_stop','collision','after_archive'):
            def run(f):
                def checkpoint(point):
                    if kind=='new_stop' and point=='before_stop_archive':(f.root/'STOP').write_bytes(b'NEW STOP')
                    elif kind=='collision' and point=='before_stop_archive':(f.op.directory(f.request)/'ACKNOWLEDGED_STOP').write_bytes(b'collision')
                    elif kind=='after_archive' and point=='after_stop_archive':(f.root/'STOP').write_bytes(b'NEW STOP')
                f.op.checkpoint=checkpoint
                with self.assertRaises(Blocked):f.stage()
                f.op.checkpoint=lambda _:None
                with self.assertRaises(Blocked):f.stage()
                self.assertFalse((f.root/'DISPATCH_ENABLED').exists())
                if kind=='collision':self.assertEqual((f.op.directory(f.request)/'ACKNOWLEDGED_STOP').read_bytes(),b'collision')
                else:self.assertEqual((f.root/'STOP').read_bytes(),b'NEW STOP')
                f.accounting()
            with self.subTest(kind=kind):self.case(run)
    def test_STOP_replaced_during_rename_is_restored_and_blocks(self):
        original=R.rename_noreplace
        def raced(source,archive):
            pathlib.Path(source).write_bytes(b'NEW STOP DURING RENAME')
            original(source,archive)
        with patch.object(R,'rename_noreplace',side_effect=raced),self.assertRaises(Blocked):self.stage()
        self.assertEqual((self.root/'STOP').read_bytes(),b'NEW STOP DURING RENAME')
        self.assertFalse((self.root/'DISPATCH_ENABLED').exists());self.accounting()
    def test_stage_crashes_recover_without_new_attempt(self):
        for point in ('after_resume_intent','before_stop_archive','after_stop_archive','after_resume_receipt','after_resume_status'):
            def run(f):
                f.op.checkpoint=lambda p:(_ for _ in ()).throw(F.Crash(p)) if p==point else None
                with self.assertRaises(F.Crash):f.stage()
                f.op.checkpoint=lambda _:None;f.stage();f.accounting();self.assertFalse((f.root/'DISPATCH_ENABLED').exists())
            with self.subTest(point=point):self.case(run)
    def test_activation_crashes_recover_without_launch(self):
        for point in ('before_dispatch_marker','after_dispatch_marker'):
            def run(f):
                f.stage();f.op.checkpoint=lambda p:(_ for _ in ()).throw(F.Crash(p)) if p==point else None
                with self.assertRaises(F.Crash):f.activate()
                f.op.checkpoint=lambda _:None;f.activate();f.accounting()
            with self.subTest(point=point):self.case(run)
    def test_new_STOP_at_activation_preempts_without_marker_removal(self):
        for point in ('before_dispatch_marker','after_dispatch_marker'):
            def run(f):
                f.stage();f.op.checkpoint=lambda p:(f.root/'STOP').write_bytes(b'NEW STOP') if p==point else None
                with self.assertRaises(Blocked):f.activate()
                self.assertEqual((f.root/'STOP').read_bytes(),b'NEW STOP');self.assertEqual((f.root/'DISPATCH_ENABLED').exists(),point=='after_dispatch_marker');f.accounting()
            with self.subTest(point=point):self.case(run)
    def test_tampered_receipt_and_foreign_marker_block(self):
        self.stage();d=self.op.directory(self.request);v=read(d/'RESUME_RECEIPT.json');v['cap']=99;atomic(d/'RESUME_RECEIPT.json',v)
        with self.assertRaises(Blocked):self.activate()
        atomic(d/'RESUME_RECEIPT.json',self.op.expected_receipt(self.request));atomic(self.root/'DISPATCH_ENABLED',{'foreign':True})
        with self.assertRaises(Blocked):self.activate()
        self.assertEqual(read(self.root/'DISPATCH_ENABLED'),{'foreign':True})
    def test_foreign_dispatch_created_at_actual_link_boundary_is_preserved(self):
        self.stage();enabled=self.root/'DISPATCH_ENABLED';original=R.os.link;foreign={}
        def raced(source,destination,*args,**kwargs):
            if pathlib.Path(destination)==enabled:
                atomic(enabled,{'foreign':'at publication syscall'});foreign.update(R.marker(enabled))
            return original(source,destination,*args,**kwargs)
        with patch.object(R.os,'link',side_effect=raced),self.assertRaises(Blocked):self.activate()
        self.assertEqual(R.marker(enabled),foreign);self.assertEqual(read(enabled),{'foreign':'at publication syscall'})
        self.assertFalse((self.op.directory(self.request)/'DISPATCH_RECEIPT.json').exists());self.accounting()
    def test_foreign_replacement_at_preempt_boundary_is_never_unlinked(self):
        self.stage();enabled=self.root/'DISPATCH_ENABLED';foreign={}
        def checkpoint(point):
            if point=='after_dispatch_marker':(self.root/'STOP').write_bytes(b'NEW STOP')
            elif point=='before_dispatch_preempt_hold':
                atomic(enabled,{'foreign':'after release, before hold'});foreign.update(R.marker(enabled))
        self.op.checkpoint=checkpoint
        with patch.object(pathlib.Path,'unlink',side_effect=AssertionError('activation must never unlink visible markers')),self.assertRaises(Blocked):self.activate()
        self.assertEqual(R.marker(enabled),foreign);self.assertEqual(read(enabled),{'foreign':'after release, before hold'})
        self.assertEqual((self.root/'STOP').read_bytes(),b'NEW STOP');self.assertFalse((self.op.directory(self.request)/'DISPATCH_RECEIPT.json').exists());self.accounting()
    def test_foreign_archive_created_at_actual_rename_boundary_is_preserved(self):
        archive=self.op.directory(self.request)/'ACKNOWLEDGED_STOP';original=R.rename_noreplace;foreign={}
        def raced(source,destination):
            pathlib.Path(destination).write_bytes(b'FOREIGN ARCHIVE AT SYSCALL');foreign.update(R.marker(destination))
            return original(source,destination)
        with patch.object(R,'rename_noreplace',side_effect=raced),self.assertRaises(Blocked):self.stage()
        self.assertEqual(R.marker(archive),foreign);self.assertEqual(R.marker(self.root/'STOP'),self.request['stop_identity'])
        self.assertFalse((self.root/'DISPATCH_ENABLED').exists());self.accounting()
    @unittest.skipUnless(os.name=='posix','real SIGKILL and POSIX durability')
    def test_real_SIGKILL_resume_boundaries_recover_with_real_locks(self):
        import signal
        for mode,point in [('stage',p) for p in ('after_resume_intent','before_stop_archive','after_stop_archive','after_resume_receipt','after_resume_status')]+[('activate',p) for p in ('before_dispatch_marker','after_dispatch_marker')]:
            f=ResumeFixture();f.setUp()
            try:
                f.op.locks=R.operator_locks
                with patch.object(V,'LOCK',str(f.root/'global.lock')):
                    if mode=='activate':f.stage()
                    child=os.fork()
                    if child==0:
                        f.op.checkpoint=lambda p:os.kill(os.getpid(),signal.SIGKILL) if p==point else None
                        try:getattr(f,mode)()
                        except BaseException:os._exit(88)
                        os._exit(89)
                    _,status=os.waitpid(child,0)
                    with self.subTest(mode=mode,point=point):
                        self.assertTrue(os.WIFSIGNALED(status));self.assertEqual(os.WTERMSIG(status),signal.SIGKILL)
                        getattr(f,mode)();f.accounting()
                        self.assertEqual(R.marker(f.op.directory(f.request)/'ACKNOWLEDGED_STOP'),f.request['stop_identity'])
                        if mode=='stage':self.assertFalse((f.root/'DISPATCH_ENABLED').exists())
            finally:f.tearDown()
    @unittest.skipUnless(os.name=='posix','production requires POSIX locking')
    def test_global_lock_excludes_second_operator(self):
        import fcntl
        lock=self.root/'global.lock'
        with patch.object(V,'LOCK',str(lock)),lock.open('a+') as owned:
            fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(Blocked):
                with R.operator_locks(self.root):pass
if __name__=='__main__':unittest.main(verbosity=2)