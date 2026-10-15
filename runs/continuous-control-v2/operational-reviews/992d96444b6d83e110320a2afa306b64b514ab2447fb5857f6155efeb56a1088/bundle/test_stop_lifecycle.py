"""Lifecycle boundary regressions; no ARC plays or external processes."""
import pathlib,unittest,os,sys
from unittest.mock import patch
import test_review as F
import supervisor as V
import observe_dispatch as O
from common import *

class LaunchStopBoundaries(unittest.TestCase):
    setUp=F.ControllerCrashPoints.setUp
    tearDown=F.ControllerCrashPoints.tearDown
    enable=F.ControllerCrashPoints.enable
    def mark(self,kind):
        if kind=='halt':self.controller.halt=True
        elif kind=='STOP':(self.root/'STOP').write_text('persistent fixture STOP')
        elif kind=='disabled':(self.root/'DISPATCH_ENABLED').unlink()
        elif kind=='cancel':
            (self.root/'cancel').mkdir(exist_ok=True);(self.root/'cancel'/self.fp).write_text('fixture cancel')
    def check_cancel(self,starts):
        state=read(self.root/'REGISTRY.json');self.assertIsNone(state['active']);self.assertEqual(state['conditions'][self.fp]['status'],'cancelled')
        self.assertEqual(state['studies'][self.ph]['attempted'],1);self.assertEqual(state['sequence'],1);self.assertEqual(self.backend.starts,starts)
        self.assertEqual(state['consecutive_failures'],0);self.assertTrue(self.backend.empty(self.backend.show()))
        run=self.controller.context(state,self.fp)[3];receipt=read(run/'TERMINAL.json');collection=read(run/'COLLECTION.json')
        self.assertEqual(collection['terminal_receipt_id'],receipt['receipt_id']);self.assertTrue(collection['unit_absent'])
        self.controller.halt=True;self.controller.tick();self.controller.verify_stopped();self.assertEqual(read(self.root/'REGISTRY.json')['sequence'],1)
    def exercise(self,fn):
        fixture=LaunchStopBoundaries();fixture.setUp()
        try:fn(fixture)
        finally:fixture.tearDown()
    def test_global_stop_after_gates_keeps_condition_unattempted(self):
        for kind in ('halt','STOP','disabled'):
            def case(f,k=kind):
                f.enable();f.controller.crash=lambda point:f.mark(k) if point=='after_gates_before_admission' else None
                f.controller.tick();s=read(f.root/'REGISTRY.json');self.assertIsNone(s['active']);self.assertEqual(s['studies'][f.ph]['attempted'],0)
                self.assertEqual(f.backend.starts,0);self.assertEqual(s['conditions'][f.fp]['status'],'pending')
            with self.subTest(kind=kind):self.exercise(case)
    def test_stop_after_intent_and_before_rpc_consumes_once_without_launch(self):
        for point in ('after_launch_intent','before_launch_rpc'):
            for kind in ('halt','STOP','disabled','cancel'):
                def case(f,q=point,k=kind):
                    f.enable();f.controller.crash=lambda p:f.mark(k) if p==q else None;f.controller.tick();f.check_cancel(0)
                with self.subTest(point=point,kind=kind):self.exercise(case)
    def test_stop_during_rpc_cancels_actual_invocation_before_return(self):
        for kind in ('halt','STOP','disabled','cancel'):
            def case(f,k=kind):
                f.enable();start=f.backend.start
                def raced(*args):actual=start(*args);f.mark(k);return actual
                with patch.object(f.backend,'start',side_effect=raced):f.controller.tick()
                f.check_cancel(1);s=read(f.root/'REGISTRY.json');run=f.controller.context(s,f.fp)[3]
                self.assertEqual(read(run/'TERMINAL.json')['evidence']['systemd_before_stop']['InvocationID'],'fixture-inv')
            with self.subTest(kind=kind):self.exercise(case)
    def test_rpc_error_after_spawn_then_stop_reconciles_owned_tree(self):
        self.enable();start=self.backend.start
        def uncertain(*args):start(*args);self.mark('STOP');raise Blocked('lost acknowledgement')
        with patch.object(self.backend,'start',side_effect=uncertain):self.controller.tick()
        self.check_cancel(1)
    def test_rpc_error_without_spawn_and_stop_is_cancelled(self):
        self.enable()
        def failed(*args):self.mark('STOP');raise Blocked('RPC error before spawn')
        with patch.object(self.backend,'start',side_effect=failed):self.controller.tick()
        self.check_cancel(0)
    def test_rpc_error_and_unknown_ownership_keeps_durable_intent(self):
        self.enable();realshow=self.backend.show
        def uncertain(*args):
            self.mark('STOP');self.backend.show=lambda:(_ for _ in ()).throw(Blocked('unknown bus'));raise Blocked('RPC timeout')
        with patch.object(self.backend,'start',side_effect=uncertain),self.assertRaises(Blocked):self.controller.tick()
        s=read(self.root/'REGISTRY.json');self.assertEqual(s['active'],self.fp);self.assertEqual(s['sequence'],0);self.assertEqual(self.backend.stops,0)
        self.backend.show=realshow;self.controller.tick();self.check_cancel(0)
    def test_stop_and_crash_recover_without_repeat_or_next_launch(self):
        for point in ('after_gates_before_admission','after_launch_intent','before_launch_rpc','after_launch','after_receipt_before_collection','after_collection_before_registry'):
            def case(f,q=point):
                f.enable()
                if 'receipt' in q or 'collection' in q:f.controller.tick();f.mark('STOP')
                def crash(p):
                    if p==q:f.mark('STOP');raise F.Crash(q)
                f.controller.crash=crash
                with self.assertRaises(F.Crash):f.controller.tick()
                recovered=V.Controller(f.root,f.backend);recovered.halt=True;recovered.tick();recovered.tick();recovered.verify_stopped();s=read(f.root/'REGISTRY.json')
                if q=='after_gates_before_admission':self.assertEqual(s['studies'][f.ph]['attempted'],0);self.assertEqual(s['sequence'],0);self.assertEqual(f.backend.starts,0)
                else:
                    self.assertEqual(s['conditions'][f.fp]['status'],'cancelled');self.assertEqual(s['studies'][f.ph]['attempted'],1);self.assertEqual(s['sequence'],1)
                    self.assertEqual(f.backend.starts,1 if q in ('after_launch','after_receipt_before_collection','after_collection_before_registry') else 0)
                self.assertEqual(s['consecutive_failures'],0);self.assertIsNone(s['active'])
            with self.subTest(point=point):self.exercise(case)
    def test_verify_stop_unknown_or_unowned_never_returns_success(self):
        with patch.object(self.backend,'show',side_effect=Blocked('unknown')),self.assertRaises(Blocked):self.controller.verify_stopped()
        self.backend.value.update(LoadState='loaded',ActiveState='active',SubState='running',MainPID='43',Description='unowned',InvocationID='other',ControlGroup=V.GROUP)
        with self.assertRaises(Blocked):self.controller.verify_stopped()
        self.assertEqual(self.backend.stops,0)
    def test_run_loop_retries_unknown_stop_until_receipt_and_collection_verified(self):
        self.enable();self.controller.tick();self.controller.halt=True;original=self.backend.show;calls=[0]
        def once():
            calls[0]+=1
            if calls[0]==1:raise Blocked('bus unavailable')
            return original()
        slept=[]
        with patch.object(self.backend,'show',side_effect=once):V.run_loop(self.controller,sleeper=lambda seconds:slept.append(seconds))
        self.assertEqual(slept,[2]);self.check_cancel(1);self.assertTrue((self.root/'STOP').exists())
    def test_explicit_stop_ticks_serially_and_preserves_existing_stop_bytes(self):
        self.enable();self.controller.tick();self.mark('STOP');before=(self.root/'STOP').read_bytes()
        with patch.object(V,'Controller',return_value=self.controller),patch.object(sys,'argv',['supervisor','--root',str(self.root),'--stop']),patch.dict(os.environ,{'SERVICE_RESULT':'success'}):V.main()
        self.check_cancel(1);self.assertEqual((self.root/'STOP').read_bytes(),before)
    def test_failed_service_guard_never_synthesizes_new_stop(self):
        with patch.object(sys,'argv',['supervisor','--root',str(self.root),'--stop']),patch.dict(os.environ,{'SERVICE_RESULT':'exit-code'}):V.main()
        self.assertFalse((self.root/'STOP').exists());self.assertEqual(self.backend.stops,0)

class ObservationSafety(unittest.TestCase):
    setUp=F.ControllerCrashPoints.setUp
    tearDown=F.ControllerCrashPoints.tearDown
    enable=F.ControllerCrashPoints.enable
    def test_observation_deadline_never_changes_markers_or_registry(self):
        self.enable();before=(self.root/'REGISTRY.json').read_bytes();clock=[0]
        def sleep(seconds):clock[0]+=seconds
        result=O.watch(self.root,18,emit=lambda _:None,sleeper=sleep,clock=lambda:clock[0],observer=lambda root:O.observe(root,self.backend))
        self.assertEqual(result['phase'],'observation_pending');self.assertEqual((self.root/'REGISTRY.json').read_bytes(),before)
        self.assertTrue((self.root/'DISPATCH_ENABLED').exists());self.assertFalse((self.root/'STOP').exists());self.assertEqual(self.backend.starts,0);self.assertEqual(self.backend.stops,0)
    def test_actual_invocation_and_resources_required_for_running_claim(self):
        self.enable();self.controller.tick();before=(self.root/'REGISTRY.json').read_bytes()
        result=O.observe(self.root,self.backend,lambda:O.EXPECTED.copy());self.assertEqual(result['phase'],'running_verified');self.assertEqual(result['invocation'],'fixture-inv');self.assertEqual((self.root/'REGISTRY.json').read_bytes(),before)
        wrong=O.EXPECTED.copy();wrong['RuntimeMaxUSec']='15s';self.assertEqual(O.observe(self.root,self.backend,lambda:wrong)['phase'],'blocked_unknown')
    def test_resource_observation_rechecks_identity_and_existing_stop(self):
        self.enable();self.controller.tick();original=self.backend.show;calls=[0]
        def replacement():
            calls[0]+=1;value=original()
            if calls[0]>1:value['InvocationID']='replaced during resources'
            return value
        with patch.object(self.backend,'show',side_effect=replacement):self.assertEqual(O.observe(self.root,self.backend,lambda:O.EXPECTED)['phase'],'blocked_unknown')
        (self.root/'STOP').write_text('fixture pending cancellation')
        self.assertEqual(O.observe(self.root,self.backend,lambda:O.EXPECTED)['phase'],'owned_running_stop_requested')
        self.assertEqual(self.backend.stops,0)
    def test_missing_invocation_is_only_reconciling(self):
        self.enable();self.controller.tick();s=read(self.root/'REGISTRY.json');s['conditions'][self.fp].pop('invocation');atomic(self.root/'REGISTRY.json',s)
        self.assertEqual(O.observe(self.root,self.backend)['phase'],'intent_reconciling')
    def test_unknown_bus_and_replacement_invocation_never_claim_running_or_mutate(self):
        self.enable();self.controller.tick();before=(self.root/'REGISTRY.json').read_bytes()
        with patch.object(self.backend,'show',side_effect=Blocked('unknown')):self.assertEqual(O.observe(self.root,self.backend)['phase'],'blocked_unknown')
        self.backend.value['InvocationID']='replacement';self.assertEqual(O.observe(self.root,self.backend)['phase'],'blocked_unknown')
        self.assertEqual((self.root/'REGISTRY.json').read_bytes(),before);self.assertEqual(self.backend.stops,0)
    def test_changed_registry_never_claims_coherent_snapshot(self):
        self.enable();self.controller.tick();original=self.backend.show
        def changed():
            value=original();s=read(self.root/'REGISTRY.json');s['status']='concurrent update';atomic(self.root/'REGISTRY.json',s);return value
        with patch.object(self.backend,'show',side_effect=changed):self.assertEqual(O.observe(self.root,self.backend,lambda:O.EXPECTED)['phase'],'snapshot_changed')
if __name__=='__main__':unittest.main(verbosity=2)