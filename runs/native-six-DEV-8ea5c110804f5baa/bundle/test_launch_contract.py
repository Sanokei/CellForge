"""Synthetic-only launch reservation, handoff ownership and envelope tests."""
import copy,json,pathlib,tempfile,unittest
from unittest import mock
from study_accounting import Accounting,Hold,digest,atomic
import study_worker as W
import native_launch_plan as P
BASE=pathlib.Path(__file__).parent
class LaunchReservation(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);self.now=[10];self.study={'study_id':'synthetic'};self.a=Accounting(self.root,self.study,lambda:self.now[0]);self.a.intake(b'{}',lambda w:None,True);self.token='a'*64
    def tearDown(self):self.tmp.cleanup()
    def test_job_burned_before_RPC_and_adopted_once(self):
        claim=self.a.reserve_launch(self.token);self.assertTrue(self.a.load()['job_used']);self.assertEqual(self.a.load()['packet_used'],0);self.assertEqual(claim['launch_token'],self.token);self.a.claim_job(self.token)
        with self.assertRaises(Hold):self.a.claim_job(self.token)
        with self.assertRaises(Hold):self.a.reserve_launch(self.token)
    def test_frozen_state_genome_cannot_be_substituted(self):
        state=self.a.load();study={**self.study,'genome_canonical_sha256':digest(state['genome'])};a=Accounting(self.root,study,lambda:10);self.assertEqual(a.load()['genome'],{})
        state['genome']={'changed':1};atomic(a.path,state)
        with self.assertRaises(Hold):a.load()
    def test_failed_or_uncertain_RPC_cannot_be_relaunched(self):
        self.a.reserve_launch(self.token)
        with self.assertRaises(Hold):Accounting(self.root,self.study,lambda:10).reserve_launch('b'*64)
    def test_wrong_token_and_expired_adoption_hold(self):
        self.a.reserve_launch(self.token)
        with self.assertRaises(Hold):self.a.claim_job('b'*64)
        self.now[0]=610
        with self.assertRaises(Hold):self.a.claim_job(self.token)
        self.assertTrue(self.a.load()['job_used'])
    def test_cohort_root_STOP_preempts_state_directory(self):
        state=self.root/'state';state.mkdir();marker=self.root/'STOP';marker.write_bytes(b'cohort STOP');a=Accounting(state,{'study_id':'synthetic','control_root':str(self.root)},lambda:10)
        with self.assertRaises(Hold):a.intake(b'{}',lambda w:None,True)
        with self.assertRaises(Hold):a.claim_job()
        self.assertEqual(marker.read_bytes(),b'cohort STOP');self.assertFalse((state/'INTAKE_CLAIM.json').exists());self.assertFalse((state/'JOB_CLAIM.json').exists())
    def test_missing_state_cannot_reset_reserved_launch(self):
        self.a.reserve_launch(self.token);self.a.path.unlink()
        with self.assertRaises(Hold):Accounting(self.root,self.study).reserve_launch('b'*64)
    def test_marker_race_after_intent_preempts_prediction_and_stays_charged(self):
        self.a.claim_job();calls=[]
        def guard():
            calls.append(True)
            if len(calls)==2:raise Hold('synthetic marker changed')
        self.a.guard=guard
        with mock.patch.object(W.A,'candidate_input') as prediction_input:
            with self.assertRaises(Hold):W.predict(self.a,{'id':'synthetic'}, {},'scripted_baseline',0)
            prediction_input.assert_not_called()
        self.assertEqual(self.a.load()['packet_used'],1);self.assertEqual(next(iter(self.a.load()['attempts'].values()))['status'],'intent')
class ExactEnvelope(unittest.TestCase):
    def setUp(self):self.study={'study_id':'synthetic','control_state_root':'/fixed/state','bundle_root':'/fixed/bundle','review_root':'/fixed/review'};self.expected=W.expected_envelope(self.study,'a'*64,123)
    def test_exact_props_and_path_order(self):
        W.validate_envelope(self.expected,self.expected);actual=copy.deepcopy(self.expected);actual['ReadWritePaths']=' '.join(reversed(actual['ReadWritePaths'].split()));W.validate_envelope(actual,self.expected)
    def test_weaker_resource_or_tree_protection_holds(self):
        for key,value in [('KillMode','process'),('SendSIGKILL','no'),('RuntimeMaxUSec','10min'),('TimeoutStopUSec','20s'),('Restart','always'),('PrivateNetwork','no'),('ProtectSystem','no'),('CapabilityBoundingSet','cap_sys_admin'),('MemoryMax','2147483648'),('MainPID','124')]:
            with self.subTest(key=key):
                actual={**self.expected,key:value}
                with self.assertRaises(Hold):W.validate_envelope(actual,self.expected)
    def test_missing_or_wider_writable_path_holds(self):
        actual={**self.expected,'ReadWritePaths':'/srv'}
        with self.assertRaises(Hold):W.validate_envelope(actual,self.expected)
        actual=copy.deepcopy(self.expected);actual.pop('KillMode')
        with self.assertRaises(Hold):W.validate_envelope(actual,self.expected)
class HandoffOwnership(unittest.TestCase):
    def setUp(self):self.pinned=json.loads((BASE/'HANDOFF_CHECKPOINT.json').read_text(encoding='utf-8'))
    def test_exact_owner_kernel_lock_and_cap(self):P.check_before_stop(self.pinned,self.pinned)
    def test_changed_owner_or_invocation_is_never_stopped(self):
        for key in ('MainPID','InvocationID'):
            actual=copy.deepcopy(self.pinned);actual['v2'][key]='changed'
            with self.assertRaises(Hold):P.check_before_stop(actual,self.pinned)
    def test_second_lock_holder_or_changed_inode_holds(self):
        actual=copy.deepcopy(self.pinned);actual['global_lock']['kernel_records']*=2
        with self.assertRaises(Hold):P.check_before_stop(actual,self.pinned)
        actual=copy.deepcopy(self.pinned);actual['global_lock']['inode']+=1
        with self.assertRaises(Hold):P.check_before_stop(actual,self.pinned)
    def test_preexisting_STOP_and_old_cap_reset_hold(self):
        actual=copy.deepcopy(self.pinned);actual['old_root_STOP_present']=True
        with self.assertRaises(Hold):P.check_before_stop(actual,self.pinned)
        actual=copy.deepcopy(self.pinned);actual['old_registry']['cap']=10
        with self.assertRaises(Hold):P.check_before_stop(actual,self.pinned)
    def test_stopped_watcher_but_live_or_foreign_tree_holds(self):
        watcher={'ActiveState':'inactive','MainPID':'0','UnitFileState':'disabled'};identity=self.pinned['global_lock'];P.check_stopped(watcher,identity,[],self.pinned,True,True)
        with self.assertRaises(Hold):P.check_stopped(watcher,identity,['someone still owns it'],self.pinned,True,True)
        with self.assertRaises(Hold):P.check_stopped(watcher,identity,[],self.pinned,False,True)
    def test_new_STOP_must_match_reviewed_stop_window_and_is_retained(self):
        identity={'nlink':1,'regular_nonsymlink':True};payload={'epoch':15,'reason':'explicit reviewed service stop'};self.assertEqual(P.check_new_handoff_stop(identity,payload,10,20),identity)
        with self.assertRaises(Hold):P.check_new_handoff_stop(identity,{**payload,'reason':'user stop'},10,20)
        with self.assertRaises(Hold):P.check_new_handoff_stop(identity,{**payload,'epoch':9},10,20)
    def test_foreign_unit_or_orphan_cgroup_prevents_terminal_certificate(self):
        unit={'LoadState':'loaded','Description':'owned','InvocationID':'one','MainPID':'0','ActiveState':'active','SubState':'exited'};P.check_owned_terminal(unit,'owned','one',False,[])
        with self.assertRaises(Hold):P.check_owned_terminal(unit,'foreign','one',False,[])
        with self.assertRaises(Hold):P.check_owned_terminal(unit,'owned','one',True,[999])
        with self.assertRaises(Hold):P.check_owned_terminal({**unit,'ActiveState':'activating'},'owned','one',False,[])
    def test_renderer_has_no_dispatch_and_retains_terminal_metadata(self):
        study=json.loads((BASE/'STUDY.json').read_text(encoding='utf-8'));plan=P.render(study,self.pinned);self.assertEqual(plan['phase'],'REVIEW_HOLD_NO_DISPATCH');self.assertEqual(plan['RuntimeMaxSec']+plan['TimeoutStopSec'],600);self.assertIn('--property=RemainAfterExit=yes',plan['argv']);self.assertIn('--property=Restart=no',plan['argv']);self.assertNotIn('--collect',plan['argv'])
if __name__=='__main__':unittest.main(verbosity=2)