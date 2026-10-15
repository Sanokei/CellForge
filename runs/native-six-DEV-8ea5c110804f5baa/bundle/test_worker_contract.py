"""Synthetic-only worker orchestration tests; no selected native inputs or predictions."""
import hashlib,json,pathlib,subprocess,sys,tempfile,unittest
from unittest import mock
import study_worker as W
from study_accounting import Accounting,Hold,atomic

class WorkerAdmission(unittest.TestCase):
    def test_default_CLI_never_opens_control_root(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);control=root/'not-provisioned';bundle=root/'not-loaded'
            out=subprocess.check_output([sys.executable,str(pathlib.Path(W.__file__)),'--bundle',str(bundle),'--control-root',str(control)],text=True)
            self.assertEqual(json.loads(out)['phase'],'REVIEW_HOLD_NO_PREDICTIONS');self.assertFalse(control.exists());self.assertFalse(bundle.exists())
    def test_mismatched_approval_blocks_before_history_or_intake(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);atomic(root/'MANIFEST.json',{'files':{}});atomic(root/'STUDY.json',{'study_id':'synthetic-fixture'});atomic(root/'approval.json',{});(root/'audit.txt').write_text('synthetic',encoding='utf-8')
            with self.assertRaises(Hold):W.verify(root,root/'approval.json',root/'audit.txt')
            self.assertFalse((root/'STATE.json').exists())
    def test_baseline_futility_never_calls_authored_or_random(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);cases=[{'id':'fixture_'+str(i)} for i in range(6)];ids=[c['id'] for c in cases];atomic(root/'SIX_INPUTS.json',cases);atomic(root/'SIX_ANSWERS.json',cases)
            a=Accounting(root,{'study_id':'synthetic-fixture','case_ids':ids,'orderings':[0,3]},lambda:10);called=[]
            def scripted(account,case,answer,arm,ordering,weights=None):
                called.append(arm);self.assertEqual(arm,'scripted_baseline');key=arm+str(ordering)+case['id'];account.reserve(key)
                row={'case_id':case['id'],'arm':arm,'ordering':ordering,'correct':case['id']!='fixture_5','schema_valid':True,'model_hash':'synthetic-script'};account.finish(key,row);return row
            with mock.patch.object(W,'predict',side_effect=scripted):result=W.run(root,a)
            self.assertEqual(result['phase'],'HOLD_BASELINE_MARGIN_IMPOSSIBLE');self.assertFalse(result['result']['gate_passed']);self.assertEqual(result['packet_used'],12);self.assertEqual(called,['scripted_baseline']*12);self.assertNotIn('random_positions_charged',result)
    def test_failed_synthetic_prediction_keeps_ID_and_spent_packet(self):
        with tempfile.TemporaryDirectory() as td:
            a=Accounting(td,{'study_id':'synthetic-fixture'},lambda:10);a.claim_job();case={'id':'synthetic-retained'}
            with mock.patch.object(W.A,'candidate_input',side_effect=ValueError('unsupported synthetic fixture')):row=W.predict(a,case,{},'scripted_baseline',0)
            self.assertEqual(row['case_id'],'synthetic-retained');self.assertFalse(row['correct']);self.assertFalse(row['schema_valid']);self.assertEqual(a.load()['packet_used'],1);self.assertEqual(len(a.load()['attempts']),1)
if __name__=='__main__':unittest.main(verbosity=2)
