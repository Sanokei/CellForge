import json,pathlib,tempfile,unittest
from study_accounting import Accounting,Hold,acceptance,baseline_futility
class PersistentBounds(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);self.now=[10];self.study={'study_id':'synthetic-fixture'};self.a=Accounting(self.root,self.study,lambda:self.now[0])
    def tearDown(self):self.tmp.cleanup()
    def test_validation_does_not_consume_intake(self):
        self.a.intake(b'{}',lambda w:None,False);self.assertFalse(self.a.path.exists())
    def test_invalid_founder_consumed_no_repair_retry(self):
        s=self.a.intake(b'{broken',lambda w:None,True);self.assertEqual(s['positions_charged'],48);self.assertEqual(s['phase'],'REJECTED_FOUNDER_CONSUMED')
        with self.assertRaises(Hold):self.a.intake(b'{}',lambda w:None,True)
        self.assertEqual(self.a.intake(b'{broken',lambda w:None,True)['intake_used'],1)
    def test_single_job_deadline_and_restart_cannot_reset(self):
        self.a.claim_job();self.assertEqual(self.a.load()['deadline_epoch'],610)
        with self.assertRaises(Hold):Accounting(self.root,self.study,lambda:20).claim_job()
        self.now[0]=610
        with self.assertRaises(Hold):self.a.reserve('late')
    def test_intent_is_charged_before_prediction_and_never_replayed(self):
        self.a.claim_job();self.assertIsNone(self.a.reserve('packet'));self.assertEqual(self.a.load()['packet_used'],1)
        with self.assertRaises(Hold):Accounting(self.root,self.study,lambda:self.now[0]).reserve('packet')
        self.a.finish('packet',{'correct':False});self.assertEqual(self.a.reserve('packet'),{'correct':False});self.assertEqual(self.a.load()['packet_used'],1)
    def test_cumulative_cap216(self):
        self.a.claim_job()
        for i in range(216):self.a.reserve(str(i));self.a.finish(str(i),{'correct':False})
        with self.assertRaises(Hold):self.a.reserve('217')
        self.assertEqual(Accounting(self.root,self.study).load()['packet_used'],216)
    def test_STOP_preempts_and_is_preserved(self):
        self.a.claim_job();marker=self.root/'STOP';marker.write_bytes(b'hold')
        with self.assertRaises(Hold):self.a.reserve('blocked')
        self.assertEqual(marker.read_bytes(),b'hold');self.assertEqual(self.a.load()['packet_used'],0)
        with self.assertRaises(Hold):self.a.intake(b'{}',lambda w:None,True)
        self.assertFalse((self.root/'INTAKE_CLAIM.json').exists())
    def test_deleted_state_cannot_reset_job_claim(self):
        self.a.claim_job();self.a.path.unlink()
        with self.assertRaises(Hold):Accounting(self.root,self.study).claim_job()
    def test_deleted_state_cannot_reset_intake(self):
        self.a.intake(b'{}',lambda w:None,True);self.a.path.unlink()
        with self.assertRaises(Hold):Accounting(self.root,self.study).intake(b'{}',lambda w:None,True)
    def test_duplicate_keys_consume_founder(self):
        s=self.a.intake(b'{"a":1,"a":2}',lambda w:None,True);self.assertEqual(s['phase'],'REJECTED_FOUNDER_CONSUMED');self.assertNotIn('genome',s)
    def test_random_positions_are_charged_once(self):
        self.a.claim_job();self.a.charge_random();self.a.charge_random();self.assertEqual(self.a.load()['random_positions_charged'],48)
    def test_new_study_identity_cannot_reuse_state(self):
        self.a.claim_job()
        with self.assertRaises(Hold):Accounting(self.root,{'study_id':'different'}).load()
class FixedGate(unittest.TestCase):
    def rows(self,auth=5,base=3,rand=3):
        rows=[]
        for order in (0,3):
            for arm,count in [('authored',auth),('scripted_baseline',base),('matched_random',rand)]:
                for i in range(6):rows.append({'case_id':str(i),'ordering':order,'arm':arm,'correct':i<count,'schema_valid':True,'model_hash':'one-genome'})
        return rows
    def test_unchanged_gate_both_orderings(self):self.assertTrue(acceptance(self.rows(),(0,3),[str(i) for i in range(6)])['gate_passed'])
    def test_baseline5_makes_margin_impossible_even_author6(self):
        rows=self.rows(6,5,3);self.assertTrue(baseline_futility([r for r in rows if r['arm']=='scripted_baseline'],(0,3)));self.assertFalse(acceptance(rows,(0,3),[str(i) for i in range(6)])['gate_passed'])
    def test_invalid_controls_cannot_earn_gain(self):
        rows=self.rows();next(r for r in rows if r['arm']=='matched_random')['schema_valid']=False;self.assertFalse(acceptance(rows,(0,3),[str(i) for i in range(6)])['gate_passed'])
    def test_duplicate_ID_or_mixed_genome_rejected(self):
        rows=self.rows();rows[0]['case_id']='1';self.assertFalse(acceptance(rows,(0,3),[str(i) for i in range(6)])['gate_passed'])
        rows=self.rows();rows[0]['model_hash']='per-case-winner';self.assertFalse(acceptance(rows,(0,3),[str(i) for i in range(6)])['gate_passed'])
if __name__=='__main__':unittest.main(verbosity=2)