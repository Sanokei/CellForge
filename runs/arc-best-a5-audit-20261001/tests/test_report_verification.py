"""Report-path enforcement tests; copied fixtures only, no saved-run mutation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from scripts.report_verification import read_gate, component_metrics, population_metrics
from scripts.weights_progress import collect, stamp_gates

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/weights-action-arguments-v1'

class TestReportVerification(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='report_gate_')
        self.root=Path(self.temp.name)
        self.run=self.root/SOURCE.name
        shutil.copytree(SOURCE,self.run)
    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):
        (self.run/name).write_text(json.dumps(value),encoding='utf-8')
    def rehash(self,name):
        integrity=json.loads((self.run/'INTEGRITY.json').read_text())
        integrity[name]=hashlib.sha256((self.run/name).read_bytes()).hexdigest()
        self.write('INTEGRITY.json',integrity)
    def assert_blocked(self,gate):
        self.assertIs(gate['scientific_headline_allowed'],False,gate)
        self.assertIs(gate['scientific_acceptance'],False,gate)
    def test_false_scientific_gate_keeps_verified_positive_component(self):
        gate=read_gate(self.run);self.assert_blocked(gate)
        self.assertTrue(gate['mechanism_replay_passed'],gate)
        component=component_metrics(self.run,gate)
        self.assertEqual(component['metric_status'],'independently_replayed')
        llm=next(m for m in component['metrics'] if m['arm']=='llm')
        self.assertEqual(llm['scores']['development']['state_correct'],4)
        self.assertEqual(llm['scores']['reporting']['state_correct'],12)
        self.assertIn('provisional and unadmitted',component['label'])
    def test_missing_receipt_cannot_pass_even_with_valid_replay(self):
        (self.run/'VERIFICATION.json').unlink()
        gate=read_gate(self.run);self.assert_blocked(gate)
        self.assertIn('missing VERIFICATION.json',gate['errors'])
        self.assertEqual(len(component_metrics(self.run,gate)['metrics']),5)
    def test_tampered_receipt_cannot_override_verifier(self):
        receipt=json.loads((self.run/'VERIFICATION.json').read_text())
        receipt['scientific_standard']['acceptance']=True
        receipt['report_policy']['scientific_headline_allowed']=True
        self.write('VERIFICATION.json',receipt)
        gate=read_gate(self.run);self.assert_blocked(gate)
        self.assertIn('stale_or_tampered_verification_receipt',gate['errors'])
    def test_rehashed_fabricated_acceptance_cannot_override_verifier(self):
        receipt=json.loads((self.run/'VERIFICATION.json').read_text())
        receipt['scientific_standard']['acceptance']=True
        receipt['report_policy']['scientific_headline_allowed']=True
        self.write('VERIFICATION.json',receipt);self.rehash('VERIFICATION.json')
        gate=read_gate(self.run);self.assert_blocked(gate)
        self.assertIn('receipt acceptance conflicts with authoritative verifier',gate['errors'])
    def test_stale_result_is_unverified_but_remains_visible(self):
        result=json.loads((self.run/'RESULT.json').read_text())
        result['results'][0]['scores']['reporting']['state_correct']=12
        self.write('RESULT.json',result)
        gate=read_gate(self.run);self.assert_blocked(gate)
        self.assertFalse(gate['mechanism_replay_passed'])
        self.assertIn('stale_or_tampered_integrity',gate['authoritative_replay']['mechanism_replay']['errors'])
        self.assertEqual(component_metrics(self.run,gate)['metric_status'],'recorded_unverified')
    def test_rehashed_fabricated_scores_fail_in_report_consumer(self):
        result=json.loads((self.run/'RESULT.json').read_text())
        result['results'][0]['scores']['reporting']['state_correct']=12
        self.write('RESULT.json',result);self.rehash('RESULT.json')
        gate=read_gate(self.run);self.assert_blocked(gate)
        self.assertFalse(gate['mechanism_replay_passed'])
        self.assertIn('score_replay_differs',gate['authoritative_replay']['mechanism_replay']['errors'])
    def test_stale_weights_fail_in_report_consumer(self):
        frozen=json.loads((self.run/'FROZEN.json').read_text())
        frozen[1]['W'][0]=[0]*8;self.write('FROZEN.json',frozen)
        gate=read_gate(self.run);self.assert_blocked(gate)
        self.assertFalse(gate['mechanism_replay_passed'])
    def test_unavailable_verifier_and_malformed_receipt_fail_closed(self):
        (self.run/'VERIFICATION.json').write_text('{broken')
        with patch('cellforge.construction.action_verification.verify',side_effect=RuntimeError('unavailable')):
            gate=read_gate(self.run)
        self.assert_blocked(gate);self.assertFalse(gate['mechanism_replay_passed'])
        self.assertTrue(any('missing_or_invalid_receipt' in e for e in gate['errors']))
        self.assertTrue(any('verifier unavailable' in e for e in gate['errors']))
    def test_standard_weights_format_is_unsupported_not_zero(self):
        other=self.root/'weights-unsupported-population';other.mkdir()
        (other/'standard-weights').mkdir()
        (other/'summary.json').write_text(json.dumps({'acceptance':False,'new_children_admitted':0}))
        record=next(r for r in collect(self.root) if r['run']==other.name)
        self.assertIsNone(record['distinct_genomes'])
        self.assertEqual(record['format_status'],'unsupported_or_missing_weight_format')
        self.assertEqual(record['verification']['status'],'unsupported_unverified')
        self.assert_blocked(record['verification'])
        self.assertEqual(record['recorded_summary']['new_children_admitted'],0)
    def test_unknown_tensor_hash_format_is_visible_not_a_zero_or_abort(self):
        other=self.root/'weights-unsupported-cell';other.mkdir()
        (other/'weights').mkdir()
        (other/'weights'/'unknown-format.json').write_text(json.dumps({'tensors':{'W':[[1]]}}))
        record=next(r for r in collect(self.root) if r['run']==other.name)
        self.assertIsNone(record['distinct_genomes'])
        self.assertTrue(record['inventory_errors'])
        self.assert_blocked(record['verification'])

    def test_empty_evidence_cannot_emit_a_verified_banner(self):
        class Figure:
            def text(self,*args,**kwargs):self.banner=args[2]
        fig=Figure();stamp_gates(fig,[])
        self.assertIn('ACCEPTANCE BLOCKED',fig.banner)
        self.assertNotIn('independently verified',fig.banner)

    def test_actual_root_entrypoint_emits_blocked_report_and_positive_metrics(self):
        bash=Path(r'C:\Program Files\Git\bin\bash.exe')
        self.assertTrue(bash.is_file(),'Git Bash required for exact root entrypoint test')
        output=self.root/'report'
        env=dict(os.environ,CELLFORGE_PYTHON=os.sys.executable.replace('\\','/'),PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([str(bash),'./run_genome_report.sh','--runs',self.root.as_posix(),'--outdir',output.as_posix()],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        gates=json.loads((output/'report_verification.json').read_text())
        self.assertIs(gates['scientific_headline_allowed'],False)
        self.assertIs(gates['runs'][SOURCE.name]['scientific_headline_allowed'],False)
        components=json.loads((output/'bounded_component_results.json').read_text())
        llm=next(m for m in components[SOURCE.name]['metrics'] if m['arm']=='llm')
        self.assertEqual(llm['scores']['reporting']['state_correct'],12)
        self.assertTrue((output/'bounded_action_component.png').is_file())
        self.assertIn('scientific_acceptance_blocked',(output/'README.md').read_text())

if __name__=='__main__':unittest.main()


class TestPopulationReportQuality(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='population_report_quality_')
        self.root=Path(self.temp.name)
        self.run=self.root/'weights-action-population-corrected-g3'
        shutil.copytree(ROOT/'runs'/self.run.name,self.run)
    def tearDown(self):self.temp.cleanup()
    def test_population_replay_is_supported_but_invalid_fixture_has_no_accuracy(self):
        gate=read_gate(self.run)
        self.assertEqual(gate['authoritative_replay']['kind'],'independent-population')
        self.assertTrue(gate['population_replay_passed'],gate)
        self.assertFalse(gate['evidence_quality_passed'])
        self.assertFalse(gate['runtime_acceptance'])
        self.assertFalse(gate['scientific_headline_allowed'])
        result=population_metrics(self.run,gate)
        self.assertIsNone(result['valid_accuracy'])
        self.assertEqual(result['metric_status'],'invalid_or_unverified_fixture_raw_scores_only')
        self.assertEqual([r['raw_recorded_correct'] for r in result['raw_scores']],[4,4,4])
        self.assertIn('fixture_filename_not_copyable:reporting:newbare:JOURNAL',result['quality_errors'])
    def test_missing_quality_receipt_blocks_metric_validity(self):
        (self.run/'QUALITY_VERIFICATION.json').unlink()
        gate=read_gate(self.run)
        self.assertFalse(gate['quality_receipt_valid'])
        self.assertFalse(gate['component_metrics_allowed'])
        self.assertIsNone(population_metrics(self.run,gate)['valid_accuracy'])
        self.assertTrue(any('QUALITY_VERIFICATION.json: missing' in e for e in gate['errors']))
    def test_tampered_quality_pass_cannot_override_independent_check(self):
        receipt=json.loads((self.run/'QUALITY_VERIFICATION.json').read_text())
        receipt['evidence_quality']={'passed':True,'errors':[]}
        receipt['runtime_acceptance']=True
        (self.run/'QUALITY_VERIFICATION.json').write_text(json.dumps(receipt))
        gate=read_gate(self.run)
        self.assertFalse(gate['quality_receipt_valid'])
        self.assertFalse(gate['component_metrics_allowed'])
        self.assertFalse(gate['runtime_acceptance'])
        self.assertIsNone(population_metrics(self.run,gate)['valid_accuracy'])
        self.assertTrue(any('quality status differs' in e for e in gate['errors']))
    def test_stale_quality_binding_blocks_metric_validity(self):
        receipt=json.loads((self.run/'QUALITY_VERIFICATION.json').read_text())
        receipt['artifact_binding']['RESULT.json']='0'*64
        (self.run/'QUALITY_VERIFICATION.json').write_text(json.dumps(receipt))
        gate=read_gate(self.run)
        self.assertFalse(gate['quality_receipt_valid'])
        self.assertIsNone(population_metrics(self.run,gate)['valid_accuracy'])
        self.assertTrue(any('artifact binding differs' in e for e in gate['errors']))
    def test_rehashed_fabricated_membership_still_fails_report_replay(self):
        result=json.loads((self.run/'RESULT.json').read_text())
        result['contexts'][0]['population']['live']['action-arguments-lexical']=[]
        (self.run/'RESULT.json').write_text(json.dumps(result))
        integrity=json.loads((self.run/'FINAL_INTEGRITY.json').read_text())
        integrity['RESULT.json']=hashlib.sha256((self.run/'RESULT.json').read_bytes()).hexdigest()
        (self.run/'FINAL_INTEGRITY.json').write_text(json.dumps(integrity))
        gate=read_gate(self.run)
        self.assertFalse(gate['population_replay_passed'])
        self.assertFalse(gate['mechanism_replay_passed'])
        self.assertFalse(gate['scientific_headline_allowed'])
        self.assertTrue(any('membership' in e for e in gate['errors']))
