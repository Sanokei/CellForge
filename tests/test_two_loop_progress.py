"""Behavioral progress contracts for the real independent cell and ARC-MoE services.

Only isolated temporary roots are used. The cell fixture remeasures saved literal
weights. ARC scheduling fixtures use 100 explicitly synthetic identity tasks with
real numeric inference, assembly, admission, journaling and publication; these
unit tests do not certify official ARC performance or LLM authorship. Only the
clock, crash boundary and initialization spies are patched; phase execution is
real. Unimplemented recovery/freshness requirements deliberately remain failures.
"""
import contextlib
import copy
import io
import json
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cellforge.construction import action_cell_service as cells
from cellforge.construction import action_cell_continuation as continuation
from cellforge.construction import arc_moe_service as moe
from cellforge.construction.action_phase_protocol import FAMILY
from cellforge.construction.anchored_action_protocol import file_hash
from cellforge.construction.joint_weight_population import digest
from tests import test_action_cell_continuation as existing_cell_fixture

ROOT=pathlib.Path(__file__).resolve().parents[1]


class SimulatedCrash(RuntimeError):
    pass


class Clock:
    """Advance declared service cadence without sleeping or mocking evaluations."""
    def __init__(self):self.now=0.0
    def monotonic(self):return self.now
    def sleep(self,seconds):self.now+=max(float(seconds),0.001)


class TestTwoLoopProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates=tempfile.TemporaryDirectory(prefix='two_loop_templates_')
        base=pathlib.Path(cls.templates.name)
        cls.cell_template=base/'cells';cls.cell_template.mkdir()
        cls.cell_anchor=existing_cell_fixture.TestCellContinuationReplay().fixture(cls.cell_template)
        state=continuation.initial_state(cls.cell_template,cls.cell_anchor)
        continuation.atomic(cls.cell_template/'STATE-0.json',state)
        continuation.atomic(cls.cell_template/'PROMPTS-1.json',continuation.prompts_for(state))
        cls.moe_template=base/'moe';cls.moe_template.mkdir()
        cls.moe_anchor=cls.make_moe_fixture(cls.moe_template)
        state=moe.initial_state(cls.moe_template,cls.moe_anchor)
        continuation.atomic(cls.moe_template/'STATE-0.json',state)
        continuation.atomic(cls.moe_template/'PROMPTS-1.json',moe.prompts(state))
        moe.publish(cls.moe_template,state)

    @classmethod
    def tearDownClass(cls):cls.templates.cleanup()

    @staticmethod
    def make_moe_fixture(root):
        source=ROOT/'runs/weights-real-learning-v2'
        for name in ('parent.json','pop1_core.py'):shutil.copyfile(source/name,root/name)
        shutil.copyfile(ROOT/'ARC_FLOOR.json',root/'ARC_FLOOR.json')
        # Real installed evaluator interface, never evaluated through artifact code.
        (root/'arc_eval.py').write_text('from cellforge.construction.arc_weight_moe import predict\n',encoding='utf-8')
        continuation.atomic(root/'CONFIG.json',{'seeds':[501,502,503],'rounds':3,'range':[-9,9]})
        directory=root/'data/training';directory.mkdir(parents=True)
        records=[]
        for i in range(100):
            tid=format(i,'08x')
            task={'train':[{'input':[[1]],'output':[[1]]}],
                  'test':[{'input':[[1]],'output':[[1]]}]}
            continuation.atomic(directory/(tid+'.json'),task)
            records.append({'id':tid,'sha256':file_hash(directory/(tid+'.json'))})
        continuation.atomic(root/'TASKS.json',{'development':records,'unit_fixture_scope':'synthetic identity tasks; not official ARC evidence'})
        runtime='src/cellforge/construction/arc_moe_service.py'
        (root/runtime).parent.mkdir(parents=True)
        shutil.copyfile(ROOT/runtime,root/runtime)
        dependencies=['CONFIG.json','TASKS.json','parent.json','pop1_core.py','ARC_FLOOR.json','arc_eval.py',runtime]
        dependencies += ['data/training/'+r['id']+'.json' for r in records]
        continuation.atomic(root/'DECLARATION.json',{'schema':'prospective-ARC-MoE/1',
                              'dependencies':{n:file_hash(root/n) for n in dependencies}})
        return file_hash(root/'DECLARATION.json')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='two_loop_progress_')
        self.base=pathlib.Path(self.temp.name)
        self.cell_root=self.base/'cells';self.moe_root=self.base/'moe'
        shutil.copytree(self.cell_template,self.cell_root)
        shutil.copytree(self.moe_template,self.moe_root)
        self.cell_controller=self.base/'cell-controller.json'
        self.moe_controller=self.base/'moe-controller.json'
        self.controller('cell',{});self.controller('moe',{})

    def tearDown(self):self.temp.cleanup()

    def read(self,path):return json.loads(path.read_text(encoding='utf-8-sig'))

    def controller(self,kind,anchors):
        anchor=self.cell_anchor if kind=='cell' else self.moe_anchor
        path=self.cell_controller if kind=='cell' else self.moe_controller
        continuation.atomic(path,{'initial_anchor':anchor,
                                   'phase_anchors' if kind=='cell' else 'round_anchors':anchors})

    def cell_proposal(self,state,n,responses=None):
        prompts=continuation.prompts_for(state)
        if responses is None:
            responses=[]
            for item in prompts:
                q=item['prompt'];params=copy.deepcopy(q['parent_params'])
                params['argument.W'][0][0]-=.1*n
                raw={'model':'gpt-6.1-sol','reasoning':'low','parent_id':q['parent_id'],
                     'parent_params':q['parent_params'],'tensors':params,
                     'feature_order':q['feature_order'],'role_order':q['role_order'],
                     'prompt_sha256':digest(q),'feedback_sha256':digest(q['development_evidence']),
                     'charged_positions':39,'interface':'tool-action-joint/2'}
                responses.append({'seed':item['seed'],'response':raw})
        continuation.atomic(self.cell_root/f'RESPONSES-{n}.json',responses)
        declaration={'initial_anchor':self.cell_anchor,'phase':n,'parent_state_hash':digest(state),
                     'responses_sha256':file_hash(self.cell_root/f'RESPONSES-{n}.json'),
                     'prompts_sha256':digest(prompts)}
        continuation.atomic(self.cell_root/f'PHASE-{n}-DECLARATION.json',declaration)
        return file_hash(self.cell_root/f'PHASE-{n}-DECLARATION.json'),responses

    def moe_proposal(self,state,n,responses=None):
        requests=moe.prompts(state)
        if responses is None:
            ids=[cid for cid,c in state['pool']['cells'].items() if c['state']=='live'][:2]
            self.assertEqual(len(ids),2)
            responses=[]
            for request in requests:
                q=request['prompt']
                raw={'parent_hash':q['parent_hash'],'pool_hash':q['pool_hash'],'prompt_hash':digest(q),
                     'model':'gpt-6.1-sol','reasoning':'low','rationale':'isolated progress-contract fixture',
                     'cell_ids':ids,'router':{'route.W':[[.1*n,0],[0,.1*n]],'route.b':[0,0]},
                     'charged_positions':6}
                responses.append({'seed':request['seed'],'response':raw})
        continuation.atomic(self.moe_root/f'RESPONSES-{n}.json',responses)
        continuation.atomic(self.moe_root/f'ROUND-{n}-DECLARATION.json',
                            {'initial_anchor':self.moe_anchor,'round':n,'parent_state_hash':digest(state),
                             'prompts_hash':digest(requests),
                             'responses_sha256':file_hash(self.moe_root/f'RESPONSES-{n}.json')})
        return file_hash(self.moe_root/f'ROUND-{n}-DECLARATION.json'),responses

    def run_service(self,kind,seconds=1,crash_name=None):
        service=cells if kind=='cell' else moe
        root=self.cell_root if kind=='cell' else self.moe_root
        controller=self.cell_controller if kind=='cell' else self.moe_controller
        anchor=self.cell_anchor if kind=='cell' else self.moe_anchor
        clock=Clock();output=io.StringIO();error=None;atomic=service.atomic
        def interrupted(path,value):
            if crash_name and pathlib.Path(path).name==crash_name:
                # Model an interrupted staged write before its atomic commit.
                continuation.atomic(pathlib.Path(str(path)+'.interrupted'),value)
                raise SimulatedCrash('interrupted staging '+crash_name)
            return atomic(path,value)
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(patch.object(service.time,'monotonic',clock.monotonic))
            stack.enter_context(patch.object(service.time,'sleep',clock.sleep))
            if crash_name:stack.enter_context(patch.object(service,'atomic',side_effect=interrupted))
            try:
                if kind=='cell':service.supervise(root,anchor,controller,max_seconds=seconds,poll_seconds=1)
                else:service.supervise(root,anchor,controller,max_seconds=seconds,cadence=30)
            except Exception as caught:error=caught
        records=[json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
        return error,records,self.read(root/'SERVICE_STATUS.json')

    def begin(self,kind):
        root=self.cell_root if kind=='cell' else self.moe_root
        state=self.read(root/'STATE-0.json')
        anchor,_=self.cell_proposal(state,1) if kind=='cell' else self.moe_proposal(state,1)
        self.controller(kind,{'1':anchor})

    def assert_waiting_snapshot(self,status,counter):
        self.assertEqual(status[counter],0,status)
        explicit=(status['status'].startswith('awaiting_') or status['status'] in ('blocked','stale')
                  or bool(status.get('blocked_reason') or status.get('waiting_for')))
        self.assertTrue(explicit,'Durable last status erased the missing-maker reason: '+str(status))

    def test_cell_missing_fresh_author_remains_observably_blocked(self):
        with patch.object(cells,'compute_phase',wraps=cells.compute_phase) as phase:
            error,records,status=self.run_service('cell',seconds=3)
        self.assertIsNone(error);self.assertEqual(phase.call_count,0)
        self.assertTrue(any(r['status']=='awaiting_literal_maker_and_external_anchor' for r in records))
        self.assertFalse(list(self.cell_root.glob('CHARGES-*.jsonl')))
        self.assert_waiting_snapshot(status,'phase')

    def test_moe_missing_fresh_author_remains_observably_blocked(self):
        with patch.object(moe,'phase',wraps=moe.phase) as phase:
            error,records,status=self.run_service('moe',seconds=90)
        self.assertIsNone(error);self.assertEqual(phase.call_count,0)
        self.assertTrue(any(r['status']=='awaiting_literal_assembly_response' for r in records))
        self.assertFalse(list(self.moe_root.glob('CHARGES-*.jsonl')))
        self.assert_waiting_snapshot(status,'round')

    def test_cell_phase_two_response_without_checksum_anchor_has_durable_block_reason(self):
        self.begin('cell');error,_,_=self.run_service('cell');self.assertIsNone(error)
        state=self.read(self.cell_root/'STATE-1.json')
        self.cell_proposal(state,2)  # Response and declaration present; controller still has only phase 1.
        before=file_hash(self.cell_root/'STATE-1.json')
        with patch.object(cells,'compute_phase',wraps=cells.compute_phase) as evaluate:
            error,records,status=self.run_service('cell',seconds=3)
        self.assertIsNone(error);self.assertEqual(evaluate.call_count,0)
        self.assertTrue(any(r['status']=='awaiting_external_phase_anchor' and r['next_phase']==2 for r in records))
        self.assertEqual(file_hash(self.cell_root/'STATE-1.json'),before)
        self.assertFalse((self.cell_root/'CHARGES-2.jsonl').exists())
        explicit=(status['status']=='awaiting_external_phase_anchor' or
                  bool(status.get('blocked_reason') or status.get('waiting_for')))
        self.assertTrue(explicit,'Present phase-2 response silently loses missing-checksum reason: '+str(status))

    def test_cell_atomic_external_checksum_handoff_advances_phase_two_once(self):
        self.begin('cell');error,_,_=self.run_service('cell');self.assertIsNone(error)
        state=self.read(self.cell_root/'STATE-1.json');anchor,_=self.cell_proposal(state,2)
        first=self.read(self.cell_controller)['phase_anchors']['1']
        self.controller('cell',{'1':first,'2':anchor})
        error,_,_=self.run_service('cell');self.assertIsNone(error)
        self.assertEqual(self.read(self.cell_root/'STATE-2.json')['phase'],2)
        charges=file_hash(self.cell_root/'CHARGES-2.jsonl')
        self.assertEqual(len(continuation.read_charges(self.cell_root,2)),6)
        error,_,_=self.run_service('cell');self.assertIsNone(error)
        self.assertEqual(file_hash(self.cell_root/'CHARGES-2.jsonl'),charges)

    def test_cell_wrong_external_checksum_blocks_before_any_phase_two_charge(self):
        self.begin('cell');error,_,_=self.run_service('cell');self.assertIsNone(error)
        state=self.read(self.cell_root/'STATE-1.json');self.cell_proposal(state,2)
        first=self.read(self.cell_controller)['phase_anchors']['1']
        self.controller('cell',{'1':first,'2':'0'*64})
        error,_,status=self.run_service('cell')
        self.assertIsInstance(error,ValueError);self.assertEqual(status['status'],'blocked')
        self.assertFalse((self.cell_root/'STATE-2.json').exists())
        self.assertFalse((self.cell_root/'CHARGES-2.jsonl').exists())

    def test_cell_process_heartbeats_do_not_claim_advancing_generations(self):
        before=file_hash(self.cell_root/'STATE-0.json')
        error,records,status=self.run_service('cell',seconds=3)
        self.assertIsNone(error)
        heartbeats=[r for r in records if r['status'].startswith('awaiting_')]
        self.assertGreaterEqual(len(heartbeats),2)
        self.assertEqual({r['next_phase'] for r in heartbeats},{1})
        self.assertEqual(file_hash(self.cell_root/'STATE-0.json'),before)
        self.assertFalse((self.cell_root/'STATE-1.json').exists())
        self.assertFalse(any(r['status']=='generation_complete' for r in records))

    def test_moe_process_heartbeats_do_not_claim_advancing_generations(self):
        before=file_hash(self.moe_root/'ARC_HISTORY.json')
        error,records,status=self.run_service('moe',seconds=90)
        self.assertIsNone(error)
        heartbeats=[r for r in records if r['status'].startswith('awaiting_')]
        self.assertGreaterEqual(len(heartbeats),2)
        self.assertEqual({r['next_round'] for r in heartbeats},{1})
        self.assertEqual(file_hash(self.moe_root/'ARC_HISTORY.json'),before)
        self.assertFalse((self.moe_root/'STATE-1.json').exists())
        self.assertFalse(any(r['status']=='ARC_MoE_round_complete' for r in records))

    def test_missing_cell_maker_or_cell_lock_does_not_stop_moe_schedule(self):
        (self.cell_root/'SERVICE.lock').write_text('isolated-test-owner')
        self.begin('moe');error,records,status=self.run_service('moe')
        self.assertIsNone(error)
        self.assertEqual(self.read(self.moe_root/'STATE-1.json')['round'],1)
        self.assertFalse((self.cell_root/'STATE-1.json').exists())
        self.assertTrue((self.cell_root/'SERVICE.lock').exists())
        self.assertEqual(len(continuation.read_charges(self.moe_root,1)),6)

    def test_missing_moe_maker_or_moe_lock_does_not_stop_cell_schedule(self):
        (self.moe_root/'SERVICE.lock').write_text('isolated-test-owner')
        self.begin('cell');error,records,status=self.run_service('cell')
        self.assertIsNone(error)
        self.assertEqual(self.read(self.cell_root/'STATE-1.json')['phase'],1)
        self.assertFalse((self.moe_root/'STATE-1.json').exists())
        self.assertTrue((self.moe_root/'SERVICE.lock').exists())
        self.assertEqual(len(continuation.read_charges(self.cell_root,1)),6)

    def test_cell_interrupted_staging_resumes_committed_attempt_exactly_once(self):
        self.begin('cell');error,_,_=self.run_service('cell',crash_name='STATE-1.json')
        self.assertIsInstance(error,SimulatedCrash)
        self.assertTrue((self.cell_root/'GENERATION-1.json').exists())
        before=file_hash(self.cell_root/'CHARGES-1.jsonl')
        with patch.object(cells,'compute_phase',wraps=cells.compute_phase) as compute:
            error,records,status=self.run_service('cell')
        self.assertEqual(file_hash(self.cell_root/'CHARGES-1.jsonl'),before,'recovery recharged an attempt')
        self.assertEqual(compute.call_count,0,'persisted result was evaluated again')
        self.assertIsNone(error,'Committed result cannot resume its interrupted checkpoint: '+str(error))
        self.assertEqual(self.read(self.cell_root/'STATE-1.json')['phase'],1)
        self.assertTrue((self.cell_root/'PROMPTS-2.json').exists())

    def test_moe_interrupted_staging_resumes_charged_round_exactly_once(self):
        self.begin('moe');error,_,_=self.run_service('moe',crash_name='STATE-1.json')
        self.assertIsInstance(error,SimulatedCrash)
        before=file_hash(self.moe_root/'CHARGES-1.jsonl')
        error,records,status=self.run_service('moe')
        self.assertEqual(file_hash(self.moe_root/'CHARGES-1.jsonl'),before,'recovery recharged an attempt')
        self.assertIsNone(error,'Charged round cannot resume interrupted staging: '+str(error))
        self.assertEqual(self.read(self.moe_root/'STATE-1.json')['round'],1)
        self.assertTrue((self.moe_root/'ROUND-1.json').exists())
        self.assertTrue((self.moe_root/'PROMPTS-2.json').exists())

    def test_cell_restart_restores_survivors_without_reinitialization_or_recharge(self):
        self.begin('cell');error,_,_=self.run_service('cell');self.assertIsNone(error)
        checkpoint=file_hash(self.cell_root/'STATE-1.json');charges=file_hash(self.cell_root/'CHARGES-1.jsonl')
        with patch.object(cells,'initial_state',wraps=cells.initial_state) as initialize:
            error,_,status=self.run_service('cell')
        self.assertIsNone(error);self.assertEqual(initialize.call_count,0)
        self.assertEqual(file_hash(self.cell_root/'STATE-1.json'),checkpoint)
        self.assertEqual(file_hash(self.cell_root/'CHARGES-1.jsonl'),charges)
        self.assertEqual(status['phase'],1)

    def test_moe_restart_restores_survivors_without_reinitialization_or_recharge(self):
        self.begin('moe');error,_,_=self.run_service('moe');self.assertIsNone(error)
        checkpoint=file_hash(self.moe_root/'STATE-1.json');charges=file_hash(self.moe_root/'CHARGES-1.jsonl')
        with patch.object(moe,'initial_state',wraps=moe.initial_state) as initialize:
            error,_,status=self.run_service('moe')
        self.assertIsNone(error);self.assertEqual(initialize.call_count,0)
        self.assertEqual(file_hash(self.moe_root/'STATE-1.json'),checkpoint)
        self.assertEqual(file_hash(self.moe_root/'CHARGES-1.jsonl'),charges)
        self.assertEqual(status['round'],1)

    def test_cell_restart_recovers_next_prompt_from_committed_checkpoint(self):
        self.begin('cell');error,_,_=self.run_service('cell');self.assertIsNone(error)
        (self.cell_root/'PROMPTS-2.json').unlink()
        error,_,_=self.run_service('cell');self.assertIsNone(error)
        self.assertTrue((self.cell_root/'PROMPTS-2.json').exists(),'cell loop silently waits after missing prompt publication')
        state=self.read(self.cell_root/'STATE-1.json')
        self.assertEqual(self.read(self.cell_root/'PROMPTS-2.json'),continuation.prompts_for(state))
        self.assertEqual(len(continuation.read_charges(self.cell_root,1)),6)

    def test_moe_restart_recovers_next_prompt_from_committed_checkpoint(self):
        self.begin('moe');error,_,_=self.run_service('moe');self.assertIsNone(error)
        (self.moe_root/'PROMPTS-2.json').unlink()
        error,_,_=self.run_service('moe');self.assertIsNone(error)
        self.assertTrue((self.moe_root/'PROMPTS-2.json').exists(),'MoE loop silently waits after missing prompt publication')
        state=self.read(self.moe_root/'STATE-1.json')
        self.assertEqual(self.read(self.moe_root/'PROMPTS-2.json'),moe.prompts(state))
        self.assertEqual(len(continuation.read_charges(self.moe_root,1)),6)

    def test_cell_successive_proposals_are_fresh_and_current_parent_bound(self):
        state=self.read(self.cell_root/'STATE-0.json');hashes=[]
        for n in (1,2):
            anchor,responses=self.cell_proposal(state,n);hashes.append(digest(responses))
            result=continuation.compute_phase(self.cell_root,self.cell_anchor,n,anchor,state,lambda e:continuation.append_charge(self.cell_root,n,e))
            for c,q in zip(result['contexts'],continuation.prompts_for(state)):
                self.assertEqual(c['arms']['llm']['parent'],q['prompt']['parent_id'])
            state=continuation.successor(state,result)
        self.assertNotEqual(*hashes)
        for arm in ('llm','random'):
            self.assertTrue(all(len(c['arms'][arm]['charges'])==2 for c in state['contexts']))

    def test_moe_successive_proposals_are_fresh_and_current_parent_bound(self):
        state=self.read(self.moe_root/'STATE-0.json');hashes=[]
        for n in (1,2):
            anchor,responses=self.moe_proposal(state,n);hashes.append(digest(responses))
            previous=copy.deepcopy(state)
            state,records=moe.phase(self.moe_root,self.moe_anchor,anchor,state,lambda e:continuation.append_charge(self.moe_root,n,e))
            for new,old in zip(state['contexts'],previous['contexts']):
                for arm in ('llm','random'):
                    self.assertEqual(new['arms'][arm]['births'][-1]['parents'],[old['arms'][arm]['champion']['hash']])
        self.assertNotEqual(*hashes)
        self.assertTrue(all(len(c['arms']['llm']['births'])==2 for c in state['contexts']))

    def test_cell_reusing_previous_raw_author_response_cannot_advance_next_phase(self):
        state=self.read(self.cell_root/'STATE-0.json');anchor,responses=self.cell_proposal(state,1)
        result=continuation.compute_phase(self.cell_root,self.cell_anchor,1,anchor,state,lambda e:continuation.append_charge(self.cell_root,1,e))
        next_state=continuation.successor(state,result)
        self.assertEqual(continuation.prompts_for(state),continuation.prompts_for(next_state),'fixture must retain actual neutral parent')
        anchor,_=self.cell_proposal(next_state,2,responses=responses)
        with self.assertRaises(ValueError,msg='A previous maker receipt was silently reused as fresh phase-2 authorship'):
            continuation.compute_phase(self.cell_root,self.cell_anchor,2,anchor,next_state,lambda e:continuation.append_charge(self.cell_root,2,e))

    def test_moe_reusing_previous_raw_author_response_cannot_advance_next_round(self):
        state=self.read(self.moe_root/'STATE-0.json');anchor,responses=self.moe_proposal(state,1)
        next_state,_=moe.phase(self.moe_root,self.moe_anchor,anchor,state,lambda e:continuation.append_charge(self.moe_root,1,e))
        self.assertEqual(moe.prompts(state),moe.prompts(next_state),'fixture must retain actual neutral parent')
        anchor,_=self.moe_proposal(next_state,2,responses=responses)
        with self.assertRaises(ValueError,msg='A previous maker receipt was silently reused as fresh round-2 authorship'):
            moe.phase(self.moe_root,self.moe_anchor,anchor,next_state,lambda e:continuation.append_charge(self.moe_root,2,e))

    def test_cell_stale_parent_proposal_is_blocked_without_random_substitution(self):
        state=self.read(self.cell_root/'STATE-0.json');_,responses=self.cell_proposal(state,1)
        responses[0]['response']['parent_id']='0'*64
        anchor,_=self.cell_proposal(state,1,responses);self.controller('cell',{'1':anchor})
        error,_,status=self.run_service('cell')
        self.assertIsInstance(error,ValueError);self.assertEqual(status['status'],'blocked')
        self.assertFalse((self.cell_root/'STATE-1.json').exists())
        self.assertEqual([c['arm'] for c in continuation.read_charges(self.cell_root,1)],['llm'])

    def test_moe_random_provenance_cannot_substitute_for_a_missing_author(self):
        state=self.read(self.moe_root/'STATE-0.json');_,responses=self.moe_proposal(state,1)
        responses[0]['response']['model']='random_seed'
        anchor,_=self.moe_proposal(state,1,responses);self.controller('moe',{'1':anchor})
        error,_,status=self.run_service('moe')
        self.assertIsInstance(error,ValueError);self.assertEqual(status['status'],'blocked')
        self.assertFalse((self.moe_root/'STATE-1.json').exists())
        self.assertEqual([c['arm'] for c in continuation.read_charges(self.moe_root,1)],['llm'])


if __name__=='__main__':unittest.main()
