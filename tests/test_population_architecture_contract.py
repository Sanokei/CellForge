"""Mandatory architecture acceptance contracts, intentionally red for missing features.

Use real registry/admission/typed execution/controller APIs. Synthetic records below
are fault-injection inputs, not hard-niche performance or LLM-authorship evidence.
No provider is called: author subprocess is intercepted at its external boundary.
No active run, source module, historical test, or owner-controlled floor is changed.
"""
import copy
import itertools
import json
import pathlib
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.circuit_population import CircuitArchive
from cellforge.construction.adhesion import AdhesionCandidate, validate_adhesion
from cellforge.construction.joint_weight_population import digest, inherit_neighbors
from cellforge.construction.population_contract import require_two_level_run
from cellforge.construction import server_author_controller as controller
from cellforge.construction import arithmetic_subtraction_service as arithmetic
from cellforge.construction.multiniche_queue import MultiNicheQueue, NicheWaiting
from cellforge.construction.action_cell_continuation import atomic
from cellforge.construction.anchored_action_protocol import file_hash
from tests import test_adhesion as numeric_fixture

ROOT=pathlib.Path(__file__).resolve().parents[1]


def scores(hit=False):
    # Fault-injection scores; deliberately not asserted as real ARC evidence.
    return {label:{'partition':'development','manifest':'fault-fixture-'+label,
                   'source':'public-training-real' if label=='anchor' else 'synthetic-fault-fixture',
                   'outcomes':{'retained':True,'unsolved':hit},'false_positive_given_consistency':0.0}
            for label in ('fresh','archive','anchor')}


def population():
    p=CellPopulation()
    a=p.seed('grid-directional',{'W':[[1,0]],'b':[0]}, {'source':'test-fixture-not-LLM'},scores())
    b=p.seed('grid-directional',{'W':[[0,2]],'b':[0]}, {'source':'test-fixture-not-LLM'},scores())
    return p,a,b


def rejected_or_raised(call):
    try:return not call()['accepted']
    except (ValueError,TypeError):return True


class TestPopulationArchitectureContract(unittest.TestCase):
    def test_shared_registry_preserves_family_and_niche_separately_after_resume(self):
        p,a,b=population();restored=CellPopulation.from_snapshot(p.snapshot())
        for cid in (a,b):
            record=restored.cells[cid]
            self.assertIn('family',record,'registry conflates tensor structure and family membership')
            self.assertIn('niche',record,'registry has no persistent task-niche membership')
            self.assertIn('niche_manifest',record,'hard niche must be bound to its real task commitment')

    def test_duplicate_parent_labels_cannot_count_as_two_parent_birth(self):
        p,a,_=population()
        accepted=not rejected_or_raised(lambda:p.birth('grid-directional',{'W':[[3,4]],'b':[0]},[a,a],
                  {'operator':'two-parent-crossover-then-llm','model':'gpt-6.1-sol'},scores(True),
                  caps_passed=True,fixtures_passed=True,arm='llm_crossover'))
        self.assertFalse(accepted,'one parent repeated twice was admitted as sexual inheritance')

    def test_two_parent_labels_without_inherited_material_cannot_earn_admission(self):
        p,a,b=population()
        self.assertTrue(rejected_or_raised(lambda:p.birth('grid-directional',{'W':[[7,8]],'b':[9]},[a,b],
                  {'operator':'two-parent-crossover-then-llm','model':'gpt-6.1-sol'},scores(True),
                  caps_passed=True,fixtures_passed=True,arm='llm_crossover')),
                  'arbitrary tensors plus two IDs are not demonstrated inheritance')

    def test_retired_second_parent_cannot_be_used_as_live_mating_partner(self):
        p,a,b=population();p.cells[b]['state']='retired';p.live['grid-directional'].remove(b)
        self.assertTrue(rejected_or_raised(lambda:p.birth('grid-directional',{'W':[[1,2]],'b':[1]},[a,b],
                  {'operator':'crossover','model':'gpt-6.1-sol'},scores(True),
                  caps_passed=True,fixtures_passed=True,arm='llm_crossover')),
                  'only the recipient is checked for contemporary live membership')

    def test_post_crossover_mutation_requires_raw_literals_bound_to_inherited_child(self):
        recipient={'r:conv.W':[[1,0,0,0] for _ in range(9)],'r:conv.b':[0],'r:paint':[0,1,0,0]}
        donor={'d:cnt.W':[[0,2,0,0]],'d:cnt.b':[-1]}
        inherited,receipt=inherit_neighbors(recipient,donor,recipient_prefix='r:',donor_prefix='d:')
        p=CellPopulation();a=p.seed('conv',recipient,{'source':'fixture'},scores())
        b=p.seed('count',donor,{'source':'fixture'},scores())
        mutated=copy.deepcopy(inherited);mutated['r:conv.W'][4][0]=1.25
        # Deliberately mismatched raw response: applied values must never be silently substituted.
        provenance={'model':'gpt-6.1-sol','reasoning':'low','inheritance':receipt,
                    'mutation_parent_hash':digest(inherited),'raw_response':{'tensors':inherited},
                    'newly_authored':['r:conv.W[4][0]'],'typed_io':'four-color-grid'}
        self.assertTrue(rejected_or_raised(lambda:p.birth('conv',mutated,[a,b],provenance,scores(True),
                  caps_passed=True,fixtures_passed=True,arm='llm_crossover')),
                  'registry admitted weights different from the purported post-crossover LLM response')

    def test_actual_two_parent_transfer_retains_recipient_and_donor_numerics(self):
        recipient={'r:conv.W':[[1,0,0,0] for _ in range(9)],'r:conv.b':[0],'r:paint':[0,1,0,0]}
        donor={'d:cnt.W':[[0,2,0,0]],'d:cnt.b':[-1]}
        child,receipt=inherit_neighbors(recipient,donor,recipient_prefix='r:',donor_prefix='d:')
        self.assertEqual(child['r:conv.W'][4],recipient['r:conv.W'][4])
        for i in (0,1,2,3,5,6,7,8):self.assertEqual(child['r:conv.W'][i],donor['d:cnt.W'][0])
        self.assertEqual(child['r:paint'],recipient['r:paint'])
        self.assertEqual(child['r:conv.b'],donor['d:cnt.b'])
        self.assertEqual(receipt['recipient_hash'],digest(recipient))
        self.assertEqual(receipt['donor_hash'],digest(donor))

    def test_typed_attachment_executes_both_contributors_and_authored_glue(self):
        spec=copy.deepcopy(numeric_fixture.SPEC);cat=copy.deepcopy(numeric_fixture.CAT)
        full=numeric_fixture.score(spec,cat)
        self.assertEqual(full,16)
        for donor in ('cmp2','mux'):
            ablated=copy.deepcopy(cat)
            for layer in ablated[donor]['layers']:
                for key in ('W','b'):
                    def zero(value):return [zero(v) for v in value] if isinstance(value,list) else 0
                    layer[key]=zero(layer[key])
            self.assertLess(numeric_fixture.score(spec,ablated),full,donor+' has no causal contribution')
        glue=copy.deepcopy(spec);glue['nodes'][1]['params']={'W':[[0,0,0]],'b':[0]}
        self.assertLess(numeric_fixture.score(glue,cat),full)
        # Fixture demonstrates numerical execution only; does not earn population admission.

    def test_cross_family_attachment_is_registered_birth_with_family_membership(self):
        receipt=validate_adhesion(numeric_fixture.SPEC,numeric_fixture.CAT)
        self.assertIsNotNone(receipt['population_child_id'],
                             'typed fixture is detached from the shared population birth/admission registry')
        self.assertIn('parent_families',receipt,'typed bit ports alone do not prove cross-family inheritance')
        self.assertGreaterEqual(len(set(receipt['parent_families'])),2)

    def test_topology_change_affects_real_execution(self):
        spec=copy.deepcopy(numeric_fixture.SPEC)
        baseline=AdhesionCandidate(spec,numeric_fixture.CAT,numeric_fixture.forward)
        changed=copy.deepcopy(spec)
        # Exchange declared low/high outputs, preserving the executable typed graph.
        keys=list(changed['outputs']);self.assertEqual(len(keys),2)
        changed['outputs'][keys[0]],changed['outputs'][keys[1]]=changed['outputs'][keys[1]],changed['outputs'][keys[0]]
        rewired=AdhesionCandidate(changed,numeric_fixture.CAT,numeric_fixture.forward)
        self.assertTrue(any(baseline.execute({'bits':list(x)})['packet']['bits']!=
                            rewired.execute({'bits':list(x)})['packet']['bits']
                            for x in itertools.product((0,1),repeat=4)))

    def circuit(self):
        return CircuitArchive({'topology':'baseline'},{'a':False,'b':False},['cmp','mux'],
                   founder_weights={'cmp':{'W':[[1]],'b':[0]},'mux':{'W':[[1]],'b':[0]}})

    def test_circuit_cannot_admit_caller_claims_without_execution_and_controls(self):
        archive=self.circuit()
        proposal={'spec':{'topology':'unexecuted-claimed-circuit'},'new_numeric_positions':4}
        decision=archive.propose(proposal,valid=True,dev_outcomes={'a':True,'b':True},contribution=True,arm='llm')
        self.assertFalse(decision['accepted'],'booleans supplied by caller replaced executed parent/random/zero ablation receipts')

    def test_handwired_half_adder_truth_table_is_not_architecture_acceptance(self):
        archive=self.circuit()
        proposal={'spec':{'topology':'handwired-half-adder','source':'toy-handwritten',
                          'domain':'two-bit-truth-table'},'new_numeric_positions':4}
        decision=archive.propose(proposal,valid=True,dev_outcomes={'a':True,'b':True},contribution=True,arm='llm')
        self.assertFalse(decision['accepted'],'toy truth-table gain bypassed genuine LLM/parent/control admission')

    def test_rejected_hashes_cannot_enter_live_counts_or_assembly(self):
        p,a,_=population();params={'W':[[9,9]],'b':[9]}
        decision=p.birth('grid-directional',params,[a],{'source':'fixture'},scores(),
                         caps_passed=True,fixtures_passed=True,arm='random')
        self.assertFalse(decision['accepted'])
        cid=digest({'structure':'grid-directional','params':params})
        self.assertEqual(p.cells[cid]['state'],'rejected')
        self.assertNotIn(cid,[v for ids in p.live.values() for v in ids])
        with self.assertRaises(ValueError):p.assemble({'route.W':[[1]],'route.b':[0]},[cid])

    def test_resume_rejects_rejected_hash_hidden_in_live_registry(self):
        p,a,_=population();snap=copy.deepcopy(p.snapshot());snap['cells'][a]['state']='rejected'
        with self.assertRaises(ValueError):CellPopulation.from_snapshot(snap)

    def test_architecture_gate_cannot_accept_label_only_birth_and_assembly(self):
        p,a,b=population()
        p.birth('grid-directional',{'W':[[7,8]],'b':[9]},[a],{'label':'llm'},scores(True),
                caps_passed=True,fixtures_passed=True,arm='llm')
        run={'population':p.snapshot(),'ensemble_population':{'histories':{'fake':['unexecuted']}},
             'selection_partition':'development','assembly':{'cell_ids':[a,b]}}
        self.assertTrue(rejected_or_raised(lambda:{'accepted':require_two_level_run(run)}),
                         'architecture gate accepted fake ensemble history and single-parent labelled mutation')

    def test_saturated_easy_niche_does_not_dispatch_another_author_attempt(self):
        with tempfile.TemporaryDirectory(prefix='architecture_saturation_') as tmp:
            root=pathlib.Path(tmp)/'run';root.mkdir();workspace=pathlib.Path(tmp)/'author'
            atomic(root/'STATE-0.json',{'phase':0})
            atomic(root/'PROMPTS-1.json',[{'seed':1,'prompt':{'partition':'development',
                  'niche':'saturated-easy','development_evidence':{'outcomes':{'easy':True},'counterexamples':[]}}}])
            with patch.object(controller.subprocess,'run',side_effect=RuntimeError('external boundary intercepted')) as provider:
                try:controller.author(root,'cells',workspace)
                except (ValueError,RuntimeError):pass
            self.assertEqual(provider.call_count,0,'saturated easy task still monopolizes provider dispatch; no hard-niche routing')

    def test_continuous_controller_does_not_stop_at_six_overall_generations(self):
        with tempfile.TemporaryDirectory(prefix='architecture_continuous_') as tmp:
            root=pathlib.Path(tmp)/'run';root.mkdir();outside=pathlib.Path(tmp)
            atomic(root/'DECLARATION.json',{'dependencies':{}})
            atomic(root/'STATE-6.json',{'phase':6})
            atomic(root/'PROMPTS-7.json',[{'seed':1,'prompt':{'niche':'hard-unsolved'}}])
            with patch.object(controller,'author',side_effect=RuntimeError('author boundary intercepted')) as author:
                controller.drive(root,'cells',file_hash(root/'DECLARATION.json'),outside/'controller.json',outside/'work',max_seconds=1)
            self.assertGreater(author.call_count,0,'continuous controller declared completion at fixed batch cap instead of prospective next batch')

    def test_complementary_primitives_can_reach_joint_evaluation_without_whole_task_solves(self):
        p=CellPopulation();unsolved=scores()
        for section in unsolved.values():section['outcomes']={'joint-task':False}
        a=p.seed('comparison',{'W':[[1]],'b':[0]},{'source':'fault-fixture'},unsolved)
        b=p.seed('selection',{'W':[[2]],'b':[0]},{'source':'fault-fixture'},unsolved)
        # Candidate eligibility is distinct from admission. The latter still needs
        # numerical joint contribution, controls and typed IO checks.
        try:
            candidate=p.assemble({'route.W':[[1,0],[0,1]],'route.b':[0,0]},[a,b])
        except ValueError as error:
            self.fail('Complementary primitives excluded before joint contribution can be measured: '+str(error))
        self.assertEqual(set(candidate['cell_ids']),{a,b})

    def broad_competitor(self,p):
        # Proposed integration seam on the actual shared population. No fake runner
        # substitutes for the absent broad competition implementation.
        compete=getattr(p,'compete',None)
        self.assertTrue(callable(compete),
            'CellPopulation has no executable broad top-cell competition interface; a ranking report is insufficient')
        return compete

    def executed_broad_fixture(self):
        # Actual arithmetic forward/adapter; independently declared synthetic DEV
        # cases test execution, not model learning or real LLM authorship.
        p=CellPopulation()
        p.seed('decimal-addition',copy.deepcopy(arithmetic.adapter.BASELINE_PARAMS),
               {'source':'synthetic-contract-fixture','input_type':'decimal-column'},scores())
        subtraction={'sum.W':[[1,-1,-1]],'sum.b':[0],'carry.W':[[-1]],'carry.b':[-.5],
                     'readout.W':[[1,10],[0,1]],'readout.b':[0,0]}
        p.seed('decimal-subtraction',subtraction,
               {'source':'synthetic-contract-fixture','input_type':'decimal-column'},scores())
        tasks=[{'id':'addition-no-carry','a':7,'b':2,'incoming':0,'expected':{'digit':9,'status':0}},
               {'id':'addition-carry','a':9,'b':2,'incoming':0,'expected':{'digit':1,'status':1}},
               {'id':'subtraction-borrow','a':2,'b':7,'incoming':0,'expected':{'digit':5,'status':1}},
               {'id':'subtraction-no-borrow','a':7,'b':2,'incoming':0,'expected':{'digit':5,'status':0}}]
        environment={'id':'mixed-arithmetic-development','input_type':'decimal-column',
                     'partition':'development','manifest':digest(tasks),'tasks':tasks}
        calls=[]
        def evaluate(cell,view):
            self.assertEqual(set(view),{'a','b','incoming'},'executor received targets or task identity')
            calls.append((cell['id'],copy.deepcopy(view)))
            if cell['family']=='decimal-addition':
                out=arithmetic.adapter._actual.predict_adder(cell['params'],
                    {'a':view['a'],'b':view['b'],'carry':view['incoming']})
                return {'digit':out['digit'],'status':out['carry']}
            self.assertEqual(cell['family'],'decimal-subtraction')
            out=arithmetic.adapter.predict(cell['params'],
                {'a':view['a'],'b':view['b'],'borrow':view['incoming']})
            return {'digit':out['digit'],'status':out['borrow']}
        return p,environment,evaluate,calls

    def assert_executed_broad(self,result,calls):
        self.assertEqual(len(calls),8,'each eligible family must execute all four actual numerical cases')
        self.assertEqual(len(result['execution_receipts']),2)
        for row in result['execution_receipts']:
            self.assertEqual(row['status'],'executed')
            self.assertEqual(row['scored_tasks'],4)
            self.assertEqual(row['correct'],2)
            self.assertEqual(len(row['outcomes']),4)
        self.assertGreater(result['competition_iteration'],0)

    def test_broad_competition_selects_every_eligible_family_and_preserves_niche_diversity(self):
        p,environment,evaluate,calls=self.executed_broad_fixture()
        compete=self.broad_competitor(p);before=copy.deepcopy(p.snapshot())
        result=compete(environments=[environment],evaluator=evaluate)
        self.assertEqual(set(result['representatives']),{'decimal-addition','decimal-subtraction'})
        self.assertEqual(result['registry_hash_before'],digest(before))
        self.assert_executed_broad(result,calls)
        self.assertEqual(set(p.live),set(before['live']))
        self.assertTrue(all(p.live[family] for family in before['live']))
        self.assertEqual(p.live,before['live'],'broad competition erased niche representatives')

    def test_broad_competition_rejects_missing_family_and_single_family_monopoly(self):
        p,a,b=population();c=p.seed('count-grid',{'W':[[1]],'b':[0]},{'source':'fixture'},scores())
        compete=self.broad_competitor(p)
        with self.assertRaises(ValueError,msg='eligible family silently excluded from broad competition'):
            compete(environments=[{'id':'broad','partition':'development'}],representatives={'grid-directional':a})
        with self.assertRaises(ValueError,msg='two representatives from one family counted as broad diversity'):
            compete(environments=[{'id':'broad','partition':'development'}],representatives={'grid-directional':a,'count-grid':b})

    def test_broad_competition_does_not_score_incompatible_inputs_as_model_errors_or_success(self):
        p,a,b=population();compete=self.broad_competitor(p)
        result=compete(environments=[{'id':'foreign-text','input_type':'text',
                                      'partition':'development','manifest':'typed-foreign-fault-fixture'}])
        self.assertTrue(result['execution_receipts'])
        for row in result['execution_receipts']:
            self.assertEqual(row['status'],'inapplicable')
            self.assertIsNone(row['correct'])
            self.assertEqual(row['scored_tasks'],0)
            self.assertTrue(row['reason'])

    def test_broad_competition_advances_independently_of_blocked_family_and_moe(self):
        p,environment,evaluate,calls=self.executed_broad_fixture();compete=self.broad_competitor(p)
        with tempfile.TemporaryDirectory(prefix='architecture_competition_') as tmp:
            root=pathlib.Path(tmp)
            atomic(root/'FAMILY_STATUS.json',{'status':'blocked_missing_author'})
            atomic(root/'MOE_STATUS.json',{'status':'awaiting_external_round_anchor'})
            result=compete(environments=[environment],evaluator=evaluate,status_root=root)
            self.assert_executed_broad(result,calls)
            self.assertEqual(json.loads((root/'FAMILY_STATUS.json').read_text())['status'],'blocked_missing_author')
            self.assertEqual(json.loads((root/'MOE_STATUS.json').read_text())['status'],'awaiting_external_round_anchor')

    def test_multiple_hard_niches_have_independent_service_queue_interface(self):
        # Current controller is the actual production queue, not a substitute scheduler.
        # Lack of a niche queue makes this an explicit missing-interface failure.
        import inspect
        parameters=inspect.signature(controller.drive).parameters
        self.assertTrue(any(k in parameters for k in ('niches','registry','niche_roots','scheduler')),
             'drive(root,kind,...) exposes one fixed niche; no shared registry/multi-niche queue for concurrent blocked-niche isolation')


    def test_unscored_only_broad_receipts_do_not_advance_competition_progress(self):
        p,a,b=population();before=p.competition_iteration
        result=p.compete(environments=[{'id':'grid-without-tasks','partition':'development','input_type':'grid','manifest':'fault-fixture'}])
        self.assertTrue(all(r['status']=='unscored' and r['scored_tasks']==0 for r in result['execution_receipts']))
        self.assertEqual(p.competition_iteration,before,'zero evaluated tasks counted as broad competition progress')
        self.assertEqual(result['competition_iteration'],before)

    def test_inapplicable_only_broad_receipts_do_not_advance_competition_progress(self):
        p,a,b=population();before=p.competition_iteration
        result=p.compete(environments=[{'id':'foreign-only','partition':'development','input_type':'text','manifest':'fault-fixture'}])
        self.assertTrue(all(r['status']=='inapplicable' and r['scored_tasks']==0 for r in result['execution_receipts']))
        self.assertEqual(p.competition_iteration,before,'no compatible execution counted as broad competition progress')

    def crossover_fault(self,mutate=False,model='gpt-6.1-sol'):
        p=CellPopulation();left={'W':[[1,0]],'b':[0]};right={'W':[[0,2]],'b':[2]}
        a=p.seed('grid-directional',left,{'source':'fault-fixture'},scores())
        b=p.seed('grid-directional',right,{'source':'fault-fixture'},scores())
        inherited={'W':[[1,0]],'b':[2]};applied=copy.deepcopy(inherited)
        if mutate:applied['W'][0][1]=.25
        receipt={'recipient_hash':digest(left),'donor_hash':digest(right),'child_hash':digest(inherited),
                 'mapping':{'tensor_sources':{'W':'recipient','b':'donor'}}}
        raw={'model':model,'tensors':applied}
        provenance={'model':model,'reasoning':'low','inheritance':receipt,'mutation_parent_hash':digest(inherited),
                    'raw_response':raw,'newly_authored':['W[0][1]'],'typed_io':'fault-fixture'}
        return p,a,b,applied,provenance

    def test_unchanged_inherited_child_is_not_fresh_llm_mutation(self):
        p,a,b,applied,provenance=self.crossover_fault()
        self.assertTrue(rejected_or_raised(lambda:p.birth('grid-directional',applied,[a,b],provenance,scores(True),
            caps_passed=True,fixtures_passed=True,arm='llm_crossover')),'unchanged inherited tensors admitted as fresh mutation')

    def test_invalid_model_label_cannot_author_fresh_crossover_mutation(self):
        p,a,b,applied,provenance=self.crossover_fault(mutate=True,model='not-an-LLM')
        self.assertTrue(rejected_or_raised(lambda:p.birth('grid-directional',applied,[a,b],provenance,scores(True),
            caps_passed=True,fixtures_passed=True,arm='llm_crossover')),'arbitrary nonempty model label counted as LLM authorship')

    def test_requested_model_string_without_actual_bound_author_receipt_cannot_admit(self):
        p,a,b,applied,provenance=self.crossover_fault(mutate=True)
        # Correct model string and matching literals still do not prove actual author invocation.
        self.assertTrue(rejected_or_raised(lambda:p.birth('grid-directional',applied,[a,b],provenance,scores(True),
            caps_passed=True,fixtures_passed=True,arm='llm_crossover')),'model string/raw literals replaced actual parent/prompt/response/exit receipt binding')

    def test_self_asserted_receipt_with_wrong_parent_commitment_cannot_admit(self):
        p,a,b,applied,provenance=self.crossover_fault(mutate=True)
        provenance['author_receipt']={'model':'gpt-6.1-sol','auth_verified':True,'returncode':0,
            'parent_state_hash':'0'*64,'prompt_sha256':'1'*64,'response_sha256':digest(provenance['raw_response']),
            'independent_commitment':'2'*64}
        self.assertTrue(rejected_or_raised(lambda:p.birth('grid-directional',applied,[a,b],provenance,scores(True),
            caps_passed=True,fixtures_passed=True,arm='llm_crossover')),'self-asserted receipt with wrong parent was accepted')


class AuthorBoundary(RuntimeError):pass


class TestActualArithmeticArchitecture(unittest.TestCase):
    """Actual service semantics with local fault receipts, never certified authorship."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='architecture_arithmetic_');self.addCleanup(self.tmp.cleanup)
        base=pathlib.Path(self.tmp.name);self.root=base/'run';self.control=base/'recorder'
        self.anchor=arithmetic.initialize(self.root,seeds=(401,))
        self.state=arithmetic.read(self.root/'STATE-0.json')
        self.rows=arithmetic.prompts(self.state)

    def response(self,row=None,mutate=True):
        row=row or self.rows[0];p=row['prompt'];params=copy.deepcopy(p['base_params'])
        if mutate:params['sum.b'][0]+=.1
        return {'seed':row['seed'],'model':arithmetic.MODEL,'reasoning':arithmetic.REASONING,
                'interface':arithmetic.adapter.INTERFACE,'parent_id':p['parent_id'],'parent_params':p['parent_params'],
                'base_sha256':p['base_sha256'],'prompt_sha256':row['prompt_sha256'],
                'feedback_sha256':row['feedback_sha256'],'charged_positions':12,'params':params}

    def stage_fault_receipts(self):
        arithmetic.atomic(self.root/'PROMPTS-1.json',self.rows)
        work=self.root/'AUTHOR-1';work.mkdir()
        arithmetic.atomic(work/'response.json',[self.response()])
        # Explicit fabricated fault input: there was no real author invocation.
        arithmetic.atomic(work/'START.json',{'model':arithmetic.MODEL,'reasoning':arithmetic.REASONING,
             'auth':'ChatGPT subscription','auth_verified':True,'paid_api_fallback':False,
             'parent_state_hash':arithmetic.digest(self.state),'prompts_hash':arithmetic.digest(self.rows)})
        arithmetic.atomic(work/'EXIT.json',{'returncode':0,'author_seconds':0.0})
        return work

    def crossed_row(self):
        state=copy.deepcopy(self.state);pop=state['contexts'][0]['arms']['llm']
        recipient=state['cells'][pop['live']];donor=copy.deepcopy(recipient)
        donor['params']['carry.W']=[[1.01]];donor['params']['carry.b']=[-9.25]
        donor['id']=arithmetic.identity(donor);donor['evidence']=arithmetic.grade(donor)
        self.assertFalse(donor['evidence']['errors'])
        state['cells'][donor['id']]=donor;pop['history'].append(donor['id'])
        row=arithmetic.prompts(state)[0]
        self.assertEqual(row['prompt']['crossover']['status'],'measured_compatible_crossover')
        self.assertEqual(set(row['prompt']['crossover']['parents']),{recipient['id'],donor['id']})
        return row

    def test_actual_arithmetic_two_parent_crossover_then_mutation_retains_both_distinguishing_contributions(self):
        row=self.crossed_row();response=self.response(row)
        cell=arithmetic.mutation_cell(response,row)
        self.assertNotEqual(cell['params'],row['prompt']['base_params'])
        cross=row['prompt']['crossover'];contributed=set()
        for item in cross['inheritance']:
            if item['position'] in cross['distinguishing_positions']:
                value=cell['params'][item['parameter']]
                for index in item['index']:value=value[index]
                if value==item['value']:contributed.add(item['parent'])
        self.assertEqual(contributed,{'left','right'})

    def test_actual_arithmetic_single_parent_numerics_cannot_replace_labelled_two_parent_mutation(self):
        row=self.crossed_row();response=self.response(row)
        for item in row['prompt']['crossover']['inheritance']:
            if item['position'] not in row['prompt']['crossover']['distinguishing_positions']:continue
            parent=row['prompt']['parent_params'][item['parameter']]
            for index in item['index']:parent=parent[index]
            dest=response['params'][item['parameter']]
            for index in item['index'][:-1]:dest=dest[index]
            dest[item['index'][-1]]=parent
        with self.assertRaises(ValueError):arithmetic.mutation_cell(response,row)

    def test_actual_arithmetic_rejects_unchanged_inherited_base(self):
        with self.assertRaises(ValueError):arithmetic.mutation_cell(self.response(mutate=False),self.rows[0])

    def test_actual_arithmetic_rejects_wrong_parent_prompt_and_feedback_hashes(self):
        for field in ('parent_id','prompt_sha256','feedback_sha256','base_sha256','model'):
            with self.subTest(field=field):
                response=self.response();response[field]='forged'
                with self.assertRaises(ValueError):arithmetic.mutation_cell(response,self.rows[0])

    def test_actual_arithmetic_recorder_rejects_correct_model_labels_without_invocation_receipt(self):
        self.stage_fault_receipts()
        with self.assertRaises(ValueError,msg='self-asserted START/EXIT flags alone cannot certify actual author invocation'):
            arithmetic.record(self.root,self.control,self.anchor,1)
        self.assertFalse((self.control/'GENERATION-1.json').exists())

    def test_actual_arithmetic_recorder_rejects_parent_state_receipt_mismatch(self):
        work=self.stage_fault_receipts();start=arithmetic.read(work/'START.json');start['parent_state_hash']='0'*64
        arithmetic.atomic(work/'START.json',start)
        with self.assertRaises(ValueError):arithmetic.record(self.root,self.control,self.anchor,1)
        self.assertFalse((self.control/'GENERATION-1.json').exists())

    def test_actual_arithmetic_evaluator_rejects_raw_bytes_changed_after_commitment(self):
        work=self.stage_fault_receipts()
        # Construct a fault commitment without asserting successful author provenance.
        event={'anchor':self.anchor,'generation':1,'parent_state_hash':arithmetic.digest(self.state),
          'prompts_hash':arithmetic.digest(self.rows),'raw_response_sha256':arithmetic.file_hash(work/'response.json'),
          'start_sha256':arithmetic.file_hash(work/'START.json'),'exit_sha256':arithmetic.file_hash(work/'EXIT.json')}
        self.control.mkdir();arithmetic.atomic(self.control/'GENERATION-1.json',event)
        response=self.response();response['params']['sum.b'][0]+=.1
        arithmetic.atomic(work/'response.json',[response])
        with self.assertRaises(ValueError):arithmetic.evaluate(self.root,self.control,self.anchor,1)
        self.assertFalse((self.root/'CHARGES-1.jsonl').exists())

    def test_actual_arithmetic_run_reaches_generation_seven_author_boundary(self):
        # Synthetic checkpoint exercises the actual unbounded loop, not learned six-generation history.
        state=copy.deepcopy(self.state);state['generation']=6
        arithmetic.atomic(self.root/'STATE-6.json',state)
        with patch.object(arithmetic,'author',side_effect=AuthorBoundary('provider boundary intercepted')) as author:
            with self.assertRaises(AuthorBoundary):arithmetic.run(self.root,self.control,self.anchor,max_seconds=5,min_free_bytes=0)
        self.assertEqual(author.call_count,1);self.assertEqual(author.call_args.args[1],7)
        self.assertTrue((self.root/'PROMPTS-7.json').exists());self.assertFalse((self.root/'SERVICE.lock').exists())


class TestActualMultiNicheArchitecture(unittest.TestCase):
    """Real scheduler, real numerical adapter evaluation; no real LLM provider."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='architecture_queue_');self.addCleanup(self.tmp.cleanup)
        self.path=pathlib.Path(self.tmp.name)/'queue.json'

    def advance(self,queue,predicate):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            state=queue.tick()
            if predicate(state['niches']):return state
            threading.Event().wait(.002)
        self.fail('actual queue did not advance: '+repr(queue.status()))

    def evaluate(self,niche,proposal):
        evidence=arithmetic.grade(proposal)
        return {'payload':proposal,'partition':'development','numeric_evaluation':evidence,'saturated':False}

    def test_actual_queue_two_niches_numerically_evaluate_despite_blocked_author(self):
        calls=[]
        def author(niche,payload):
            calls.append(niche)
            if niche=='blocked':raise NicheWaiting('missing_fresh_author',retry_after=3600)
            return copy.deepcopy(payload)
        queue=MultiNicheQueue(self.path,author,self.evaluate)
        try:
            for name in ('blocked','borrow-chain','underflow'):
                queue.add_niche(name,copy.deepcopy(arithmetic.adapter.BASELINE_CELL))
            state=self.advance(queue,lambda rows:rows['borrow-chain']['generation']>=1 and rows['underflow']['generation']>=1 and rows['blocked']['status']=='waiting')
            self.assertEqual(state['niches']['blocked']['generation'],0)
            self.assertEqual(state['niches']['blocked']['reason'],'missing_fresh_author')
            for name in ('borrow-chain','underflow'):
                evidence=state['niches'][name]['last_receipt']['numeric_evaluation']
                self.assertTrue(evidence['traces']);self.assertEqual(evidence['partition'],'development')
                self.assertTrue(any(t['id']=='dev-chain' for t in evidence['traces']))
        finally:queue.close()

    def test_actual_queue_saturated_niche_cannot_monopolize_author_budget(self):
        calls=[];queue=MultiNicheQueue(self.path,lambda n,p:calls.append(n) or p,self.evaluate)
        try:
            queue.add_niche('saturated-easy',copy.deepcopy(arithmetic.adapter.BASELINE_CELL),saturated=True)
            queue.add_niche('hard-borrow',copy.deepcopy(arithmetic.adapter.BASELINE_CELL))
            state=self.advance(queue,lambda rows:rows['hard-borrow']['generation']>=1)
            self.assertNotIn('saturated-easy',calls)
            self.assertEqual(state['niches']['saturated-easy']['dispatches']['author'],0)
            self.assertGreater(state['niches']['hard-borrow']['dispatches']['evaluate'],0)
        finally:queue.close()

    def test_actual_queue_numerical_evaluation_continues_past_six_generations(self):
        queue=MultiNicheQueue(self.path,lambda n,p:p,self.evaluate)
        try:
            queue.add_niche('hard-borrow',copy.deepcopy(arithmetic.adapter.BASELINE_CELL))
            state=self.advance(queue,lambda rows:rows['hard-borrow']['generation']>=7)
            self.assertGreaterEqual(state['niches']['hard-borrow']['dispatches']['evaluate'],7)
            self.assertTrue(state['niches']['hard-borrow']['last_receipt']['numeric_evaluation']['traces'])
        finally:queue.close()

    def test_actual_queue_unexecuted_evaluation_receipt_cannot_count_as_progress(self):
        queue=MultiNicheQueue(self.path,lambda n,p:p,lambda n,p:{'status':'unscored','scored_tasks':0,'payload':p})
        try:
            queue.add_niche('hard-borrow',copy.deepcopy(arithmetic.adapter.BASELINE_CELL))
            state=self.advance(queue,lambda rows:rows['hard-borrow']['generation']>=1 or rows['hard-borrow']['status']=='waiting')
            self.assertEqual(state['niches']['hard-borrow']['generation'],0,'queue promoted an unscored callback receipt into evaluated generation progress')
        finally:queue.close()


if __name__=='__main__':unittest.main()
