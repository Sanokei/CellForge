"""Fault-injection tests: runtime binding, not scientific or ARC evidence."""
import copy
import unittest
from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.joint_weight_population import digest, inherit_neighbors
from cellforge.construction.population_contract import (_measurement, _controls, _control_genome,
                                                       _lineage, _ensemble, require_two_level_run)


def forward(genome, request):
    return genome['params']['W'][0] * request['x']


def measured(genome, cases, executor=forward):
    predictions={c['id']:executor(copy.deepcopy(genome),copy.deepcopy(c['request'])) for c in cases}
    return {'partition':'development','genome_hash':digest(genome),'manifest_hash':digest(cases),
            'cases':copy.deepcopy(cases),'predictions':predictions,
            'outcomes':{c['id']:predictions[c['id']]==c['target'] for c in cases}}


class PopulationExecutionBindingTests(unittest.TestCase):
    def setUp(self):
        self.genome={'structure':'affine','params':{'W':[2]}}
        self.cases=[{'id':'fault-case','request':{'x':3},'target':6}]
        self.receipt=measured(self.genome,self.cases)

    def test_actual_weight_execution_reproduces_receipt(self):
        calls=[]
        def execute(genome,request):
            calls.append(request);return forward(genome,request)
        predictions,outcomes=_measurement(self.receipt,self.genome,execute)
        self.assertEqual(predictions,{'fault-case':6});self.assertEqual(outcomes,{'fault-case':True})
        self.assertEqual(calls,[{'x':3}])

    def test_rehashing_fabricated_prediction_does_not_make_it_executed(self):
        receipt=copy.deepcopy(self.receipt);receipt['predictions']['fault-case']=99
        receipt['outcomes']['fault-case']=True
        with self.assertRaisesRegex(ValueError,'do not reproduce'):_measurement(receipt,self.genome,forward)

    def test_wrong_genome_and_changed_manifest_rejected(self):
        other=copy.deepcopy(self.genome);other['params']['W'][0]=3
        with self.assertRaisesRegex(ValueError,'wrong literal genome'):_measurement(self.receipt,other,forward)
        receipt=copy.deepcopy(self.receipt);receipt['cases'][0]['request']['x']=4
        with self.assertRaisesRegex(ValueError,'frozen cases'):_measurement(receipt,self.genome,forward)

    def test_reporting_targets_and_duplicate_cases_cannot_enter_replay(self):
        for kind in ('reporting','target','duplicate'):
            receipt=copy.deepcopy(self.receipt)
            if kind=='reporting':receipt['partition']='reporting'
            if kind=='target':receipt['cases'][0]['request']['answer']=6
            if kind=='duplicate':receipt['cases'].append(copy.deepcopy(receipt['cases'][0]))
            receipt['manifest_hash']=digest(receipt['cases'])
            with self.assertRaises(ValueError):_measurement(receipt,self.genome,forward)

    def test_strict_boolean_outcomes_required_even_if_one_equals_true(self):
        receipt=copy.deepcopy(self.receipt);receipt['outcomes']['fault-case']=1
        with self.assertRaisesRegex(ValueError,'strict boolean'):_measurement(receipt,self.genome,forward)

    def test_control_label_cannot_replace_actual_zero_intervention(self):
        genome={'spec':{'nodes':[{'id':'glue','kind':'glue','params':{'W':[[2]],'b':[1]}}]},
                'founders':{'parent':{'layers':[{'W':[[3]],'b':[1]}]}}}
        def execute(g,request):return g['spec']['nodes'][0]['params']['W'][0][0]*request['x']
        bundle={'measurement':measured(genome,self.cases,execute),'controls':{}}
        wrong=copy.deepcopy(genome);wrong['spec']['nodes'][0]['params']['W'][0][0]=1
        bundle['controls']['zero']={'genome':wrong,'measurement':measured(wrong,self.cases,execute)}
        with self.assertRaisesRegex(ValueError,'control numerics'):_controls(bundle,genome,execute,['zero'])
        zero=_control_genome(genome,'zero')
        bundle['controls']['zero']={'genome':zero,'measurement':measured(zero,self.cases,execute)}
        self.assertEqual(_controls(bundle,genome,execute,['zero']),{'fault-case':6})

    def population(self):
        p=CellPopulation()
        scores={k:{'partition':'development','manifest':k,'source':'public-training-real',
                   'outcomes':{'a':True,'b':False},'false_positive_given_consistency':0.0}
                for k in ('fresh','archive','anchor')}
        r={'r:conv.W':[[1,0,0,0] for _ in range(9)],'r:conv.b':[0],'r:paint':[0,1,0,0]}
        d={'d:cnt.W':[[0,2,0,0]],'d:cnt.b':[-1]}
        author=lambda params:{'model':'gpt-6.1-sol','reasoning':'low','prompt':{'source':'fault-fixture'},'raw_response':{'tensors':params}}
        a=p.seed('conv',r,author(r),scores)
        b=p.seed('count',d,author(d),scores)
        child,receipt=inherit_neighbors(r,d,recipient_prefix='r:',donor_prefix='d:')
        result=p.birth('conv',child,[a,b],{'inheritance':receipt},scores,caps_passed=True,fixtures_passed=True,arm='crossover')
        self.assertFalse(result['accepted'])
        return p,a,b,child,receipt

    def test_rejected_but_executed_sexual_child_can_satisfy_lineage_coverage(self):
        p,a,b,child,receipt=self.population()
        ids=_lineage(p)
        self.assertEqual(len(ids),1);self.assertEqual(p.cells[ids[0]]['state'],'rejected')

    def test_forged_actual_parent_hash_in_snapshot_fails_lineage_gate(self):
        p,a,b,child,receipt=self.population()
        cid=next(c for c,v in p.cells.items() if v['parents'])
        p.cells[cid]['provenance']['inheritance']['recipient_hash']=digest({'W':[999]})
        with self.assertRaisesRegex(ValueError,'both actual parent'):_lineage(p)

    def test_fake_ensemble_history_and_label_only_birth_fail_full_gate(self):
        p,a,b,child,receipt=self.population()
        run={'population':p.snapshot(),'selection_partition':'development',
             'ensemble_population':{'histories':{'fake':['unexecuted']}},'assembly':{'cell_ids':[a,b]}}
        with self.assertRaisesRegex(ValueError,'execution bundle'):require_two_level_run(run)

    def test_receipt_bundle_cannot_claim_execution_without_trusted_replay(self):
        p,a,b,child,receipt=self.population()
        run={'population':p.snapshot(),'selection_partition':'development','assembly':{'cell_ids':[a,b]},
             'architecture_execution':{'format':'cellforge-two-level-execution/1'}}
        with self.assertRaisesRegex(ValueError,'trusted target-blind'):require_two_level_run(run)

    def moe(self):
        p=CellPopulation()
        scores={k:{'partition':'development','manifest':k,'source':'public-training-real',
                   'outcomes':{'fixture':False},'false_positive_given_consistency':0.0}
                for k in ('fresh','archive','anchor')}
        ids=[]
        for family, weight in (('comparison',2),('selection',3)):
            params={'W':[weight]}
            ids.append(p.seed(family,params,{'source':'fault-fixture'},scores))
        base={'cells':{c:p.cells[c]['params'] for c in ids},
              'router':{'route.W':[[1,0],[0,1]],'route.b':[0,0]},'topology':{'cell_order':ids}}
        candidate=copy.deepcopy(base);candidate['router']['route.W']=[[2,0],[0,2]]
        def execute(genome,request):
            return sum(genome['cells'][cid]['W'][0]*(sum(genome['router']['route.W'][i])+genome['router']['route.b'][i])
                       for i,cid in enumerate(genome['topology']['cell_order']))*request['x']
        cases=[{'id':'whole-moe-fault','request':{'x':1},'target':10}]
        a,b=digest(base),digest(candidate)
        def provenance(genome,parent=None):
            prompt={'cell_hashes':{cid:digest(params) for cid,params in genome['cells'].items()}}
            if parent:prompt['parent_hash']=parent
            return {'model':'gpt-6.1-sol','reasoning':'low','prompt':prompt,'parent_hash':parent,'raw_response':{'genome':genome}}
        archive={'live':a,'historical_opponents':[a],
                 'records':{a:{'genome':base,'state':'bootstrap-control','parents':[],
                               'outcomes':{'whole-moe-fault':False},'provenance':provenance(base)},
                            b:{'genome':candidate,'state':'rejected','parents':{'moe':a},
                               'outcomes':{'whole-moe-fault':True},'provenance':provenance(candidate,a)}},
                 'charges':[{'charged':True,'parent':a,'child':b}],
                 'events':[{'event':'proposal','id':b,'parent':a,'accepted':False}]}
        bundle={'measurement':measured(candidate,cases,execute),'controls':{}}
        for label in ['parent','random','zero','uniform-router']+['ablate-cell:'+cid for cid in ids]:
            control=base if label=='parent' else _control_genome(candidate,label,123)
            bundle['controls'][label]={'genome':control,'measurement':measured(control,cases,execute),'seed':123}
        execution={'ensemble':{a:{'measurement':measured(base,cases,execute)},b:bundle}}
        return p,archive,execution,execute,b

    def test_separate_whole_moe_executes_router_and_each_actual_cell_without_requiring_admission(self):
        p,archive,execution,execute,cid=self.moe()
        _ensemble({'ensemble_population':archive},execution,execute,p)
        self.assertEqual(archive['records'][cid]['state'],'rejected')

    def test_whole_moe_rejects_fabricated_parent_uniform_router_and_archive_membership(self):
        for kind in ('parent','uniform-router','rejected-history'):
            p,archive,execution,execute,cid=self.moe()
            if kind=='parent':execution['ensemble'][cid]['controls']['parent']['genome']['router']['route.b']=[99,99]
            if kind=='uniform-router':
                execution['ensemble'][cid]['controls']['uniform-router']['genome']['router']['route.b']=[99,99]
            if kind=='rejected-history':archive['historical_opponents'].append(cid)
            with self.assertRaises(ValueError):_ensemble({'ensemble_population':archive},execution,execute,p)

    def test_socket_or_circuit_spec_is_not_whole_moe_population(self):
        p,archive,execution,execute,cid=self.moe()
        first=archive['live'];del archive['records'][first]['genome']
        archive['records'][first]['spec']={'topology':'socket-fixture'}
        with self.assertRaisesRegex(ValueError,'whole-MoE requires'):
            _ensemble({'ensemble_population':archive},execution,execute,p)
    def test_labelled_post_crossover_llm_requires_exact_raw_tensor_response(self):
        p,a,b,child,receipt=self.population()
        cid=next(c for c,v in p.cells.items() if v['parents'])
        p.charges[0]['arm']='llm_crossover'
        p.cells[cid]['provenance'].update({'model':'gpt-6.1-sol','reasoning':'low','prompt':{'source':'fixture'},
           'raw_response':{'tensors':{'W':[99]}},'mutation_parent_hash':digest(child),
           'newly_authored':['r:conv.b'],'typed_io':'grid'})
        with self.assertRaisesRegex(ValueError,'raw author response'):_lineage(p)


if __name__=='__main__':unittest.main()


