import copy
import unittest
from cellforge.construction.cell_population import CellPopulation, require_retained_material
from cellforge.construction.joint_weight_population import digest, inherit_neighbors


def scores(hit=False):
    return {k: {'partition':'development', 'manifest':k, 'source':'public-training-real' if k=='anchor' else 'synthetic-fault-fixture',
                'outcomes':{'a':True,'b':hit}, 'false_positive_given_consistency':0.0}
            for k in ('fresh','archive','anchor')}


class PopulationLineageValidationTests(unittest.TestCase):
    def pair(self):
        p=CellPopulation(author_verifier=lambda receipt,expected:True)
        r={'r:conv.W':[[1,0,0,0] for _ in range(9)], 'r:conv.b':[0], 'r:paint':[0,1,0,0]}
        d={'d:cnt.W':[[0,2,0,0]], 'd:cnt.b':[-1]}
        a=p.seed('conv',r,{'source':'fixture'},scores(),family='spatial',niche='neighbors',niche_manifest='frozen-neighbors')
        b=p.seed('count',d,{'source':'fixture'},scores(),family='arithmetic',niche='counts',niche_manifest='frozen-counts')
        child,receipt=inherit_neighbors(r,d,recipient_prefix='r:',donor_prefix='d:')
        return p,a,b,child,receipt

    def birth(self,p,a,b,child,receipt):
        return p.birth('conv',child,[a,b],{'inheritance':receipt},scores(True),
                       caps_passed=True,fixtures_passed=True,arm='crossover')

    def test_replayed_crossover_admits_actual_parent_material(self):
        p,a,b,child,receipt=self.pair()
        self.assertTrue(self.birth(p,a,b,child,receipt)['accepted'])
        c=p.live['conv'][0]
        self.assertEqual(p.cells[c]['family'],'spatial')
        self.assertEqual(p.cells[c]['niche'],'neighbors')
        self.assertEqual(p.cells[c]['niche_manifest'],'frozen-neighbors')
        self.assertEqual(p.cells[a]['state'],'retired')
        self.assertEqual(p.cells[b]['state'],'live')

    def test_forged_receipt_charged_without_registry_mutation(self):
        p,a,b,child,receipt=self.pair();before=copy.deepcopy(p.cells)
        receipt['donor_hash']=digest({'wrong':[1]})
        self.assertTrue(self.birth(p,a,b,child,receipt)['invalid'])
        self.assertEqual(p.cells,before);self.assertEqual(len(p.charges),1)

    def test_duplicate_unknown_and_unregistered_live_parents_are_charged(self):
        for kind in ('duplicate','unknown','missing-live'):
            p,a,b,child,receipt=self.pair()
            if kind=='duplicate':b=a
            if kind=='unknown':b='unknown'
            if kind=='missing-live':p.live['count'].remove(b)
            self.assertFalse(self.birth(p,a,b,child,receipt)['accepted'])
            self.assertEqual(len(p.charges),1)

    def test_literal_llm_response_and_inherited_hash_admit_valid_mutation(self):
        p,a,b,child,receipt=self.pair();inherited=copy.deepcopy(child)
        child['r:conv.W'][4][0]=1.25
        provenance={'inheritance':receipt,'mutation_parent_hash':digest(inherited),
                    'raw_response':{'tensors':child},'model':'gpt-6.1-sol','reasoning':'low',
                    'prompt':{'mutation_parent_hash':digest(inherited)},
                    'newly_authored':['r:conv.W[4][0]'],'typed_io':'four-color-grid'}
        provenance['author_receipt']={'model':'gpt-6.1-sol','reasoning':'low','parent_ids':[a,b],
            'parent_hashes':[digest(p.cells[c]['params']) for c in (a,b)],
            'mutation_parent_hash':digest(inherited),'prompt_hash':digest(provenance['prompt']),
            'response_hash':digest(provenance['raw_response']),'returncode':0,'invocation_commitment':'1'*64}
        result=p.birth('conv',child,[a,b],provenance,scores(True),caps_passed=True,fixtures_passed=True,arm='llm_crossover')
        self.assertTrue(result['accepted'])
        self.assertEqual(p.cells[p.live['conv'][0]]['provenance'],provenance)

    def test_post_crossover_unchanged_tensors_or_unrequested_model_cannot_admit(self):
        for kind in ('unchanged','wrong-model','wrong-reasoning','wrong-prompt'):
            p,a,b,child,receipt=self.pair();inherited=copy.deepcopy(child)
            if kind!='unchanged':child['r:conv.W'][4][0]=1.25
            provenance={'inheritance':receipt,'mutation_parent_hash':digest(inherited),
                        'raw_response':{'tensors':child},'model':'gpt-6.1-sol','reasoning':'low',
                        'prompt':{'mutation_parent_hash':digest(inherited)},
                        'newly_authored':['r:conv.W[4][0]'],'typed_io':'four-color-grid'}
            if kind=='wrong-model':provenance['model']='not-an-LLM'
            if kind=='wrong-reasoning':provenance['reasoning']='high'
            if kind=='wrong-prompt':provenance['prompt']['mutation_parent_hash']='wrong'
            result=p.birth('conv',child,[a,b],provenance,scores(True),caps_passed=True,
                           fixtures_passed=True,arm='llm_crossover')
            self.assertTrue(result['invalid']);self.assertFalse(result['accepted'])
            self.assertEqual(len(p.charges),1)
    def test_self_asserted_correct_receipt_requires_external_host_verification(self):
        for verified in (None, lambda receipt,expected:False):
            p,a,b,child,receipt=self.pair();p.author_verifier=verified
            inherited=copy.deepcopy(child);child['r:conv.W'][4][0]=1.25
            provenance={'inheritance':receipt,'mutation_parent_hash':digest(inherited),
                        'raw_response':{'tensors':child},'model':'gpt-6.1-sol','reasoning':'low',
                        'prompt':{'mutation_parent_hash':digest(inherited)},
                        'newly_authored':['r:conv.W[4][0]'],'typed_io':'four-color-grid'}
            provenance['author_receipt']={'model':'gpt-6.1-sol','reasoning':'low','parent_ids':[a,b],
                'parent_hashes':[digest(p.cells[c]['params']) for c in (a,b)],
                'mutation_parent_hash':digest(inherited),'prompt_hash':digest(provenance['prompt']),
                'response_hash':digest(provenance['raw_response']),'returncode':0,'invocation_commitment':'1'*64}
            result=p.birth('conv',child,[a,b],provenance,scores(True),caps_passed=True,
                           fixtures_passed=True,arm='llm_crossover')
            self.assertTrue(result['invalid']);self.assertFalse(result['accepted'])
            self.assertIn('trusted independent author',result['reasons'][0])
            self.assertEqual(len(p.charges),1)
    def test_mutation_cannot_erase_a_parent_or_hide_edits_outside_authored_positions(self):
        for kind in ('donor','recipient','undeclared'):
            p,a,b,child,receipt=self.pair();inherited=copy.deepcopy(child)
            if kind=='donor':
                child=copy.deepcopy(p.cells[a]['params']);authored=['r:conv.W','r:conv.b']
            elif kind=='recipient':
                child['r:conv.W'][4]=[9,9,9,9];child['r:paint']=[9,9,9,9]
                authored=['r:conv.W','r:paint']
            else:
                child['r:conv.W'][4][0]=1.25;child['r:conv.b'][0]=2
                authored=['r:conv.W[4][0]']
            with self.assertRaises(ValueError):
                require_retained_material(p.cells[a]['params'],inherited,child,receipt,authored)
    def test_snapshot_is_detached_and_restoration_rejects_live_omission_or_duplicate(self):
        p,a,b,child,receipt=self.pair();snap=p.snapshot();snap['cells'][a]['family']='changed'
        self.assertEqual(p.cells[a]['family'],'spatial')
        for duplicate in (False,True):
            snap=p.snapshot()
            if duplicate:snap['live']['conv'].append(a)
            else:snap['live']['conv'].remove(a)
            with self.assertRaises(ValueError):CellPopulation.from_snapshot(snap)

    def test_competition_executes_target_stripped_inputs_and_preserves_niches(self):
        p,a,b,child,receipt=self.pair();calls=[];live=copy.deepcopy(p.live)
        def executor(cell,task):
            self.assertNotIn('expected',task);self.assertNotIn('id',task);calls.append(cell['id'])
            return sum(cell['params'].get('d:cnt.W',[[0]])[0])
        result=p.compete(environments=[{'id':'frozen','partition':'development','manifest':'fixture',
                            'tasks':[{'id':'t','input':[[1]],'expected':2}]}],evaluator=executor)
        self.assertEqual(set(calls),{a,b});self.assertEqual(p.live,live)
        self.assertTrue(all(r['status']=='executed' and r['scored_tasks']==1 for r in result['execution_receipts']))
        self.assertEqual(sum(r['correct'] for r in result['execution_receipts']),1)
        self.assertFalse(result['scientific_acceptance'])

    def test_same_structure_niches_keep_independent_archives(self):
        p=CellPopulation(author_verifier=lambda receipt,expected:True)
        left=scores();right=scores()
        for section in right.values():section['manifest']='other-'+section['manifest']
        a=p.seed('grid',{'W':[0]},{'source':'fixture'},left,niche='spatial')
        b=p.seed('grid',{'W':[1]},{'source':'fixture'},right,niche='arithmetic')
        decision=p.birth('grid',{'W':[2]},[a],{'source':'fixture'},scores(True),
                         caps_passed=True,fixtures_passed=True,arm='random')
        self.assertTrue(decision['accepted']);self.assertEqual(p.cells[b]['state'],'live')
        restored=CellPopulation.from_snapshot(p.snapshot())
        self.assertEqual(len(restored.histories),2)
        self.assertEqual(len(restored.histories[restored.cells[b]['history_key']].records),1)
    def test_missing_executor_cannot_claim_executed_accuracy(self):
        p,*_=self.pair()
        result=p.compete(environments=[{'id':'no-tasks','partition':'development'}])
        self.assertFalse(result['accepted']);self.assertFalse(result['admission'])
        self.assertEqual(result['competition_iteration'],0)
        for row in result['execution_receipts']:
            self.assertEqual(row['status'],'unscored');self.assertIsNone(row['correct'])
            self.assertEqual(row['scored_tasks'],0)


if __name__=='__main__':unittest.main()








