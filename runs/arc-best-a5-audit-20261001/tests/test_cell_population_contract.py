import unittest
from cellforge.construction.cell_population import CellPopulation
from cellforge.construction.population_contract import require_two_level_run
from cellforge.construction.patch_cell import predict_patch
def evidence(hit):
    return {k:{'partition':'development','manifest':k,'source':'public-training-real',
    'outcomes':{'a':hit,'b':False},'false_positive_given_consistency':0.0} for k in ('fresh','archive','anchor')}
class PopulationContractTests(unittest.TestCase):
    def test_selection_changes_live_population_and_keeps_old_opponent(self):
        p=CellPopulation();a=p.seed('x',{'W':[0]}, {'raw_response':'actual'},evidence(False))
        decision=p.birth('x',{'W':[1]},[a],{'raw_response':'actual'},evidence(True),caps_passed=True,fixtures_passed=True,arm='llm')
        self.assertTrue(decision['accepted']);self.assertEqual(p.cells[a]['state'],'retired')
        self.assertEqual(len(p.histories['x'].records),2)
        child=p.live['x'][0]
        decision=p.birth('x',{'W':[2]},[child],{'raw_response':'actual'},evidence(False),caps_passed=True,fixtures_passed=True,arm='llm')
        self.assertFalse(decision['accepted']);self.assertEqual(p.live['x'],[child]);self.assertEqual(len(p.charges),2)
    def test_unearned_cell_cannot_assemble(self):
        p=CellPopulation();p.seed('x',{'W':[0]},{'raw_response':'actual'},evidence(False))
        with self.assertRaises(ValueError):p.assemble({'route.W':[[1]],'route.b':[0]})
    def test_reporting_cannot_enter_archive(self):
        scores=evidence(True);scores['anchor']['partition']='reporting'
        with self.assertRaises(ValueError):CellPopulation().seed('x',{'W':[0]},{'raw_response':'actual'},scores)
    def test_old_ensemble_and_slots_fail_architecture_gate(self):
        for run in ({'rows':[{'arm':'llm'}]}, {'population':{'cells':{'slot':{}},'live':{},'histories':{}}}):
            with self.assertRaises(ValueError):require_two_level_run(run)
    def test_authored_patch_emits_snapshot_footprint_and_preserves_fallback(self):
        p={'patch.zero.W':[[1]+[0]*9],'patch.detect.b':[-8.5],'patch.paint':[0,1]+[0]*8,
           'patch.emit.W':[[1]*3 for _ in range(3)],'patch.emit.b':[-.5],'patch.guard.density':[.3]}
        grid=[[2]*5 for _ in range(5)]
        for r in range(1,4):
            for c in range(1,4):grid[r][c]=0
        out=predict_patch(p,grid,lambda _,g:g)
        self.assertEqual(out[1:4],[[2,1,1,1,2]]*3);self.assertEqual(grid[2][2],0)
        sparse=[[0]*5 for _ in range(5)];sparse[0][0]=2
        self.assertEqual(predict_patch(p,sparse,lambda _,g:g),sparse)


    def test_non_numeric_slots_and_unproven_structural_repairs_rejected(self):
        with self.assertRaises(ValueError):CellPopulation().seed('x',{'slot':['conv']},{'raw_response':'actual'},evidence(True))
        p=CellPopulation();a=p.seed('x',{'W':[0]},{'raw_response':'actual'},evidence(False))
        decision=p.birth('new-interface',{'W':[1]},[a],{'raw_response':'actual'},evidence(True),caps_passed=True,fixtures_passed=True,arm='llm_repair')
        self.assertTrue(decision['invalid']);self.assertEqual(len(p.charges),1)
        decision=p.birth('x',{'W':['socket']},[a],{'raw_response':'actual'},evidence(True),caps_passed=True,fixtures_passed=True,arm='llm')
        self.assertTrue(decision['invalid']);self.assertEqual(len(p.charges),2)

    def test_executed_child_predictions_and_numeric_parent_contributions(self):
        import json
        from pathlib import Path
        import numpy as np
        from cellforge.construction.typed_population import PaletteExecutor
        root=Path(__file__).resolve().parents[1];run=root/'runs/weights-cell-population-g1'
        population=json.loads((run/'population.json').read_text())
        results=json.loads((run/'summary.json').read_text())
        dev=json.loads((run/'development.json').read_text())[4:]
        core=PaletteExecutor(root/'runs/weights-real-learning-v2/pop1_core.py',10)
        patch=next(r for r in results['cells'] if r['arm']=='llm_patch')
        params=population['cells'][patch['cell']]['params']
        recomputed={}
        for row in dev:
            recomputed[row['id']]=all(predict_patch(params,pair['input'],lambda p,g:core.predict('conv3x3_detect',p,g))==pair['output'] for pair in row['task']['test'])
        self.assertEqual(recomputed,patch['scores']['fresh']['outcomes'])
        cross=next(r for r in results['cells'] if r['arm']=='crossover')
        child=population['cells'][cross['cell']]
        recipient=population['cells'][child['parents'][0]]['params'];donor=population['cells'][child['parents'][1]]['params']
        for i in (0,1,2,3,5,6,7,8):self.assertEqual(child['params']['conv.W'][i],donor['cnt.W'][0])
        self.assertEqual(child['params']['conv.W'][4],recipient['conv.W'][4])
        self.assertEqual(child['params']['paint'],recipient['paint'])
        self.assertFalse(np.array_equal(child['params']['conv.W'],recipient['conv.W']))
        receipt=json.loads((run/'two-level-receipt.json').read_text())
        self.assertEqual(len(receipt['ensemble_population']['charges']),1)
        for cid in receipt['assembly']['cell_ids']:
            self.assertEqual(population['cells'][cid]['state'],'live')
            self.assertTrue(any(v for sec in population['cells'][cid]['scores'].values() for v in sec['outcomes'].values()))
