import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,r'C:\Users\wkeif\Documents\GitHub\CellForge\src')
from cellforge.construction.red_queen_history import OpponentHistory

def scores():
    return {k:dict(partition='development',manifest=k,source='public-training-real' if k=='anchor' else 'authored-generator',outcomes={'a':True,'b':False},false_positive_given_consistency=0.) for k in ('fresh','archive','anchor')}

class Tests(unittest.TestCase):
    def test_resume_and_tamper_rejection(self):
        import tempfile,json
        history=OpponentHistory();history.register(graph_hash='g',family='n',structure_hash='s',provenance={'actor':'fixture'},scores=scores())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'history.json';history.save(path)
            self.assertEqual(OpponentHistory.load(path).records,history.records)
            payload=json.loads(path.read_text());payload['records'][0]['family']='changed';path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):OpponentHistory.load(path)
    def test_all_predecessors_guarded_and_snapshot(self):
        parent=scores(); earlier=scores();earlier['archive']['outcomes']['b']=True
        history=OpponentHistory();history.register(graph_hash='earlier',family='mirror',structure_hash='s',provenance={'actor':'fixture'},scores=earlier)
        earlier['archive']['outcomes']['b']=False
        child=copy.deepcopy(parent);child.update(caps_passed=True,fixtures_passed=True);child['fresh']['outcomes']['b']=True
        result=history.admission(parent,child)
        self.assertFalse(result['accepted']);self.assertEqual(result['reasons'][-1]['gate'],'historical_opponent_regression')
        child['archive']['outcomes']['b']=True
        self.assertTrue(history.admission(parent,child)['accepted'])
    def test_report_and_contract_mismatch_fail_closed(self):
        history=OpponentHistory();old=scores();old['archive']['partition']='report'
        with self.assertRaises(ValueError):history.register(graph_hash='g',family='n',structure_hash='s',provenance={'actor':'fixture'},scores=old)
        old=scores();history.register(graph_hash='g',family='n',structure_hash='s',provenance={'actor':'fixture'},scores=old)
        child=scores();child.update(caps_passed=True,fixtures_passed=True);child['archive']['manifest']='different'
        with self.assertRaises(ValueError):history.admission(old,child)

if __name__=='__main__': unittest.main()

