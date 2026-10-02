import pytest
from scripts.build_deduction_specialist import build_deduction
from cellforge.runtime.deduction import HornDeductionPredictor
from cellforge.ir.graph import GraphError,Limits


def test_cycles_chain_permutations_unseen_names():
    graph = build_deduction()
    graph.validate(Limits(strict_microcells=True))
    predict = HornDeductionPredictor(graph)
    names = ['unseen_'+str(i) for i in range(32)]
    rules = [{'if':[names[i]],'then':names[i+1]} for i in range(31)]
    assert predict([names[0]],rules,names[-1])['entailed']
    assert predict([names[0]],list(reversed(rules)),names[-1])['entailed']
    assert predict([],rules,names[-1])['status'] == 'not_proven'
    cyclic = [{'if':['a'],'then':'b'},{'if':['b'],'then':'a'}]
    assert not predict([],cyclic,'a')['entailed']
    assert predict(['a'],cyclic,'b')['entailed']


def test_conjunction_empty_duplicate_and_provenance():
    predict = HornDeductionPredictor(build_deduction())
    rules = [{'if':[],'then':'a'},{'if':['a','b','b'],'then':'c'}]
    assert not predict([],rules,'c')['entailed']
    out = predict(['b'],rules,'c')
    assert out['entailed']
    assert out['fired_rules'] == [{'rule_index':0,'round':1},{'rule_index':1,'round':2}]
    assert predict(['known'],[],'known')['entailed']


@pytest.mark.parametrize('facts,rules,query',[
    (['!a'],[],'a'),(['not a'],[],'a'),([], [{'if':['a']*5,'then':'b'}],'b'),
    ([],[{'if':'a','then':'b'}],'b'),([], [{'if':[],'then':'a','x':1}],'a'),
    ([str(i) for i in range(33)],[],'q')])
def test_invalid_or_negation(facts,rules,query):
    with pytest.raises(GraphError):
        HornDeductionPredictor(build_deduction())(facts,rules,query)
