import pytest
from cellforge.construction.living_population import LivingPopulation


def evaluation(key,solves,**changes):
    return dict({'genotype_hash':key,'task_manifest':'fixed-tasks-v1','solves':solves},**changes)


def population(capacity=4,max_age=3):
    return LivingPopulation(capacity=capacity,max_age=max_age,task_manifest='fixed-tasks-v1',n_tasks=3)


def test_shared_rewards_conserve_one_per_solved_task_and_niche_novelty():
    p=population(); p.admit([evaluation('a',[True,True,False]),evaluation('b',[True,False,False]),evaluation('c',[False,False,True])])
    rewards=p.metrics()['shared_rewards']
    assert rewards=={'a':1.5,'b':.5,'c':1.}
    assert sum(rewards.values())==3
    assert p.metrics()['novelty']['c']>p.metrics()['novelty']['a']


def test_age_evicts_elite_even_after_reevaluation_and_retains_archive():
    p=population(max_age=2); p.admit([evaluation('elite',[True]*3)])
    p.advance_age(); p.admit([evaluation('elite',[True]*3)])
    assert p.members[0].age==1
    p.advance_age(); assert p.members==()
    assert len(p.archive)==2 and any(e['event']=='age_eviction' for e in p.events)
    with pytest.raises(ValueError): p.sample_parents(seed=4)


def test_capacity_hash_ties_and_failed_archive_no_living_reward():
    p=population(capacity=2)
    p.admit([evaluation(k,[True,False,False]) for k in ('c','b','a')]+[evaluation('failed',[True]*3,failed=True)])
    assert [m.genotype_hash for m in p.members]==['a','b']
    assert len(p.archive)==4 and p.archive[-1]['failed']
    assert set(p.metrics()['shared_rewards'])=={'a','b'}
    assert any(e['event']=='capacity_eviction' and e['genotype_hash']=='c' for e in p.events)


def test_sampling_deterministic_contemporary_only_and_input_order_independent():
    entries=[evaluation('a',[True,False,False]),evaluation('b',[False,True,False]),evaluation('c',[False,False,True])]
    a=population(); b=population()
    a.admit(entries); b.admit(list(reversed(entries)))
    assert a.sample_parents(seed=42,count=20)==b.sample_parents(seed=42,count=20)
    a.advance_age(); a.advance_age(); a.advance_age()
    a.admit([evaluation('new',[False]*3)])
    result=a.sample_parents(seed=42,count=20)
    assert set(result['parent_hashes'])=={'new'} and result['probabilities']=={'new':1.}
    assert a.metrics()['archive_genotypes']==4


@pytest.mark.parametrize('changes',[{'task_manifest':'other'}, {'solves':[True]}, {'solves':[1,0,0]}, {'failed':1}, {'genotype_hash':''}])
def test_invalid_manifest_or_vectors_rejected_atomically(changes):
    p=population()
    invalid=dict(evaluation('bad',[False]*3),**changes)
    with pytest.raises(ValueError): p.admit([evaluation('good',[True]*3),invalid])
    assert not p.members and not p.archive


def test_archive_and_member_views_cannot_modify_population():
    p=population(); p.admit([evaluation('a',[True]*3)])
    p.archive[0]['failed']=True
    assert not p.archive[0]['failed']
    assert p.metrics()['recombination_implemented'] is False
