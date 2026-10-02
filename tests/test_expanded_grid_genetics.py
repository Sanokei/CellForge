import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import expanded_grid_genetics as genetics
from cellforge.ir.graph import Graph


@pytest.fixture
def base():return Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/expanded-seven-slot-v1/expanded')


def test_atomic_slot_repair_and_stale_record_rejection(base):
    from cellforge.construction.transactions import apply_ops, TransactionError
    parent_hash = base.hash()
    child, _ = apply_ops(base, [{"op": "set_slot_gene", "slot": 10, "gene": "geometry_0"}])
    genetics.canonical(base, child)
    assert child.meta['expanded_genome']['parent_hashes'] == [parent_hash]
    assert base.hash() == parent_hash
    neutral, _ = apply_ops(child, [{"op": "set_slot_gene", "slot": 10, "gene": "geometry_0"}])
    assert neutral.hash() == child.hash()
    for op in ({"op": "set_slot_gene", "slot": True, "gene": "geometry_0"},
               {"op": "set_slot_gene", "slot": 10, "gene": "missing"},
               {"op": "put_def", "name": "expanded_slot_10", "def": base.defs['geometry_0']}):
        with pytest.raises(TransactionError): apply_ops(base, [op])
        assert base.hash() == parent_hash


def test_new_gene_import_is_atomic_and_hash_bound(base):
    from cellforge.construction.transactions import apply_ops, TransactionError
    from cellforge.ir.profiles import check_profile
    from cellforge.runtime.grid import GridPredictor
    donor=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/tile2-v1')
    op=dict(op='import_slot_gene',slot=8,prefix='repair_tile_',name='repeat.gene',**{'from':donor.hash()})
    parent_hash=base.hash()
    child,_=apply_ops(base,[op],resolver=lambda ref:donor)
    genetics.validate_genome(child);check_profile(child,'ladder-micro/1')
    assert len(child.meta['expanded_genome']['available_genes'])==8
    assert child.meta['expanded_genome']['parent_hashes']==[parent_hash]
    assert child.meta['slot_gene_imports'][0]['donor_hash']==donor.hash()
    x=[[1,2],[3,4]];y=[[1,2,1,2],[3,4,3,4],[1,2,1,2],[3,4,3,4]]
    assert y in GridPredictor(child,backend='compiled')([{'input':x,'output':y}],x)
    assert base.hash()==parent_hash
    with pytest.raises(TransactionError):apply_ops(child,[op],resolver=lambda ref:donor)
    bad=dict(op);bad['from']='wrong'
    with pytest.raises(TransactionError):apply_ops(base,[bad],resolver=lambda ref:donor)
    assert base.hash()==parent_hash


def test_mutation_can_activate_eighth_imported_gene(base):
    from cellforge.construction.transactions import apply_ops
    donor=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/tile2-v1')
    imported,_=apply_ops(base,[dict(op='import_slot_gene',slot=8,
        prefix='mutation_test_',name='repeat.gene',**{'from':donor.hash()})],
        resolver=lambda ref:donor)
    parent=genetics.instantiate(imported,['geometry_0']*7)
    bank=parent.meta['expanded_genome']['available_genes']
    seed=next(seed for seed in range(100) if _second_draw(seed)==7)
    child=genetics.mutate(parent,seed)
    assert child.meta['expanded_genome']['new_allele']==bank[7]
    assert bank[7] in child.meta['expanded_genome']['alleles']
    genetics.canonical(parent,child)


def _second_draw(seed):
    import numpy as np
    rng=np.random.default_rng(seed)
    rng.integers(7)
    return int(rng.integers(8))


def test_multiple_typed_donors_coexist_and_bank_is_bounded(base):
    import copy
    from cellforge.construction.transactions import apply_ops, TransactionError, MAX_GRID_BANK_GENES
    from build_grid_marker_mirror_gene import build_marker_mirror
    tile=Graph.load(Path(__file__).parents[1]/'runs/arc-grid-genes/tile2-v1')
    mirror=build_marker_mirror()
    original=base.hash()
    child=base
    for slot,prefix,donor in [(8,'multi_tile_',tile),(9,'multi_mirror_',mirror)]:
        before=child.hash()
        child,_=apply_ops(child,[dict(op='import_slot_gene',slot=slot,
            prefix=prefix,name='repeat.gene',**{'from':donor.hash()})],
            resolver=lambda ref,d=donor:d)
        assert child.meta['expanded_genome']['parent_hashes']==[before]
    assert len(child.meta['expanded_genome']['available_genes'])==9
    assert len(child.meta['slot_gene_imports'])==2
    assert base.hash()==original
    genetics.canonical(child,genetics.crossover(child,child,123))
    full=child.copy()
    while len(full.meta['expanded_genome']['available_genes'])<MAX_GRID_BANK_GENES:
        name='capacity_alias_'+str(len(full.meta['expanded_genome']['available_genes']))
        full.defs[name]=copy.deepcopy(full.defs['geometry_0'])
        full.meta['expanded_genome']['available_genes'].append(name)
    genetics.validate_genome(full)
    full_hash=full.hash()
    with pytest.raises(TransactionError,match='capacity'):
        apply_ops(full,[dict(op='import_slot_gene',slot=10,prefix='overflow_',
            name='repeat.gene',**{'from':tile.hash()})],resolver=lambda ref:tile)
    assert full.hash()==full_hash


def test_lineage_does_not_change_genotype_and_contract_tampering_rejected(base):
    alleles=base.meta['expanded_genome']['alleles']
    a=genetics.instantiate(base,alleles,seed=1);b=genetics.instantiate(base,alleles,seed=2)
    assert a.hash()!=b.hash() and genetics.canonical(base,a)==genetics.canonical(base,b)
    bad=a.copy();bad.defs['expanded_slot_9']=bad.defs['geometry_0']
    with pytest.raises(ValueError):genetics.canonical(base,bad)
    bad=a.copy();bad.meta['grid_attempts']=1
    with pytest.raises(ValueError):genetics.canonical(base,bad)
    bad=a.copy();bad.meta['gradients']=True
    with pytest.raises(ValueError):genetics.canonical(base,bad)
    bad=a.copy();key=next(iter(bad.params));bad.params[key].flat[0]+=1
    with pytest.raises(ValueError):genetics.canonical(base,bad)


def test_seven_slot_recombination_mutation_lineage_and_parent_immutability(base):
    bank=base.meta['expanded_genome']['available_genes'];identity='geometry_0'
    left=genetics.instantiate(base,bank[:3]+[identity]*4)
    right=genetics.instantiate(base,[identity]*3+bank[3:]);hashes=[left.hash(),right.hash()]
    for seed in range(16):
        child=genetics.crossover(left,right,seed);genetics.canonical(base,child)
        assert child.hash()==genetics.crossover(left,right,seed).hash()
        r=child.meta['expanded_genome'];assert r['parent_hashes']==hashes
        for i,d in enumerate(r['donor_per_slot']):assert r['alleles'][i]==(left,right)[d].meta['expanded_genome']['alleles'][i]
        mutant=genetics.mutate(child,seed);genetics.canonical(base,mutant)
        changed=sum(x!=y for x,y in zip(r['alleles'],mutant.meta['expanded_genome']['alleles']))
        assert changed<=1 and mutant.meta['expanded_genome']['no_change']==(changed==0)
    assert [left.hash(),right.hash()]==hashes
