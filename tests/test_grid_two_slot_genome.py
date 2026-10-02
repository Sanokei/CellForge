import importlib.util
import sys
from pathlib import Path
import itertools
import numpy as np
from cellforge.ir.graph import Limits
from cellforge.ir.profiles import check_profile
from cellforge.runtime.grid import GridPredictor

scripts=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('genome_builder',scripts/'build_grid_two_slot_genome.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_all16_genomes_typed_profile_same_capacity_reference_compiled():
    base=builder.build_base();before=base.hash()
    train=[{'input':[[2,4,0],[0,1,0]],'output':[[2,0],[4,1],[0,0]]}]
    query=[[2,0,4],[0,1,0]]
    for genome in itertools.product(builder.GENES,repeat=2):
        child=builder.instantiate_genome(base,genome)
        child.validate(Limits(strict_microcells=True));check_profile(child,'ladder-micro/1')
        assert child.params.keys()==base.params.keys() and child.defs.keys()==base.defs.keys()
        assert child.meta['state_dim']==1010 and child.meta['grid_attempts']==2
        assert child.stats()['unique_param_storage']==base.stats()['unique_param_storage']==195
        for name in base.params:np.testing.assert_array_equal(child.params[name],base.params[name])
        assert all(base.defs[name]==child.defs[name] for name in base.defs if name not in builder.SLOTS)
        assert GridPredictor(child)(train,query)==GridPredictor(child,backend='compiled')(train,query)
    assert base.hash()==before


def test_homologous_crossover_and_one_slot_mutation_seed_lineage():
    base=builder.build_base();left=builder.instantiate_genome(base,(builder.GENES[2],builder.GENES[0]));right=builder.instantiate_genome(base,(builder.GENES[0],builder.GENES[3]))
    saw_nochange=False
    for seed in range(20):
        child=builder.crossover(left,right,seed)
        assert child.hash()==builder.crossover(left,right,seed).hash()
        record=child.meta['two_slot_genome'];assert record['parent_hashes']==[left.hash(),right.hash()]
        for i,gene in enumerate(record['alleles']):assert gene in (left.meta['two_slot_genome']['alleles'][i],right.meta['two_slot_genome']['alleles'][i])
        mutation=builder.mutate(left,seed);change=mutation.meta['two_slot_genome']
        assert mutation.hash()==builder.mutate(left,seed).hash()
        assert sum(a!=b for a,b in zip(change['alleles'],left.meta['two_slot_genome']['alleles']))<=1
        assert change['no_change']==(change['old_allele']==change['new_allele'])
        saw_nochange |= change['no_change']
    assert saw_nochange
