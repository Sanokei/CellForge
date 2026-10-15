import sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_object_genome import build_object_base
from build_grid_two_slot_genome import instantiate_genome,crossover,mutate,assert_compatible
from run_grid_population_matched import canonical_genotype
from cellforge.ir.graph import Graph
from cellforge.ir.profiles import check_profile


def test_all_object_bank_slots_are_valid_and_same_capacity():
    base=build_object_base();genes=base.meta['two_slot_genome']['available_genes']
    sizes=set()
    for a in genes:
        for b in genes:
            child=instantiate_genome(base,(a,b))
            canonical_genotype(base,child);check_profile(child,'ladder-micro/1')
            sizes.add((child.stats()['unique_param_storage'],child.meta['state_dim']))
    assert len(genes)==5 and sizes=={(206,1010)}


def test_object_allele_recombines_and_mutation_can_reach_it():
    base=build_object_base();object_gene='object_bank_largest4.gene'
    a=instantiate_genome(base,(object_gene,'geometry_0'))
    b=instantiate_genome(base,('geometry_0','bank_adjacency_recolor.gene'))
    children=[crossover(a,b,i) for i in range(20)]
    assert any(x.meta['two_slot_genome']['alleles']==[object_gene,'bank_adjacency_recolor.gene'] for x in children)
    assert any(object_gene in mutate(b,i).meta['two_slot_genome']['alleles'] for i in range(30))
    duplicate=a.copy();duplicate.meta['neutral_lineage_note']='new metadata'
    assert canonical_genotype(base,a)==canonical_genotype(base,duplicate)
    duplicate.meta['two_slot_genome']['available_genes']=duplicate.meta['two_slot_genome']['available_genes'][:-1]
    with pytest.raises(ValueError,match='banks differ'):assert_compatible(a,duplicate)
