import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_scaling_genome import build_scaling_base
from build_grid_two_slot_genome import instantiate_genome,crossover,mutate
from run_grid_population_matched import canonical_genotype
from cellforge.runtime.grid import GridPredictor


def test_seven_gene_bank_crossover_and_compiled_scaling_behavior():
    base=build_scaling_base();bank=base.meta['two_slot_genome']['available_genes']
    assert len(bank)==7 and base.stats()['unique_param_storage']==268
    left=instantiate_genome(base,('scaling_bank2_repeat.gene','geometry_0'))
    right=instantiate_genome(base,('geometry_0','scaling_bank3_repeat.gene'))
    children=[crossover(left,right,s) for s in range(12)]
    child=next(g for g in children if g.meta['two_slot_genome']['donor_per_slot']==[0,1])
    canonical_genotype(base,child)
    assert child.meta['two_slot_genome']['parent_hashes']==[left.hash(),right.hash()]
    for seed in range(12):canonical_genotype(base,mutate(child,seed))
    predictor=GridPredictor(child,backend='compiled')
    parents=[GridPredictor(g,backend='compiled') for g in (left,right)]
    x=[[0,2,1],[3,4,5]];query=[[6,0],[2,8]]
    for f in (2,3,2):
        scale=lambda a:np.repeat(np.repeat(a,f,axis=0),f,axis=1).tolist()
        demos=[dict(input=x,output=scale(x))]
        assert scale(query) in predictor(demos,query)
        assert [scale(query) in p(demos,query) for p in parents]==([True,False] if f==2 else [False,True])
