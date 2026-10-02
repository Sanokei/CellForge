import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from evolve_grid_context_masks import masked_graph, offspring, FOUNDERS
from cellforge.runtime.compiled import CompiledInterpreter
from cellforge.runtime.grid import pack_grid


def test_projected_keys_match_scalar_with_padding_and_color_zero():
    x=[[0,9],[1,2]]
    xv,_,xh,xw=pack_grid(x)
    args=dict(input=xv,output=xv,input_height=xh,input_width=xw,
              output_height=xh,output_width=xw,phase=np.zeros((1,1)),state=np.zeros((1,3602)))
    for mask in FOUNDERS:
        g=masked_graph(mask)
        keys=CompiledInterpreter(g).run_bundle('context_memory_update',args,'/u')['keys'][0]
        for r in range(30):
            for c in range(30):
                scalar=0
                for i,(dr,dc) in enumerate(( (dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) )):
                    rr,cc=r-dr,c-dc
                    token=x[rr][cc]+1 if 0<=rr<2 and 0<=cc<2 else 0
                    if mask & (1<<i): scalar+=token*11**i
                assert keys[r*30+c]==scalar
        assert g.stats()['unique_param_storage']==23


def test_discrete_children_keep_typed_center_and_parent_provenance():
    rng=np.random.default_rng(1)
    for crossover in (False,True):
        for _ in range(20):
            mask=offspring(56,146,rng,crossover)
            assert 0<=mask<512 and mask & 16
            g=masked_graph(mask,['parent-a','parent-b'],'crossover')
            assert g.meta['parent_hashes']==['parent-a','parent-b']
    with pytest.raises(ValueError):masked_graph(0)
