import importlib.util
import sys
from pathlib import Path
import numpy as np
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

scripts=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('recolor_builder',scripts/'build_grid_adjacency_recolor_gene.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_adjacency_independent_fixtures_zero_padding_corners_and_simultaneous():
    cases=[((2,3,4),[[2,4,2],[2,8,2],[8,2,8]],[[3,4,3],[2,8,2],[8,2,8]]),
           ((2,4,4),[[4,2,2,2]],[[4,4,2,2]]),
           ((2,3,0),[[2]],[[2]]),((2,3,0),[[2,0,9]],[[3,0,9]]),
           ((0,3,4),[[0,4,9],[9,0,9]],[[3,4,9],[9,3,9]]),
           ((2,0,4),[[2],[4],[2]],[[0],[4],[0]])]
    for colors,x,expected in cases:
        graph=builder.build_adjacency_recolor_gene(*colors)
        for backend in ('reference','compiled'):
            assert GridPredictor(graph,backend=backend)([{'input':[[9]],'output':[[9]]}],x)==[expected]
        v,_,h,w=pack_grid(x);inputs={'input':v,'input_height':h,'input_width':w}
        a=Interpreter(graph).run_bundle('adjacency_recolor.gene',inputs,'/gene')
        b=CompiledInterpreter(graph).run_bundle('adjacency_recolor.gene',inputs,'/gene')
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        assert not a['colors'][0,len(x):].any()
        assert not a['colors'][0,:,len(x[0]):].any()
        assert graph.stats()['unique_param_storage']==32


def test_generator_positive_negative_groundtruth_without_model_scoring():
    fixture=builder.generate_adjacency_family(tasks=100,seed=91)
    assert fixture==builder.generate_adjacency_family(tasks=100,seed=91)
    for task in fixture['tasks']:
        for pair in task['train']+task['test']:
            x,y=pair['input'],pair['output'];positive=negative=0
            for r,row in enumerate(x):
                for c,color in enumerate(row):
                    adjacent=any((rr,cc) in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)] for rr,other in enumerate(x) for cc,value in enumerate(other) if value==4)
                    expected=3 if color==2 and adjacent else color
                    assert y[r][c]==expected
                    positive+=color==2 and adjacent;negative+=color==2 and not adjacent
            assert positive and negative
