import importlib.util
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from cellforge.runtime.grid import GridPredictor,pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

scripts=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('count_builder',scripts/'build_grid_count_strip_gene.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_independent_count_fixtures_empty_ties_padding_cap_and_full_parity():
    graph=builder.build_count_strip_gene()
    capped=[[7 if r*10+c<35 else 0 for c in range(10)] for r in range(10)]
    cases=[([[9,9,1],[9,2,9],[1,9,2]],[[1]*4]),([[8,3]],[[8]]),
           ([[5,5],[5,5]],[[5]]),([[0]],[[0]]),(capped,[[7]*30]),
           ([[2,2,2],[2,8,2],[7,8,2]],[[8]*3])]
    for backend in ('reference','compiled'):
        predict=GridPredictor(graph,backend=backend)
        for x,expected in cases:assert predict([{'input':[[0]],'output':[[0]]}],x)==[expected]
    for x,_ in cases:
        values,_,h,w=pack_grid(x);inputs={'input':values,'input_height':h,'input_width':w}
        a=Interpreter(graph).run_bundle('count_strip.gene',inputs,'/gene')
        b=CompiledInterpreter(graph).run_bundle('count_strip.gene',inputs,'/gene')
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert graph.stats()['unique_param_storage']==11


def test_new_family_generator_groundtruth_independent_no_model_scoring():
    fixture=builder.generate_count_family(tasks=100,seed=81)
    assert fixture==builder.generate_count_family(tasks=100,seed=81)
    for task in fixture['tasks']:
        counts=[]
        for pair in task['train']+task['test']:
            x=pair['input'];assert 3<=len(x)<=9 and 3<=len(x[0])<=9
            histogram=Counter(c for row in x for c in row);background=histogram.most_common(1)[0][0]
            foreground=[c for row in x for c in row if c!=background]
            assert len(foreground)<=20 and histogram[background]>len(foreground)
            assert pair['output']==[[foreground[0]]*len(foreground)]
            counts.append(len(foreground))
        assert len(set(counts))==3 and task['family']=='count_nonbg_to_strip'
