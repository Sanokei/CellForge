import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_stacked_xor_gene import build_stacked_xor_gene,stacked_xor_import_ops
from cellforge.runtime.grid import pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

def predict(g,x):
    packed,_,h,w=pack_grid(x);inputs=dict(input=packed,input_height=h,input_width=w)
    values=[e(g).run_bundle('stacked_xor.gene',inputs) for e in (Interpreter,CompiledInterpreter)]
    for key in values[0]:np.testing.assert_array_equal(values[0][key],values[1][key])
    v=values[0];height=int(v['height'][0].argmax())+1
    return v['colors'][0].argmax(axis=2)[:height,:len(x[0])].tolist()

def test_observed_stacked_train_pair():
    path=Path(__file__).parents[1]/'runs/priority-a-native-diag-v1/775101-maker_repair.json'
    example=next(e for e in json.loads(path.read_text())['counterexamples'] if e['task_id']=='3428a4f5')
    assert predict(build_stacked_xor_gene(),example['input'])==example['expected']

def test_dynamic_shape_and_xor_truth_table():
    g=build_stacked_xor_gene();rng=np.random.default_rng(89753)
    for height,width in ((1,4),(6,5),(14,30)):
        a=rng.integers(0,2,size=(height,width));b=rng.integers(0,2,size=(height,width))
        x=np.concatenate([a*2,np.full((1,width),4),b*7],axis=0)
        assert predict(g,x.tolist())==((a^b)*3).tolist()

def test_declared_entry_and_contextual_import_execute_xor():
    from cellforge.ir.graph import Graph
    from cellforge.runtime.grid import GridPredictor
    from cellforge.construction.transactions import apply_ops
    donor=build_stacked_xor_gene()
    path=Path(__file__).parents[1]/'runs/priority-a-native-diag-v1/775101-maker_repair.json'
    pair=next(e for e in json.loads(path.read_text())['counterexamples'] if e['task_id']=='3428a4f5')
    demos=[dict(input=pair['input'],output=pair['expected'])]
    assert pair['expected'] in GridPredictor(donor,backend='compiled')(demos,pair['input'])
    parent=Graph.load(Path(__file__).parents[1]/'runs/priority-a-batch04-v1/775101-maker_repair-selected')
    child,_=apply_ops(parent,stacked_xor_import_ops(parent,donor,13,'a06_'),resolver=lambda _:donor)
    assert child.defs['expanded_slot_13']['inputs']==['input','input_height','input_width','mode']
    assert pair['expected'] in GridPredictor(child,backend='compiled')(demos,pair['input'])
