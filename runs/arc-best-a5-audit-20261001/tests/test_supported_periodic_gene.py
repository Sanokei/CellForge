import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_grid_supported_periodic_gene import build_supported_periodic_gene,build_joint_supported_periodic_gene,supported_periodic_import_ops
from build_grid_periodic_marker_gene import examples
from cellforge.runtime.grid import pack_grid
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.compiled import CompiledInterpreter

def predict(g,x,gene='supported_periodic.gene'):
    packed,_,h,w=pack_grid(x);inputs=dict(input=packed,input_height=h,input_width=w)
    values=[e(g).run_bundle(gene,inputs) for e in (Interpreter,CompiledInterpreter)]
    for key in values[0]:np.testing.assert_array_equal(values[0][key],values[1][key])
    return values[0]['colors'][0].argmax(axis=2)[:len(x),:len(x[0])].tolist()

def test_observed_train_failure_reconstructed():
    path=Path(__file__).parents[1]/'runs/priority-a-native-diag-v1/775101-maker_repair.json'
    example=next(e for e in json.loads(path.read_text())['counterexamples'] if e['task_id']=='cbded52d')
    assert predict(build_supported_periodic_gene(),example['input'])==example['expected']

def test_singletons_and_different_residues_do_not_spread():
    x=np.ones((8,2),dtype=int);x[0,0]=3;x[1,0]=3;x[0,1]=7
    assert predict(build_supported_periodic_gene(),x.tolist())==x.tolist()

def test_joint_cell_preserves_sparse_template_and_corrects_observed_example():
    g=build_joint_supported_periodic_gene()
    pair=examples(987160,1)[0]['train'][0]
    assert predict(g,pair['input'],'joint_supported_periodic.gene')==pair['output']
    path=Path(__file__).parents[1]/'runs/priority-a-native-diag-v1/775101-maker_repair.json'
    pair=next(e for e in json.loads(path.read_text())['counterexamples'] if e['task_id']=='cbded52d')
    assert predict(g,pair['input'],'joint_supported_periodic.gene')==pair['expected']

def test_import_preserves_context_blueprint_and_routes_observed_demo():
    from cellforge.ir.graph import Graph
    from cellforge.construction.transactions import apply_ops
    from cellforge.runtime.grid import GridPredictor
    parent=Graph.load(Path(__file__).parents[1]/'runs/priority-a-batch04-v1/775101-maker_repair-selected')
    donor=build_joint_supported_periodic_gene()
    child,_=apply_ops(parent,supported_periodic_import_ops(parent,donor,9,'a05_'),resolver=lambda _:donor)
    assert child.defs['expanded_slot_9']['inputs']==['input','input_height','input_width','mode']
    pair=next(e for e in json.loads((Path(__file__).parents[1]/'runs/priority-a-native-diag-v1/775101-maker_repair.json').read_text())['counterexamples'] if e['task_id']=='cbded52d')
    assert pair['expected'] in GridPredictor(child,backend='compiled')([dict(input=pair['input'],output=pair['expected'])],pair['input'])
    assert child.stats()['unique_param_storage']<=1024

def test_all_supplied_train_pairs_and_marker_relabeling():
    path=Path(__file__).parents[1]/'runs/priority-a-train-context-v1.json'
    pairs=json.loads(path.read_text())['tasks'][0]['train']
    g=build_joint_supported_periodic_gene(color_adaptive=True)
    for pair in pairs:assert predict(g,pair['input'],'joint_supported_periodic.gene')==pair['output']
    remap={3:6,7:2}
    convert=lambda grid:[[remap.get(c,c) for c in row] for row in grid]
    assert predict(g,convert(pairs[0]['input']),'joint_supported_periodic.gene')==convert(pairs[0]['output'])
