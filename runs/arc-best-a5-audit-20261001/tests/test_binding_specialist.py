import importlib.util
from pathlib import Path
import pytest
from cellforge.ir.graph import Graph,GraphError
from cellforge.runtime.binding import BindingPredictor

spec = importlib.util.spec_from_file_location('binding_builder',Path(__file__).parents[1]/'scripts/build_binding_specialist.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_retrieves_unseen_unicode_and_resets(tmp_path):
    graph = builder.build_binding()
    graph.save(tmp_path)
    predictor = BindingPredictor(Graph.load(tmp_path))
    assert predictor([{'key':'狐','value':'🦊 café'}],'狐') == {'status':'ok','value':'🦊 café','matches':1}
    assert predictor([],'狐')['status']=='missing'


def test_missing_duplicate_and_prefix_are_not_answers():
    p = BindingPredictor(builder.build_binding())
    pairs = [{'key':'ab','value':'one'},{'key':'abc','value':'two'}]
    assert p(pairs,'abc')['value']=='two'
    assert p(pairs,'a')['status']=='missing'
    assert p(pairs+[{'key':'abc','value':'three'}],'abc') == {'status':'ambiguous','value':None,'matches':2}


def test_rejects_overflow_and_bad_types():
    p = BindingPredictor(builder.build_binding())
    with pytest.raises(GraphError):
        p([],'a'*16)
    with pytest.raises(GraphError):
        p([{'key':'x','value':42}],'x')
