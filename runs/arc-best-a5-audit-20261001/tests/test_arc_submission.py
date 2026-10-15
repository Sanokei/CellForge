import numpy as np
import pytest
from cellforge.ir.graph import Graph
from cellforge.runtime import arc_submission as adapter
from cellforge.runtime.grid import GridPredictor


class LabelSentinel(dict):
    def __getitem__(self,key):
        if key=='output':raise AssertionError('query label accessed')
        return super().__getitem__(key)
    def get(self,key,*args):
        if key=='output':raise AssertionError('query label accessed')
        return super().get(key,*args)


def test_two_attempts_multiquery_no_query_label_or_task_features(monkeypatch):
    seen=[]
    class Predictor:
        attempts=2;last_exec={}
        def __init__(self,graph,backend):assert backend=='compiled'
        def __call__(self,train,query):
            assert all(set(pair)=={'input','output'} for pair in train)
            seen.append((train,query));return [[[1]],[[2]]]
    monkeypatch.setattr(adapter,'GridPredictor',Predictor)
    tasks={'ignored-task-id':{'family':'never graph input','train':[{'input':[[0]],'output':[[1]],'extra':'ignored'}],
                             'test':[LabelSentinel(input=[[3]],output='poison'),LabelSentinel(input=[[4]],output='poison')]}}
    submission,metrics=adapter.run_arc_submission(Graph(),tasks)
    assert submission=={'ignored-task-id':[{'attempt_1':[[1]],'attempt_2':[[2]]}]*2}
    assert len(seen)==2 and metrics['query_count']==2 and metrics['task_count']==1


@pytest.mark.parametrize('bad', [[],[[[1]],[[2]],[[3]]],[[[True]],[[2]]],[[[10]],[[2]]],[[[1],[1,2]],[[2]]]])
def test_invalid_attempts_fail_visibly(monkeypatch,bad):
    class Predictor:
        attempts=2;last_exec={}
        def __init__(self,*args,**kwargs):pass
        def __call__(self,*args):return bad
    monkeypatch.setattr(adapter,'GridPredictor',Predictor)
    with pytest.raises(adapter.ARCSubmissionError):adapter.run_arc_submission(Graph(),{'x':{'train':[{'input':[[0]],'output':[[0]]}],'test':[{'input':[[0]]}]}})


def test_graph_error_is_not_repaired(monkeypatch):
    class Predictor:
        attempts=2;last_exec={}
        def __init__(self,*args,**kwargs):pass
        def __call__(self,*args):raise RuntimeError('graph failure')
    monkeypatch.setattr(adapter,'GridPredictor',Predictor)
    with pytest.raises(RuntimeError,match='graph failure'):adapter.run_arc_submission(Graph(),{'x':{'train':[{'input':[[0]],'output':[[0]]}],'test':[{'input':[[0]]}]}})


def test_actual_small_graph_compiled_reference_and_single_attempt_duplication():
    graph=Graph(defs={'root':{'kind':'bundle','inputs':['input','input_height','input_width','state'],'nodes':[
        {'id':'c','use':'op:reshape','in':{'x':'$input'},'args':{'shape':[1,30,30,10]}},
        {'id':'h','use':'op:reshape','in':{'x':'$input_height'},'args':{'shape':[1,30]}},
        {'id':'w','use':'op:reshape','in':{'x':'$input_width'},'args':{'shape':[1,30]}}],
        'outputs':{'color_logits':'c','height_logits':'h','width_logits':'w','state':'$state'}}},entries={'grid_step':'root'},
        meta={'task':'grid','grid_interface':'grid-graph/1','state_dim':0,'grid_attempts':1})
    train=[{'input':[[0]],'output':[[0]]}];query=[[1,2],[3,4]]
    submission,metrics=adapter.run_arc_submission(graph,{'x':{'train':train,'test':[LabelSentinel(input=query,output='poison')]}})
    expected=GridPredictor(graph)(train,query)[0]
    assert submission['x']==[{'attempt_1':expected,'attempt_2':expected}]
    assert metrics['duplicated_single_attempt_queries']==1
    assert metrics['backend']=='compiled' and metrics['graph_hash']==graph.hash()
    # The actual two-attempt tensor contract also travels through the compiler.
    two=graph.copy();two.meta['grid_attempts']=2;root=two.defs['root']
    for port,ref in list(root['outputs'].items()):
        if port=='state':continue
        name=ref+'_two';root['nodes'].append({'id':name,'use':'op:concat','in':{'x0':ref,'x1':ref},'args':{'axis':0}})
        root['outputs'][port]=name
    submission,metrics=adapter.run_arc_submission(two,{'x':{'train':train,'test':[{'input':query}]}})
    assert submission['x']==[{'attempt_1':expected,'attempt_2':expected}]
    assert metrics['model_attempts']==2 and metrics['duplicated_single_attempt_queries']==0
