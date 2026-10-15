from pathlib import Path
import pytest
from cellforge.ir.graph import Graph
from cellforge.runtime.grid import GridPredictor
from cellforge.construction.maker_diagnostics import DevelopmentTracingPredictor,train_counterexamples

def test_trace_matches_prediction_and_does_not_accumulate():
 g=Graph.load(Path(__file__).parents[1]/'runs/expanded-breed-repair-pilot-v1/breeding_only')
 demos=[{'input':[[1,2]],'output':[[3,4]]}];x=[[2,1]]
 p=DevelopmentTracingPredictor(g);expected=GridPredictor(g,backend='compiled')(demos,x)
 assert p(demos,x)==expected;first=p.last_cells
 assert len(first)==2 and first[-1]['phase']=='query'
 assert any(s['calls'] for s in first)
 assert p(demos,x)==expected and p.last_cells==first
 with pytest.raises(ValueError,match='train-only'):
  train_counterexamples(g,[{'train':demos,'test':[]}])
 failing=[{'input':[[1]],'output':[[8,8]]}]*2
 diverse=train_counterexamples(g,[{'task_id':'a','train':failing},
        {'task_id':'b','train':failing}],limit=2,per_task_limit=1)
 assert [x['task_id'] for x in diverse['counterexamples']]==['a','b']
 assert diverse['diagnostic_predictor_calls']==2
