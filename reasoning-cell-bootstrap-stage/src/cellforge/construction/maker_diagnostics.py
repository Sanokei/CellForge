"""Bounded development train-pair diagnostics with actual reference cell calls."""
from collections import Counter
from ..runtime.interp import Interpreter
from ..runtime.grid import GridPredictor

class _CellTrace(Interpreter):
    def __init__(self, graph):
        super().__init__(graph.copy())
        self.graph_hash=graph.hash();self.steps=[];self.current=None
    def assert_frozen(self):
        if self.g.hash()!=self.graph_hash:raise ValueError('traced graph mutated')
    def run_bundle(self,name,inputs,path=''):
        if name==self.g.entries['grid_step']:
            self.current=dict(phase='query' if float(inputs['phase'][0,0]) else 'demonstration',cells=Counter())
            self.steps.append(self.current)
        return super().run_bundle(name,inputs,path)
    def run_cell(self,name,x):
        if self.current is not None:self.current['cells'][name]+=1
        return super().run_cell(name,x)

class DevelopmentTracingPredictor(GridPredictor):
    def __init__(self,graph):
        super().__init__(graph,backend='reference')
        self._compiled=_CellTrace(graph)  # reference semantics, same adapter reset
    def __call__(self,demonstrations,grid):
        self._compiled.steps=[];self._compiled.current=None
        result=super().__call__(demonstrations,grid)
        self.last_cells=[dict(phase=s['phase'],calls=dict(s['cells'])) for s in self._compiled.steps]
        return result

def train_counterexamples(graph,tasks,limit=5,per_task_limit=None):
    if type(limit) is not int or not 1<=limit<=5:raise ValueError('bounded counterexample limit required')
    if not isinstance(tasks,list) or not 1<=len(tasks)<=100:raise ValueError('bounded development tasks required')
    if per_task_limit is not None and (type(per_task_limit) is not int or not 1<=per_task_limit<=limit):raise ValueError('bounded per-task counterexamples required')
    if any(set(t)-{'train','task_id','family'} for t in tasks):raise ValueError('train-only diagnostics required')
    predictor=DevelopmentTracingPredictor(graph);examples=[];calls=0
    for task in tasks:
        task_examples=0
        for demo in task['train']:
            predictions=predictor(task['train'],demo['input']);calls+=1
            if demo['output'] not in predictions:
                examples.append(dict(input=demo['input'],expected=demo['output'],predicted=predictions,
                    cells_fired=predictor.last_cells,task_id=task.get('task_id'),family=task.get('family')))
                task_examples+=1
                if len(examples)==limit:break
                if per_task_limit is not None and task_examples>=per_task_limit:break
        if len(examples)==limit:break
    return dict(partition='development',evidence='train-pairs',counterexamples=examples,
                graph_hash=graph.hash(),diagnostic_predictor_calls=calls,
                trace_scope='reference cell-definition calls per adaptation/query step; not compiled physical-work or node-path trace')
