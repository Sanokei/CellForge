"""Generated adjacency niche tasks scored by the real graph runtime."""
import numpy as np
from .niche_archive import default_generators
from .maker_diagnostics import train_counterexamples
from ..runtime.grid import GridPredictor, pack_grid


def adjacency_context(graph, shown_seed, fresh_seed):
    if 'grid_step' not in graph.entries:
        raise ValueError('adjacency adapter requires a grid_step gene; supply a domain adapter')
    def tasks(seed):
        return [{'task_id': gen.niche, 'family': gen.niche,
                 'train': [{'input': t['input'], 'output': t['expected']}
                           for t in gen.sample(seed, 3)]}
                for gen in default_generators()]
    shown, fresh = tasks(shown_seed), tasks(fresh_seed)
    x, mask, h, w = pack_grid([[1]])
    inputs = dict(input=x, input_mask=mask, input_height=h, input_width=w,
                  output=x, output_mask=mask, output_height=h, output_width=w,
                  phase=[[0.]], index=[[0.]], state=np.zeros((1, graph.meta.get('state_dim', 0))))
    fixture = {'entry': graph.entries['grid_step'],
               'inputs': {n: np.asarray(v).tolist() for n, v in inputs.items()},
               'output_shapes': dict(state=[1, graph.meta.get('state_dim', 0)],
                   color_logits=[1, 1, 30, 30, 10], height_logits=[1, 1, 30], width_logits=[1, 1, 30])}
    def evaluate(candidate):
        predictor = GridPredictor(candidate, backend='compiled')
        scores, false_positives = {}, 0
        for task in fresh:
            solved = 0
            for pair in task['train']:
                predictions = predictor(task['train'], pair['input'])
                solved += pair['output'] in predictions
                prediction = predictions[0]
                if len(prediction) != len(pair['input']) or any(len(a) != len(b) for a, b in zip(prediction, pair['input'])):
                    false_positives += 1
                else:
                    false_positives += any(prediction[r][c] != pair['input'][r][c]
                        and pair['output'][r][c] == pair['input'][r][c]
                        for r in range(len(prediction)) for c in range(len(prediction[r])))
            scores[task['family']] = solved
        return {'solved': sum(scores.values()), 'per_niche': scores, 'false_positive_grids': false_positives,
                'tasks': sum(len(t['train']) for t in fresh)}
    def accept(parent, child):
        return (child['solved'] > parent['solved']
                and child['false_positive_grids'] <= parent['false_positive_grids']
                and all(child['per_niche'][n] >= v for n, v in parent['per_niche'].items()))
    return {'diagnostics': train_counterexamples(graph, shown), 'fixtures': [fixture],
            'evaluate': evaluate, 'accept': accept}
