"""New depth-three receding lookahead; no changes to the full-horizon template."""
import numpy as np
import skillcore as S
import planning_skill as P

TEMPLATE = 'resource_lookahead3'
NICHENAME = 'resource-lookahead3'
SHAPES = {'reward': (6,), 'action': (4,), 'horizon': (3,), 'beam': (6,)}
schema_rows = P.schema_rows
generated = P.generated
score = P.score


def predict(params, q, demos=None):
    reward, bias = np.asarray(params['reward']), np.asarray(params['action'])
    horizon, width = int(np.argmax(params['horizon'])) + 1, (int(np.argmax(params['beam'])) + 1) * 8
    state, plan = tuple(q['initial']), []
    for _ in range(q['max_steps']):
        if state[2] == q['goal']:
            return {'complete': True, 'steps': len(plan)}

        def utility(item):
            (m, g, d, e), path = item
            return float(reward @ [d / q['goal'], float(d == q['goal']), -len(path),
                                  e / q['max_energy'], m / q['capacity'], g / q['capacity']] + sum(bias[a] for a in path))

        frontier, completed = [(state, [])], []
        for depth in range(horizon):
            expanded = {}
            for current, path in frontier:
                if current[2] == q['goal']:
                    completed.append((current, path))
                    continue
                for action in range(4):
                    nxt = P.transition(current, action, q)
                    if nxt is None:
                        continue
                    item = (nxt, path + [action])
                    if nxt not in expanded or utility(item) > utility(expanded[nxt]):
                        expanded[nxt] = item
            frontier = sorted(expanded.values(), key=utility, reverse=True)[:width]
            if not frontier:
                break
        options = [(s, p) for s, p in completed + frontier if p]
        if not options:
            return {'complete': False, 'steps': len(plan)}
        _, rollout = max(options, key=utility)
        action = rollout[0]
        state = P.transition(state, action, q)
        plan.append(action)
    return {'complete': state[2] == q['goal'], 'steps': len(plan)}


S.TEMPLATES[TEMPLATE] = {'shapes': SHAPES, 'fn': predict, 'aware': False,
    'doc': 'New receding-depth-three resource planning template. Input initial=[materials,goods,delivered,energy], '
           'goal,capacity,max_energy,costs=[acquire,assemble,deliver],max_steps. Actions 0 acquire,1 assemble,2 deliver,3 recharge. '
           'Legal physical transitions enforce inventory, capacity and energy. Output {complete:bool,steps:int} must be shortest completion. '
           'Each step simulates horizon=argmax(horizon)+1 (1..3) actions, beam=(argmax(beam)+1)*8; '
           'score prospective state/path by reward dot [delivered/goal,complete,-rollout_steps,energy/max_energy,materials/capacity,goods/capacity] '
           'plus sum(action[a]) along rollout. Keep max scoring path per state each depth, choose highest scoring terminal or horizon state, '
           'execute its FIRST action, then replan. You must avoid recharge/material/assembly cycles and preserve optimal energy batching. '
           'Weights reward[6],action[4],horizon[3],beam[6] finite [-9,9].'}
